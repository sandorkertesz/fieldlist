# (C) Copyright 2022 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import datetime
import logging
import os
import threading
import tempfile
import time

from contextlib import contextmanager

import eccodes
import numpy as np


LOG = logging.getLogger(__name__)


class TmpFile:
    """The TmpFile objects are designed to be used for temporary files.
    It ensures that the file is unlinked when the object is
    out-of-scope (with __del__).
    Parameters
    ----------
    path : str
        Actual path of the file.
    """

    def __init__(self, suffix=".tmp"):
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        self.path = path

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        if self.path is not None:
            os.unlink(self.path)
        self.path = None


def missing_is_none(x):
    return None if x == 2147483647 else x


# This does not belong here, should be in the C library
def get_messages_positions(path):
    fd = os.open(path, os.O_RDONLY)
    try:

        def get(count):
            buf = os.read(fd, count)
            assert len(buf) == count
            return int.from_bytes(
                buf,
                byteorder="big",
                signed=False,
            )

        offset = 0
        while True:
            code = os.read(fd, 4)
            if len(code) < 4:
                break

            if code != b"GRIB":
                offset = os.lseek(fd, offset + 1, os.SEEK_SET)
                continue

            length = get(3)
            edition = get(1)

            if edition == 1:
                if length & 0x800000:
                    sec1len = get(3)
                    os.lseek(fd, 4, os.SEEK_CUR)
                    flags = get(1)
                    os.lseek(fd, sec1len - 8, os.SEEK_CUR)

                    if flags & (1 << 7):
                        sec2len = get(3)
                        os.lseek(fd, sec2len - 3, os.SEEK_CUR)

                    if flags & (1 << 6):
                        sec3len = get(3)
                        os.lseek(fd, sec3len - 3, os.SEEK_CUR)

                    sec4len = get(3)

                    if sec4len < 120:
                        length &= 0x7FFFFF
                        length *= 120
                        length -= sec4len
                        length += 4

            if edition == 2:
                length = get(8)

            yield offset, length
            offset = os.lseek(fd, offset + length, os.SEEK_SET)

    finally:
        os.close(fd)


class GribHandle:
    """The GRIB message handle"""

    MISSING_VALUE = np.finfo(np.float32).max
    _DEFAULT_ACCURACY = 24
    ACCURACY = _DEFAULT_ACCURACY
    KEY_TYPES = {
        "s": str,
        "l": int,
        "d": float,
        "str": str,
        "int": int,
        "float": float,
        "": None,
    }

    def __init__(self, handle, path, offset):
        self.handle = handle
        self.path = path
        self.offset = offset

    def __del__(self):
        # print(f"Handle delete {self.path} {self.offset}")
        eccodes.codes_release(self.handle)

    def get(self, name, key_type=None):
        """Get the value of a given ecCodes key"""

        if name == "values":
            return self.values()

        if key_type is None:
            name, _, key_type_str = name.partition(":")
        try:
            key_type = self.KEY_TYPES[key_type_str]
        except KeyError:
            raise ValueError(f"Key type={key_type_str} not supported")

        try:
            size = eccodes.codes_get_size(self.handle, name)
            if size is not None and size > 1:
                return eccodes.codes_get_array(self.handle, name, key_type)
            return eccodes.codes_get(self.handle, name, key_type)
        except eccodes.KeyValueNotFoundError:
            return None

    def get_long(self, name):
        try:
            return eccodes.codes_get_long(self.handle, name)
        except eccodes.KeyValueNotFoundError:
            return None

    def get_string(self, name):
        try:
            return eccodes.codes_get_string(self.handle, name)
        except eccodes.KeyValueNotFoundError:
            return None

    def get_double(self, name):
        try:
            return eccodes.codes_get_double(self.handle, name)
        except eccodes.KeyValueNotFoundError:
            return None

    def get_double_array(self, name):
        try:
            return eccodes.codes_get_double_array(self.handle, name)
        except eccodes.KeyValueNotFoundError:
            return None

    def get_data(self):
        return eccodes.codes_grib_get_data(self.handle)

    def as_mars(self, param="shortName"):
        r = {}
        it = eccodes.codes_keys_iterator_new(self.handle, "mars")

        try:
            while eccodes.codes_keys_iterator_next(it):
                key = eccodes.codes_keys_iterator_get_name(it)
                r[key] = self.get(param if key == "param" else key)
        finally:
            eccodes.codes_keys_iterator_delete(it)

        return r

    def values(self):
        vals = eccodes.codes_get_values(self.handle)
        if self.get_long("bitmapPresent"):
            missing_value = eccodes.codes_get_double(self.handle, "missingValue")
            vals[vals == missing_value] = np.nan
        return vals

    def set(self, name, value):
        """Set the value of a given ecCodes key"""
        if name == "values":
            self.set_values(value)
        else:
            if not isinstance(value, (np.ndarray, list, tuple)):
                eccodes.codes_set(self.handle, name, value)
            else:
                eccodes.codes_set_array(self.handle, name, value)

    def set_string(self, name, value):
        eccodes.codes_set_string(self.handle, name, value)

    def set_long(self, name, value):
        eccodes.codes_set_long(self.handle, name, value)

    def set_double(self, name, value):
        eccodes.codes_set_double(self.handle, name, value)

    def set_double_array(self, name, value):
        eccodes.codes_set_double_array(self.handle, name, value)

    def best_accuracy(self, accuracy):
        if self.ACCURACY > 0:
            return self.ACCURACY
        elif self.ACCURACY == -1:
            return accuracy
        else:
            return self._DEFAULT_ACCURACY

    def set_values(self, values):
        assert self.path is None, "Only cloned handles can have values changed"

        # set bitsPerValue when needed
        accuracy_current = self.get_long("bitsPerValue")
        accuracy = self.best_accuracy(accuracy_current)
        if accuracy != accuracy_current:
            eccodes.codes_set_long(self.handle, "bitsPerValue", accuracy)

        # replace nans with missing values
        has_missing = np.isnan(np.dot(values, values))
        if has_missing:
            values = np.nan_to_num(values, copy=True, nan=self.MISSING_VALUE)
            eccodes.codes_set_double(self.handle, "missingValue", self.MISSING_VALUE)

        eccodes.codes_set_long(self.handle, "bitmapPresent", int(has_missing))
        eccodes.codes_set_values(self.handle, values.flatten())
        # eccodes.codes_set_long(self.handle, "generatingProcessIdentifier", 254)

    def clone(self):
        return GribHandle(eccodes.codes_clone(self.handle), None, None)

    def write(self, fp, path):
        self.offset = fp.tell()
        eccodes.codes_write(self.handle, fp)
        if path:
            self.path = path

    def save(self, path):
        with open(path, "wb") as f:
            eccodes.codes_write(self.handle, f)
            self.path = path
            self.offset = 0

    def read_bytes(self, offset, length):
        with open(self.path, "rb") as f:
            f.seek(offset)
            return f.read(length)


class GribReader:
    """Represents the GRIB file object. Provides iterator through the GRIB handles"""

    def __init__(self, path):
        """Should not be called directly"""
        self.path = path
        self.lock = threading.Lock()
        self.file = None

        if self.path is not None:
            if not os.path.exists(self.path):
                raise FileNotFoundError(f"{self.path}")

    def open(self):
        if self.file is None:
            self.file = open(self.path, "rb")

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def scan(self):
        for offset, length in get_messages_positions(self.path):
            yield GribField(self, None, offset, length)

    def at_offset(self, offset):
        self.open()
        with self.lock:
            self.last = time.time()
            self.file.seek(offset, 0)
            handle = eccodes.codes_new_from_file(
                self.file,
                eccodes.CODES_PRODUCT_GRIB,
            )
            assert handle is not None
            return GribHandle(handle, self.path, offset)


class TmpGribReader(GribReader):
    def __init__(self):
        self.tmp_file = TmpFile()
        super().__init__(self.tmp_file.path)


class FieldReleaser:
    def __init__(self, field):
        self.field = field

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.field.release()


class GribField:
    def __init__(self, reader, handle=None, offset=None, length=None):
        self.reader = reader
        self._handle = handle
        self._offset = offset
        self._length = length

    # def __enter__(self):
    #     return self

    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     self.release

    @contextmanager
    def expand(self):
        try:
            yield self
        finally:
            self.release()

    @property
    def handle(self):
        if self._handle is None:
            assert self._offset is not None
            assert self.reader is not None
            self._handle = self.reader.at_offset(self._offset)
        return self._handle

    def release(self):
        if self._handle is not None:
            # print(f"{self.__class__.__name__}.release offset={self._offset}")
            self._handle = None

    @property
    def path(self):
        return self.reader.path if self.reader is not None else ""

    @property
    def values(self):
        return self.handle.values()

    @property
    def latitudes(self):
        return self.handle.get_double_array("latitudes")

    @property
    def longitudes(self):
        return self.handle.get_double_array("longitudes")

    # TODO: review it
    @property
    def offset(self):
        if self._offset is None:
            self._offset = int(self.get("offset"))
        return self._offset

    @property
    def shape(self):
        Nj = missing_is_none(self.get("Nj"))
        Ni = missing_is_none(self.get("Ni"))
        if Ni is None or Nj is None:
            return self.get("numberOfDataPoints")
        return (Nj, Ni)

    def __repr__(self):
        return "{}({},{},{},{},{},{})".format(
            self.___class__.__name__,
            self.get("shortName"),
            self.get("levelist"),
            self.get("date"),
            self.get("time"),
            self.get("step"),
            self.get("number"),
        )

    def field_metadata(self):
        m = self._grid_definition()
        for n in ("shortName", "units", "paramId"):
            p = self.get(n)
            if p is not None:
                m[n] = str(p)
        m["shape"] = self.shape
        return m

    def datetime(self):
        date = self.get("date")
        time = self.get("time")
        return datetime.datetime(
            date // 10000, date % 10000 // 100, date % 100, time // 100, time % 100
        )

    def valid_datetime(self):
        step = self.get("endStep")
        return self.datetime() + datetime.timedelta(hours=step)

    def to_datetime_list(self):
        return [self.valid_datetime()]

    def to_numpy(self):
        # return self.values.reshape(self.shape)
        return self.values

    def grib_index(self):
        return (self.handle.path, self.handle.offset)

    def _attributes(self, names):
        return {name: self.get(name) for name in names}

    def __getitem__(self, name):
        return self.get(name)

    def get(self, name):
        if isinstance(name, (list, tuple)):
            return [self.handle.get(k) for k in name]
        else:
            return self.handle.get(name)

    @staticmethod
    def args_to_dict(*args, **kwargs):
        # print(f"args={args}")
        # print(f"kwargs={kwargs}")
        if args:
            if kwargs:
                raise ValueError()
            elif len(args) == 1:
                if not isinstance(args[0], dict):
                    raise ValueError()
                return args[0]
            else:
                raise ValueError()
        elif kwargs:
            return kwargs
        else:
            raise ValueError()

    def set(self, *args, **kwargs):
        vals = self.args_to_dict(*args, **kwargs)
        result = self.handle.clone()
        for k, v in vals.items():
            result.set(k, v)
        return result

    def set_values(self, values):
        result = self.handle.clone()
        result.set_values(values)
        return result

    def clone(self):
        return GribField(None, handle=self.handle.clone())

    def write(self, fp, path):
        self.handle.write(fp, path)
        # if self.path is None:
        #     self.path = path
        # if self._offset is None:
        #     self._offset = self.handle.offset

    @staticmethod
    def _compute_1(func, f):
        result = f.handle.clone()
        result.set_values(func(f.values))
        return result

    @staticmethod
    def _compute_2(func, f1, f2):
        result = None
        if isinstance(f1, GribField):
            result = f1.handle.clone()
            f1 = f1.values
        if isinstance(f2, GribField):
            if result is None:
                result = f2.handle.clone()
            f2 = f2.values

        result.set_values(func(f1, f2))
        return result

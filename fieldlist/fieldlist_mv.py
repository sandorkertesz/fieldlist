# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import datetime
import glob
import logging
import os
import tempfile
import threading
import time

from contextlib import contextmanager
from inspect import signature

import eccodes
import numpy as np
import pandas as pd

from fieldlist.codes import GribField, GribReader, TmpGribReader
from fieldlist import maths


# from .._dev import utils
# from .._dev.indexer import FieldListIndex


# from reader import Reader

LOG = logging.getLogger(__name__)

LS_KEYS = [
    "centre",
    "shortName",
    "typeOfLevel",
    "level",
    "dataDate",
    "dataTime",
    "stepRange",
    "dataType",
    "number",
    "gridType",
]

DESCRIBE_KEYS = [
    "shortName"
    "typeOfLevel"
    "level"
    "date"
    "time"
    "step"
    "number"
    "paramId"
    "marsClass"
    "marsStream"
    "marsType"
    "experimentVersionNumber"
]


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


def get_file_list(path):
    return sorted(glob.glob(path))


# decorator to implement math functions for FieldList
def wrap_maths(cls):

    WRAP_MATHS_ATTRS = {
        "abs": maths.abs,
        "acos": maths.acos,
        "asin": maths.asin,
        "atan": maths.atan,
        "atan2": maths.atan2,
        "cos": maths.cos,
        "div": maths.floor_div,
        "exp": maths.exp,
        "log": maths.log,
        "log10": maths.log10,
        "mod": maths.mod,
        "sgn": maths.sgn,
        "sin": maths.sin,
        "square": maths.square,
        "sqrt": maths.sqrt,
        "tan": maths.tan,
        "__neg__": maths.neg,
        "__pos__": maths.pos,
        "__invert__": maths.not_func,
        "__add__": maths.add,
        "__radd__": (maths.add, "r"),
        "__sub__": maths.sub,
        "__rsub__": (maths.sub, "r"),
        "__mul__": maths.mul,
        "__rmul__": (maths.mul, "r"),
        "__truediv__": maths.div,
        "__rtruediv__": (maths.div, "r"),
        "__pow__": maths.pow,
        "__rpow__": (maths.pow, "r"),
        "__ge__": maths.ge,
        "__gt__": maths.gt,
        "__le__": maths.le,
        "__lt__": maths.lt,
        "__eq__": maths.eq,
        "__ne__": maths.ne,
        "__and__": maths.and_func,
        "__or__": maths.or_func,
        "bitmap": (maths.bitmap, {"use_first_from_other": True}),
        "nobitmap": maths.nobitmap,
    }

    def wrap_single_method(fn):
        def wrapper(self):
            return cls.computeA(fn, self)

        return wrapper

    def wrap_double_method(fn):
        def wrapper(self, other):
            return cls.compute_2A(fn, self, other)

        return wrapper

    def wrap_double_method_r(fn):
        def wrapper(self, other):
            return cls.compute_2A(fn, other, self)

        return wrapper

    for name, it in WRAP_MATHS_ATTRS.items():
        if not isinstance(it, tuple):
            it = (it, {})
        fn, opt = it
        n = len(signature(fn).parameters)
        if n == 1:
            setattr(cls, name, wrap_single_method(fn))
        elif n == 2:
            if opt != "r":
                setattr(cls, name, wrap_double_method(fn))
            else:
                setattr(cls, name, wrap_double_method_r(fn))
    return cls


@wrap_maths
class FieldList:
    def __init__(self, paths=None, fields=[]):
        self._fields = []
        self._indexer = None

        if isinstance(fields, GribField):
            self._fields = [fields]
        else:
            self._fields = [f for f in fields]

        if paths is not None:
            if not isinstance(paths, (list, tuple)):
                paths = [paths]
            for p in paths:

                for fp in get_file_list(p):
                    reader = GribReader(fp)
                    for f in reader.scan():
                        self._fields.append(f)

    @classmethod
    def from_tmp_handles(cls, handles):
        fields = []
        reader = TmpGribReader()
        with open(reader.path, "wb") as fp:
            for handle in handles:
                handle.write(fp, reader.path)
                length = fp.tell() - handle.offset
                # handle.release()
                f = GribField(reader, handle=None, offset=handle.offset, length=length)
                fields.append(f)

        return cls(fields=fields)

    def __getitem__(self, index):
        if isinstance(index, (np.ndarray, list)):
            return FieldList(fields=[self._fields[i] for i in index])
        elif isinstance(index, str):
            return self.get(index)
        elif isinstance(index, int):
            return self._fields[index]
        else:
            return FieldList(fields=self._fields[index])

    # def __setitem__(self, index, value):
    #     if isinstance(index, int):
    #         if isinstance(value, FieldList):
    #             if len(value) == 1:
    #                 self._fields[index] = value[0]
    #             else:
    #                 raise ValueError(f"assigned FieledsList ")
    #     else:
    #         result = []
    #         for field in self:
    #             field[index] = value

    def __len__(self):
        return len(self._fields)

    def append(self, other):
        if isinstance(other, GribField):
            self._fields.append(other)
        else:
            self._fields = self._fields + other._fields

    def merge(self, other):
        result = FieldList(fields=self._fields)
        result.append(other)
        return result

    def _attributes(self, names):
        result = []
        for f in self:
            with f.expand() as f:
                result.append(f._attributes(names))
        return result

    # def to_xarray(self):
    #     assert self._unfiltetered
    #     assert self.path

    #     import xarray as xr

    #     params = self.source.cfgrib_options()
    #     if isinstance(self.path, (list, tuple)):
    #         ds = xr.open_mfdataset(self.path, engine="cfgrib", **params)
    #     else:
    #         ds = xr.open_dataset(self.path, engine="cfgrib", **params)
    #     return self.source.post_xarray_open_dataset_hook(ds)

    # def _apply(self, func_name, *args, **kwargs):
    #     for f in self:
    #         with f.expand() as f:
    #             yield getattr(f, func_name)(*args, **kwargs)

    def set(self, *args, **kwargs):
        print(f"args={args}")
        print(f"kwargs={kwargs}")

        def _set():
            for f in self._fields:
                with f.expand() as f:
                    yield f._set_handle(*args, **kwargs)

        return FieldList.from_tmp_handles(_set())

    def get(self, keys, group="field"):
        if group not in ["field", "key"]:
            raise ValueError(f"get: invalid group={group}. Must be field or key")
        result = []
        for f in self._fields:
            with f.expand() as f:
                result.append(f.get(keys))
        if group == "key":
            result = list(map(list, zip(*result)))  # transpose lists of lists
        return result

    def write(self, path):
        with open(path, "wb") as out:
            for f in self._fields:
                f.write(out, path)

    def _get_indexer(self):
        if self._indexer is None:
            self._indexer = FieldListIndex(self)
        assert self._indexer is not None
        return self._indexer

    def sel(self, **kwargs):
        options = kwargs
        self._get_indexer().update(list(options.keys()))

        fs = FieldList()
        s = self._get_indexer().subindex(filter_by_keys=options)
        items = []
        for f, item in self._get_indexer().subindex(filter_by_keys=options).fields():
            fs.append(f)
            items.append(item)
        fs._indexer = FieldListIndex(fs, index_keys=s.index_keys, items=items)
        return fs

    def head(self, rows=5):
        return self.ls().head(rows)

    def tail(self, rows=5):
        return self.ls().tail(rows)

    def ls(self, extra_keys=None, filter=None, no_print=False):
        keys = list(LS_KEYS)
        extra_keys = [] if extra_keys is None else extra_keys
        if extra_keys is not None:
            [keys.append(x) for x in extra_keys if x not in keys]

        self._get_indexer().update(keys)

        filter = {} if filter is None else filter
        if filter:
            df = pd.DataFrame(
                self._get_indexer().subindex(filter_by_keys=filter).metadata(keys)
            )
        else:
            df = pd.DataFrame(self._get_indexer().metadata(keys))

        return utils.process_ls(df, no_print)

    def describe(self, *args, **kwargs):
        keys = DESCRIBE_KEYS
        self._get_indexer().update(keys)
        df = pd.DataFrame(self._get_indexer().metadata(keys))

        param = args[0] if len(args) == 1 else None
        if param is None:
            param = kwargs.pop("param", None)

        return utils.process_describe(df, param=param, **kwargs)

    def count_open_handles(self):
        cnt = 0
        for f in self._fields:
            if f._handle is not None:
                cnt += 1
        return cnt

    @staticmethod
    def _make_2d_array(v):
        """Forms a 2D ndarray from a list of 1D ndarrays"""
        v = FieldList._list_or_single(v)
        return np.stack(v, axis=0) if isinstance(v, list) else v

    @property
    def values(self):
        result = []
        for f in self._fields:
            with f.expand() as f:
                result.append(f.values)
        return np.stack(result, axis=0)

    def set_values(self, values):
        def _set():
            for f, v in zip(self._fields, values):
                with f.expand() as f:
                    yield f.set_values(v)

        if len(values) != len(self):
            raise ValueError(
                f"set_values: number 1d-arrays={len(values)} does not match the number of fields={len(self)}"
            )

        return FieldList.from_tmp_handles(_set())

    def _first_path(self):
        """For debugging purposes"""
        if len(self) > 0:
            return self._fields[0].reader.path
        return ""

    def compute(self, func):
        return self.computeA(func, self)

    @classmethod
    def computeA(cls, func, fl):
        def _compute():
            for f in fl:
                with f.expand() as f:
                    yield GribField._compute_handle(func, f)

        return cls.from_tmp_handles(_compute())

    @classmethod
    def compute_2A(cls, func, fl1, fl2):
        def _compute_2(d1, d2):
            for f1, f2 in zip(d1, d2):
                with f1.expand() as f1:
                    with f2.expand() as f2:
                        yield GribField._compute_2_handle(func, f1, f2)

        def _compute_2_left(d1, d2):
            for f in d1:
                with f.expand() as f:
                    yield GribField._compute_2_handle(func, f, d2)

        def _compute_2_right(d1, d2):
            for f in d2:
                with f.expand() as f:
                    yield GribField._compute_2_handle(func, d1, f)

        if not (isinstance(fl1, FieldList) or isinstance(fl2, FieldList)):
            raise TypeError("")

        if isinstance(fl1, FieldList) and isinstance(fl2, FieldList):
            if len(fl1) == len(fl2):
                return cls.from_tmp_handles(_compute_2(fl1, fl2))
            elif len(fl1) == 1:
                return cls.from_tmp_handles(_compute_2_left(fl1, fl2))
            elif len(fl2) == 1:
                return cls.from_tmp_handles(_compute_2_right(fl1, fl2))
            else:
                raise Exception(
                    f"FieldLists must have the same number of fields for this operation! {len(fl1)} != {len(fl2)}"
                )
        else:
            if isinstance(fl1, FieldList):
                return cls.from_tmp_handles(_compute_2_left(fl1, fl2))
            elif isinstance(fl2, FieldList):
                return cls.from_tmp_handles(_compute_2_right(fl1, fl2))
            else:
                raise ValueError("Invalid arguments")

    # def __add__(self, other):
    #     return FieldList.compute_2(maths.add, self, other)

    # def __sub__(self, other):
    #     return FieldList.compute_2(maths.sub, self, other)

    # def __radd__(self, other):
    #     return FieldList._compute_2(maths.add, other, self)

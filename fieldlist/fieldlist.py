# (C) Copyright 2022 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import glob
import logging

import numpy as np
import pandas as pd

from fieldlist.codes import GribField, GribReader, TmpGribReader
from fieldlist.compute import wrap_maths
from fieldlist import utils
from fieldlist.indexer import FieldListDb


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
    "shortName",
    "typeOfLevel",
    "level",
    "date",
    "time",
    "step",
    "number",
    "paramId",
    "marsClass",
    "marsStream",
    "marsType",
    "experimentVersionNumber",
]

DEFAULT_SORT_KEYS = ["date", "time", "step", "number", "level", "paramId"]


def get_file_list(path):
    return sorted(glob.glob(path))


@wrap_maths
class FieldList:
    def __init__(self, paths=None, fields=[]):
        self._fields = []
        self._db = FieldListDb(self)

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
            return FieldList(fields=self._fields[index])
        else:
            return FieldList(fields=self._fields[index])

    def __len__(self):
        return len(self._fields)

    def append(self, other):
        if isinstance(other, GribField):
            self._fields.append(other)
        else:
            self._fields = self._fields + other._fields
        self._db = FieldListDb(self)

    def merge(self, other):
        result = FieldList(fields=self._fields)
        result.append(other)
        return result

    def _attributes(self, names):
        return self._collect(GribField._attributes, names)

    # used by cfgrib
    def items(self):
        for i, f in enumerate(self):
            yield (i, f)

    def to_numpy(self):
        return self.values

    def to_xarray(self, **kwarg):
        # soft dependency on cfgrib
        try:
            import xarray as xr
        except ImportError:
            print("Package xarray not found. Try running 'pip install xarray'.")
            raise
        return xr.open_dataset(self, engine="cfgrib", backend_kwargs=kwarg)

    def _make_new(self, func, *args, **kwargs):
        def _call():
            for f in self._fields:
                with f.manage_handle():
                    yield func(f, *args, **kwargs)

        return FieldList.from_tmp_handles(_call())

    def _make_new_each(self, func, field_args, *args, **kwargs):
        def _call():
            for f, x in zip(self._fields, field_args):
                with f.manage_handle():
                    yield func(f, x, *args, **kwargs)

        return FieldList.from_tmp_handles(_call())

    def _collect(self, func, *args, **kwargs):
        result = []
        for f in self._fields:
            with f.manage_handle():
                if callable(func):
                    result.append(func(f, *args, **kwargs))
                else:
                    result.append(getattr(f, func))
        return result

    def set(self, *args, **kwargs):
        return self._make_new(GribField.set, *args, **kwargs)

    def get(self, keys, group="field"):
        if group not in ["field", "key"]:
            raise ValueError(f"get: invalid group={group}. Must be field or key")

        result = self._collect(GribField.get, keys)

        if group == "key":
            result = list(map(list, zip(*result)))  # transpose lists of lists
        if isinstance(keys, (list, tuple)):
            return result
        return utils.list_or_single_value(result)

    def write(self, path):
        with open(path, "wb") as out:
            for f in self._fields:
                f.write(out, path)

    def sel(self, *args, **kwargs):
        options = GribField.args_to_dict(*args, **kwargs)
        fields = []
        md = []
        for idx, md_item in self._db.filter(options):
            fields.append(self._fields[idx])
            md.append(md_item)
        res = FieldList(fields=fields)
        res._db = FieldListDb(res, items=md, index_keys=self._db.index_keys)
        return res

    def head(self, n=5):
        if n <= 0:
            raise ValueError("n must be > 0")
        num = len(self)
        if num > 0:
            return self[: min(num + 1, n + 1) :].ls()
        return None

    def tail(self, n=5):
        if n <= 0:
            raise ValueError("n must be > 0")
        num = len(self)
        if num > 0:
            return self[-min(num, n) :].ls()
        return None

    def ls(self, extra_keys=None, filter=None, no_print=False):
        keys = list(LS_KEYS)
        extra_keys = [] if extra_keys is None else extra_keys
        if extra_keys is not None:
            [keys.append(x) for x in extra_keys if x not in keys]

        filter = {} if filter is None else filter
        df = pd.DataFrame.from_records(self._db.metadata(keys, filter_by_keys=filter))
        return utils.process_ls(df, no_print)

    def describe(self, *args, **kwargs):
        keys = DESCRIBE_KEYS
        df = pd.DataFrame.from_records(self._db.metadata(keys))

        param = args[0] if len(args) == 1 else None
        if param is None:
            param = kwargs.pop("param", None)

        return utils.process_describe(df, param=param, **kwargs)

    def sort(self, keys=[], ascending=True):
        # handle arguments
        asc = True
        if keys is None:
            keys = []
        if not isinstance(keys, (list, tuple)):
            keys = [keys]

        if len(keys) == 0:
            keys = DEFAULT_SORT_KEYS

        # print(f"keys={keys} asc={asc}")
        # get metadata
        df = pd.DataFrame.from_records(self._db.metadata(keys))
        if df is not None:
            dfs = utils.sort_dataframe(df, columns=keys, ascending=ascending)
            fields = []
            for idx in dfs.index:
                fields.append(self._fields[idx])
            res = FieldList(fields=fields)
            return res

        return FieldList()

    def count_open_handles(self):
        cnt = 0
        for f in self._fields:
            if f._handle is not None:
                cnt += 1
        return cnt

    @staticmethod
    def _make_2d_array(v):
        """Forms a 2D ndarray from a list of 1D ndarrays"""
        v = utils.list_or_single_value(v)
        return np.stack(v, axis=0) if isinstance(v, list) else v

    @property
    def values(self):
        return self._make_2d_array(self._collect("values"))

    def set_values(self, values):
        if isinstance(values, list):
            list_of_arrays = values
        else:
            if len(values.shape) > 1:
                list_of_arrays = [a for a in values]
            else:
                list_of_arrays = [values]
        if len(list_of_arrays) != len(self):
            raise ValueError(
                f"set_values: number of 1d-arrays={len(values)} does not match the number of fields={len(self)}"
            )
        return self._make_new_each(GribField.set_values, list_of_arrays)

    def _first_path(self):
        """For debugging purposes"""
        if len(self) > 0:
            return self._fields[0].path
        return ""

    def compute(self, func, *args):
        if len(args) == 0:
            return self._compute_1(func, self)
        elif len(args) == 1:
            return self._compute_2(func, self, args[0])
        raise ValueError("compute")

    def accumulate(self):
        result = []
        for f in self._fields:
            with f.manage_handle():
                result.append(np.nansum(f.values))
        return result

    def sum(self):
        if len(self._fields) > 0:
            f0 = self._fields[0]
            with f0.manage_handle():
                v = f0.values

            for i in range(1, len(self._fields)):
                f = self._fields[i]
                with f.manage_handle():
                    v += f.values

            return self.from_tmp_handles([f0.set_values(v)])
        return None

    @classmethod
    def _compute_1(cls, func, fl):
        def _compute():
            for f in fl._fields:
                with f.manage_handle():
                    yield GribField._compute_1(func, f)

        return cls.from_tmp_handles(_compute())

    @classmethod
    def _compute_2(cls, func, fl1, fl2):
        def _compute(d1, d2):
            for f1, f2 in zip(d1._fields, d2._fields):
                with f1.manage_handle():
                    with f2.manage_handle():
                        yield GribField._compute_2(func, f1, f2)

        def _compute_left(d1, d2):
            for f in d1._fields:
                with f.manage_handle():
                    yield GribField._compute_2(func, f, d2)

        def _compute_right(d1, d2):
            for f in d2._fields:
                with f.manage_handle():
                    yield GribField._compute_2(func, d1, f)

        if not (isinstance(fl1, FieldList) or isinstance(fl2, FieldList)):
            raise TypeError("")

        if isinstance(fl1, FieldList) and isinstance(fl2, FieldList):
            if len(fl1) == len(fl2):
                return cls.from_tmp_handles(_compute(fl1, fl2))
            elif len(fl1) == 1:
                return cls.from_tmp_handles(_compute_left(fl1, fl2))
            elif len(fl2) == 1:
                return cls.from_tmp_handles(_compute_right(fl1, fl2))
            else:
                raise Exception(
                    f"FieldLists must have the same number of fields for this operation! {len(fl1)} != {len(fl2)}"
                )
        else:
            if isinstance(fl1, FieldList):
                return cls.from_tmp_handles(_compute_left(fl1, fl2))
            elif isinstance(fl2, FieldList):
                return cls.from_tmp_handles(_compute_right(fl1, fl2))
            else:
                raise ValueError("Invalid arguments")

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
import sys

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

    def tail(self, n=5):
        if n <= 0:
            raise ValueError("n must be > 0")
        num = len(self)
        if num > 0:
            return self[-min(num, n) :].ls()

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

        def _call():
            for f, x in zip(self._fields, list_of_arrays):
                with f.manage_handle():
                    yield f.set_values(x)

        return FieldList.from_tmp_handles(_call())

    def _first_path(self):
        """For debugging purposes"""
        if len(self) > 0:
            return self._fields[0].path
        return ""

    def compute(self, func, *args):
        return self._compute_any(func, self, *args)

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
    def _compute_any(cls, func, *args):
        def _comp():
            # TODO: ensure single fieldlists are only expanded once
            for it in zip(*x):
                # print("next")
                # for h in it:
                #     print(f" {type(h)}")
                z = [v.values for v in it]
                c = it[template_idx].set_values(func(*z))
                for v in it:
                    v.release()
                yield c

        num = 0
        template_idx = None

        for i, v in enumerate(args):
            if isinstance(v, FieldList):
                num = max(num, len(v))

        x = []
        for i, item in enumerate(args):
            if isinstance(item, FieldList):
                if len(item) == num:
                    x.append(MultiFLIterVar(item, num))
                elif len(item) == 1:
                    x.append(SingleFLIterVar(item, num))
                else:
                    raise ValueError(
                        f"Wrong number of fields={len(item)} in positional arg {i}. FieldLists must have the same number of fields={num} for this operation or they should contain a single field!"
                    )
                if template_idx is None:
                    template_idx = i
            else:
                x.append(ScalarIterVar(item, num))

        return cls.from_tmp_handles(_comp())


# expose all FieldList functions as a module level function
def _make_module_func(name):
    def wrapped(fs, *args):
        return getattr(fs, name)(*args)

    return wrapped


module_obj = sys.modules[__name__]
for fn in dir(FieldList):
    if callable(getattr(FieldList, fn)) and not fn.startswith("_") and fn != "compute":
        setattr(module_obj, fn, _make_module_func(fn))


def bind_functions(namespace, module_name=None):
    """Add to the module globals all FieldList functions except operators like: +, &, etc."""
    namespace["compute"] = compute
    for fn in dir(FieldList):
        if (
            callable(getattr(FieldList, fn))
            and not fn.startswith("_")
            and fn != "compute"
        ):
            namespace[fn] = _make_module_func(fn)
        namespace["FieldList"] = FieldList


def compute(func, *args):
    return FieldList._compute_any(func, *args)


class ScalarWrapped:
    def __init__(self, v):
        self.values = v

    def release(self):
        pass


class IterVar:
    def __init__(self, f, num):
        self.f = f
        self.num = num

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        raise NotImplementedError()


class SingleFLIterVar(IterVar):
    def __init__(self, *args):
        super().__init__(*args)

    def __getitem__(self, idx):
        return self.f._fields[0]


class MultiFLIterVar(IterVar):
    def __init__(self, *args):
        super().__init__(*args)

    def __getitem__(self, idx):
        return self.f._fields[idx]


class ScalarIterVar(IterVar):
    def __init__(self, *args):
        super().__init__(*args)
        self.w = ScalarWrapped(self.f)

    def __getitem__(self, idx):
        return self.w

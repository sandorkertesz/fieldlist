# (C) Copyright 2022 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from inspect import signature

from . import maths

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

# # decorator to implement math functions for FieldList
def wrap_maths(cls):
    def wrap_single_method(fn):
        def wrapper(self):
            return cls._compute_any(fn, self)

        return wrapper

    def wrap_double_method(fn):
        def wrapper(self, other):
            return cls._compute_any(fn, self, other)

        return wrapper

    def wrap_double_method_r(fn):
        def wrapper(self, other):
            return cls._compute_any(fn, other, self)

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

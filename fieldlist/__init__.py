# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

# from .fieldlist import FieldList

from . import fieldlist

fieldlist.bind_functions(globals(), module_name=__name__)

from .codes import set_write_accuracy

# __all__ = [
#     "FieldList",
# ]


from .version import __version__

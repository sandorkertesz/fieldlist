# (C) Copyright 2022 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

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


class FieldListDb:
    def __init__(self, data, items=None, index_keys=None):
        self.data = data
        self.items = items if items is not None else []
        self.index_keys = list(LS_KEYS) if index_keys is None else index_keys

    @property
    def empty(self):
        return len(self.items) == 0 if self.items else True

    def update(self, keys):
        if self._add_keys(keys) or len(self.items) == 0:
            self._scan()

    def _scan(self):
        self.items = self.data._attributes(self.index_keys)

    def _valid(self, item, filter_by_keys):
        for k, v in filter_by_keys.items():
            if isinstance(v, (list, tuple)):
                if item.get(k) not in v:
                    return False
            elif item.get(k) != v:
                return False
        return True

    def filter(self, filter_by_keys):
        self.update(list(filter_by_keys.keys()))

        for i, v in enumerate(self.items):
            if self._valid(v, filter_by_keys):
                yield i, v

    def metadata(self, keys=[], filter_by_keys={}):
        if keys:
            self.update(keys)
            if filter_by_keys:
                for _, item in self.filter(filter_by_keys):
                    yield {k: item.get(k, None) for k in keys}

            else:
                for item in self.items:
                    yield {k: item.get(k, None) for k in keys}

        else:
            return self._metadata(filter_by_keys)

    def _metadata(self, filter_by_keys={}):
        if filter_by_keys:
            for _, item in self.filter(filter_by_keys):
                yield item
        else:
            for item in self.items:
                yield item

    def _add_keys(self, keys):
        n = len(self.index_keys)
        for k in keys:
            if k not in self.index_keys:
                self.index_keys.append(k)
        return n < len(self.index_keys)

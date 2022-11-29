import datetime
import functools

import pandas as pd

# tuple-> 0: ecCodes type, 1: pandas type, 2: Python type 3: use in duplicate check
DEFAULT_KEYS = {
    "shortName": ("s", str, str, False),
    "paramId": ("l", "Int32", int, False),
    "date": ("l", "Int64", int, True),
    "time": ("l", "Int64", int, True),
    "step": ("l", "Int32", int, True),
    "level": ("l", "Int32", int, True),
    "typeOfLevel": ("s", str, str, False),
    "number": ("s", str, str, True),
    "experimentVersionNumber": ("s", str, str, False),
    "marsClass": ("s", str, str, False),
    "marsStream": ("s", str, str, False),
    "marsType": ("s", str, str, False),
}

DATE_KEYS = {
    k: ("l", "Int64", int)
    for k in ["date", "dataDate", "validityDate", "mars.date", "marsDate"]
}

TIME_KEYS = {
    k: ("l", "Int64", int)
    for k in ["time", "dataTime", "validityTime", "mars.time", "marsTime"]
}

DATETIME_KEYS = {
    "_dateTime": ("date", "time"),
    "_dataDateTime": ("dataDate", "dataTime"),
    "_validityDateTime": ("validityDate", "validityTime"),
}

KEYS_TO_REPLACE = {
    ("type", "mars.type"): "marsType",
    ("stream", "mars.stream"): "marsStream",
    ("class", "mars.class", "class_"): "marsClass",
    ("perturbationNumber"): "number",
    ("mars.date", "marsDate"): "date",
    ("mars.time", "marsTime"): "time",
}


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
        print(f"SCAN {self.index_keys}")
        self.items = self.data._attributes(self.index_keys)

    def _valid(self, item, filter_by_keys):
        for k, v in filter_by_keys.items():
            if isinstance(v, (list, tuple)):
                if item.get(k) not in v:
                    return False
            elif item.get(k) != v:
                return False
        return True

    # def _valid_items(self):
    #     if self.items is None:
    #         self.items = self.source._attributes(self.index_keys)

    #     return [i for i in self.items if self._valid(i)]

    def filter(self, filter_by_keys):
        print(f"Filter by keys={filter_by_keys}")
        self.update(list(filter_by_keys.keys()))

        for i, v in enumerate(self.items):
            if self._valid(v, filter_by_keys):
                yield i, v

    def metadata(self, keys=[], filter_by_keys={}):
        # print(f"keys={keys}")
        if keys:
            self.update(keys)
            if filter_by_keys:
                for _, item in self.filter(filter_by_keys):
                    yield {k: item.get(k, None) for k in keys}

            else:
                for item in self.items:
                    # print(self.index_keys)
                    # print(f"RES", {k: item.get(k, None) for k in keys})
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


class FieldListIndex:
    def __init__(
        self,
        source,
        *,
        filter_by_keys={},
        grib_errors="warn",
        index_keys=[],
        read_keys=[],
        items=None,
    ):
        self.source = source
        self.filter_by_keys = filter_by_keys
        self.grib_errors = grib_errors
        self.index_keys = index_keys
        self.read_keys = read_keys
        self.items = items

    def __call__(self, *, grib_errors, index_keys, read_keys, **kwargs):
        return FieldListIndex(
            source=self.source,
            filter_by_keys=self.filter_by_keys,
            grib_errors=grib_errors,
            index_keys=index_keys,
            read_keys=read_keys,
            items=self.items,
        )

    def subindex(self, filter_by_keys={}, **kwargs):
        query = dict(**self.filter_by_keys)
        query.update(filter_by_keys)
        query.update(kwargs)
        return FieldListIndex(
            source=self.source,
            filter_by_keys=query,
            grib_errors=self.grib_errors,
            index_keys=self.index_keys,
            read_keys=self.read_keys,
            items=self.items,
        )

    def update(self, keys):
        if self._add_keys(keys) or self.items is None:
            print("CALL")
            self.items = self.source._attributes(self.index_keys)

    def _valid(self, item):
        for k, v in self.filter_by_keys.items():
            if item.get(k) != v:
                return False
        return True

    def _valid_items(self):
        if self.items is None:
            self.items = self.source._attributes(self.index_keys)

        return [i for i in self.items if self._valid(i)]

    def __getitem__(self, key):

        x = [i[key] for i in self._valid_items()]
        if None in x:
            return []

        print("GET", key, x)
        return list(set(x))

    def getone(self, key):
        return self[key][0]

    @property
    def offsets(self):
        for i, item in enumerate(self._valid_items()):
            yield item, i

    def metadata(self, keys=[]):
        if keys:
            for item in self._valid_items():
                yield {k: item.get(k, None) for k in keys}
        else:
            for item in self._valid_items():
                yield item

    @property
    def filestream(self):
        return self

    @property
    def path(self):
        return None

    def first(self):
        return Field(self.source[0])

    def fields(self):
        if self.items is None:
            self.items = self.source._attributes(self.index_keys)

        for i, v in enumerate(self.items):
            if self._valid(v):
                yield self.source[i], v

    def _add_keys(self, keys):
        n = len(self.index_keys)
        for k in keys:
            if k not in self.index_keys:
                self.index_keys.append(k)
        return n < len(self.index_keys)


def ls(fs, extra_keys=None, filter=None, no_print=False):
    keys = list(LS_KEYS)
    extra_keys = [] if extra_keys is None else extra_keys
    if extra_keys is not None:
        [keys.append(x) for x in extra_keys if x not in keys]

    filter = {} if filter is None else filter
    idx = FieldListIndex(fs, index_keys=keys, filter_by_keys=filter)
    df = pd.DataFrame(idx.metadata(keys))

    # only show the column for number in the default set of keys if
    # there are any valid values in it
    if "number" not in extra_keys:
        r = df["number"].unique()
        if len(r) == 1 and r[0] in ["0", None]:
            df.drop("number", axis=1, inplace=True)

    # init_pandas_options()

    # test whether we're in the Jupyter environment
    if is_ipython_active():
        return df
    elif not no_print:
        print(df)
    return df


# taken from eccodes stepUnits.table
GRIB_STEP_UNITS_TO_SECONDS = [
    60,
    3600,
    86400,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    10800,
    21600,
    43200,
    1,
    900,
    1800,
]
DEFAULT_EPOCH = datetime.datetime(1970, 1, 1)


def from_grib_date_time(
    message, date_key="dataDate", time_key="dataTime", epoch=DEFAULT_EPOCH
):
    """
    Return the number of seconds since the ``epoch`` from the values of the ``message`` keys,
    using datetime.total_seconds().

    :param message: the target GRIB message
    :param date_key: the date key, defaults to "dataDate"
    :param time_key: the time key, defaults to "dataTime"
    :param epoch: the reference datetime
    """
    date = message[date_key]
    time = message[time_key]
    hour = time // 100
    minute = time % 100
    year = date // 10000
    month = date // 100 % 100
    day = date % 100
    data_datetime = datetime.datetime(year, month, day, hour, minute)
    # Python 2 compatible timestamp implementation without timezone hurdle
    # see: https://docs.python.org/3/library/datetime.html#datetime.datetime.timestamp
    return int((data_datetime - epoch).total_seconds())


def to_grib_date_time(
    message, time_ns, date_key="dataDate", time_key="dataTime", epoch=DEFAULT_EPOCH
):
    time_s = int(time_ns) * 1e-9
    time = epoch + datetime.timedelta(seconds=time_s)
    datetime_iso = str(time)
    message[date_key] = int(datetime_iso[:10].replace("-", ""))
    message[time_key] = int(datetime_iso[11:16].replace(":", ""))


def from_grib_step(message, step_key="endStep", step_unit_key="stepUnits"):
    step_unit = message[step_unit_key]
    to_seconds = GRIB_STEP_UNITS_TO_SECONDS[step_unit]
    if to_seconds is None:
        raise ValueError("unsupported stepUnit %r" % step_unit)
    assert isinstance(to_seconds, int)  # mypy misses this
    return int(message[step_key]) * to_seconds / 3600.0


def to_grib_step(
    message, step_ns, step_unit=1, step_key="endStep", step_unit_key="stepUnits"
):
    step_s = step_ns * 1e-9
    to_seconds = GRIB_STEP_UNITS_TO_SECONDS[step_unit]
    if to_seconds is None:
        raise ValueError("unsupported stepUnit %r" % step_unit)
    message[step_key] = int(step_s / to_seconds)
    message[step_unit_key] = step_unit


def from_grib_month(message, verifying_month_key="verifyingMonth", epoch=DEFAULT_EPOCH):
    date = message[verifying_month_key]
    year = date // 100
    month = date % 100
    data_datetime = datetime.datetime(year, month, 1, 0, 0)
    return int((data_datetime - epoch).total_seconds())


def to_grib_dummy(message, value):
    pass


def build_valid_time(time, step):
    """
    Return dimensions and data of the valid_time corresponding to the given ``time`` and ``step``.
    The data is seconds from the same epoch as ``time`` and may have one or two dimensions.

    :param time: given in seconds from an epoch, as returned by ``from_grib_date_time``
    :param step: given in hours, as returned by ``from_grib_step``
    """
    step_s = step * 3600
    if len(time.shape) == 0 and len(step.shape) == 0:
        data = time + step_s
        dims = ()  # type: T.Tuple[str, ...]
    elif len(time.shape) > 0 and len(step.shape) == 0:
        data = time + step_s
        dims = ("time",)
    elif len(time.shape) == 0 and len(step.shape) > 0:
        data = time + step_s
        dims = ("step",)
    else:
        data = time[:, None] + step_s[None, :]
        dims = ("time", "step")
    return dims, data


COMPUTED_KEYS = {
    "time": (from_grib_date_time, to_grib_date_time),
    "step": (from_grib_step, to_grib_step),
    "valid_time": (
        functools.partial(
            from_grib_date_time, date_key="validityDate", time_key="validityTime"
        ),
        functools.partial(
            to_grib_date_time, date_key="validityDate", time_key="validityTime"
        ),
    ),
    "verifying_time": (from_grib_month, to_grib_dummy),
    "indexing_time": (
        functools.partial(
            from_grib_date_time, date_key="indexingDate", time_key="indexingTime"
        ),
        functools.partial(
            to_grib_date_time, date_key="indexingDate", time_key="indexingTime"
        ),
    ),
}  # type: messages.ComputedKeysType


# @attr.attrs(auto_attribs=True)
# class CfMessage(messages.ComputedKeysMessage):
#     computed_keys: messages.ComputedKeysType = COMPUTED_KEYS

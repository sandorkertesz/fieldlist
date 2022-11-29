# (C) Copyright 2022 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import pandas as pd

ipython_active = None


PANDAS_ORI_OPTIONS = {}


def init_pandas_options():
    global PANDAS_ORI_OPTIONS
    if len(PANDAS_ORI_OPTIONS) == 0:
        opt = {
            "display.max_colwidth": 300,
            "display.colheader_justify": "center",
            "display.max_columns": 100,
            "display.max_rows": 500,
            "display.width": None,
        }
        for k, _ in opt.items():
            PANDAS_ORI_OPTIONS[k] = pd.get_option(k)
        for k, v in opt.items():
            pd.set_option(k, v)


def reset_pandas_options():
    global PANDAS_ORI_OPTIONS
    if len(PANDAS_ORI_OPTIONS) > 0:
        for k, v in PANDAS_ORI_OPTIONS.items():
            pd.set_option(k, v)
        PANDAS_ORI_OPTIONS = {}


def is_ipython_active():
    global ipython_active
    if ipython_active is None:
        try:
            from IPython import get_ipython

            ipython_active = get_ipython() is not None
        except Exception:
            ipython_active = False
    return ipython_active


def list_or_single_value(v):
    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            return v[0]
        return v
    return v


def format_list(v, full=False):
    if isinstance(v, list):
        if full is True:
            return ",".join([str(x) for x in v])
        else:
            if len(v) == 1:
                return v[0]
            if len(v) > 2:
                return ",".join([str(x) for x in [v[0], v[1], "..."]])
            else:
                return ",".join([str(x) for x in v])
    else:
        return v


def make_unique(x, full=False):
    v = set(x)
    r = []
    for t in v:
        r.append(str(t))
    return format_list(r, full=full)


def drop_unwanted_series(df, key=None, axis=1):
    # only show the column for number in the default set of keys if
    # there are any valid values in it
    r = None
    if axis == 1 and key in df.columns:
        r = df[key].unique()
    elif axis == 0 and key in df.index:
        r = df.loc[key].unique()
    if len(r) == 1 and r[0] in ["0", None]:
        df.drop(key, axis=axis, inplace=True)


def process_ls(df, no_print):
    drop_unwanted_series(df, key="number", axis=1)

    # test whether we're in the Jupyter environment
    if is_ipython_active():
        return df
    elif not no_print:
        print(df)
    return df


def process_describe(df, param=None, groupby=[], no_print=False):
    init_pandas_options()
    labels = {"marsClass": "class", "marsStream": "stream", "marsType": "type"}
    no_header = False
    main_axis = 1
    if param is None:
        df = df.groupby(["shortName", "typeOfLevel"]).agg(make_unique)
        df.rename(labels, axis=1, inplace=True)
    elif isinstance(param, int):
        df = pd.DataFrame(df[df["paramId"] == param].agg(make_unique, full=True).T)
        df.rename(labels, axis=0, inplace=True)
        no_header = True
        main_axis = 0
    elif isinstance(param, str):
        df = pd.DataFrame(df[df["shortName"] == param].agg(make_unique, full=True).T)
        df.rename(labels, axis=0, inplace=True)
        no_header = True
        main_axis = 0
    else:
        return pd.DataFrame()

    drop_unwanted_series(df, key="number", axis=main_axis)

    df = df.style.set_properties(**{"text-align": "left"})
    df.set_table_styles([dict(selector="th", props=[("text-align", "left")])])
    if no_header:
        df.hide(axis="columns")

    if is_ipython_active:
        return df
    elif not no_print:
        print(df)
    return df


def sort_dataframe(df, columns=None, ascending=True):
    if columns is None:
        columns = list(df.columns)
    elif not isinstance(columns, list):
        columns = [columns]

    # mergesoft is a stable sorting algorithm
    df = df.sort_values(by=columns, ascending=ascending, kind="mergesort")
    # df = df.reset_index(drop=True)
    return df

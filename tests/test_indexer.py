# (C) Copyright 2017- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import datetime
import os

import numpy as np
import pandas as pd
import pytest

from fieldlist import FieldList
from fieldlist.fieldlist import DEFAULT_SORT_KEYS
from fieldlist.utils import sort_dataframe

PATH = os.path.dirname(__file__)


def file_in_testdir(filename):
    return os.path.join(PATH, filename)


def file_in_sort_dir(filename):
    return os.path.join(PATH, "sort", filename)


def build_metadata_dataframe(fs, keys):
    val = fs.get(keys, "key")
    md = {k: v for k, v in zip(keys, val)}
    return pd.DataFrame.from_dict(md)


def read_sort_meta_from_csv(name):
    f_name = file_in_sort_dir(f"{name}.csv.gz")
    return pd.read_csv(f_name, index_col=None)  # dtype=str)


def write_sort_meta_to_csv(name, md):
    f_name = file_in_sort_dir(f"{name}.csv.gz")
    md.to_csv(path_or_buf=f_name, header=True, index=False, compression="gzip")


def test_fieldset_select_single_file():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    assert f._db.empty == True

    # ------------------------
    # single resulting field
    # ------------------------
    g = f.sel(shortName="u", level=700)
    assert len(g) == 1
    assert g.get(["shortName", "level:l"]) == [["u", 700]]

    g1 = f[7]
    d = g - g1
    v = d.values
    assert np.allclose(v, np.zeros(len(v)))

    # # check indexer contents
    assert f._db.empty == False
    assert g._db.empty == False
    md = {
        "centre": "ecmf",
        "shortName": "u",
        "typeOfLevel": "isobaricInhPa",
        "level": 700,
        "dataDate": 20180801,
        "dataTime": 1200,
        "stepRange": "0",
        "dataType": "an",
        "number": 0,
        "gridType": "regular_ll",
    }

    assert [md] == g._db.items

    # ------------------------------------
    # single resulting field - paramId
    # ------------------------------------
    g = f.sel(paramId=131, level=700)
    assert len(g) == 1
    assert g.get(["paramId:l", "level:l"]) == [[131, 700]]
    # g1 = f[7]
    # d = g - g1
    # assert np.allclose(d.values(), np.zeros(len(d.values())))

    # -------------------------
    # multiple resulting fields
    # -------------------------
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    assert f._db.empty == True

    g = f.sel(shortName=["t", "u"], level=[700, 500])
    assert len(g) == 4
    assert g.get(["shortName", "level:l"]) == [
        ["t", 700],
        ["u", 700],
        ["t", 500],
        ["u", 500],
    ]

    assert f._db.empty == False
    assert g._db.empty == False
    assert len(g._db.items) == 4
    md = {
        "centre": ["ecmf"] * 4,
        "shortName": ["t", "u", "t", "u"],
        "typeOfLevel": ["isobaricInhPa"] * 4,
        "level": [700, 700, 500, 500],
        "dataDate": [20180801] * 4,
        "dataTime": [1200] * 4,
        "stepRange": ["0"] * 4,
        "dataType": ["an"] * 4,
        "number": [0] * 4,
        "paramId": [130, 131, 130, 131],
        "gridType": ["regular_ll"] * 4,
    }

    for i, item in enumerate(g._db.items):
        for k, v in item.items():
            assert md[k][i] == v, f"k={k}, i={i}"

    # -------------------------
    # empty result
    # -------------------------
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    g = f.sel(shortName="w")
    assert isinstance(g, FieldList)
    assert len(g) == 0

    # -------------------------
    # invalid key
    # -------------------------
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    g = f.sel(INVALIDKEY="w")
    assert isinstance(g, FieldList)
    assert len(g) == 0

    # -------------------------
    # str or int values
    # -------------------------
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    assert f._db.empty == True

    g = f.sel(shortName=["t"], level=[500, 700], marsType="an")
    # g = f.sel(shortName=["t"], level=["500", 700], marsType="an")
    assert len(g) == 2
    assert g.get(["shortName", "level:l", "marsType"]) == [
        ["t", 700, "an"],
        ["t", 500, "an"],
    ]

    f = FieldList(file_in_testdir("t_time_series.grib"))
    assert f._db.empty == True

    g = f.sel(shortName=["t"], step=[3, 6])
    assert len(g) == 2
    assert g.get(["shortName", "level:l", "step:l"]) == [
        ["t", 1000, 3],
        ["t", 1000, 6],
    ]

    g = f.sel(shortName=["t"], step=[3, 6])
    # g = f.sel(shortName=["t"], step=["3", "06"])
    assert len(g) == 2
    assert g.get(["shortName", "level:l", "step:l"]) == [
        ["t", 1000, 3],
        ["t", 1000, 6],
    ]

    # -------------------------
    # repeated use
    # -------------------------
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    assert f._db.empty == True

    g = f.sel(shortName=["t"], level=[500, 700], marsType="an")
    assert len(g) == 2
    assert g.get(["shortName", "level:l", "marsType"]) == [
        ["t", 700, "an"],
        ["t", 500, "an"],
    ]

    g = f.sel(shortName=["t"], level=[500], marsType="an")
    assert len(g) == 1
    assert g.get(["shortName", "level:l", "marsType"]) == [
        ["t", 500, "an"],
    ]

    # -------------------------
    # mars keys
    # -------------------------
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    assert f._db.empty == True

    g = f.sel(shortName=["t"], level=[500, 700], marsType="an")
    assert len(g) == 2
    assert g.get(["shortName", "level:l", "marsType"]) == [
        ["t", 700, "an"],
        ["t", 500, "an"],
    ]

    g = f.sel(shortName=["t"], level=[500, 700], type="an")
    assert len(g) == 2
    assert g.get(["shortName", "level:l", "type"]) == [
        ["t", 700, "an"],
        ["t", 500, "an"],
    ]
    # check the index db contents. "type" must be mapped to the "marsType" column of the
    # db so no rescanning should happen. The db should only contain the default set of columns.
    # assert g._db is not None
    # assert "scalar" in g._db.blocks
    # assert len(g._db.blocks) == 1
    # assert list(g._db.blocks["scalar"].keys())[:-1] == DB_DEFAULT_COLUMN_NAMES

    g = f.sel(shortName=["t"], level=[500, 700], type="fc")
    assert len(g) == 0

    g = f.sel({"shortName": "t", "level": [500, 700], "mars.type": "an"})
    assert len(g) == 2
    assert g.get(["shortName", "level:l", "mars.type"]) == [
        ["t", 700, "an"],
        ["t", 500, "an"],
    ]

    # -------------------------
    # custom keys
    # -------------------------
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    assert f._db.empty == True

    g = f.sel(shortName=["t"], level=[500, 700], gridType="regular_ll")
    assert len(g) == 2
    assert g.get(["shortName", "level:l", "gridType"]) == [
        ["t", 700, "regular_ll"],
        ["t", 500, "regular_ll"],
    ]

    g = f.sel({"shortName": ["t"], "level": [500, 700], "mars.param:s": "130.128"})
    assert len(g) == 2
    assert g.get(["shortName", "level:l", "mars.param"]) == [
        ["t", 700, "130.128"],
        ["t", 500, "130.128"],
    ]

    # assert g._db is not None
    # assert "scalar" in g._db.blocks
    # assert len(g._db.blocks) == 1
    # assert list(g._db.blocks["scalar"].keys())[:-1] == [
    #     *DB_DEFAULT_COLUMN_NAMES,
    #     "gridType",
    #     "mars.param:s",
    # ]


def test_fieldset_select_date():
    # date and time
    f = FieldList(file_in_testdir("t_time_series.grib"))
    assert f._db.empty == True

    g = f.sel(date=20201221, time=1200, step=9)
    # g = f.sel(date="20201221", time="12", step="9")
    assert len(g) == 2

    ref_keys = ["shortName", "date", "time", "step"]
    ref = [
        ["t", 20201221, 1200, 9],
        ["z", 20201221, 1200, 9],
    ]

    assert g.get(ref_keys) == ref

    return

    g = f.sel(date=20201221, time="1200", step=9)
    assert len(g) == 2
    assert mv.grib_get(g, ref_keys) == ref

    g = f.select(date=20201221, time="12:00", step=9)
    assert len(g) == 2
    assert mv.grib_get(g, ref_keys) == ref

    g = f.select(date=20201221, time=12, step=9)
    assert len(g) == 2
    assert mv.grib_get(g, ref_keys) == ref

    g = f.select(date="2020-12-21", time=1200, step=9)
    assert len(g) == 2
    assert mv.grib_get(g, ref_keys) == ref

    g = f.select(
        date=datetime.datetime(2020, 12, 21),
        time=datetime.time(hour=12, minute=0),
        step=9,
    )
    assert len(g) == 2
    assert mv.grib_get(g, ref_keys) == ref

    # dataDate and dataTime
    g = f.select(dataDate="20201221", dataTime="12", step=9)
    assert len(g) == 2
    assert mv.grib_get(g, ref_keys) == ref

    g = f.select(dataDate="2020-12-21", dataTime="12:00", step=9)
    assert len(g) == 2
    assert mv.grib_get(g, ref_keys) == ref

    # validityDate and validityTime
    g = f.select(validityDate="20201221", validityTime="21")
    assert len(g) == 2
    assert mv.grib_get(g, ref_keys) == ref

    g = f.select(validityDate="2020-12-21", validityTime="21:00")
    assert len(g) == 2
    assert mv.grib_get(g, ref_keys) == ref

    # dateTime
    g = f.select(dateTime="2020-12-21 12:00", step=9)
    assert len(g) == 2
    assert mv.grib_get(g, ref_keys) == ref

    # dataDateTime
    g = f.select(dataDateTime="2020-12-21 12:00", step=9)
    assert len(g) == 2
    assert mv.grib_get(g, ref_keys) == ref

    # validityDateTime
    g = f.select(validityDateTime="2020-12-21 21:00")
    assert len(g) == 2
    assert mv.grib_get(g, ref_keys) == ref

    # ------------------------------------
    # check multiple dates/times
    # ------------------------------------

    ref = [
        ["t", "20201221", "1200", "3"],
        ["z", "20201221", "1200", "3"],
        ["t", "20201221", "1200", "9"],
        ["z", "20201221", "1200", "9"],
    ]

    # date and time
    g = f.select(date="2020-12-21", time=12, step=[3, 9])
    assert len(g) == 4
    assert mv.grib_get(g, ref_keys) == ref

    # dateTime
    g = f.select(dateTime="2020-12-21 12:00", step=[3, 9])
    assert len(g) == 4
    assert mv.grib_get(g, ref_keys) == ref

    # validityDate and validityTime
    g = f.select(validityDate="2020-12-21", validityTime=[15, 21])
    assert len(g) == 4
    assert mv.grib_get(g, ref_keys) == ref

    # validityDateTime
    g = f.select(validityDateTime=["2020-12-21 15:00", "2020-12-21 21:00"])
    assert len(g) == 4
    assert mv.grib_get(g, ref_keys) == ref

    # ------------------------------------
    # check times with 1 digit hours
    # ------------------------------------

    # we create a new fieldset
    f = mv.merge(f[0], mv.grib_set_long(f[2:4], ["time", 600]))

    ref = [
        ["t", "20201221", "0600", "3"],
        ["z", "20201221", "0600", "3"],
    ]

    g = f.select(date="20201221", time="6", step="3")
    assert len(g) == 2
    assert mv.grib_get(g, ref_keys) == ref

    g = f.select(date=20201221, time="06", step=3)
    assert len(g) == 2
    assert mv.grib_get(g, ref_keys) == ref

    g = f.select(date=20201221, time="0600", step=3)
    assert len(g) == 2
    assert mv.grib_get(g, ref_keys) == ref

    g = f.select(date=20201221, time="06:00", step=3)
    assert len(g) == 2
    assert mv.grib_get(g, ref_keys) == ref

    g = f.select(validityDate="2020-12-21", validityTime=9)
    assert len(g) == 2
    assert mv.grib_get(g, ref_keys) == ref

    g = f.select(validityDate="2020-12-21", validityTime="09")
    assert len(g) == 2
    assert mv.grib_get(g, ref_keys) == ref

    g = f.select(validityDate="2020-12-21", validityTime=18)
    assert len(g) == 0

    # ------------------------------------
    # daily climatology dates (no year)
    # ------------------------------------

    f = mv.read(file_in_testdir("daily_clims.grib"))

    g = f.select(date="apr-01")
    assert len(g) == 1
    assert int(mv.grib_get_long(g, "date")) == 401

    g = f.select(date="Apr-02")
    assert len(g) == 1
    assert int(mv.grib_get_long(g, "date")) == 402

    g = f.select(date="402")
    assert len(g) == 1
    assert int(mv.grib_get_long(g, "date")) == 402

    g = f.select(date="0402")
    assert len(g) == 1
    assert int(mv.grib_get_long(g, "date")) == 402

    g = f.select(date=401)
    assert len(g) == 1
    assert int(mv.grib_get_long(g, "date")) == 401

    g = f.select(date=[401, 402])
    assert len(g) == 2
    assert [int(v) for v in mv.grib_get_long(g, "date")] == [402, 401]

    g = f.select(dataDate="apr-01")
    assert len(g) == 1
    assert int(mv.grib_get_long(g, "dataDate")) == 401


def test_fieldset_select_multi_file():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    f.append(FieldList(file_in_testdir("ml_data.grib")))
    assert f._db.empty == True

    # single resulting field
    g = f.sel(shortName="t", level=61)
    assert len(g) == 1
    assert g.get(["shortName", "level:l", "typeOfLevel"]) == [["t", 61, "hybrid"]]

    g1 = f[34]
    d = g - g1
    assert np.allclose(d.values, np.zeros(len(d.values)))

    assert g._db.empty == False
    md = {
        "centre": "ecmf",
        "shortName": "t",
        "typeOfLevel": "hybrid",
        "level": 61,
        "dataDate": 20180111,
        "dataTime": 1200,
        "stepRange": "12",
        "dataType": "fc",
        "number": None,
        "gridType": "regular_ll",
    }

    assert g._db.items == [md]


def test_indexer_dataframe_sort_value_with_key():

    md = {
        "paramId": [1, 2, 1, 2, 3],
        "level": [925, 850, 925, 850, 850],
        "step": [12, 110, 1, 3, 1],
        "rest": ["1", "2", "aa", "b1", "1b"],
    }

    md_ref = {
        "paramId": [1, 1, 2, 2, 3],
        "level": [925, 925, 850, 850, 850],
        "step": [1, 12, 3, 110, 1],
        "rest": ["aa", "1", "b1", "2", "1b"],
    }

    df = pd.DataFrame(md)
    df = sort_dataframe(df)
    df = df.reset_index(drop=True)
    df_ref = pd.DataFrame(md_ref)

    if not df.equals(df_ref):
        print(df.compare(df_ref))
        assert False


def test_describe():

    f = FieldList(file_in_testdir("tuv_pl.grib"))

    # full contents
    df = f.describe(no_print=True)
    df = df.data

    # ref = {
    #     "typeOfLevel": {
    #         "t": "isobaricInhPa",
    #         "u": "isobaricInhPa",
    #         "v": "isobaricInhPa",
    #     },
    #     "level": {"t": "300,400,...", "u": "300,400,...", "v": "300,400,..."},
    #     "date": {"t": 20180801, "u": 20180801, "v": 20180801},
    #     "time": {"t": 1200, "u": 1200, "v": 1200},
    #     "step": {"t": 0, "u": 0, "v": 0},
    #     "paramId": {"t": 130, "u": 131, "v": 132},
    #     "class": {"t": "od", "u": "od", "v": "od"},
    #     "stream": {"t": "oper", "u": "oper", "v": "oper"},
    #     "type": {"t": "an", "u": "an", "v": "an"},
    #     "experimentVersionNumber": {"t": "0001", "u": "0001", "v": "0001"},
    # }

    ref = {
        "level": {
            ("t", "isobaricInhPa"): "1000,300,...",
            ("u", "isobaricInhPa"): "1000,300,...",
            ("v", "isobaricInhPa"): "1000,300,...",
        },
        "date": {
            ("t", "isobaricInhPa"): "20180801",
            ("u", "isobaricInhPa"): "20180801",
            ("v", "isobaricInhPa"): "20180801",
        },
        "time": {
            ("t", "isobaricInhPa"): "1200",
            ("u", "isobaricInhPa"): "1200",
            ("v", "isobaricInhPa"): "1200",
        },
        "step": {
            ("t", "isobaricInhPa"): "0",
            ("u", "isobaricInhPa"): "0",
            ("v", "isobaricInhPa"): "0",
        },
        "paramId": {
            ("t", "isobaricInhPa"): "130",
            ("u", "isobaricInhPa"): "131",
            ("v", "isobaricInhPa"): "132",
        },
        "class": {
            ("t", "isobaricInhPa"): "od",
            ("u", "isobaricInhPa"): "od",
            ("v", "isobaricInhPa"): "od",
        },
        "stream": {
            ("t", "isobaricInhPa"): "oper",
            ("u", "isobaricInhPa"): "oper",
            ("v", "isobaricInhPa"): "oper",
        },
        "type": {
            ("t", "isobaricInhPa"): "an",
            ("u", "isobaricInhPa"): "an",
            ("v", "isobaricInhPa"): "an",
        },
        "experimentVersionNumber": {
            ("t", "isobaricInhPa"): "0001",
            ("u", "isobaricInhPa"): "0001",
            ("v", "isobaricInhPa"): "0001",
        },
    }

    ref_full = ref
    assert ref == df.to_dict()

    # repeated use
    df = f.describe()
    df = df.data
    assert ref == df.to_dict()

    # single param by shortName
    df = f.describe("t", no_print=True)
    df = df.data

    # ref = {
    #     "val": {
    #         "shortName": "t",
    #         "name": "Temperature",
    #         "paramId": 130,
    #         "units": "K",
    #         "typeOfLevel": "isobaricInhPa",
    #         "level": "300,400,500,700,850,1000",
    #         "date": "20180801",
    #         "time": "1200",
    #         "step": "0",
    #         "class": "od",
    #         "stream": "oper",
    #         "type": "an",
    #         "experimentVersionNumber": "0001",
    #     }
    # }

    ref = {
        0: {
            "shortName": "t",
            "typeOfLevel": "isobaricInhPa",
            "level": "1000,300,400,850,500,700",
            "date": "20180801",
            "time": "1200",
            "step": "0",
            "paramId": "130",
            "class": "od",
            "stream": "oper",
            "type": "an",
            "experimentVersionNumber": "0001",
        }
    }

    assert ref[0] == df[0].to_dict()

    # repeated use
    df = f.describe(param="t", no_print=True)
    df = df.data
    assert ref[0] == df[0].to_dict()

    df = f.describe("t")
    df = df.data
    assert ref[0] == df[0].to_dict()

    df = f.describe(param="t")
    df = df.data
    assert ref[0] == df[0].to_dict()

    # single param by paramId
    df = f.describe(130, no_print=True)
    df = df.data

    ref = {
        0: {
            "shortName": "t",
            "typeOfLevel": "isobaricInhPa",
            "level": "1000,300,400,850,500,700",
            "date": "20180801",
            "time": "1200",
            "step": "0",
            "paramId": "130",
            "class": "od",
            "stream": "oper",
            "type": "an",
            "experimentVersionNumber": "0001",
        }
    }

    # ref = {
    #     "val": {
    #         "shortName": "t",
    #         "name": "Temperature",
    #         "paramId": 130,
    #         "units": "K",
    #         "typeOfLevel": "isobaricInhPa",
    #         "level": "300,400,500,700,850,1000",
    #         "date": "20180801",
    #         "time": "1200",
    #         "step": "0",
    #         "class": "od",
    #         "stream": "oper",
    #         "type": "an",
    #         "experimentVersionNumber": "0001",
    #     }
    # }

    assert ref[0] == df[0].to_dict()

    df = f.describe(param=130, no_print=True)
    df = df.data
    assert ref[0] == df[0].to_dict()

    df = f.describe(130)
    df = df.data
    assert ref[0] == df[0].to_dict()

    df = f.describe(param=130)
    df = df.data
    assert ref[0] == df[0].to_dict()

    # append
    g = f + 0
    df = g.describe(no_print=True)
    df = df.data
    assert ref_full == df.to_dict()

    g.append(f[0].set({"level": 25}))
    df = g.describe(no_print=True)
    df = df.data

    ref = {
        "level": {
            ("t", "isobaricInhPa"): "1000,300,...",
            ("u", "isobaricInhPa"): "1000,300,...",
            ("v", "isobaricInhPa"): "1000,300,...",
        },
        "date": {
            ("t", "isobaricInhPa"): "20180801",
            ("u", "isobaricInhPa"): "20180801",
            ("v", "isobaricInhPa"): "20180801",
        },
        "time": {
            ("t", "isobaricInhPa"): "1200",
            ("u", "isobaricInhPa"): "1200",
            ("v", "isobaricInhPa"): "1200",
        },
        "step": {
            ("t", "isobaricInhPa"): "0",
            ("u", "isobaricInhPa"): "0",
            ("v", "isobaricInhPa"): "0",
        },
        "paramId": {
            ("t", "isobaricInhPa"): "130",
            ("u", "isobaricInhPa"): "131",
            ("v", "isobaricInhPa"): "132",
        },
        "class": {
            ("t", "isobaricInhPa"): "od",
            ("u", "isobaricInhPa"): "od",
            ("v", "isobaricInhPa"): "od",
        },
        "stream": {
            ("t", "isobaricInhPa"): "oper",
            ("u", "isobaricInhPa"): "oper",
            ("v", "isobaricInhPa"): "oper",
        },
        "type": {
            ("t", "isobaricInhPa"): "an",
            ("u", "isobaricInhPa"): "an",
            ("v", "isobaricInhPa"): "an",
        },
        "experimentVersionNumber": {
            ("t", "isobaricInhPa"): "0001",
            ("u", "isobaricInhPa"): "0001",
            ("v", "isobaricInhPa"): "0001",
        },
    }

    # ref = {
    #     "typeOfLevel": {
    #         "t": "isobaricInhPa",
    #         "u": "isobaricInhPa",
    #         "v": "isobaricInhPa",
    #     },
    #     "level": {"t": "25,300,...", "u": "300,400,...", "v": "300,400,..."},
    #     "date": {"t": 20180801, "u": 20180801, "v": 20180801},
    #     "time": {"t": 1200, "u": 1200, "v": 1200},
    #     "step": {"t": 0, "u": 0, "v": 0},
    #     "paramId": {"t": 130, "u": 131, "v": 132},
    #     "class": {"t": "od", "u": "od", "v": "od"},
    #     "stream": {"t": "oper", "u": "oper", "v": "oper"},
    #     "type": {"t": "an", "u": "an", "v": "an"},
    #     "experimentVersionNumber": {"t": "0001", "u": "0001", "v": "0001"},
    # }

    assert ref == df.to_dict()


def test_ls():
    f = FieldList(file_in_testdir("tuv_pl.grib"))

    # default keys
    df = f[:4].ls(no_print=True)

    ref = {
        "centre": {0: "ecmf", 1: "ecmf", 2: "ecmf", 3: "ecmf"},
        "shortName": {0: "t", 1: "u", 2: "v", 3: "t"},
        "typeOfLevel": {
            0: "isobaricInhPa",
            1: "isobaricInhPa",
            2: "isobaricInhPa",
            3: "isobaricInhPa",
        },
        "level": {0: 1000, 1: 1000, 2: 1000, 3: 850},
        "dataDate": {0: 20180801, 1: 20180801, 2: 20180801, 3: 20180801},
        "dataTime": {0: 1200, 1: 1200, 2: 1200, 3: 1200},
        "stepRange": {0: "0", 1: "0", 2: "0", 3: "0"},
        "dataType": {0: "an", 1: "an", 2: "an", 3: "an"},
        "number": {0: 0, 1: 0, 2: 0, 3: 0},
        "gridType": {
            0: "regular_ll",
            1: "regular_ll",
            2: "regular_ll",
            3: "regular_ll",
        },
    }

    assert ref == df.to_dict()

    # extra keys
    df = f[:2].ls(extra_keys=["paramId"], no_print=True)

    ref = {
        "centre": {0: "ecmf", 1: "ecmf"},
        "shortName": {0: "t", 1: "u"},
        "typeOfLevel": {0: "isobaricInhPa", 1: "isobaricInhPa"},
        "level": {0: 1000, 1: 1000},
        "dataDate": {0: 20180801, 1: 20180801},
        "dataTime": {0: 1200, 1: 1200},
        "stepRange": {0: "0", 1: "0"},
        "dataType": {0: "an", 1: "an"},
        "number": {0: 0, 1: 0},
        "gridType": {0: "regular_ll", 1: "regular_ll"},
        "paramId": {0: 130, 1: 131},
    }

    assert ref == df.to_dict()

    # filter
    df = f.ls(filter={"shortName": ["t", "v"], "level": 850}, no_print=True)

    ref = {
        "centre": {0: "ecmf", 1: "ecmf"},
        "shortName": {0: "t", 1: "v"},
        "typeOfLevel": {0: "isobaricInhPa", 1: "isobaricInhPa"},
        "level": {0: 850, 1: 850},
        "dataDate": {0: 20180801, 1: 20180801},
        "dataTime": {0: 1200, 1: 1200},
        "stepRange": {0: "0", 1: "0"},
        "dataType": {0: "an", 1: "an"},
        "number": {0: 0, 1: 0},
        "gridType": {0: "regular_ll", 1: "regular_ll"},
    }

    # ref = {
    #     "centre": {3: "ecmf", 5: "ecmf"},
    #     "shortName": {3: "t", 5: "v"},
    #     "typeOfLevel": {3: "isobaricInhPa", 5: "isobaricInhPa"},
    #     "level": {3: 850, 5: 850},
    #     "dataDate": {3: 20180801, 5: 20180801},
    #     "dataTime": {3: 1200, 5: 1200},
    #     "stepRange": {3: "0", 5: "0"},
    #     "dataType": {3: "an", 5: "an"},
    #     "number": {0: 0, 1: 0},
    #     "gridType": {3: "regular_ll", 5: "regular_ll"},
    # }

    assert ref == df.to_dict()

    # append
    g = f[:2]
    df = g.ls(no_print=True)

    ref = {
        "centre": {0: "ecmf", 1: "ecmf"},
        "shortName": {0: "t", 1: "u"},
        "typeOfLevel": {
            0: "isobaricInhPa",
            1: "isobaricInhPa",
        },
        "level": {0: 1000, 1: 1000},
        "dataDate": {0: 20180801, 1: 20180801},
        "dataTime": {0: 1200, 1: 1200},
        "stepRange": {0: "0", 1: "0"},
        "dataType": {0: "an", 1: "an"},
        "number": {0: 0, 1: 0},
        "gridType": {
            0: "regular_ll",
            1: "regular_ll",
        },
    }

    assert ref == df.to_dict()

    g.append(f[2].set({"level": 500}))
    df = g.ls(no_print=True)

    ref = {
        "centre": {0: "ecmf", 1: "ecmf", 2: "ecmf"},
        "shortName": {0: "t", 1: "u", 2: "v"},
        "typeOfLevel": {0: "isobaricInhPa", 1: "isobaricInhPa", 2: "isobaricInhPa"},
        "level": {0: 1000, 1: 1000, 2: 500},
        "dataDate": {0: 20180801, 1: 20180801, 2: 20180801},
        "dataTime": {0: 1200, 1: 1200, 2: 1200},
        "stepRange": {0: "0", 1: "0", 2: "0"},
        "dataType": {0: "an", 1: "an", 2: "an"},
        "number": {0: 0, 1: 0, 2: 0},
        "gridType": {0: "regular_ll", 1: "regular_ll", 2: "regular_ll"},
    }

    assert ref == df.to_dict()


def test_sort():

    # In each message the message index (1 based) is encoded
    # into latitudeOfLastGridPoint!
    fs = FieldList(file_in_sort_dir("sort_data.grib"))

    default_sort_keys = ["date", "time", "step", "number", "level", "paramId"]
    assert DEFAULT_SORT_KEYS == default_sort_keys

    # Note: shortName, units and latitudeOfLastGridPoint are non-default sort keys!
    keys = [
        "date",
        "time",
        "step",
        "number",
        "level",
        "paramId",
        "shortName",
        "units",
        "latitudeOfLastGridPoint",
    ]

    # the reference csv files were generated like this:
    #   write_sort_meta_to_csv(f_name, md)
    #
    # the correctness of the reference was tested by generating md_ref using
    # the GribIndexer and comparing it to md. E.g.:
    #   md_ori = build_metadata_dataframe(fs, keys)
    #   md_ref = GribIndexer._sort_dataframe(md_ori, columns=default_sort_keys)
    #   assert md.equals(md_ref)

    # default sorting
    f_name = "default"
    r = fs.sort()
    assert len(fs) == len(r)
    md = build_metadata_dataframe(r, keys)
    md_ref = read_sort_meta_from_csv(f_name)
    if not md.equals(md_ref):
        print(md.compare(md_ref))
        assert False

    # -----------------------------
    # default sorting direction
    # -----------------------------

    sort_keys = {
        k: k
        for k in [
            "date",
            "time",
            "step",
            "number",
            "level",
            "paramId",
            "shortName",
            "units",
        ]
    }
    sort_keys["date_level"] = ["date", "level"]
    sort_keys["level_units"] = ["level", "units"]
    sort_keys["keys"] = keys

    for f_name, key in sort_keys.items():
        r = fs.sort(key)
        assert len(fs) == len(r)
        md = build_metadata_dataframe(r, keys)
        md_ref = read_sort_meta_from_csv(f_name)
        if not md.equals(md_ref):
            print(md.compare(md_ref))
            assert False, f"key={key}"

    # single key as list
    key = f_name = "level"
    r = fs.sort([key])
    assert len(fs) == len(r)
    md = build_metadata_dataframe(r, keys)
    md_ref = read_sort_meta_from_csv(f_name)
    if not md.equals(md_ref):
        print(md.compare(md_ref))
        assert False, f"key={key}"

    # -----------------------------
    # custom sorting direction
    # -----------------------------

    # default keys

    # ascending
    f_name = "default"
    r = fs.sort(ascending=True)
    assert len(fs) == len(r)
    md = build_metadata_dataframe(r, keys)
    md_ref = read_sort_meta_from_csv(f_name)
    if not md.equals(md_ref):
        print(md.compare(md_ref))
        assert False, "default ascending"

    # descending
    f_name = "default_desc"
    r = fs.sort(ascending=False)
    assert len(fs) == len(r)
    md = build_metadata_dataframe(r, keys)
    md_ref = read_sort_meta_from_csv(f_name)
    if not md.equals(md_ref):
        print(md.compare(md_ref))
        assert False, "default descending"

    # single key
    key = "level"

    # ascending
    f_name = f"{key}_asc"
    md_ref = read_sort_meta_from_csv(f_name)

    r = fs.sort(key, ascending=True)
    assert len(fs) == len(r)
    md = build_metadata_dataframe(r, keys)
    if not md.equals(md_ref):
        print(md.compare(md_ref))
        assert False, f"key={key}"

    # descending
    f_name = f"{key}_desc"
    md_ref = read_sort_meta_from_csv(f_name)

    r = fs.sort(key, ascending=False)
    assert len(fs) == len(r)
    md = build_metadata_dataframe(r, keys)
    if not md.equals(md_ref):
        print(md.compare(md_ref))
        assert False, f"key={key}"

    # multiple keys
    key = ["level", "paramId", "date"]

    f_name = "multi_asc"
    md_ref = read_sort_meta_from_csv(f_name)

    r = fs.sort(key, ascending=True)
    assert len(fs) == len(r)
    md = build_metadata_dataframe(r, keys)
    if not md.equals(md_ref):
        print(md.compare(md_ref))
        assert False, f"key={key}"

    f_name = "multi_desc"
    md_ref = read_sort_meta_from_csv(f_name)

    r = fs.sort(key, ascending=False)
    assert len(fs) == len(r)
    md = build_metadata_dataframe(r, keys)
    if not md.equals(md_ref):
        print(md.compare(md_ref))
        assert False, f"key={key}"

    f_name = "multi_mixed"
    md_ref = read_sort_meta_from_csv(f_name)

    r = fs.sort(key, ascending=[True, False, True])
    assert len(fs) == len(r)
    md = build_metadata_dataframe(r, keys)
    if not md.equals(md_ref):
        print(md.compare(md_ref))
        assert False, f"key={key}"

    # invalid arguments
    with pytest.raises(ValueError):
        r = fs.sort(key, "1")

    with pytest.raises(ValueError):
        r = fs.sort(key, ascending=["True", "False"])

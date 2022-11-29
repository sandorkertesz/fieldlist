# (C) Copyright 2017- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import gc
import os
import pytest

import numpy as np

from fieldlist import FieldList
from fieldlist.codes import GribField

PATH = os.path.dirname(__file__)


def file_in_testdir(filename):
    return os.path.join(PATH, filename)


def test_empty_fieldlist_constructor():
    f = FieldList()
    assert isinstance(f, FieldList)
    assert len(f) == 0


# def test_fieldlist_contructor_bad_file_path():
#     with pytest.raises(FileNotFoundError):
#         f = FieldList("does/not/exist")


def test_non_empty_fieldlist_contructor_len():
    f = FieldList(file_in_testdir("test.grib"))
    assert isinstance(f, FieldList)
    assert len(f) == 1


def test_non_empty_fieldlist_contructor_len_18():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    assert isinstance(f, FieldList)
    assert len(f) == 18


def test_fieldlist_create_from_list_of_paths():
    paths = [file_in_testdir("t_for_xs.grib"), file_in_testdir("ml_data.grib")]
    f = FieldList(paths)
    assert len(f) == 42
    assert f[0:2].get("level") == [1000, 850]
    assert f[5:9].get("level") == [300, 1, 1, 5]
    assert f[40:42].get("level") == [133, 137]


def test_fieldlist_create_from_glob_path_single():
    f = FieldList(file_in_testdir("test.g*ib"))
    assert isinstance(f, FieldList)
    assert len(f) == 1


def test_fieldlist_create_from_glob_path_multi():
    f = FieldList(file_in_testdir("t_[f,w]*.grib"))
    assert isinstance(f, FieldList)
    assert len(f) == 7
    par_ref = [
        ["t", 1000],
        ["t", 850],
        ["t", 700],
        ["t", 500],
        ["t", 400],
        ["t", 300],
        ["t", 1000],
    ]
    assert par_ref == f.get(["shortName", "level"])


def test_fieldlist_create_from_glob_paths():
    f = FieldList([file_in_testdir("test.g*ib"), file_in_testdir("t_[f,w]*.grib")])
    assert isinstance(f, FieldList)
    assert len(f) == 8
    par_ref = [
        ["2t", 0],
        ["t", 1000],
        ["t", 850],
        ["t", 700],
        ["t", 500],
        ["t", 400],
        ["t", 300],
        ["t", 1000],
    ]
    assert par_ref == f.get(["shortName", "level"])


def test_read_1():
    f = FieldList(file_in_testdir("test.grib"))
    assert isinstance(f, FieldList)
    assert len(f) == 1


def test_grib_get_string_1():
    f = FieldList(file_in_testdir("test.grib"))
    for name in ("shortName", "shortName:s", "shortName:str"):
        sn = f.get(name)
        assert sn == "2t"
        sn = f[name]
        assert sn == "2t"


def test_grib_get_string_18():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    for name in ("shortName", "shortName:s", "shortName:str"):
        sn = f.get(name)
        assert sn == ["t", "u", "v"] * 6
        sn = f[name]
        assert sn == ["t", "u", "v"] * 6


def test_grib_get_long_1():
    f = FieldList(file_in_testdir("test.grib"))
    for name in ("level", "level:l", "level:int"):
        r = f.get(name)
        assert r == 0
        r = f[name]
        assert r == 0


def test_grib_get_long_18():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    ref = (
        ([1000] * 3)
        + ([850] * 3)
        + ([700] * 3)
        + ([500] * 3)
        + ([400] * 3)
        + ([300] * 3)
    )

    for name in ("level", "level:l", "level:int"):
        r = f.get(name)
        assert r == ref
        r = f[name]
        assert r == ref


def test_grib_get_double_1():
    f = FieldList(file_in_testdir("test.grib"))
    for name in ("max", "max:d", "max:float"):
        r = f.get(name)
        assert np.isclose(r, 316.061)
        r = f[name]
        assert np.isclose(r, 316.061)


def test_grib_get_double_18():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    ref = [
        320.564,
        21.7131,
        19.8335,
        304.539,
        43.1016,
        28.661,
        295.265,
        44.1455,
        31.6385,
        275.843,
        52.74,
        47.0099,
        264.003,
        62.2138,
        55.9496,
        250.653,
        66.4555,
        68.9203,
    ]
    for name in ("max", "max:d", "max:float"):
        r = f.get(name)
        np.testing.assert_allclose(r, ref, 0.001)
        r = f[name]
        np.testing.assert_allclose(r, ref, 0.001)


def test_grib_get_long_array_1():
    f = FieldList(file_in_testdir("rgg_small_subarea_cellarea_ref.grib"))
    pl = f.get("pl")
    assert isinstance(pl, np.ndarray)
    assert len(pl) == 73
    assert pl[0] == 24
    assert pl[1] == 28
    assert pl[20] == 104
    assert pl[72] == 312


def test_grib_get_double_array_values_1():
    f = FieldList(file_in_testdir("test.grib"))
    v = f.get("values")
    assert isinstance(v, np.ndarray)
    assert len(v) == 115680
    assert np.isclose(v[0], 260.4356)
    assert np.isclose(v[24226], 276.1856)
    assert np.isclose(v[36169], 287.9356)
    assert np.isclose(v[115679], 227.1856)


def test_grib_get_double_array_values_18():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    v = f.get("values")
    assert isinstance(v, list)
    assert len(v) == 18
    assert isinstance(v[0], np.ndarray)
    assert isinstance(v[17], np.ndarray)
    assert len(v[0]) == 2664
    assert len(v[17]) == 2664
    eps = 0.001
    assert np.isclose(v[0][0], 272.5642, eps)
    assert np.isclose(v[0][1088], 304.5642, eps)
    assert np.isclose(v[17][0], -3.0797, eps)
    assert np.isclose(v[17][2663], -11.0797, eps)


def test_grib_get_double_array_1():
    f = FieldList(file_in_testdir("ml_data.grib"))[0]
    v = f.get("pv")
    assert isinstance(v, np.ndarray)
    assert len(v) == 276
    assert np.isclose(v[0], 0.0)
    assert np.isclose(v[1], 2.0003650188446045)
    assert np.isclose(v[20], 316.4207458496094)
    assert np.isclose(v[275], 1.0)


def test_grib_get_double_array_18():
    f = FieldList(file_in_testdir("ml_data.grib"))
    v = f.get("pv")
    assert isinstance(v, list)
    assert len(v) == 36
    for row in v:
        assert isinstance(row, np.ndarray)
        assert len(row) == 276

    eps = 0.001
    assert np.isclose(v[0][1], 2.0003650188446045, eps)
    assert np.isclose(v[0][20], 316.4207458496094, eps)
    assert np.isclose(v[17][1], 2.0003650188446045, eps)
    assert np.isclose(v[17][20], 316.4207458496094, eps)


def test_grib_get_generic():
    f = FieldList(file_in_testdir("tuv_pl.grib"))[0:4]
    sn = f.get(["shortName"])
    assert sn == [["t"], ["u"], ["v"], ["t"]]
    cs = f.get(["centre:s"])
    assert cs == [["ecmf"], ["ecmf"], ["ecmf"], ["ecmf"]]
    cl = f.get(["centre:l"])
    assert cl == [[98], [98], [98], [98]]
    lg = f.get(["level:d", "cfVarName"])
    assert lg == [[1000, "t"], [1000, "u"], [1000, "v"], [850, "t"]]
    lgk = f.get(["level:d", "cfVarName"], "key")
    assert lgk == [[1000, 1000, 1000, 850], ["t", "u", "v", "t"]]
    with pytest.raises(ValueError):
        lgk = f.get(["level:d", "cfVarName"], "silly")

    f = FieldList(file_in_testdir("tuv_pl.grib"))[0]
    lg = f.get(["level", "cfVarName"])
    assert lg == [[1000, "t"]]

    # ln = f.get(["level:n"])
    # assert ln == [[1000], [1000], [1000], [850]]
    # cn = f.get(["centre:n"])
    # assert cn == [["ecmf"], ["ecmf"], ["ecmf"], ["ecmf"]]
    # vn = f[0].grib_get(["longitudes:n"])
    # assert vn[0][0][0] == 0
    # assert vn[0][0][1] == 5
    # assert vn[0][0][5] == 25


# def test_grib_get_generic_key_not_exist():
#     f = FieldList(file_in_testdir("tuv_pl.grib"))[0:2]
#     kv = f.get(["silly"])
#     assert kv == [[None], [None]]
#     with pytest.raises(Exception):
#         kv = f.get(["silly"])


def test_values_1():
    f = FieldList(file_in_testdir("test.grib"))
    v = f[0].values
    assert isinstance(v, np.ndarray)
    assert len(v) == 115680
    assert np.isclose(v[0], 260.4356)
    assert np.isclose(v[24226], 276.1856)
    assert np.isclose(v[36169], 287.9356)
    assert np.isclose(v[115679], 227.1856)


def test_values_18():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    v = f.values
    assert isinstance(v, np.ndarray)
    assert v.shape == (18, 2664)
    assert isinstance(v[0], np.ndarray)
    assert isinstance(v[17], np.ndarray)
    assert len(v[0]) == 2664
    assert len(v[17]) == 2664
    eps = 0.001
    assert np.isclose(v[0][0], 272.5642, eps)
    assert np.isclose(v[0][1088], 304.5642, eps)
    assert np.isclose(v[17][0], -3.0797, eps)
    assert np.isclose(v[17][2663], -11.0797, eps)


def test_values_with_missing():
    f = FieldList(file_in_testdir("t_with_missing.grib"))
    v = f.values
    assert isinstance(v, np.ndarray)
    assert v.shape == (2664,)
    eps = 0.001
    assert np.count_nonzero(np.isnan(v)) == 254
    assert np.isclose(v[0], 272.5642, eps)
    assert np.isnan(v[798])
    assert np.isnan(v[806])
    assert np.isnan(v[1447])
    assert np.isclose(v[2663], 240.5642, eps)


def test_grib_set_string():
    f = FieldList(file_in_testdir("tuv_pl.grib"))[0:2]
    assert len(f) == 2

    opt = {"typeOfLevel": "hybrid"}
    g1 = f.set(opt)
    g2 = f.set(**opt)
    for g in (g1, g2):
        assert g.get("typeOfLevel") == ["hybrid"] * 2
        assert f.get("typeOfLevel") == ["isobaricInhPa"] * 2

    opt = {"typeOfLevel": "hybrid", "shortName": "q"}
    g1 = f.set(opt)
    g2 = f.set(**opt)
    for g in (g1, g2):
        assert g.get("typeOfLevel") == ["hybrid"] * 2
        assert g.get("shortName") == ["q", "q"]
        assert f.get("typeOfLevel") == ["isobaricInhPa"] * 2
        assert f.get("shortName") == ["t", "u"]


def test_grib_set_long():
    f = FieldList(file_in_testdir("tuv_pl.grib"))[0:2]
    assert len(f) == 2

    opt = {"level": 95}
    g1 = f.set(opt)
    g2 = f.set(**opt)
    for g in (g1, g2):
        assert g.get("level") == [95] * 2
        assert f.get("level") == [1000] * 2

    opt = {"level": 95, "time": 1800}
    g1 = f.set(opt)
    g2 = f.set(**opt)
    for g in (g1, g2):
        assert g.get("level") == [95] * 2
        assert g.get("time") == [1800] * 2
        assert f.get("level") == [1000] * 2
        assert f.get("time") == [1200] * 2


def test_grib_set_double():
    f = FieldList(file_in_testdir("tuv_pl.grib"))[0:2]
    assert len(f) == 2

    opt = {"level": 95}
    g1 = f.set(opt)
    g2 = f.set(**opt)
    for g in (g1, g2):
        assert g.get("level") == [95] * 2
    assert f.get("level") == [1000] * 2

    key = "longitudeOfFirstGridPointInDegrees"
    orig_point = f.get(key)
    g = f.set({key: 95.6})
    assert g.get(key) == [95.6] * 2
    assert f.get(key) == orig_point


# def test_grib_set_generic():
#     f = FieldList(file_in_testdir("tuv_pl.grib"))[0:2]
#     assert len(f) == 2

#     g = f.set({"shortName": "r"})
#     assert g.get("shortName") == ["r"] * 2
#     assert f.get("shortName") == ["t", "u"]
#     g = f.set({"shortName:s": "q"})
#     assert g.get("shortName") == ["q"] * 2
#     assert f.get("shortName") == ["t", "u"]

#     g = f.grib_set(["level:l", 500, "shortName", "z"])
#     assert g.grib_get_long("level") == [500] * 2
#     assert g.grib_get_string("shortName") == ["z"] * 2
#     assert f.grib_get_long("level") == [1000] * 2
#     assert f.grib_get_string("shortName") == ["t", "u"]

#     g = f.grib_set(["level:d", 500])
#     np.testing.assert_allclose(
#         np.array(g.grib_get_double("level")), np.array([500] * 2)
#     )
#     np.testing.assert_allclose(
#         np.array(f.grib_get_double("level")), np.array([1000] * 2)
#     )

#     g = f.grib_set_double(["longitudeOfFirstGridPointInDegrees", 95.6])
#     np.testing.assert_allclose(
#         np.array(g.grib_get_double("longitudeOfFirstGridPointInDegrees")),
#         np.array([95.6] * 2),
#     )
#     np.testing.assert_allclose(
#         np.array(f.grib_get_double("longitudeOfFirstGridPointInDegrees")), [0, 0]
#     )


def test_write_fieldset():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    temp_path = "written_tuv_pl.grib"
    f.write(temp_path)
    assert os.path.isfile(temp_path)
    g = FieldList(temp_path)
    assert isinstance(g, FieldList)
    assert len(g) == 18
    sn = g.get("shortName")
    assert sn == ["t", "u", "v"] * 6
    f = None
    os.remove(temp_path)


def test_temporary_file():
    # create a temp file, then delete the fieldset - temp should be removed
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    g = f.set(level=925)
    temp_path = g._first_path()
    assert os.path.isfile(temp_path)
    g = None
    gc.collect()
    assert not os.path.isfile(temp_path)


def test_permanent_file_not_accidentally_deleted():
    path = file_in_testdir("tuv_pl.grib")
    f = FieldList(path)
    assert os.path.isfile(path)
    f = None
    gc.collect()
    assert os.path.isfile(path)


def test_single_index_0():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    r = f[0]
    assert isinstance(r, FieldList)
    assert len(r) == 1
    assert r.get("shortName") == "t"
    v = r.values
    eps = 0.001
    assert len(v) == 2664
    assert np.isclose(v[1088], 304.5642, eps)


def test_single_index_17():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    r = f[17]
    assert isinstance(r, FieldList)
    assert len(r) == 1
    assert r.get("shortName") == "v"
    v = r.values
    eps = 0.001
    assert len(v) == 2664
    assert np.isclose(v[2663], -11.0797, eps)


def test_single_index_minus_1():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    r = f[-1]
    assert isinstance(r, FieldList)
    assert len(r) == 1
    assert r.get("shortName") == "v"
    v = r.values
    eps = 0.001
    assert len(v) == 2664
    assert np.isclose(v[2663], -11.0797, eps)


def test_single_index_bad():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    with pytest.raises(IndexError):
        r = f[27]


def test_slice_0_5():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    r = f[0:5]
    assert isinstance(r, FieldList)
    assert len(r) == 5
    assert r.get("shortName") == ["t", "u", "v", "t", "u"]
    v = r.values
    assert v.shape == (5, 2664)
    # check the original fieldset
    assert len(f) == 18
    assert f.get("shortName") == ["t", "u", "v"] * 6


def test_array_indexing():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    indexes = np.array([1, 16, 5, 9])
    r = f[indexes]
    assert isinstance(r, FieldList)
    assert len(r) == 4
    assert r.get("shortName") == ["u", "u", "v", "t"]
    # check with bad indexes
    indexes = np.array([1, 36, 5, 9])
    with pytest.raises(IndexError):
        r = f[indexes]


def test_fieldset_iterator():
    g = FieldList(file_in_testdir("tuv_pl.grib"))
    sn = g.get("shortName")
    assert len(sn) == 18
    iter_sn = []
    for f in g:
        iter_sn.append(f.get("shortName"))
    assert len(iter_sn) == len(sn)
    assert iter_sn == sn
    iter_sn = [f.get("shortName") for f in g]
    assert iter_sn == sn


def test_fieldset_iterator_multiple():
    g = FieldList(file_in_testdir("tuv_pl.grib"))
    sn = g.get("shortName")
    assert len(sn) == 18
    for i in [1, 2, 3]:
        iter_sn = []
        for f in g:
            iter_sn.append(f.get("shortName"))
        assert len(iter_sn) == len(sn)
        for i in range(0, 18):
            assert sn[i] == iter_sn[i]


def test_fieldset_iterator_with_zip():
    # this tests something different with the iterator - this does not try to
    # 'go off the edge' of the fieldset, because the length is determined by
    # the list of levels
    g = FieldList(file_in_testdir("tuv_pl.grib"))
    ref_levs = g.get("level")
    assert len(ref_levs) == 18
    levs1 = []
    levs2 = []
    for k, f in zip(g.get("level"), g):
        levs1.append(k)
        levs2.append(f.get("level"))
    assert levs1 == ref_levs
    assert levs2 == ref_levs


def test_fieldset_iterator_with_zip_multiple():
    # same as test_fieldset_iterator_with_zip() but multiple times
    g = FieldList(file_in_testdir("tuv_pl.grib"))
    ref_levs = g.get("level")
    assert len(ref_levs) == 18
    for i in [1, 2, 3]:
        levs1 = []
        levs2 = []
        for k, f in zip(g.get("level"), g):
            levs1.append(k)
            levs2.append(f.get("level"))
        print(g.get("level"))
        assert levs1 == ref_levs
        assert levs2 == ref_levs


def test_fieldset_reverse_iterator():
    g = FieldList(file_in_testdir("tuv_pl.grib"))
    sn = g.get("shortName")
    sn_reversed = list(reversed(sn))
    assert sn_reversed[0] == "v"
    assert sn_reversed[17] == "t"
    gr = reversed(g)
    iter_sn = [f.get("shortName") for f in gr]
    assert len(iter_sn) == len(sn_reversed)
    assert iter_sn == sn_reversed
    assert iter_sn == ["v", "u", "t"] * 6


def test_fieldset_append():
    g = FieldList(file_in_testdir("tuv_pl.grib"))
    h = FieldList(file_in_testdir("all_missing_vals.grib"))
    r = g[0:3]
    r.append(h)
    assert r.get("shortName") == ["t", "u", "v", "z"]


def test_fieldset_merge():
    g = FieldList(file_in_testdir("tuv_pl.grib"))
    h = FieldList(file_in_testdir("all_missing_vals.grib"))
    i = g[0:3]
    j = i.merge(h)  # does not alter the original fieldset
    assert i.get("shortName") == ["t", "u", "v"]
    assert j.get("shortName") == ["t", "u", "v", "z"]


def test_str():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    # assert str(f) == "Fieldset (18 fields)"


def test_set_values_single_field():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    f0 = f[0]
    f0_vals = f0.values
    vals_plus_10 = f0_vals + 10
    f0_modified = f0.set_values(vals_plus_10)
    f0_mod_vals = f0_modified.values
    np.testing.assert_allclose(f0_mod_vals, vals_plus_10)

    # write to disk, read and check again
    testpath = "written_f0_modified.grib"
    f0_modified.write(testpath)
    f0_read = FieldList(testpath)
    np.testing.assert_allclose(f0_read.values, vals_plus_10)
    os.remove(testpath)


def test_set_values_multiple_fields():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    f03 = f[0:3]
    f47 = f[4:7]
    f03_modified = f03.set_values(f47.values)
    np.testing.assert_allclose(f03_modified.values, f47.values)
    # same, but with a list of arrays instead of a 2D array
    list_of_arrays = [f.values for f in f47]
    f03_modified_2 = f03.set_values(list_of_arrays)
    np.testing.assert_allclose(f03_modified_2.values, f47.values)
    # wrong number of arrays
    f48 = f[4:8]
    with pytest.raises(ValueError):
        f03_modified_3 = f03.set_values(f48.values)


def test_set_values_with_missing_values():
    f = FieldList(file_in_testdir("t_with_missing.grib"))
    new_vals = f.values + 40
    g = f.set_values(new_vals)
    v = g.values
    assert v.shape == (2664,)
    eps = 0.001
    assert np.isclose(v[0], 272.5642 + 40, eps)
    assert np.isnan(v[798])
    assert np.isnan(v[806])
    assert np.isnan(v[1447])
    assert np.isclose(v[2663], 240.5642 + 40, eps)


def test_set_values_with_missing_values_2():
    f = FieldList(file_in_testdir("t_with_missing.grib"))
    g = f[0]
    v = g.values
    v[1] = np.nan
    h = g.set_values(v)
    hv = h.values[:10]
    assert np.isclose(hv[0], 272.56417847)
    assert np.isnan(hv[1])
    assert np.isclose(hv[2], 272.56417847)


# def test_set_values_resize():
#     # NOTE: the current change in behavour - in 'standard Metview' the user
#     # has to supply "resize" as an optional argument in order to allow an array
#     # of different size to be used; if not supplied, and the given array is not the
#     # same size as the original field, an error is thrown; here, we allow resizing
#     # without the need for an extra argument - do we want to do this check?
#     f = mv.Fieldset(path=os.path.join(PATH, "tuv_pl.grib"))
#     f0 = f[0]
#     f0_20vals = f0.values()[0:20]
#     f0_modified = f0.set_values(f0_20vals)
#     f0_mod_vals = f0_modified.values()
#     eps = 0.001
#     np.testing.assert_allclose(f0_mod_vals, f0_20vals, eps)


def test_handle_released():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    r = f["typeOfLevel"]
    assert r == ["isobaricInhPa"] * 18
    assert f.count_open_handles() == 0

    g = f.set(level=200)
    assert f.count_open_handles() == 0
    assert g.count_open_handles() == 0

    r = g["level"]
    assert r == [200] * 18
    assert g.count_open_handles() == 0

    v = f[0].values
    assert f.count_open_handles() == 0
    g = f[0].set_values(v * 0 + 1)
    assert f.count_open_handles() == 0
    assert g.count_open_handles() == 0

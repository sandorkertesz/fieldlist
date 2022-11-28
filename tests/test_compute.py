# (C) Copyright 2017- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from inspect import ArgInfo
import numpy as np
import os
import pytest

from fieldlist import FieldList
from fieldlist.codes import GribField

PATH = os.path.dirname(__file__)


def file_in_testdir(filename):
    return os.path.join(PATH, filename)


def test_write_modified_fieldlist_binop():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    fp20 = f + 20
    temp_path = "written_tuv_pl.grib"
    fp20.write(temp_path)
    assert os.path.isfile(temp_path)
    g = FieldList(temp_path)
    assert isinstance(g, FieldList)
    assert len(g) == 18
    sn = g["shortName"]
    assert sn == ["t", "u", "v"] * 6
    np.testing.assert_allclose(g.values, f.values + 20)
    f = None
    os.remove(temp_path)


def test_write_modified_fieldlist_unop():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    negf = -f
    temp_path = "written_tuv_pl_unop.grib"
    negf.write(temp_path)
    assert os.path.isfile(temp_path)
    g = FieldList(temp_path)
    assert isinstance(g, FieldList)
    assert len(g) == 18
    sn = g.get("shortName")
    assert sn == ["t", "u", "v"] * 6
    np.testing.assert_allclose(g.values, -f.values, 0.0001)
    f = None
    os.remove(temp_path)


def test_field_compute_func():
    def sqr_func(x):
        return x * x

    f = FieldList(file_in_testdir("tuv_pl.grib"))
    g = f.compute(sqr_func)
    assert isinstance(g, FieldList)
    assert len(g) == 18
    vf = f.values
    vg = g.values
    np.testing.assert_allclose(vg, vf * vf, 0.0001)


def test_field_compute_neg():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    g = -f
    assert isinstance(g, FieldList)
    assert len(g) == 18
    vf = f.values
    vg = g.values
    np.testing.assert_allclose(vg, -vf)


def test_field_compute_pos():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    g = +f  # should return values unaltered
    assert isinstance(g, FieldList)
    assert len(g) == 18
    vf = f.values
    vg = g.values
    np.testing.assert_allclose(vg, vf)


def test_field_compute_abs():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    g = f.abs()
    assert isinstance(g, FieldList)
    assert len(g) == 18
    vf = f.values
    vg = g.values
    np.testing.assert_allclose(vg, np.abs(vf))


def test_field_scalar_func():
    f = FieldList(file_in_testdir("tuv_pl.grib"))[0:3]
    # fieldlist op scalar
    g = f + 10
    assert isinstance(g, FieldList)
    assert len(g) == 3
    np.testing.assert_allclose(g.values, f.values + 10)
    q = f - 5
    np.testing.assert_allclose(q.values, f.values - 5)
    m = f * 1.5
    np.testing.assert_allclose(m.values, f.values * 1.5)
    d = f / 3.0
    np.testing.assert_allclose(d.values, f.values / 3.0, 0.0001)
    p = f**2
    np.testing.assert_allclose(p.values, f.values**2)
    first_val = f.values[0][0]  # 272
    ge = f >= first_val
    v = ge.values
    assert v[0][0] == 1  # 272
    assert v[0][2645] == 0  # 240
    assert v[0][148] == 1  # 280
    assert v[1][0] == 0  # -6
    gt = f > first_val
    v = gt.values
    assert v[0][0] == 0  # 272
    assert v[0][2645] == 0  # 240
    assert v[0][148] == 1  # 280
    assert v[1][0] == 0  # - 6
    lt = f < first_val
    v = lt.values
    assert v[0][0] == 0  # 272
    assert v[0][2645] == 1  # 240
    assert v[0][148] == 0  # 280
    assert v[1][0] == 1  # - 6
    lt = f <= first_val
    v = lt.values
    assert v[0][0] == 1  # 272
    assert v[0][2645] == 1  # 240
    assert v[0][148] == 0  # 280
    assert v[1][0] == 1  # - 6
    e = f == first_val
    v = e.values
    assert v[0][0] == 1  # 272
    assert v[0][2645] == 0  # 240
    assert v[0][148] == 0  # 280
    assert v[1][0] == 0  # - 6
    ne = f != first_val
    v = ne.values
    assert v[0][0] == 0  # 272
    assert v[0][2645] == 1  # 240
    assert v[0][148] == 1  # 280
    assert v[1][0] == 1  # - 6
    andd = (f > 270) & (f < 290)  # and
    v = andd.values
    assert v[0][0] == 1  # 272
    assert v[0][2645] == 0  # 240
    assert v[0][148] == 1  # 280
    assert v[1][0] == 0  # - 6
    orr = (f < 270) | (f > 279)  # or
    v = orr.values
    assert v[0][0] == 0  # 272
    assert v[0][2645] == 1  # 240
    assert v[0][148] == 1  # 280
    assert v[1][0] == 1  # - 6
    nott = ~((f > 270) & (f < 290))  # not
    v = nott.values
    assert v[0][0] == 0  # 272
    assert v[0][2645] == 1  # 240
    assert v[0][148] == 0  # 280
    assert v[1][0] == 1  # - 6

    # scalar op fieldlist
    h = 20 + f
    assert isinstance(h, FieldList)
    assert len(h) == 3
    np.testing.assert_allclose(h.values, f.values + 20)
    r = 25 - f
    np.testing.assert_allclose(r.values, 25 - f.values)
    mr = 3 * f
    np.testing.assert_allclose(mr.values, f.values * 3)
    dr = 200 / f
    np.testing.assert_allclose(dr.values, 200 / f.values, 0.0001)
    pr = 2**f
    np.testing.assert_allclose(pr.values, 2**f.values, 1)


def test_fieldlist_compute_oper():
    f = FieldList(file_in_testdir("tuv_pl.grib"))[0:3]
    g = FieldList(file_in_testdir("tuv_pl.grib"))[5:8]

    r = f + g
    np.testing.assert_allclose(r.values, f.values + g.values)

    r = g + f
    np.testing.assert_allclose(r.values, g.values + f.values)

    r = f - g
    np.testing.assert_allclose(r.values, f.values - g.values)

    r = g - f
    np.testing.assert_allclose(r.values, g.values - f.values)

    r = g * f
    np.testing.assert_allclose(r.values, g.values * f.values, 0.0001)

    r = g / f
    np.testing.assert_allclose(r.values, g.values / f.values, 0.0001)

    r = f > g
    np.testing.assert_array_equal(r.values, f.values > g.values)

    r = f >= g
    np.testing.assert_array_equal(r.values, f.values >= g.values)

    r = f < g
    np.testing.assert_array_equal(r.values, f.values < g.values)

    r = f <= g
    np.testing.assert_array_equal(r.values, f.values <= g.values)


# def test_fieldlist_multiple_funcs():
#     f = FieldList(file_in_testdir("tuv_pl.grib"))
#     g = 1 - ((f[0] + f[3]) - 5)
#     np.testing.assert_allclose(g.values, 1 - ((f[0].values + f[3].values) - 5))


def test_fieldlist_funcs_with_read():
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    assert isinstance(f, FieldList)
    g = f + 18
    assert isinstance(g, FieldList)
    diff = g - f
    assert isinstance(diff, FieldList)
    # assert np.isclose(diff.minvalue(), 18)
    # assert np.isclose(diff.maxvalue(), 18)


def test_field_maths_funcs():
    return
    f = FieldList(file_in_testdir("tuv_pl.grib"))
    f = f[0]
    v = f.values()

    # no arg
    r = f.abs()
    np.testing.assert_allclose(r.values(), np.fabs(v), rtol=1e-05)

    r = f.cos()
    np.testing.assert_allclose(r.values(), np.cos(v), rtol=1e-05)

    f1 = f / 100
    r = f1.exp()
    np.testing.assert_allclose(r.values(), np.exp(f1.values()), rtol=1e-05)

    r = f.log()
    np.testing.assert_allclose(r.values(), np.log(v), rtol=1e-05)

    r = f.log10()
    np.testing.assert_allclose(r.values(), np.log10(v), rtol=1e-05)

    r = f.sgn()
    np.testing.assert_allclose(r.values(), np.sign(v), rtol=1e-05)

    r = f.sin()
    np.testing.assert_allclose(r.values(), np.sin(v), rtol=1e-05)

    r = f.sqrt()
    np.testing.assert_allclose(r.values(), np.sqrt(v), rtol=1e-05)

    r = f.tan()
    np.testing.assert_allclose(r.values(), np.tan(v), rtol=1e-04)

    # inverse functions
    # scale input between [-1, 1]
    f1 = (f - 282) / 80
    v1 = f1.values()
    r = f1.acos()
    np.testing.assert_allclose(r.values(), np.arccos(v1), rtol=1e-05)

    r = f1.asin()
    np.testing.assert_allclose(r.values(), np.arcsin(v1), rtol=1e-05)

    r = f1.atan()
    np.testing.assert_allclose(r.values(), np.arctan(v1), rtol=1e-05)

    # 1 arg
    f1 = f - 274
    v1 = f1.values()

    r = f.atan2(f1)
    np.testing.assert_allclose(r.values(), np.arctan2(v, v1), rtol=1e-05)

    r = f.div(f1)
    np.testing.assert_allclose(r.values(), np.floor_divide(v, v1), rtol=1e-05)

    r = f.mod(f1)
    np.testing.assert_allclose(r.values(), np.mod(v, v1), rtol=1e-04)


# def test_accumulate():
#     f = mv.fieldlist(path=os.path.join(PATH, "t1000_LL_7x7.grb"))
#     v = mv.accumulate(f)
#     assert isinstance(v, float)
#     assert np.isclose(v, 393334.244141)

#     f = mv.fieldlist(path=os.path.join(PATH, "monthly_avg.grib"))
#     v = mv.accumulate(f)
#     assert isinstance(v, list)
#     v_ref = [
#         408058.256226,
#         413695.059631,
#         430591.282776,
#         428943.981812,
#         422329.622498,
#         418016.024231,
#         409755.097961,
#         402741.786194,
#     ]
#     assert len(v) == len(f)
#     np.testing.assert_allclose(v, v_ref)


# def test_average():
#     fs = mv.fieldlist(path=os.path.join(PATH, "test.grib"))

#     # const fields
#     v = mv.average(fs * 0 + 1)
#     assert isinstance(v, float)
#     assert np.isclose(v, 1)

#     # # single field
#     v = mv.average(fs)
#     assert isinstance(v, float)
#     assert np.isclose(v, 279.06647863)

#     # multiple fields
#     f = mv.fieldlist(path=os.path.join(PATH, "monthly_avg.grib"))
#     v = mv.average(f)
#     assert isinstance(v, list)
#     v_ref = [
#         290.639783636,
#         294.654600877,
#         306.688947846,
#         305.515656561,
#         300.804574428,
#         297.732210991,
#         291.848360371,
#         286.85312407,
#     ]

#     assert len(v) == len(f)
#     np.testing.assert_allclose(v, v_ref)


# def test_latitudes():
#     fs = mv.fieldlist(path=os.path.join(PATH, "t1000_LL_2x2.grb"))

#     v = mv.latitudes(fs)
#     assert isinstance(v, np.ndarray)
#     assert len(v) == 16380
#     assert np.isclose(v[0], 90)
#     assert np.isclose(v[1], 90)
#     assert np.isclose(v[8103], 0)
#     assert np.isclose(v[11335], -34)
#     assert np.isclose(v[16379], -90)

#     f = fs.merge(fs)
#     lst = mv.latitudes(f)
#     assert len(lst) == 2
#     for v in lst:
#         assert np.isclose(v[0], 90)
#         assert np.isclose(v[1], 90)
#         assert np.isclose(v[8103], 0)
#         assert np.isclose(v[11335], -34)
#         assert np.isclose(v[16379], -90)


# def test_longitudes():
#     fs = mv.fieldlist(path=os.path.join(PATH, "t1000_LL_2x2.grb"))

#     v = mv.longitudes(fs)
#     assert isinstance(v, np.ndarray)
#     assert len(v) == 16380
#     assert np.isclose(v[0], 0)
#     assert np.isclose(v[1], 2)
#     assert np.isclose(v[8103], 6)
#     assert np.isclose(v[11335], 350)
#     assert np.isclose(v[16379], 358)

#     f = fs.merge(fs)
#     lst = mv.longitudes(f)
#     assert len(lst) == 2
#     for v in lst:
#         assert np.isclose(v[0], 0)
#         assert np.isclose(v[1], 2)
#         assert np.isclose(v[8103], 6)
#         assert np.isclose(v[11335], 350)
#         assert np.isclose(v[16379], 358)


# def test_coslat():
#     fs = mv.fieldlist(path=os.path.join(PATH, "t_time_series.grib"))

#     # WARN: it is important that the data should be at least 16 bit
#     #  to keep accuracy in resulting fields

#     f = fs[0]
#     r = mv.coslat(f)
#     np.testing.assert_allclose(
#         r.values(), np.cos(np.deg2rad(f.latitudes())), rtol=1e-06
#     )

#     f = fs[:2]
#     r = mv.coslat(f)
#     assert len(r) == 2
#     for i in range(len(r)):
#         np.testing.assert_allclose(
#             r[i].values(), np.cos(np.deg2rad(f[i].latitudes())), rtol=1e-06
#         )


# def test_mean():
#     fs = mv.fieldlist(path=os.path.join(PATH, "test.grib"))

#     # single fields
#     f = fs
#     r = mv.mean(f)
#     v_ref = mv.values(fs)
#     assert len(r) == 1
#     np.testing.assert_allclose(r.values(), v_ref, rtol=1e-05)

#     # known mean
#     f = fs.merge(2 * fs)
#     f = f.merge(3 * fs)
#     r = f.mean()
#     v_ref = mv.values(fs) * 2
#     assert len(r) == 1
#     np.testing.assert_allclose(r.values(), v_ref, rtol=1e-05)


# def test_maxvalue():
#     fs = mv.fieldlist(path=os.path.join(PATH, "test.grib"))

#     f = fs
#     f = f.merge(3 * fs)
#     f = f.merge(2 * fs)
#     v = mv.maxvalue(f)
#     assert isinstance(v, float)
#     assert np.isclose(v, 948.1818237304688)


# def test_minvalue():
#     fs = mv.fieldlist(path=os.path.join(PATH, "test.grib"))

#     f = 3 * fs
#     f = f.merge(fs)
#     f = f.merge(2 * fs)
#     v = mv.minvalue(f)
#     assert isinstance(v, float)
#     assert np.isclose(v, 206.93560791015625)


# def test_sinlat():
#     fs = mv.fieldlist(path=os.path.join(PATH, "t_time_series.grib"))

#     # WARN: it is important that the data should be at least 16 bit
#     #  to keep accuracy in resulting fields

#     f = fs[0]
#     r = mv.sinlat(f)
#     np.testing.assert_allclose(
#         r.values(), np.sin(np.deg2rad(f.latitudes())), rtol=1e-06
#     )

#     f = fs[:2]
#     r = mv.sinlat(f)
#     assert len(r) == 2
#     for i in range(len(r)):
#         np.testing.assert_allclose(
#             r[i].values(), np.sin(np.deg2rad(f[i].latitudes())), rtol=1e-06
#         )


# def test_tanlat():
#     fs = mv.fieldlist(path=os.path.join(PATH, "t_time_series.grib"))

#     # WARN: it is important that the data should be at least 16 bit
#     #  to keep accuracy in resulting fields

#     # TODO: use pole_limit value from fieldlist

#     pole_limit = 90.0 - 1e-06

#     f = fs[0]
#     r = mv.tanlat(f)
#     lat = f.latitudes()
#     lat[np.fabs(lat) > pole_limit] = np.nan
#     np.testing.assert_allclose(
#         r.values(), np.tan(np.deg2rad(lat)), rtol=1e-06, atol=1e-06
#     )

#     f = fs[:2]
#     r = mv.tanlat(f)
#     assert len(r) == 2
#     for i in range(len(r)):
#         lat = f[i].latitudes()
#         lat[np.fabs(lat) > pole_limit] = np.nan
#         np.testing.assert_allclose(
#             r[i].values(), np.tan(np.deg2rad(lat)), rtol=1e-06, atol=1e-06
#         )


# def test_stdev():
#     fs = mv.fieldlist(path=os.path.join(PATH, "t1000_LL_7x7.grb"))

#     # single field
#     r = mv.stdev(fs)
#     assert len(r) == 1
#     np.testing.assert_allclose(r.values(), fs.values() * 0)

#     # known variance
#     f = fs.merge(4 * fs)
#     f = f.merge(10 * fs)
#     r = mv.stdev(f)
#     assert len(r) == 1
#     np.testing.assert_allclose(r.values(), np.sqrt(np.square(fs.values()) * 42 / 3))

#     # real life example
#     fs = mv.fieldlist(path=os.path.join(PATH, "monthly_avg.grib"))
#     r = mv.stdev(fs)
#     assert len(r) == 1

#     v_ref = np.ma.std(np.array([x.values() for x in fs]), axis=0)
#     np.testing.assert_allclose(r.values(), v_ref, rtol=1e-03)


# def test_sum():
#     fs = mv.fieldlist(path=os.path.join(PATH, "t1000_LL_7x7.grb"))

#     # single fields
#     f = fs
#     r = mv.sum(f)
#     assert len(r) == 1
#     np.testing.assert_allclose(r.values(), fs.values())

#     # known sum
#     f = fs.merge(fs)
#     f = f.merge(fs)
#     r = f.sum()
#     assert len(r) == 1
#     np.testing.assert_allclose(r.values(), fs.values() * 3)

#     # real life example
#     f = mv.fieldlist(path=os.path.join(PATH, "monthly_avg.grib"))
#     r = f.sum()
#     assert len(r) == 1
#     v_ref = r.values() * 0
#     for g in f:
#         v_ref += g.values()
#     np.testing.assert_allclose(r.values(), v_ref)


# def test_var():
#     fs = mv.fieldlist(path=os.path.join(PATH, "t1000_LL_7x7.grb"))

#     # single field
#     r = mv.var(fs)
#     assert len(r) == 1
#     np.testing.assert_allclose(r.values(), fs.values() * 0)

#     # known variance
#     f = fs.merge(4 * fs)
#     f = f.merge(10 * fs)
#     r = mv.var(f)
#     assert len(r) == 1
#     np.testing.assert_allclose(r.values(), np.square(fs.values()) * 42 / 3)

#     # real life example
#     fs = mv.fieldlist(path=os.path.join(PATH, "monthly_avg.grib"))
#     r = mv.var(fs)
#     assert len(r) == 1

#     v_ref = np.ma.var(np.array([x.values() for x in fs]), axis=0)
#     np.testing.assert_allclose(r.values(), v_ref, rtol=1e-03)


# def test_date():

#     fs = mv.fieldlist(path=os.path.join(PATH, "monthly_avg.grib"))

#     # analysis, so valid=base
#     bdate_ref = [
#         "2016-01-01 00:00:00",
#         "2016-02-01 00:00:00",
#         "2016-03-01 00:00:00",
#         "2016-04-01 00:00:00",
#         "2016-05-01 00:00:00",
#         "2016-06-01 00:00:00",
#         "2016-07-01 00:00:00",
#         "2016-08-01 00:00:00",
#     ]
#     vdate_ref = bdate_ref

#     v = mv.base_date(fs)
#     assert len(v) == len(fs)
#     for i, d in enumerate(v):
#         assert d == utils.date_from_str(bdate_ref[i])

#     v = mv.valid_date(fs)
#     assert len(v) == len(fs)
#     for i, d in enumerate(v):
#         assert d == utils.date_from_str(vdate_ref[i])


# def test_bitmap():
#     fs = mv.fieldlist(path=os.path.join(PATH, "t1000_LL_2x2.grb"))

#     # -- const field
#     f = fs * 0 + 1

#     # non missing
#     r = mv.bitmap(f, 0)
#     np.testing.assert_allclose(r.values(), f.values())

#     # all missing
#     r = mv.bitmap(f, 1)
#     np.testing.assert_allclose(r.values(), f.values() * np.nan)

#     # -- non const field
#     f = fs

#     # bitmap with value
#     f_mask = f > 300
#     r = mv.bitmap(f_mask, 1)
#     v_ref = f_mask.values()
#     v_ref[v_ref == 1] = np.nan
#     np.testing.assert_allclose(r.values(), v_ref)

#     f_mask = f > 300
#     r = mv.bitmap(f_mask * 2, 2)
#     v_ref = f_mask.values() * 2
#     v_ref[v_ref == 2] = np.nan
#     np.testing.assert_allclose(r.values(), v_ref)

#     # bitmap with field
#     f = mv.bitmap(fs > 300, 0)
#     r = mv.bitmap(fs, f)
#     v_ref = fs.values() * f.values()
#     np.testing.assert_allclose(r.values(), v_ref)

#     # multiple fields
#     f = mv.fieldlist(path=os.path.join(PATH, "monthly_avg.grib"))
#     f = f[0:2]

#     # with value
#     f_mask = f > 300
#     r = mv.bitmap(f_mask, 1)
#     assert len(r) == len(f)
#     for i in range(len(r)):
#         v_ref = f_mask[i].values()
#         v_ref[v_ref == 1] = np.nan
#         np.testing.assert_allclose(r[i].values(), v_ref)

#     # with field
#     f1 = mv.bitmap(f > 300, 0)
#     r = mv.bitmap(f, f1)
#     assert len(r) == len(f1)
#     for i in range(len(r)):
#         v_ref = f[i].values() * f1[i].values()
#         np.testing.assert_allclose(r[i].values(), v_ref)

#     # with single field
#     f1 = mv.bitmap(f[0] > 300, 0)
#     r = mv.bitmap(f, f1)
#     assert len(r) == len(f)
#     for i in range(len(r)):
#         v_ref = f[i].values() * f1.values()
#         np.testing.assert_allclose(r[i].values(), v_ref)


# def test_nobitmap():

#     fs = mv.fieldlist(path=os.path.join(PATH, "t_with_missing.grib"))

#     # single field
#     f = fs
#     r = mv.nobitmap(f, 1)
#     assert len(r) == 1
#     v_ref = f.values()
#     v_ref[np.isnan(v_ref)] = 1
#     np.testing.assert_allclose(r.values(), v_ref)

#     # multiple fields
#     f = fs.merge(2 * fs)
#     r = mv.nobitmap(f, 1)
#     assert len(r) == 2

#     for i in range(len(r)):
#         v_ref = f[i].values()
#         v_ref[np.isnan(v_ref)] = 1
#         np.testing.assert_allclose(r[i].values(), v_ref)


# def test_grib_index_0():
#     # empty fieldlist
#     fs = mv.fieldlist()
#     gi = fs.grib_index()
#     assert gi == []


# def test_grib_index_1():
#     # single field
#     grib_path = os.path.join(PATH, "test.grib")
#     fs = mv.fieldlist(path=grib_path)
#     gi = fs.grib_index()
#     assert gi == [(grib_path, 0)]


# def test_grib_index_2():
#     # multiple fields
#     grib_path = os.path.join(PATH, "tuv_pl.grib")
#     fs = mv.fieldlist(path=grib_path)
#     gi = fs.grib_index()
#     assert isinstance(gi, list)
#     assert len(gi) == 18
#     for f, g in zip(fs, gi):
#         assert g == (grib_path, f.grib_get_long("offset"))
#     assert gi[5] == (grib_path, 7200)


# def test_grib_index_3():
#     # merged fields from different files
#     gp1 = os.path.join(PATH, "tuv_pl.grib")
#     gp2 = os.path.join(PATH, "t_time_series.grib")
#     fs1 = mv.fieldlist(path=gp1)
#     fs2 = mv.fieldlist(path=gp2)
#     fs3 = fs1[4:7]
#     fs3.append(fs2[1])
#     fs3.append(fs1[2])
#     gi = fs3.grib_index()
#     assert isinstance(gi, list)
#     assert len(gi) == 5
#     # assert gi == [(gp1, 5760), (gp1, 7200), (gp1, 8640), (gp2, 5520), (gp1, 2880)]
#     assert gi == [(gp1, 5760), (gp1, 7200), (gp1, 8640), (gp2, 5436), (gp1, 2880)]


# def test_grib_index_4():
#     # test with a derived fieldlist
#     fs = mv.fieldlist(os.path.join(PATH, "t_time_series.grib"))[0:4]
#     fs1 = fs + 1
#     gi = fs1.grib_index()
#     for i in gi:
#         assert is_temp_file(i[0])
#     offsets = [i[1] for i in gi]
#     assert offsets == [0, 8440, 16880, 25320]


# def test_grib_index_5():
#     # test with grib written with write() function
#     f_orig = mv.fieldlist(path=os.path.join(PATH, "tuv_pl.grib"))
#     f = (f_orig[0:4]).merge(f_orig[7])
#     p = "written_tuv_pl_5.grib"
#     f.write(p)
#     gi = f.grib_index()
#     assert gi == [(p, 0), (p, 1440), (p, 2880), (p, 4320), (p, 5760)]
#     f = 0
#     os.remove(p)

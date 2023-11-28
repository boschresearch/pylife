# Copyright (c) 2019-2023 - for information on the respective copyright owner
# see the NOTICE file and/or the repository
# https://github.com/boschresearch/pylife
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np
import pandas as pd
from scipy import signal as sg
from pylife.stress import timesignal as pts


def create_input_DF():
    fs = 2048
    t = np.arange(0, 30, 1/fs)
    ts_df = pd.DataFrame({"sin1": np.sin(10* 2 * np.pi * t),
                          "sin2": 2 * np.sin(10* 2 * np.pi * t + 0.05),
                          "cos": 10 * np.cos(5* 2 * np.pi * t),
                          "wn": np.random.rand(len(t))},
                         index=t)
    ts_df.index.name = "t"
    return ts_df


def test_fs_calc_true():
    df = create_input_DF()
    fs = 2048
    assert fs == pts.fs_calc(df)


def test_fs_calc_false():
    df = create_input_DF()
    df.index = df.index.astype(str)
    assert 1 == pts.fs_calc(df)


def test_resample_acc_sine():
    # sine
    omega = 10*2*np.pi  # Hz
    ts_sin = pd.DataFrame(np.sin(
        omega*np.arange(0, 2, 1/1024)), index=np.arange(0, 2, 1/1024))
    expected_sin = ts_sin.describe().drop(['count', 'mean', '50%', '25%', '75%'])
    test_sin = pts.resample_acc(ts_sin, int(12*omega)).describe().drop(
        ['count', 'mean', '50%', '25%', '75%'])
    pd.testing.assert_frame_equal(test_sin, expected_sin, rtol=2)


def test_resample_acc_wn():
    # white noise
    ts_wn = pd.DataFrame(np.random.randn(129), index=np.linspace(0, 1, 129))
    expected_wn = ts_wn.describe()
    test_wn = pts.resample_acc(ts_wn, fs = pts.fs_calc(ts_wn)).describe()
    pd.testing.assert_frame_equal(test_wn, expected_wn, check_exact=True)


def test_resample_acc_Sor():
    # SoR
    t = np.arange(0, 20, 1/4096)
    ts_sin = pd.DataFrame(np.sin(10 * 2 * np.pi * t), index=t)
    expected_sin = ts_sin.describe().drop(
        ['count', 'mean', '50%', '25%', '75%'])
    test_sin = pts.resample_acc(ts_sin, 2048).describe().drop(
        ['count', 'mean', '50%', '25%', '75%'])
    pd.testing.assert_frame_equal(test_sin, expected_sin, rtol=1e-2)


def test_resample_acc_sawtooth():
    # sawtooth
    t=t = np.arange(0, 10, 1/4096)
    ts_st = pd.DataFrame(sg.sawtooth(2 * np.pi * 1 * t), index=t)
    expected_st = ts_st.describe().drop(['count', '50%', 'mean'])
    test_st = pts.resample_acc(ts_st, 1024).describe().drop(['count', '50%', 'mean'])
    pd.testing.assert_frame_equal(test_st, expected_st, rtol=1e-2)


def test_ps_df():
    sample_frequency = 512
    t = np.linspace(0, 60, 60 * sample_frequency)
    ts_df = pd.DataFrame(data = np.array([np.random.randn(len(t)), np.sin(2 * np.pi * 10 * t)]).T,
                         columns=["wn", "sine"],
                         index=t)
    test_psd = pts.psd_df(ts_df, 512)

    np.testing.assert_allclose(test_psd.sum().values, np.array([1, 0.5]), rtol=1e-1)

# def test_running_stats_filt():
#     t = np.linspace(0, 1, 2048+1)
#     sin1 = -abs(150*np.sin(2*np.pi*10*t)+152)
#     sin2 = 300*np.sin(2*np.pi*10*t)
#     wn1 = np.ones(len(t))
#     wn1[500] = -250
#     wn2 = np.zeros(len(t))
#     wn2[0] = 350
#     df = pd.DataFrame(np.hstack((
#         sin1,
#         sin2,
#         wn1,
#         wn2)), columns=['data'])
#     df_sep = pd.DataFrame(np.vstack((
#         sin1,
#         sin2,
#         wn1,
#         wn2))).T
#     df_sep.columns = ['sin1', 'sin2', 'wn1', 'wn2']

#     test_rms = pts.TimeSignalPrep(df).running_stats_filt(
#         col='data', window_length=2049, buffer_overlap=0.0, limit=0.95, method="rms")
#     test_rms.columns = ['sin2']
#     pd.testing.assert_frame_equal(test_rms, pd.DataFrame(df_sep['sin2']),
#                                   check_index_type=False)

#     test_max = pts.TimeSignalPrep(df).running_stats_filt(
#         col='data', window_length=2049, buffer_overlap=0.0, limit=0.95, method="max")
#     test_max.columns = ['wn2']
#     pd.testing.assert_frame_equal(test_max, pd.DataFrame(df_sep['wn2']),
#                                   check_index_type=False)

#     test_min = pts.TimeSignalPrep(df).running_stats_filt(
#         col='data', window_length=2049, buffer_overlap=0.0, limit=1, method="min")
#     test_min.columns = ['sin1']
#     pd.testing.assert_frame_equal(test_min, pd.DataFrame(df_sep['sin1']),
#                                   check_index_type=False)

#     test_abs = pts.TimeSignalPrep(df).running_stats_filt(
#         col='data', window_length=2049, buffer_overlap=0.0, limit=0.1, method="abs")
#     pd.testing.assert_frame_equal(test_abs, df,
#                                   check_index_type=False)
#     test_sor = pts.resample_acc(df, 2048).describe().drop(
#         ['count', 'mean', '50%', '25%', '75%'])
#     # return expected_sor, test_sor
#     pd.testing.assert_frame_equal(test_sor, expected_sor, rtol=1e-2)
#     return


# %%
# Test timeseries

@pytest.fixture
def ts_inp():
    size_df = 10
    return pd.DataFrame(
        index=np.arange(size_df)+1,
        data={'x': np.arange(size_df)**2}
    )


@pytest.fixture
def ts_prep_rolling(ts_inp):
    return pts._prepare_rolling(ts_inp)


@pytest.fixture
def df_rolled(ts_prep_rolling):
    return pts._roll_dataset(ts_prep_rolling, window_size=3, overlap=1)


@pytest.fixture
def extracted_features(df_rolled):
    return pts._extract_feature_df(df_rolled, feature="maximum")


@pytest.fixture
def grid_points(ts_prep_rolling, extracted_features):
    return pts._select_relevant_windows(ts_prep_rolling, extracted_features, 'x__maximum',
                                        fraction_max=0.24, window_size=3,
                                        overlap=1, n_gridpoints=3)


@pytest.fixture
def poly_gridpoints(grid_points, ts_prep_rolling):
    ts_time = ts_prep_rolling.copy()
    delta_t = ts_time.index[1]-ts_time.index[0]
    line = pd.DataFrame(ts_time.iloc[:1], index=[- delta_t])
    ts_time = pd.concat([ts_time, line], ignore_index=False).sort_index()
    ts_time.index = ts_time.index + delta_t
    ts_time['time'] = ts_time.index.values

    return pts._polyfit_gridpoints(
        grid_points, ts_time, order=1, verbose=False, n_gridpoints=3
    )


@pytest.fixture
def ts_cleaned(ts_inp):
    return pts.clean_timeseries(
        ts_inp, 'x', window_size=3,
        overlap=1, feature="maximum", n_gridpoints=2,
        percentage_max=0.24, order=1
    )


def test_prepare_rolling(ts_inp, ts_prep_rolling):
    size_df = 10
    exact_res = pd.DataFrame({
        'x': np.arange(size_df)**2,
        'id': 0,
        'time': np.int64(np.arange(size_df))
    })
    exact_res.index = exact_res["time"]

    pd.testing.assert_frame_equal(exact_res, ts_prep_rolling, check_dtype=False, check_index_type=False)


def test_roll_dataset(df_rolled):
    x = np.array([0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8])

    df = pd.DataFrame({
        'id': np.int64(np.zeros(len(x), dtype=int)),
        'max_time': [2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8]
    })

    exact_res = pd.DataFrame({'x': x**2}, index=np.arange(len(x)))

    exact_res['id'] = pd.MultiIndex.from_frame(df)
    exact_res['time'] = np.int64(x)

    pd.testing.assert_frame_equal(exact_res, df_rolled, check_dtype=False, check_index_type=False)


# %%

@pytest.mark.skipif(not pts._HAVE_TSFRESH, reason="Don't have tsfresh")
def test_extract_feature_df(extracted_features):
    exact_res = pd.DataFrame({'x__maximum': [4., 16., 36., 64.]})
    pd.testing.assert_frame_equal(exact_res, extracted_features)


# %%


@pytest.mark.skipif(not pts._HAVE_TSFRESH, reason="Don't have tsfresh")
def test_select_relevant_windows(ts_prep_rolling, grid_points):
    ts_prep_selected = ts_prep_rolling.copy()
    ts_prep_selected.iloc[0:3, 0] = None

    exact_res = ts_prep_selected
    pd.testing.assert_frame_equal(exact_res, grid_points)


# %%


@pytest.mark.skipif(not pts._HAVE_TSFRESH, reason="Don't have tsfresh")
def test_polyfit_gridpoints1(grid_points, poly_gridpoints):
    x = [0, 4]
    y = [0, 9]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    delta_t = grid_points.index[1]-grid_points.index[0]
    line = pd.DataFrame(grid_points.iloc[:1], index=[- delta_t])
    grid_points = pd.concat([grid_points, line], ignore_index=False).sort_index()
    grid_points["time"] = grid_points.index + delta_t
    grid_points.index = grid_points["time"]
    grid_points.iloc[0, :] = 0

    grid_points.iloc[1, 0] = p(1)
    grid_points.iloc[2, 0] = p(2)
    grid_points.iloc[3, 0] = p(3)
    exact_res = grid_points
    exact_res["time"] = exact_res.index.values
    pd.testing.assert_frame_equal(exact_res, poly_gridpoints)


# %%

@pytest.mark.skipif(not pts._HAVE_TSFRESH, reason="Don't have tsfresh")
def test_clean_timeseries(poly_gridpoints, ts_cleaned):
    poly_gridpoints.pop("id")
    poly_gridpoints.index = poly_gridpoints["time"]
    pd.testing.assert_frame_equal(poly_gridpoints, ts_cleaned)


# %% Test timeseries 2

@pytest.fixture
def ts_gaps():
    t = np.linspace(0, np.pi, 20)
    y1 = np.sin(4*t)+1
    y2 = np.sin(t)+1
    return pd.DataFrame(index=t,
                        data=np.array([y1, y2]).T)


@pytest.fixture
def df_gaps_prep(ts_gaps):
    df_gaps_prep = pts._prepare_rolling(ts_gaps)
    return df_gaps_prep


@pytest.fixture
def df_gaps_rolled(df_gaps_prep):
    df_gaps_rolled = pts._roll_dataset(df_gaps_prep, window_size=5, overlap=2)
    return df_gaps_rolled


@pytest.fixture
def extraced_feature_gaps(df_gaps_rolled):
    extraced_feature_gaps = pts._extract_feature_df(
        df_gaps_rolled, feature="maximum")
    return extraced_feature_gaps


@pytest.fixture
def grid_points_gaps(df_gaps_prep, extraced_feature_gaps):
    return pts._select_relevant_windows(
        df_gaps_prep, extraced_feature_gaps,
        "0__maximum", fraction_max=0.80,
        window_size=5, overlap=2
    )


@pytest.mark.skipif(not pts._HAVE_TSFRESH, reason="Don't have tsfresh")
def test_select_relevant_windows2(df_gaps_prep, grid_points_gaps):
    ts_gaps_selected = df_gaps_prep.copy()
    ts_gaps_selected.iloc[3*2:3*2+5, 0:2] = np.nan
    ts_gaps_selected.iloc[15:20, 0:2] = np.nan
    list = [ts_gaps_selected.index[6], ts_gaps_selected.index[7],
            ts_gaps_selected.index[15], ts_gaps_selected.index[16]]

    exact_res = ts_gaps_selected.drop(list, axis=0)
    pd.testing.assert_frame_equal(exact_res, grid_points_gaps)


# %%

@pytest.mark.skipif(not pts._HAVE_TSFRESH, reason="Don't have tsfresh")
def test_clean_timeseries2(ts_gaps, df_gaps_prep, grid_points_gaps):
    poly_gridpoints_gaps = pts._polyfit_gridpoints(
        grid_points_gaps, df_gaps_prep, order=3, verbose=False, n_gridpoints=3)
    poly_gridpoints_gaps = poly_gridpoints_gaps.dropna(axis=0, how='any')
    poly_gridpoints_gaps.pop("id")
    exact_res = poly_gridpoints_gaps
    ts_cleaned = pts.clean_timeseries(ts_gaps, "0", window_size=5,
                                       overlap=2, feature="maximum", n_gridpoints=3,
                                       percentage_max=0.91, order=3)
    pd.testing.assert_frame_equal(exact_res, ts_cleaned)

# %% Test timeseries 3


@pytest.mark.skipif(not pts._HAVE_TSFRESH, reason="Don't have tsfresh")
def test_clean_timeseries3():
    t1 = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    t2 = np.linspace(2*np.pi, 4*np.pi, 1000)
    y1 = 100 * np.cos(8*t1)
    y2 = 0.1 * np.sin(8*t2)
    t = np.append(t1, t2)
    y = np.append(y1, y2)
    ts_sin = pd.DataFrame(index=t,
                          data=y)

    ts_sin_test = ts_sin.copy()
    ts_cleaned = pts.clean_timeseries(ts_sin_test, "0", window_size=100,
                                       overlap=99, feature="maximum", n_gridpoints=5,
                                       percentage_max=0.3, order=3)

    ts_time = pts._prepare_rolling(ts_sin)

    exact_res = pd.DataFrame(index=t1, data=y1)
    exact_res = pts._prepare_rolling(exact_res)

    delta_t = exact_res.index[1]-exact_res.index[0]
    line = pd.DataFrame(exact_res.iloc[:1], index=[- delta_t])
    exact_res = pd.concat([exact_res, line], ignore_index=False).sort_index()
    exact_res.iloc[0, :] = 0
    exact_res["time"] = exact_res.index + delta_t
    exact_res.index = exact_res["time"]
    exact_res.pop('id')

    pd.testing.assert_frame_equal(exact_res, ts_cleaned)

# Copyright (c) 2019-2021 - for information on the respective copyright owner
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


import numpy as np
import pandas as pd
from scipy import signal as sg
from pylife.stress import timesignal as tsig

# %%


def test_resample_acc():
    # sine
    omega = 10*2*np.pi  # Hz
    ts_sin = pd.DataFrame(np.sin(
        omega*np.arange(0, 2, 1/1024)), index=np.arange(0, 2, 1/1024))
    expected_sin = ts_sin.describe().drop(['count', 'mean', '50%', '25%', '75%'])
    test_sin = tsig.resample_acc(ts_sin, int(12*omega)).describe().drop(
        ['count', 'mean', '50%', '25%', '75%'])
    pd.testing.assert_frame_equal(test_sin, expected_sin, check_less_precise=2)
    # white noise
    ts_wn = pd.DataFrame(np.random.randn(129), index=np.linspace(0, 1, 129))
    expected_wn = ts_wn.describe()
    test_wn = tsig.resample_acc(ts_wn, 128).describe()
    pd.testing.assert_frame_equal(test_wn, expected_wn, check_exact=True)
    # SoR
    t = np.arange(0, 20, 1/4096)
    f = np.logspace(np.log10(1), np.log10(10), num=len(t))
    df = pd.DataFrame(data=np.multiply(
        np.interp(f, np.array([1, 10]), np.array([0.5, 5]), left=0, right=0),
        sg.chirp(t, 1, 20, 10, method="logarithmic", phi=-90)),
        index=t, columns=["Sweep"])
    df['sor'] = df['Sweep'] + 0.1*np.random.randn(len(df))
    expected_sor = df.describe().drop(['count', 'mean', '50%', '25%', '75%'])
    test_sor = tsig.resample_acc(df, 2048).describe().drop(['count', 'mean', '50%', '25%', '75%'])
    # return expected_sor, test_sor
    pd.testing.assert_frame_equal(test_sor, expected_sor, check_less_precise=1)


# %%
# Test timeseries 1
print("doing timeseries 1")
size_df = 10
ts_inp = pd.DataFrame(index=np.arange(size_df)+1,
                      data=np.arange(size_df)**2)
ts_inp_test = ts_inp.copy()


def test_prepare_rolling():

    exact_res = pd.DataFrame()
    exact_res.loc[:, 0] = np.arange(size_df)**2
    exact_res['id'] = 0
    exact_res['time'] = np.int64(np.arange(size_df))
    exact_res.index = exact_res["time"]
    global ts_prep_rolling
    ts_prep_rolling = tsig._prepare_rolling(ts_inp)

    pd.testing.assert_frame_equal(exact_res, ts_prep_rolling)
    return ts_prep_rolling





def test_roll_dataset():
    global df_rolled
    df_rolled = tsig._roll_dataset(ts_prep_rolling, timeshift=3, rolling_direction=2)
    exact_res = pd.DataFrame()
    x = np.array([0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8])
    df = pd.DataFrame()

    df["id"] = np.int64(np.zeros(len(x), dtype=int))
    df["max_time"] = [2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8]

    exact_res.loc[:, 0] = x**2

    exact_res.index = np.arange(len(x))
    exact_res["id"] = pd.MultiIndex.from_frame(df)
    exact_res['time'] = np.int64(x)

    pd.testing.assert_frame_equal(exact_res, df_rolled)
    return df_rolled




def test_extract_features_df():
    global extracted_features
    extracted_features = tsig._extract_features_df(df_rolled, feature="maximum")
    exact_res = pd.DataFrame()
    exact_res["0__maximum"] = [4., 16., 36., 64.]
    pd.testing.assert_frame_equal(exact_res, extracted_features)
    return extracted_features



# %%


def test_select_relevant_windows():

    ts_prep_selected = ts_prep_rolling.copy()
    global grid_points
    global liste
    grid_points = tsig._select_relevant_windows(
        ts_prep_rolling, extracted_features, "0__maximum", fraction_max=0.24, timeshift=3, rolling_direction=2, n_gridpoints=3)
    ts_prep_selected.iloc[0:3, 0] = None
    global exact_res
    exact_res = ts_prep_selected
    pd.testing.assert_frame_equal(exact_res, grid_points)
    return grid_points





# %%
def test_polyfit_gridpoints1():
    global ts_grid_test
    ts_grid_test = grid_points.copy()
    global poly_gridpoints
    ts_time = ts_prep_rolling.copy()
    delta_t = ts_time.index[1]-ts_time.index[0]
    ts_time.loc[-delta_t] = ts_time.loc[0]  # adding a row
    ts_time.index = ts_time.index + delta_t  # shifting index
    ts_time = ts_time.sort_index()  # sorting by index

    poly_gridpoints = tsig._polyfit_gridpoints(
        grid_points, ts_time, order=1, verbose=False, n_gridpoints=3)
    x = [0, 4]
    y = [0, 9]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    ts_grid_test.loc[-1] = ts_grid_test.loc[0]  # adding a row
    ts_grid_test.index = ts_grid_test.index + 1  # shifting index
    ts_grid_test = ts_grid_test.sort_index()  # sorting by index
    ts_grid_test.loc[0] = 0

    ts_grid_test.iloc[1, 0] = p(1)
    ts_grid_test.iloc[2, 0] = p(2)
    ts_grid_test.iloc[3, 0] = p(3)
    exact_res = ts_grid_test
    exact_res["time"] = exact_res.index.values

    pd.testing.assert_frame_equal(exact_res, poly_gridpoints)
    return poly_gridpoints




# %%


def test_clean_dataset():
    global ts_cleaned
    ts_cleaned = tsig.clean_dataset(ts_inp_test, "0__maximum", timeshift=3, rolling_direction=2,
                                    feature="maximum", n_gridpoints=2, percentage_max=0.24, order=1)
    poly_gridpoints.pop("id")
    poly_gridpoints.index = poly_gridpoints["time"]
    pd.testing.assert_frame_equal(poly_gridpoints, ts_cleaned)
    return ts_cleaned,




# %% Test timeseries 2
print("doing timeseries 2")
t = np.linspace(0, np.pi, 20)
y1 = np.sin(4*t)+1
y2 = np.sin(t)+1

ts_gaps = pd.DataFrame(index=t,
                       data=np.array([y1, y2]).T)
ts_gaps_test = ts_gaps.copy()


df_gaps_prep = tsig._prepare_rolling(ts_gaps)
df_gaps_rolled = tsig._roll_dataset(df_gaps_prep, timeshift=5, rolling_direction=3)
extraced_feature_gaps = tsig._extract_features_df(df_gaps_rolled, feature="maximum")


def test_select_relevant_windows2():
    ts_gaps_selected = df_gaps_prep.copy()
    global grid_points_gaps
    grid_points_gaps = tsig._select_relevant_windows(
        df_gaps_prep, extraced_feature_gaps, "0__maximum", fraction_max=0.93, timeshift=5, rolling_direction=3)

    ts_gaps_selected.iloc[3*2:3*2+5, 0:2] = np.nan
    list = [ts_gaps_selected.index[6], ts_gaps_selected.index[7]]
    global exact_res
    exact_res = ts_gaps_selected.drop(list, axis=0)

    pd.testing.assert_frame_equal(exact_res, grid_points_gaps)
    return grid_points_gaps




# %%


def test_clean_dataset2():
    poly_gridpoints_gaps = tsig._polyfit_gridpoints(
        grid_points_gaps, df_gaps_prep, order=3, verbose=False, n_gridpoints=3)
    poly_gridpoints_gaps.pop("id")
    exact_res = poly_gridpoints_gaps
    ts_cleaned = tsig.clean_dataset(ts_gaps_test, "0__maximum", timeshift=5, rolling_direction=3,
                                    feature="maximum", n_gridpoints=3, percentage_max=0.91, order=3)
    pd.testing.assert_frame_equal(exact_res, ts_cleaned)
    return ts_cleaned



# %% Test timeseries 3
print("doing timeseries 3")
t1 = np.linspace(0, 2*np.pi, 1000, endpoint=False)
t2 = np.linspace(2*np.pi, 4*np.pi, 1000)
y1 = 100 * np.cos(8*t1)
y2 = 0.1 * np.sin(8*t2)
t = np.append(t1, t2)
y = np.append(y1, y2)
ts_sin = pd.DataFrame(index=t,
                      data=y)


# %%
def test_clean_dataset3():

    ts_sin_test = ts_sin.copy()
    ts_cleaned = tsig.clean_dataset(ts_sin_test,"0__maximum", timeshift=100, rolling_direction=1,
                                    feature="maximum", n_gridpoints=5, percentage_max=0.2, order=3)
    #ts_cleaned.plot(subplots=True, sharex=True, figsize=(10,10))
    # plt.show()
    ts_time = tsig._prepare_rolling(ts_sin)

    exact_res = pd.DataFrame(index=t1, data=y1)
    exact_res = tsig._prepare_rolling(exact_res)

    top_row = exact_res.iloc[0, :]
    top_row.name = 0.
    top_row = pd.DataFrame(top_row)

    top_row = top_row.transpose()
    top_row.iloc[0, :] = 0

    exact_res = pd.concat([top_row, exact_res])

    ts_time = ts_time.head(len(exact_res))
    exact_res["time"] = ts_time["time"].values
    exact_res.index = exact_res["time"]
    exact_res.pop("id")
    pd.testing.assert_frame_equal(exact_res, ts_cleaned)
    return ts_cleaned

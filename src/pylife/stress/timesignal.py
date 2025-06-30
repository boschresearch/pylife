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

"""A module for time signal handling

Warning
-------

This module is not considered finalized even though it is part of pylife-2.0.
Breaking changes might occur in upcoming minor releases.
"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal

try:
    import tsfresh as ts

    _HAVE_TSFRESH = True
except ModuleNotFoundError:
    _HAVE_TSFRESH = False


class TimeSignalGenerator:
    r"""Generates mixed time signals

    The generated time signal is a mixture of random sets of sinus signals

    For each set the user supplys a dict describing the set::

      sinus_set = {
          'number': number of signals
          'amplitude_median':
          'amplitude_std_dev':
          'frequency_median':
          'frequency_std_dev':
          'offset_median':
          'offset_std_dev':
      }

    The amplitudes (:math:`A`), fequencies (:math:`\omega`) and
    offsets (:math:`c`) are then norm distributed. Each sinus signal
    looks like

            :math:`s = A \sin(\omega t + \phi) + c`

    where :math:`phi` is a random value between 0 and :math:`2\pi`.

    So the whole sinus :math:`S` set is given by the following expression:

            :math:`S = \sum^n_i A_i \sin(\omega_i t + \phi_i) + c_i`.
    """

    def __init__(self, sample_rate, sine_set, gauss_set, log_gauss_set):
        sine_amplitudes = stats.norm.rvs(
            loc=sine_set["amplitude_median"],
            scale=sine_set["amplitude_std_dev"],
            size=sine_set["number"],
        )
        sine_frequencies = stats.norm.rvs(
            loc=sine_set["frequency_median"],
            scale=sine_set["frequency_std_dev"],
            size=sine_set["number"],
        )
        sine_offsets = stats.norm.rvs(
            loc=sine_set["offset_median"],
            scale=sine_set["offset_std_dev"],
            size=sine_set["number"],
        )
        sine_phases = 2.0 * np.pi * np.random.rand(sine_set["number"])

        self.sine_set = list(
            zip(sine_amplitudes, sine_frequencies, sine_phases, sine_offsets)
        )

        self.sample_rate = sample_rate
        self.time_position = 0.0

    def query(self, sample_num):
        """Gets a sample chunk of the time signal

        Parameters
        ----------
        sample_num : int
            number of the samples requested

        Returns
        -------
        samples : 1D numpy.ndarray
            the requested samples


        You can query multiple times, the newly delivered samples
        will smoothly attach to the previously queried ones.
        """
        samples = np.zeros(sample_num)
        end_time_position = self.time_position + (sample_num - 1) / self.sample_rate

        for ampl, omega, phi, offset in self.sine_set:
            periods = np.floor(self.time_position / omega)
            start = self.time_position - periods * omega
            end = end_time_position - periods * omega
            time = np.linspace(start, end, sample_num)
            samples += ampl * np.sin(omega * time + phi) + offset

        self.time_position = end_time_position + 1.0 / self.sample_rate

        return samples

    def reset(self):
        """Resets the generator

        A resetted generator behaves like a new generator.
        """
        self.time_position = 0.0


def fs_calc(df):
    """
    Calculates the sample frequency of a DataFrame time series

    Parameters
    ----------
    df : DataFrame
        time series.

    Returns
    -------
    fs : int, float
        sample freqency

    """
    try:
        fs = np.rint(1 / np.mean(np.diff(df.index)))
    except TypeError:
        print("Index has to be a number not a string. We assume fs = 1")
        fs = 1
    return fs


def resample_acc(df, fs=1):
    """Resamples a pandas time series DataFrame

    Parameters
    ----------
    df: DataFrame

    time_col: str
        column name of the time column
    fs: float
        sample rate of the resampled time series

    Returns
    -------
    DataFrame
    """
    index_new = np.arange(df.index.min(), df.index.max() + 1 / fs, 1 / fs)

    df_rs = pd.DataFrame(
        df.apply(lambda x: np.interp(index_new, df.index, x)).values,
        index=index_new,
        columns=df.columns,
    )
    return df_rs


def butter_bandpass(df, lowcut, highcut, order=5):
    """Use the functonality of scipy


    Parameters
    ----------

    df: DataFrame
    lowcut : float
        low frequency
    highcut : float
        high freqency.
    order : int, optional
        Butterworth filter order. The default is 5.

    Returns
    -------
    TSout : DataFrame

    """
    fs = fs_calc(df)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="bandpass")
    return df.apply(lambda x: signal.filtfilt(b, a, x, padlen=int(fs / 2)))


def psd_df(df_ts, nfft=512, nperseg=256):
    """
    calculates the psd using Welch algorithm from matplotlib functionality

    Parameters
    ----------
    df_ts : DataFram
        Time series dataframe
    nfft : int, optional
        Length of the FFT. The default is 512.

    Returns
    -------
    df_psd : DataFrame
        PSD.

    """

    nperseg = min(nperseg, nfft)
    fs = fs_calc(df_ts)
    df_psd = pd.DataFrame()
    for col in df_ts:
        freq, df_psd[col] = signal.welch(
            df_ts[col].values, fs=fs, nperseg=nperseg, nfft=nfft
        )
    df_psd.index = pd.Index(freq, name="frequency")
    return df_psd


def _prepare_rolling(df):
    """
    Adds ID, time to the dataset for TsFresh, We would need different ID's if we had
    independant timeseries -like timeseries for different robots.

    Parameters
    ----------
    df: pandas DataFrame
        input data
    self : TimeSignalPrep class


    Returns
    -------
    df : pandas DataFrame
        output DataFrame with added id, time

    """
    prep_roll = df.copy()
    prep_roll["id"] = 0
    prep_roll["time"] = df.index.values
    prep_roll["time"] = prep_roll["time"].subtract(prep_roll["time"].values[0])
    prep_roll.index = prep_roll["time"]

    return prep_roll


def _roll_dataset(prep_roll_df, window_size=1000, overlap=200):
    """
    Rolls dataset in windows so we can later extract features from every window
    Parameters
    ----------
    prep_roll: output from prepare_rolling

    window_size : int , optional
         window size of the rolled segments  -the default is 1000.
    overlap : int, optional
         overlap between 2 adjecent windows -The default is 200.

    Returns
    -------
    df_rolled : pandas DataFrame
        rolled DataFrame

    """

    # Create Rolled Dataset with Parameter rolling_direction & window_size
    # throws away the last halfshift
    rolling_direction = window_size - overlap
    cycles = int((len(prep_roll_df) - window_size) / rolling_direction) + 1

    parts = []
    # shiften
    for i in range(cycles):
        position = (rolling_direction) * i
        shift = prep_roll_df.iloc[position : position + window_size, :].copy()
        # change IDs to format (id,time)
        df = pd.DataFrame(
            {
                "id": np.int64(np.zeros(len(shift), dtype=int)),
                "max_time": shift.iloc[-1, -1],
            }
        )

        shift["id"] = pd.MultiIndex.from_frame(df).to_numpy()

        parts.append(shift)

    return pd.concat(parts, ignore_index=True)


def _extract_feature_df(df_rolled, feature="maximum"):
    """Extracts features like "abs_energy" or "maximum" from the rolled dataset with TsFresh

    Parameters
    ----------
    df_rolled : pandas DataFrame
        rolled DataFrame from roll_dataset
    feature : string, optional
        Extracted feature - only supports one at a time -
        and only features form tsfresh that dont need extra parameters.
        The default is "maximum".

    Returns
    -------
    extracted_features : pandas DataFrame
        Dataframe of extracted features

    """
    # extract features

    # fc_parameters = {"abs_energy", "maximum"}
    fc_parameters = {
        feature: None,
    }
    extracted_features = ts.extract_features(
        df_rolled,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc_parameters,
        n_jobs=0,
    )
    extracted_features.index = range(len(extracted_features))
    return extracted_features


def _select_relevant_windows(
    prep_roll,
    extracted_features,
    comparison_column_ex,
    fraction_max=0.25,
    window_size=1000,
    overlap=200,
    n_gridpoints=3,
    method="keep",
):
    """Writes n_gridpoints NaN's into the window_sizes with extracted features
    lower than fraction_max

    Parameters
    ----------
    prep_roll : pandas DataFrame
        input data - normally output from perpare_rolling(df)
    extracted_features : pandas Dataframe
        DataFrame of features
    comparison_column_ex: string - name of the extraced feature column
        it is build: comparison_column + '__' + feauture
    fraction_max : float
        percentage of the maximum of the extraced feature.
    window_size : int
        window size of the rolled segments  -the default is 1000.
    overlap : int, optional
         overlap between 2 adjecent windows -The default is 200.


    Returns
    -------
    df : pandas DataFrame relevant_windows
        dataframe with NaN's in the windows with too low extracted features

    """
    # get added up abs energy of interval x, if too low set None
    rolling_direction = window_size - overlap

    relevant_feature = extracted_features[comparison_column_ex]
    relevant_windows = prep_roll.copy()
    just_added_NaNs = False
    liste = []
    for i in range(len(extracted_features)):
        if relevant_feature[i] <= relevant_feature.max() * fraction_max:
            if just_added_NaNs is True:
                liste.append(
                    list(
                        range(
                            0 + i * rolling_direction,
                            window_size + i * rolling_direction,
                        )
                    )
                )

            else:
                liste.append(
                    list(
                        range(
                            0 + i * rolling_direction,
                            window_size + i * rolling_direction - n_gridpoints,
                        )
                    )
                )
                relevant_windows.iloc[
                    i * rolling_direction
                    + window_size
                    - n_gridpoints : i * rolling_direction
                    + window_size,
                    0 : relevant_windows.shape[1] - 2,
                ] = None
                just_added_NaNs = True
        else:
            just_added_NaNs = False

    index_liste = []
    """
    tail = (len(prep_roll)-window_size) % rolling_direction+1
    for i in range(tail):
        liste.append(len(prep_roll)-i-1)
    """
    liste = list(pd.core.common.flatten(liste))
    liste = list(set(liste))
    for i in range(len(liste)):
        index_liste.append(relevant_windows.index[liste[i]])
    if method == "keep":
        relevant_windows = relevant_windows.drop(index_liste, axis=0)
    elif method == "remove":
        relevant_windows = relevant_windows.loc[index_liste]
    return relevant_windows


def _polyfit_gridpoints(grid_points, prep_roll, order=3, verbose=False, n_gridpoints=3):
    """Fills gridpoints with polynomial regression

    Parameters
    ----------
    gridpoints : pandas DataFrame
        DataFrame with NaN's as gridpoints
    prep_roll : pandas DataFrame used to create time axis.
        DataFrame used to create time axis.
    order : int, optional
        Order of polynom The default is 3.
    verbose : boolean, optional
        If true plots polyfits. The default is False.
    n_gridpoints : TYPE, optional
        Number of gridpoints. The default is 3.

    Returns
    -------
    df : pandas DataFrame
        DataFrame with polynomial values at the gridpoints.
    """

    # add a null row at the start and reset time index
    delta_t = prep_roll.index[1] - prep_roll.index[0]
    line = pd.DataFrame(grid_points.iloc[:1], index=[-delta_t])
    grid_points = pd.concat([grid_points, line], ignore_index=False)
    grid_points.index = grid_points.index + delta_t
    poly_gridpoints = grid_points.sort_index()
    poly_gridpoints.iloc[0, :] = 0
    ts_time = prep_roll.iloc[: len(poly_gridpoints)]

    poly_gridpoints["time"] = ts_time.index.values
    poly_gridpoints.index = poly_gridpoints["time"]

    # %% smooth the gaps with polynomial values
    poly_gridpoints.interpolate(method="polynomial", order=order, inplace=True)

    return poly_gridpoints


def clean_timeseries(
    df,
    comparison_column,
    window_size=1000,
    overlap=800,
    feature="abs_energy",
    method="keep",
    n_gridpoints=3,
    percentage_max=0.05,
    order=3,
):
    """Removes segments of the data in which the extracted feature value is lower as
    percentage_max and fills the gaps with polynomial regression

    Parameters
    ----------
    df : input pandas DataFrame that shall be cleaned
    comparison_column: str, column that is used for the feature
        comparison with percentage max
    window_size : int, optional
        window size of the rolled segments - The default is 1000.
    overlap : int, optional
         overlap between 2 adjecent windows -The default is 200.
    feature : string, optional
        extracted feature - only supports one at a time -
        and only features form tsfresh that dont need extra parameters.
        The default is "maximum".
    method: string, optional
        * 'keep': keeps the windows which are extracted,
        * 'remove': removes the windows which are extracted
    n_gridpoints : TYPE, optional
        number of gridpoints. The default is 3.
    percentage_max : float, optional
        min percentage of the maximum to keep the window. The default is 0.05.
    order : int, optional
        order of polynom The default is 3.

    Returns
    -------
    df_poly : pandas DataFrame
        cleaned DataFrame

    """

    if not _HAVE_TSFRESH:
        raise ImportError(
            "tsfresh and dependencies are not installed. "
            "Use `pip install pylife[tsfresh]` to install it."
        )

    df_prep = _prepare_rolling(df)
    ts_time = df_prep.copy()
    # adding a row
    delta_t = ts_time.index[1] - ts_time.index[0]
    line = pd.DataFrame(ts_time.iloc[:1], index=[-delta_t])
    ts_time = pd.concat([ts_time, line], ignore_index=False)
    ts_time.index = ts_time.index + delta_t
    ts_time = ts_time.sort_index()

    ts_time["time"] = ts_time.index.values

    comparison_column_ex = comparison_column + "__" + feature
    df_rolled = _roll_dataset(df_prep, window_size=window_size, overlap=overlap)
    extracted_features = _extract_feature_df(df_rolled, feature)
    grid_points = _select_relevant_windows(
        df_prep,
        extracted_features,
        comparison_column_ex,
        percentage_max,
        window_size,
        overlap,
        method=method,
    )

    poly_gridpoints = _polyfit_gridpoints(
        grid_points, ts_time, order=order, verbose=False, n_gridpoints=n_gridpoints
    )

    # Remove NaN's at the end - should be maximum 2n
    cleaned = poly_gridpoints.dropna(axis=0, how="any")
    cleaned.pop("id")

    return cleaned

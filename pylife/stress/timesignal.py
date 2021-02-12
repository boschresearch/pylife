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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.pyplot as plt
import time
from tsfresh import extract_features
from pandas.core.common import flatten


class TimeSignalGenerator:
    '''Generates mixed time signals

    The generated time signal is a mixture of random sets of

    * sinus signals
    * gauss signals (not yet)
    * log gauss signals (not yet)

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
    '''

    def __init__(self, sample_rate, sine_set, gauss_set, log_gauss_set):
        sine_amplitudes = stats.norm.rvs(loc=sine_set['amplitude_median'],
                                         scale=sine_set['amplitude_std_dev'],
                                         size=sine_set['number'])
        sine_frequencies = stats.norm.rvs(loc=sine_set['frequency_median'],
                                          scale=sine_set['frequency_std_dev'],
                                          size=sine_set['number'])
        sine_offsets = stats.norm.rvs(loc=sine_set['offset_median'],
                                      scale=sine_set['offset_std_dev'],
                                      size=sine_set['number'])
        sine_phases = 2. * np.pi * np.random.rand(sine_set['number'])

        self.sine_set = list(zip(sine_amplitudes, sine_frequencies, sine_phases, sine_offsets))

        self.sample_rate = sample_rate
        self.time_position = 0.0

    def query(self, sample_num):
        '''Gets a sample chunk of the time signal

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
        '''
        samples = np.zeros(sample_num)
        end_time_position = self.time_position + (sample_num-1) / self.sample_rate

        for ampl, omega, phi, offset in self.sine_set:
            periods = np.floor(self.time_position / omega)
            start = self.time_position - periods * omega
            end = end_time_position - periods * omega
            time = np.linspace(start, end, sample_num)
            samples += ampl * np.sin(omega * time + phi) + offset

        self.time_position = end_time_position + 1. / self.sample_rate

        return samples

    def reset(self):
        ''' Resets the generator

        A resetted generator behaves like a new generator.
        '''
        self.time_position = 0.0


def resample_acc(df, fs=1):
    """ Resamples a pandas time series DataFrame

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
    index_new = np.linspace(
        df.index.min(),
        df.index.min() + np.floor((df.index.max()-df.index.min())*fs)/fs,
        int(np.floor(df.index.max()-df.index.min())*fs + 1))

    df_rs = pd.DataFrame(df.apply(lambda x: np.interp(index_new, df.index, x)).values,
                         index=index_new, columns=df.columns)
    return df_rs


def butter_bandpass(df, lowcut, highcut, fs, order=5):
    """ Use the functonality of scipy


    Parameters
    ----------

    df: DataFrame
    lowcut : float
        low frequency
    highcut : float
        high freqency.
    fs: float
        sample rate of the resampled time series
    order : int, optional
        Butterworth filter order. The default is 5.

    Returns
    -------
    TSout : DataFrame

    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return df.apply(lambda x: signal.filtfilt(b, a, x))


def _prepare_rolling(df):
    """
    Adds ID, time to the dataset for TsFresh

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
    start = time.time()
    prep_roll["id"] = 0
    prep_roll["time"] = df.index.values
    prep_roll["time"] = prep_roll["time"].subtract(prep_roll["time"].values[0])
    prep_roll.index = prep_roll["time"]
    end = time.time()

    print("Prepare_rolling: {:5.3f}s".format(end - start))
    return prep_roll


def _roll_dataset(prep_roll_df, timeshift=1000, rolling_direction=800):
    """
    rolls dataset
    Parameters
    ----------
    prep_roll: output from prepare_rolling

    timeshift : int , optional
         window size -the default is 1000.
    rolling_direction : int, optional
         windowshift -The default is 800.

    Returns
    -------
    df_rolled : pandas DataFrame
        rolled DataFrame

    """

    # Create Rolled Dataset with Parameter rolling_direction & timeshift
    # throws away the last halfshift
    pd.options.mode.chained_assignment = None  # stops the copyslice warning
    start = time.time()
    cycles = int(len(prep_roll_df) / rolling_direction) - 1
    df_rolled_is_empty = True
    # shiften
    for i in range(cycles):
        position = (rolling_direction) * i
        shift = prep_roll_df.iloc[position: position + timeshift, :]
        # change IDs to format (id,time)
        shift.loc[:, ("max_time")] = max(shift.loc[:, ("time")])
        df = shift.loc[:, ("id", "max_time")]
        shift.loc[:, ("id")] = pd.MultiIndex.from_frame(df)
        # delete max_time
        shift.pop("max_time")
        if df_rolled_is_empty:
            df_rolled = shift
            df_rolled_is_empty = False
        else:
            df_rolled = df_rolled.append(shift, ignore_index=True)

    ende = time.time()
    print("roll_dataset: {:5.3f}s".format(ende - start))
    return df_rolled


def _extract_features_df(df_rolled, feature="maximum"):
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
        Dataframe of extraced features

    """
    # extract features
    start = time.time()

    # fc_parameters = {"abs_energy", "maximum"}
    fc_parameters = {
        feature: None,
    }
    extracted_features = extract_features(
        df_rolled,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc_parameters,
        n_jobs=0,
    )
    extracted_features.index = range(len(extracted_features))
    ende = time.time()
    print("extract_features_df: {:5.3f}s".format(ende - start))
    return extracted_features


def _select_relevant_windows(prep_roll, extracted_features, comparison_column, fraction_max=0.25,
                             timeshift=1000, rolling_direction=800, n_gridpoints=3):
    """Writes NaN's into the timeshifts with extracted features lower than fraction_max

    Parameters
    ----------
    prep_roll : pandas DataFrame
        input data - normally output from perpare_rolling(df)
    extracted_features : pandas Dataframe
        DataFrame of features
    fraction_max : float
        percentage of the maximum of the extraced feature.
    timeshift : int
        window size -the default is 1000.
    rolling_direction : TYPE
        window shift- the default is 800

    Returns
    -------
    df : pandas DataFrame relevant_windows
        dataframe with NaN's in the windows with too low extracted features

    """
    # get added up abs energy of interval x, if too low set None
    start = time.time()
    added_feature = np.zeros(len(extracted_features))
    for i in range(prep_roll.shape[1] - 2):
        added_feature += extracted_features.iloc[:, i]
    relevant_feature = extracted_features[comparison_column]

    relevant_windows = prep_roll.copy()
    just_added_NaNs = False
    liste = []
    for i in range(len(extracted_features)):
        if relevant_feature.iloc[i] <= (max(relevant_feature) * fraction_max):
            if just_added_NaNs is True:
                liste.append(list(range(0 + i * rolling_direction,
                                        timeshift + i * rolling_direction)))

            else:
                liste.append(list(range(0 + i * rolling_direction,
                                        timeshift + i * rolling_direction - n_gridpoints)))
                relevant_windows.iloc[i * rolling_direction + timeshift - n_gridpoints:i *
                                      rolling_direction + timeshift,
                                      0:relevant_windows.shape[1]-2] = None
                just_added_NaNs = True
        else:
            just_added_NaNs = False

    index_liste = []

    liste = list(flatten(liste))
    liste = list(set(liste))
    print(liste)
    for i in range(len(liste)):
        index_liste.append(relevant_windows.index[liste[i]])

    relevant_windows = relevant_windows.drop(index_liste, axis=0)
    ende = time.time()
    print("select_relevant_windows: {:5.3f}s".format(ende - start))
    return relevant_windows


def _polyfit_gridpoints(grid_points, prep_roll, order=3,
                        verbose=False, n_gridpoints=3):
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
    start = time.time()
    # add a null row at the start and reset time index

    delta_t = prep_roll.index[1]-prep_roll.index[0]
    grid_points.loc[-delta_t] = grid_points.loc[0]  # adding a row
    grid_points.index = grid_points.index + delta_t  # shifting index
    poly_gridpoints = grid_points.sort_index()  # sorting by index

    ts_time = prep_roll.head(len(poly_gridpoints))

    poly_gridpoints["time"] = ts_time.index.values
    poly_gridpoints.index = poly_gridpoints["time"]
    poly_gridpoints.iloc[0, :] = 0
    # %% smooth the gaps with polynomial values
    print(poly_gridpoints)
    poly_gridpoints.interpolate(method='polynomial', order=order, inplace=True)
    ende = time.time()
    print('Total Cleaning: {:5.3f}s'.format(ende-start))
    return poly_gridpoints


def clean_dataset(df, comparison_column, timeshift=1000, rolling_direction=800,
                  feature="abs_energy", n_gridpoints=3,
                  percentage_max=0.05, order=3):
    """ Removes irrelevant parts of the data and fills the gaps with polynomial regression

    Parameters
    ----------
    timeshift : int, optional
        window size - The default is 1000.
    rolling_direction : int, optional
        window shift - The default is 800.
    feature : string, optional
        extracted feature - only supports one at a time -
        and only features form tsfresh that dont need extra parameters.
        The default is "maximum".
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

    start = time.time()

    df_prep = _prepare_rolling(df)
    ts_time = df_prep.copy()

    delta_t = ts_time.index[1]-ts_time.index[0]
    ts_time.loc[-delta_t] = ts_time.loc[0]  # adding a row
    ts_time.index = ts_time.index + delta_t  # shifting index
    ts_time = ts_time.sort_index()  # sorting by index

    df_rolled = _roll_dataset(df_prep, timeshift=timeshift,
                              rolling_direction=rolling_direction)
    extracted_features = _extract_features_df(df_rolled, feature)
    grid_points = _select_relevant_windows(df_prep, extracted_features, comparison_column,
                                           percentage_max, timeshift, rolling_direction)
    print(grid_points)
    poly_gridpoints = _polyfit_gridpoints(grid_points, ts_time, order=order, verbose=False,
                                          n_gridpoints=n_gridpoints)

    # Remove NaN's at the end - should be maximum 2n
    len_poly = poly_gridpoints.shape[0]
    cleaned = poly_gridpoints.dropna(axis=0, how='any', thresh=None, subset=None)
    cleaned.pop("id")

    print("Number of NaN's dropped at END:", len_poly-poly_gridpoints.shape[0])
    ende = time.time()
    print('Total Cleaning: {:5.3f}s'.format(ende-start))

    return cleaned

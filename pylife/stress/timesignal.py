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


class TimeSignalPrep:

    def __init__(self,df):

        self.df = df

    def resample_acc(self,sample_rate_new = 1):
        """ Resampling the time series

        Parameters
        ----------
        self: DataFrame

        time_col: str
            column name of the time column
        sample_rate_new: float
            sample rate of the resampled time series

        Returns
        -------
        DataFrame
        """
#        dfResample.index =  np.arange(self.df.index.min(),self.df.index.max(),1/sample_rate_new)
        index_new =  np.linspace(self.df.index.min(),
                                 self.df.index.min() + np.floor((self.df.index.max()-self.df.index.min())*sample_rate_new)/sample_rate_new,
                                 int(np.floor(self.df.index.max()-self.df.index.min())*sample_rate_new + 1))
        dfResample = pd.DataFrame(index = index_new)
        for colakt in self.df.columns:
            dfResample[colakt] = np.interp(dfResample.index,self.df.index,self.df[colakt])
        return dfResample

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        """Use the functonality of scipy"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        TSout = signal.filtfilt(b, a, self.df)
        return TSout
    def prepare_rolling(self):
        """
        Adds ID, time for TsFresh

        Parameters
        ----------
        self.df: pandas DataFrame
            input data
        self : TimeSignalPrep class
            

        Returns
        -------
        df : pandas DataFrame
            output DataFrame with added id, time

        """
        self.prep_roll = self.df
        start = time.time()
        self.prep_roll["id"] = 0
        self.prep_roll["time"] = self.df.index.values
        self.prep_roll["time"] = self.prep_roll["time"].subtract(self.prep_roll["time"].values[0])
        self.prep_roll.index = self.prep_roll["time"]
        end = time.time()
        self.ts_time = self.prep_roll.copy()

        print("Prepare_rolling: {:5.3f}s".format(end - start))
        return self.prep_roll

    def roll_dataset(self, timeshift=1000, rolling_direction=800):
        """

        Parameters
        ----------
        self.prep_roll: output from prepare_rolling
             
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
        cycles = int(len(self.prep_roll) / rolling_direction) - 1
        df_rolled_is_empty = True
        # shiften
        for i in range(cycles):
            position = (rolling_direction) * i
            shift = self.prep_roll.iloc[position : position + timeshift, :]
            # change IDs to format (id,time)
            shift.loc[:, ("max_time")] = max(shift.loc[:, ("time")])
            df = shift.loc[:, ("id", "max_time")]
            shift.loc[:, ("id")] = pd.MultiIndex.from_frame(df)
            # delete max_time
            shift.pop("max_time")
            if df_rolled_is_empty == True:
                self.df_rolled = shift
                df_rolled_is_empty = False
            else:
                self.df_rolled = self.df_rolled.append(shift, ignore_index=True)

        ende = time.time()
        print("roll_dataset: {:5.3f}s".format(ende - start))
        return self.df_rolled

    def extract_features_df(self, feature="maximum"):

        """

        Parameters
        ----------
        self.df_rolled : pandas DataFrame
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
        self.extracted_features = extract_features(
            self.df_rolled,
            column_id="id",
            column_sort="time",
            default_fc_parameters=fc_parameters,
            n_jobs=0,
        )
        self.extracted_features.index = range(len(self.extracted_features))
        ende = time.time()
        print("extract_features_df: {:5.3f}s".format(ende - start))
        return self.extracted_features

    def select_relevant_windows(self, percentage_max,
                                timeshift=1000, rolling_direction=800):

        """

        Parameters
        ----------
       self.prep_roll : pandas DataFrame
            input data - normally output from perpare_rolling(df)
        percentage_max : float
            min percentage of the maximum of the extraced feature.
        self.extracted_features : pandas Dataframe
            DataFrame of features
        timeshift : int
            window size -the default is 1000.
        rolling_direction : TYPE
            window shift- the default is 800

        Returns
        -------
        df : pandas DataFrame
            dataframe with NaN's in the windows with too low extracted features

        """
        # get added up abs energy of interval x, if too low set None
        start = time.time()
        added_feature = np.zeros(len(self.extracted_features))
        for i in range(self.prep_roll.shape[1] - 2):
            added_feature += self.extracted_features.iloc[:, i]

        plt.show()
        self.relevant_windows = self.prep_roll.copy()
        for i in range(len(self.extracted_features)):
            if added_feature.iloc[i] <= (max(added_feature) * percentage_max):
                # set those rows 0 in ts_data
                self.relevant_windows.iloc[
                    0 + i * rolling_direction : timeshift + i * rolling_direction,
                    0 : self.relevant_windows.shape[1] - 2,
                ] = None
        ende = time.time()
        print("select_relevant_windows: {:5.3f}s".format(ende - start))
        return self.relevant_windows

        
    def create_gridpoints(self,  n_gridpoints=3):
        """
    
        Parameters
        ----------
        self.relevant_windows : pandas DataFrame
            dataframe with NaN's from select_relevant_windows(...)
        n_gridpoints : int, optional
            Number of gridpoints for polynomial smoothing. The default is 3.
    
        Returns
        -------
        df : pandas DataFrame
            reduces number of NaN's to n_gridpoints
    
        """
        # let at any gap exactly Parameter n NaN's - these will be used to remove jumps
        # let at any gap exactly Parameter n NaN's -
        # these will be used to remove jumps
        start = time.time()
        list = []
        for i in range(len(self.relevant_windows) - n_gridpoints):
            istrue = True
            if ~(self.relevant_windows.iloc[i : i + n_gridpoints + 1, 0].isna().all()):
                istrue = False
            if istrue == True:
                # add to delete list
                list.append(self.relevant_windows.index[i])

        ende = time.time()
        print("create_gridpoints Part 1: {:5.3f}s".format(ende - start))

        start = time.time()
        self.grid_points = self.relevant_windows.drop(list, axis=0)
        ende = time.time()
        print("create_gridpoints Part 2: {:5.3f}s".format(ende - start))
        return self.grid_points
        
    def polyfit_gridpoints(self,  order=3, verbose=False, n_gridpoints=3):
        """
        
        
        Parameters
        ----------
        self.gridpoints : pandas DataFrame
            DataFrame with NaN's as gridpoints
        self.ts_time : pandas DataFrame used to create time axis.
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
        top_row = self.grid_points.iloc[0, :]
        top_row.name = (
            self.grid_points.iloc[0, self.grid_points.shape[1] - 1]
            - self.grid_points.iloc[2, self.grid_points.shape[1] - 1]
            + self.grid_points.iloc[1, self.grid_points.shape[1] - 1]
        )
        top_row = pd.DataFrame(top_row)
        top_row = top_row.transpose()
        top_row.head()
        g = 0
        self.poly_gridpoints = pd.concat([top_row, self.grid_points])  # ,ignore_index=True)
        self.ts_time = self.ts_time.head(len(self.poly_gridpoints))
        self.poly_gridpoints["time"] = self.ts_time["time"].values
        self.poly_gridpoints.index = self.poly_gridpoints["time"]
        self.poly_gridpoints.iloc[0, :] = 0
        #%% smooth the gaps with polynomial values
        # plot polynomials
        for i in range(len(self.poly_gridpoints) - n_gridpoints - 1):
            if self.poly_gridpoints.iloc[i, :].isna()[0]:
                g += 1
                # calculate time axis
                t_2 = self.poly_gridpoints.iloc[i - 1, self.poly_gridpoints.shape[1] - 1]
                t_3 = self.poly_gridpoints.iloc[i + n_gridpoints, self.poly_gridpoints.shape[1] - 1]
                t_4 = self.poly_gridpoints.iloc[i + n_gridpoints + 1, self.poly_gridpoints.shape[1] - 1]
                if i - 3 < 0:
                    t_1 = t_3 - t_4
                else:
                    t_1 = self.poly_gridpoints.iloc[i - 2, self.poly_gridpoints.shape[1] - 1]

                x = np.array([t_1, t_2, t_3, t_4])
                # print(x)
                for j in range(self.poly_gridpoints.shape[1] - 2):
                    ##calculate values instead of Non, y-values polynomials
                    p_2 = self.poly_gridpoints.iloc[i - 1, j]

                    if i - 3 < 0:  # |np.isnan(p_1)):
                        p_1 = p_2
                    else:
                        p_1 = self.poly_gridpoints.iloc[i - 2, j]

                    p_3 = self.poly_gridpoints.iloc[i + n_gridpoints, j]

                    # if(np.isnan(p_3)):
                    # p_3=0

                    p_4 = self.poly_gridpoints.iloc[i + n_gridpoints + 1, j]

                    if np.isnan(p_4):
                        p_4 = p_3

                    y = np.array([p_1, p_2, p_3, p_4])
                    # print(y)
                    z = np.polyfit(x, y, order)
                    p = np.poly1d(z)
                    x1 = []
                    y1 = []
                    for k in range(n_gridpoints):
                        # write elements into dataset
                        self.poly_gridpoints.iloc[i + k, j] = p(self.poly_gridpoints.iloc[i + k, self.poly_gridpoints.shape[1] - 1])
                        x1.append(self.poly_gridpoints.iloc[i + k, self.poly_gridpoints.shape[1] - 1])
                        y1.append(p(self.poly_gridpoints.iloc[i + k, self.poly_gridpoints.shape[1] - 1]))

                    xp = np.linspace(t_1, t_4, 200)
                    if verbose == True:
                        plt.plot(xp, p(xp))
                        plt.plot(x, y, "o")
                        plt.plot(x1, y1, "*")
                        plt.show()
        print("Number of gaps:", g)
        ende = time.time()
        print("polyfit_gridpoints: {:5.3f}s".format(ende - start))
        return self.poly_gridpoints
        
    

    def clean_dataset(self, timeshift=1000, rolling_direction = 800, feature="abs_energy", n_gridpoints=3,percentage_max=0.05,order=3):
        
        
        """
    
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
    
        self.prepare_rolling()
    
        #add 1line to ts_time -we add a line to the data--> If we dont delete anything ts_data>ts_time--> add one line at the end
        last_row = self.ts_time.iloc[0,:]
        #calculate t of the next step
        last_row.name=self.ts_time.iloc[len(self.ts_time)-1, self.df.shape[1]-1]
        +self.ts_time.iloc[len(self.ts_time)-1, self.df.shape[1]-1]
        -self.ts_time.iloc[len(self.ts_time)-2, self.df.shape[1]-1]
        
        last_row=pd.DataFrame(last_row)
        last_row=last_row.transpose()
        self.ts_time=self.ts_time.append(last_row)
    
    
        self.roll_dataset(timeshift=timeshift, rolling_direction = rolling_direction)
        self.extract_features_df(feature)
        self.select_relevant_windows(percentage_max, timeshift, rolling_direction)
       
        self.create_gridpoints(n_gridpoints=n_gridpoints)
        self.polyfit_gridpoints(order=order, verbose=False, n_gridpoints=n_gridpoints)
    
        #Remove NaN's at the end - should be maximum 2n
        l=self.poly_gridpoints.shape[0]
        self.cleaned = self.poly_gridpoints.dropna(axis=0, how='any', thresh=None, subset=None)
        self.cleaned.pop("id")
        print("Number of NaN's dropped at END:", l-self.poly_gridpoints.shape[0])
        ende = time.time()
        print('Total Cleaning: {:5.3f}s'.format(ende-start))
        
        return self.cleaned
        
        

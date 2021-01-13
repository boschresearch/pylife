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

    def running_stats_filt(self,col,window_length = 2048,buffer_overlap = 0.1,limit = 0.05, method = "rms"):
        """
        Calculates the running statistics of one DataFrame column and drops the rejected data points from the whole DataFrame.

        **Attention**: Reset_index is used

        Parameters
        -----------

        self: DataFrame

        col: str
            column name of the signal for the runnings stats calculation
        window_length: int
            window length of the single time snippet, default is 2048
        buffer_overlap: float
            overlap parameter, 0.1 is equal to 10 % overlap of every buffer, default is 0.1
        limit: float
            limit value of skipping values, 0.05 for example skips all values which buffer method parameter is lower than 5% of the total max value,
            default is 0.05
        method: str
            method: 'rms', 'min', 'max', 'abs', default is 'rms'

        Returns
        -------
        DataFrame

        """
        df = self.df.reset_index(drop = True)
        delta_t = self.df.index.values[1]-self.df.index.values[0]
        hop = int(window_length*(1-buffer_overlap)) # absolute stepsize
        df = df.loc[:int(np.floor(len(df)/hop)*hop),:]
        n_iter = 1+int((len(df)-window_length)/(hop))
        ind_act = 0
        stats_list = []
        for ii in range (n_iter):
            if method == "rms":
                stats_list.append( np.sqrt(np.mean(df[col][ind_act:ind_act+window_length]**2)))
            elif method == "max":
                stats_list.append(np.max(df[col][ind_act:ind_act+window_length]))
            elif method == "min":
                stats_list.append(np.abs(np.min(df[col][ind_act:ind_act+window_length])))
            elif method == "abs":
                stats_list.append(np.max(np.abs(df[col][ind_act:ind_act+window_length])))
            ind_act = ind_act+hop
        try:
            stats_list = pd.DataFrame({"stats": np.asarray(stats_list)})#,
        except:
            print(str(stats_list))
                                  # index = np.arange(0,len(np.asarray(stats_list))-1,
                                  #                   np.asarray(stats_list)))
        stats_list = stats_list[stats_list["stats"] < limit*stats_list["stats"].max()]
        for ind_act in stats_list.index:
            df = df.drop(index = np.arange(ind_act*hop,ind_act*hop+window_length), errors = 'ignore')
        df.index = np.linspace(0,delta_t*(len(df)-1), len(df))
        return df

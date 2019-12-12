# Copyright (c) 2019 - for information on the respective copyright owner
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
import scipy.stats as stats


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

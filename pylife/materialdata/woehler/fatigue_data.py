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

__author__ = "Mustapha Kassem"
__maintainer__ = "Johannes Mueller"

import numpy as np
from scipy import stats, optimize

class FatigueData:
    def __init__(self, data, load_cycle_limit):

        self.data = data
        self.load_cycle_limit = load_cycle_limit
        self.__data_sort()
        self.__slope()
        self.__deviation()
        
    def __data_sort(self):

        self.loads_max = self.data.loads.max()
        self.loads_min = self.data.loads.min()

        self.cycles_max = self.data.cycles.max()
        self.cycles_min = self.data.cycles.min()

        self.fractures = self.data[self.data.cycles < self.load_cycle_limit]
        self.runouts = self.data[self.data.cycles >= self.load_cycle_limit]

        self.__calc_ld_endur_zones()

        self.zone_inf_fractures = self.fractures[self.fractures.loads < self.fatg_lim]

        self.ld_lvls = np.unique(self.data.loads, return_counts=True)
        self.ld_lvls_fin = np.unique(self.zone_fin.loads, return_counts=True)
        self.ld_lvls_inf = np.unique(self.zone_inf.loads, return_counts=True)
        self.ld_lvls_inf_frac = np.unique(self.zone_inf_fractures.loads, return_counts=True)


    def __calc_ld_endur_zones(self):
        '''
        Computes the start value of the load endurance limit. This is done by searching for the lowest load
        level before the appearance of a runout data point, and the first load level where a runout appears.
        Then the median of the two load levels is the start value.
        '''

        self.zone_fin = self.fractures[self.fractures.loads > self.runouts.loads.max()]
        zone_fin_min = self.zone_fin.loads.min()
        if zone_fin_min == 0:
            self.fatg_lim = self.runouts.loads.max()
        else:
            self.fatg_lim = np.mean([zone_fin_min, self.runouts.loads.max()])
        self.zone_inf = self.data[self.data.loads<= self.fatg_lim]
        
    def __slope(self):
        '# Computes the slope of the finite zone with the help of a linear regression function'

        self.a_wl, self.b_wl, _, _, _ = stats.linregress(np.log10(self.fractures.loads),
                                                         np.log10(self.fractures.cycles)
                                                         )

        '# Woehler Slope'
        self.k = -self.a_wl
        '# Cycle for load = 1'
        self.N0 = 10**self.b_wl
        '# Load-cycle endurance start value relative to the load endurance start value'
        self.N_E = 10**(self.b_wl + self.a_wl*(np.log10(self.fatg_lim)))            
        
    def __deviation(self):
        '''
        Pearl chain method: consists of shifting the fractured data to a median load level.
        The shifted data points are assigned to a Rossow failure probability.The scatter in load-cycle
        direction can be computed from the probability net.
        '''
        # Mean load level:
        self.Sa_shift = np.mean(self.fractures.loads)

        # Shift probes to the mean load level
        self.N_shift = self.fractures.cycles * ((self.Sa_shift/self.fractures.loads)**(-self.k))
        self.N_shift = np.sort(self.N_shift)

        fp = self.__rossow_fail_prob(self.N_shift)
        self.u = stats.norm.ppf(fp)

        self.a_pa, self.b_pa, _, _, _ = stats.linregress(np.log10(self.N_shift), self.u)

        # Scatter in load cycle direction
        self.TN = 10**(2.5631031311*(1./self.a_pa))

        # Scatter in load direction
        '# Empirical method "following Koeder" to estimate the scatter in load direction '
        self.TS = self.TN**(1./self.k)  
        
    def __rossow_fail_prob(self, x):
        """ Failure Probability estimation formula of Rossow

        'Statistics of Metal Fatigue in Engineering' page 16

        https://books.google.de/books?isbn=3752857722
        """
        i = np.arange(len(x))+1
        pa = (3.*(i)-1.)/(3.*len(x)+1.)

        return pa
  
    @property
    def initial_p_opt(self):
        return {'SD_50': self.fatg_lim, '1/TS': 1.2, 'k_1': self.k, 
                      'ND_50': self.N_E, '1/TN': self.TN}
        
    @property
    def initial_dict_bound(self):
        return {'SD_50':(-np.inf, self.fatg_lim*0.95), '1/TS':(0.5, 2),
                           'k_1':(self.k*0.9, self.k*1.1), 
                           'ND_50':(self.N_E*0.8, self.load_cycle_limit*0.8),
                           '1/TN':(self.TN*0.1, self.TN*10)}    
 

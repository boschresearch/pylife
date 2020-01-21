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

from pylife.materialdata.woehler.probability_plot import ProbabilityPlot

class ProbabilityPlotCreator:
    def __init__(self, fatigue_data):
        self.fatigue_data = fatigue_data
    
    def probability_plot_finite(self):
        probability_plot_finite = {'X': self.fatigue_data.N_shift, 'Y': self.fatigue_data.u, 'a': self.fatigue_data.a_pa, 
                                   'b': self.fatigue_data.b_pa, 'T': self.fatigue_data.TN}
        return ProbabilityPlot(self.fatigue_data, probability_plot_finite)

    def probability_plot_inifinite(self):
        # Probaility regression plot
        failure_probability_infinite = self.__rossow_failure_probability_infinite(self.fatigue_data)
        inv_cdf = stats.norm.ppf(failure_probability_infinite)
        a_ue, b_ue, _, _, _ = stats.linregress(np.log10(self.fatigue_data.ld_lvls_inf[0]), inv_cdf)
        # Deviation TS in load-cycle direction
        TS_probit = 10**(2.5631031311*(1./a_ue))
        probability_plot = {'X': self.fatigue_data.ld_lvls_inf[0], 'Y': inv_cdf, 'a': a_ue, 'b': b_ue, 'T': TS_probit}
        return ProbabilityPlot(self.fatigue_data, probability_plot)
    
    def __rossow_failure_probability_infinite(self):
        """
        Probit, probability unit, is the inverse cumulative distribution function (CDF).
        Describes the failure probability of the infinite zone.

        Parameters
        ----------
        data_probit:
            - A three column table containing:
                1st column: Load levels in the infinite zone
                2nd column: The quantity of data points found in the respective load level
                3rd column: Of these data points, the quantity of fractured data points

        Returns
        -------
        self.FP: The failure probability following rossow's method for the infinite zone
        """
        data_probit = np.zeros((len(self.fatigue_data.ld_lvls_inf[0]), 3))

        data_probit[:, 0] = self.fatigue_data.ld_lvls_inf[0]
        data_probit[:, 1] = self.fatigue_data.ld_lvls_inf[1]

        if len(self.fatigue_data.ld_lvls_inf[0]) != len(self.fatigue_data.ld_lvls_inf_frac[1]):
            x = {k:v for k,v in enumerate(~np.in1d(self.fatigue_data.ld_lvls_inf[0], self.fatigue_data.ld_lvls_inf_frac[0]))
                 if v == True
                }
            if len([*x.keys()]) > 1:
                fracs = list(self.fatigue_data.ld_lvls_inf_frac[1])
                for keys in np.arange(len([*x.keys()])):
                    fracs.insert([*x.keys()][keys], 0)
                data_probit[:, 2] = np.asarray(fracs)
            else:
                    fracs = list(self.fatigue_data.ld_lvls_inf_frac[1])
                    fracs.insert([*x.keys()][0], 0)
                    data_probit[:, 2] = np.asarray(fracs)
        else:
            data_probit[:, 2] = self.fatigue_data.ld_lvls_inf_frac[1]

        # Rossow failure probability for the transition zone
        failure_probability_infinite = []

        for i in data_probit:
            '#Number of specimen'
            n = i[1]
            '#Number of fractures'
            r = i[2]

            if r == 0:
                failure_probability_infinite.append(1. - 0.5**(1./n))
            elif r == n:
                failure_probability_infinite.append(0.5**(1./n))
            else:
                failure_probability_infinite.append((3*r-1)/(3*n+1))

        return np.asarray(failure_probability_infinite)   

    
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

import matplotlib
import matplotlib.pyplot as plt
from pylife.materialdata.woehler.probability_plot import ProbabilityPlot

class PlotProbabilityPlot:
    
    @staticmethod
    def plot_probability_plot(ProbabilityPlot, xlabel, ylabel, title):
        #Finite
        # X = WoehlerCurve.fatigue_data.N_shift
        # Y = WoehlerCurve.u
        # a = WoehlerCurve.a_pa
        # b = WoehlerCurve.b_pa
        # T = WoehlerCurve.TN
        # xlabel = 'load cycle N'
        # ylabel = 'Failure probability'    
        
        #Infinite
        # X = WoehlerCurve.ld_lvls_inf[0]
        # Y = WoehlerCurve.inv_cdf
        # a = WoehlerCurve.a_ue
        # b = WoehlerCurve.b_ue
        # T = WoehlerCurve.Probit_result['1/TS']
        # xlabel = self.amp+' ('+self.ld_typ+') in '+self.unit 
        # ylabel = 'Failure probability'
        
        scatter = '$1/T_N$ = '

        fig = plt.figure(figsize=(6, 4))
        ax = plt.subplot('111')


        plt.plot(X, Y, 'ro')
        plt.plot([10**((i-b)/a) for i in np.arange(-2.5, 2.5, 0.1)],
                 np.arange(-2.5, 2.5, 0.1), 'r')

        yticks = [1, 5, 10, 20, 50, 80, 90, 95, 99]
        plt.yticks([stats.norm.ppf(i/100.) for i in yticks],
                   [str(i)+' %' for i in yticks])

        plt.xticks([10**((stats.norm.ppf(0.1)-b)/a), 10**((stats.norm.ppf(0.9)-b)/a)],
                    ('', ''))

        plt.xscale('log') #problem with the scaling cant overwrite xticks for inf zone
        plt.grid()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        plt.text(0.15, 0.03, 'N($P_{A,10}$)='+'{:1.1e}'.format(10**((stats.norm.ppf(0.1)-b)/a), decimals=1),
                 verticalalignment='bottom',horizontalalignment='left', transform=self.ax.transAxes,
                 bbox={'facecolor':'grey', 'alpha':0.1, 'pad':10}, fontsize=11)

        plt.text(0.5, 0.88, scatter + str(np.round(T,decimals=2)),
                 verticalalignment='bottom',horizontalalignment='center', transform=self.ax.transAxes,
                 bbox={'facecolor':'grey', 'alpha':0.2, 'pad':10}, fontsize=11)

        plt.text(0.9, 0.03, 'N($P_{A,90}$)='+'{:1.1e}'.format(10**((stats.norm.ppf(0.9)-b)/a),decimals=1),
                 verticalalignment='bottom',horizontalalignment='right', transform=self.ax.transAxes,
                 bbox={'facecolor':'grey', 'alpha':0.1, 'pad':10}, fontsize=11)

        plt.xticks([10**((stats.norm.ppf(0.1)-b)/a), 10**((stats.norm.ppf(0.9)-b)/a)], ('', ''))

        fig.tight_layout()

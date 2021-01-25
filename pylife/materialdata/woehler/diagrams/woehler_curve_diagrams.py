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

__author__ = "Mustapha Kassem"
__maintainer__ = "Johannes Mueller"


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .woehler_curve_graph import WoehlerCurveGraph
from .whole_woehler_curve_graph import WholeWoehlerCurveGraph


class WoehlerCurveDiagrams:
    def __init__(self, woehler_curve, fatigue_data = None, analyzer = None,
                 y_min=None, y_max=None, x_min=None, x_max=None, ax=None):
        self.woehler_curve = woehler_curve
        self._analyzer = analyzer
        self._fd = fatigue_data
        self.y_min = self._fd.load.min()*0.8 if y_min is None else y_min
        self.y_max = self._fd.load.max()*1.2 if y_max is None else y_max
        self.x_min = self._fd.cycles.min()*0.4 if x_min is None else x_min
        self.x_max = self._fd.cycles.max()*2 if x_max is None else x_max
        
        self.xlim_WL = (self.x_min,round(self.x_max, -1))
        self.ylim_WL = (self.y_min,round(self.y_max, -1))
        
        self.reset(ax)

    def reset(self, ax=None):
        if ax is None:
            _, self._ax = plt.subplots()
        else:
            self._ax = ax
        self.__default_ax_config()
        return self

    def __default_ax_config(self):
        self._ax.set_xlim(self.xlim_WL)
        self._ax.set_ylim(self.ylim_WL)
        self._ax.set_xscale('log')
        self._ax.set_yscale('log')
        self._ax.grid(True)
        self._ax.set_xlabel('Number of cycles', fontsize=11)
        self._ax.set_ylabel('Amplitude' + ' (' + 'Stress' + ') in ' + '$N/mm^2$' + '(log scaled)')
        matplotlib.rcParams.update({'font.size': 11})

    def plot_fatigue_data(self):
        self._ax.plot(self._fd.runouts.cycles,
                      self._fd.runouts.load, 'bo', mfc='none', label='Runout')
        self._ax.plot(self._fd.fractures.cycles,
                      self._fd.fractures.load, 'bo', label='Failure')
        # self.plot_endurance_limit()
        self._ax.legend(loc='upper right', fontsize=11)
        return self

    def plot_endurance_limit(self):
        self._ax.axhline(y=self._fd.fatigue_limit, linewidth=2, color='r', label='Endurance limit')
        return self

    def plot_woehler_curve(self, failure_probablility=0.5, **kw):
        woehler_curve_graph = WoehlerCurveGraph(self.woehler_curve, self.y_min, self.y_max, failure_probablility)

        self._ax.plot(woehler_curve_graph.points[:, 1],
                      woehler_curve_graph.points[:, 0], **kw)
        self._ax.legend(loc='upper right', fontsize=11)
        return self

    def plot_pearl_chain_method(self):
        pce = self._analyzer.pearl_chain_estimator()
        normed_load = pce.normed_load
        normed_cycles = pce.normed_cycles
        self._ax.plot(normed_cycles, np.ones(len(normed_cycles))*normed_load,
                      'go', label='PCM shifted probes', marker="v")
        self._ax.plot(self.xlim_WL, np.ones(len(self.xlim_WL))*normed_load, 'g')
        self.plot_woehler_curve()
        self._ax.legend(loc='upper right', fontsize=11)
        return self

    def plot_deviation(self, **kw):
        self.plot_woehler_curve(failure_probablility=0.5, color='r', label='$P_A$=50%')
        self.plot_woehler_curve(failure_probablility=0.1, color='r', linestyle='--', label='$P_A$=10% u. 90%')
        self.plot_woehler_curve(failure_probablility=0.9, color='r', linestyle='--')

        text = '$k$ = '+str(np.round(self.woehler_curve['k_1'], decimals=2)) + '\n'
        text += '$1/T_N$ = ' + str(np.round(self.woehler_curve['1/TN'], decimals=2)) + '\n'
        text += '$1/T_S^*$ = ' + str(np.round(self.woehler_curve['1/TS'], decimals=2))

        self._ax.text(0.01, 0.03, text, verticalalignment='bottom',
                    horizontalalignment='left', transform=self._ax.transAxes,
                    bbox={'facecolor': 'grey', 'alpha': 0.2, 'pad': 10})
        self._ax.legend(loc='upper right', fontsize=11)
        return self

    def plot_fitted_curve(self, k_2=None):
        ''' This is broken now
        '''
        whole_woehler_curve_graph = WholeWoehlerCurveGraph(self.woehler_curve, k_2, self.y_min, self.y_max)
        WL_50 = whole_woehler_curve_graph.graph_50
        WL_10 = whole_woehler_curve_graph.graph_10
        WL_90 = whole_woehler_curve_graph.graph_90
        self._ax.plot(WL_50[:, 1], WL_50[:, 0], 'r', linewidth=2., label=u'WC, $P_A$=50%')
        self._ax.plot(WL_10[:, 1], WL_10[:, 0], 'r', linewidth=1.5, linestyle='--', label=u'WC, $P_A$=10% u. 90%')
        self._ax.plot(WL_90[:, 1], WL_90[:, 0], 'r', linewidth=1.5, linestyle='--')
        self._ax.legend(loc='upper right', fontsize=11)

        text = '$k_1$ = '+str(np.round(self.woehler_curve['k_1'],decimals=2)) + '\n'
        text += '$1/T_N$ = ' + str(np.round(self.woehler_curve['1/TN'],decimals=2)) + '\n'
        text += '$1/T_S^*$ = ' + str(np.round(self.woehler_curve['1/TN']**(1./self.woehler_curve['k_1']), decimals=2)) + '\n'
        text += '$S_{D,50}$ = ' + str(np.round(self.woehler_curve['SD_50'],decimals=1)) + '\n'
        text += '$N_{D,50}$ = ' + '{:1.2e}'.format(self.woehler_curve['ND_50']) + '\n'
        text += '$1/T_S$ = ' + str(np.round(self.woehler_curve['1/TS'],decimals=2))

        self._ax.text(0.01, 0.03, text,
                 verticalalignment='bottom',horizontalalignment='left',
                 transform=self._ax.transAxes, bbox={'facecolor':'grey', 'alpha':0.2, 'pad':10})
        return self

    def plot_slope(self, failure_probablility=0.5):
        text = '$k$ = '+str(np.round(self.woehler_curve['k_1'], decimals=2))
        self._ax.text(0.01, 0.03, text, verticalalignment='bottom',
            horizontalalignment='left', transform=self._ax.transAxes,
            bbox={'facecolor': 'grey', 'alpha': 0.2, 'pad': 10})
        self.plot_fatigue_data()
        self.plot_woehler_curve()
        return self

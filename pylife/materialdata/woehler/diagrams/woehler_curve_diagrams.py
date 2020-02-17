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
import matplotlib
import matplotlib.pyplot as plt

from pylife.materialdata.woehler.diagrams.woehler_curve_graph import WoehlerCurveGraph
from pylife.materialdata.woehler.diagrams.whole_woehler_curve_graph import WholeWoehlerCurveGraph


class WoehlerCurveDiagrams:
    def __init__(self, woehler_curve, y_min=None, y_max=None, ax=None):
        self.woehler_curve = woehler_curve
        self.y_min = self.woehler_curve.fatigue_data.load.min()*0.8 if y_max is None else y_max
        self.y_max = self.woehler_curve.fatigue_data.load.max()*1.2 if y_min is None else y_min
        self.xlim_WL = (round(min(self.woehler_curve.fatigue_data.cycles)*0.4, -1),
                        round(max(self.woehler_curve.fatigue_data.cycles)*2, -1))
        self.ylim_WL = (round(min(self.woehler_curve.fatigue_data.load)*0.8, -1),
                        round(max(self.woehler_curve.fatigue_data.load)*1.2, -1))

    def __default_ax_config(self, title):
        fig, ax = plt.subplots()
        # need setter for the title
        ax.set_title('Initial data')
        # fig = plt.figure()
        # ax = fig.add_axes([0,0,1,1])
        ax.set_xlim(self.xlim_WL)
        ax.set_ylim(self.ylim_WL)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        ax.set_xlabel('Number of cycles', fontsize=11)
        ax.set_ylabel('Amplitude' + ' (' + 'Stress' + ') in ' + '$N/mm^2$' + '(log scaled)')
        fig.tight_layout()
        matplotlib.rcParams.update({'font.size': 11})
        return ax

    def plot_basic_fatigue_data(self, ax=None):
        ax_local = self.__default_ax_config('Initial data') if ax is None else ax
        ax_local.plot(self.woehler_curve.fatigue_data.runouts.cycles,
                      self.woehler_curve.fatigue_data.runouts.load, 'bo', mfc='none', label='Runout')
        ax_local.plot(self.woehler_curve.fatigue_data.fractures.cycles,
                      self.woehler_curve.fatigue_data.fractures.load, 'bo', label='Failure')
        # self.plot_endurance_limit(ax_local)
        ax_local.legend(loc='upper right', fontsize=11)
        return self

    def plot_initial_data(self, ax=None):
        ax_local = self.__default_ax_config('Initial data') if ax is None else ax
        self.plot_endurance_limit(ax_local)
        self.plot_basic_fatigue_data(ax_local)
        ax_local.legend(loc='upper right', fontsize=11)
        return self

    def plot_endurance_limit(self, ax=None):
        ax_local = self.__default_ax_config('Endurance limit') if ax is None else ax
        ax_local.axhline(y=self.woehler_curve.fatigue_data.fatg_lim, linewidth=2, color='r', label='Endurance limit')
        return self

    def plot_woehler_curve(self, ax=None):
        ax_local = self.__default_ax_config('Woehler curve') if ax is None else ax
        print('plot_woehler_curve', self.woehler_curve.curve_parameters)
        woehler_curve_graph = WoehlerCurveGraph(self.woehler_curve, self.y_min, self.y_max)
        ax_local.plot(woehler_curve_graph.points[:, 1],
                      woehler_curve_graph.points[:, 0], color='r', linewidth=2., label=u'WL, $P_A$=50%')
        ax_local.legend(loc='upper right', fontsize=11)
        return self

    def plot_pearl_chain_method(self, ax=None):
        ax_local = self.__default_ax_config('Pearl chain method') if ax is None else ax
        fatigue_data = self.woehler_curve.fatigue_data
        ax_local.plot(fatigue_data.N_shift, np.ones(len(fatigue_data.N_shift))*fatigue_data.Sa_shift,
                      'go', label='PCM shifted probes', marker="v")
        ax_local.plot(self.xlim_WL, np.ones(len(self.xlim_WL))*fatigue_data.Sa_shift, 'g')
        self.plot_woehler_curve(ax_local)
        ax_local.legend(loc='upper right', fontsize=11)
        return self


    def plot_deviation(self, ax=None):
        ax_local = self.__default_ax_config('Deviation') if ax is None else ax
        woehler_curve_graph = WoehlerCurveGraph(self.woehler_curve, self.y_min, self.y_max)
        ax_local.plot(woehler_curve_graph.calc_shifted_woehlercurve_points(0.1)[:, 1],
                woehler_curve_graph.points[:, 0], 'r', linewidth=1.5,
                linestyle='--', label=u'WL, $P_A$=10% u. 90%')
        ax_local.plot(woehler_curve_graph.calc_shifted_woehlercurve_points(0.9)[:, 1],
                woehler_curve_graph.points[:, 0], 'r', linewidth=1.5,
                linestyle='--')
        self.plot_woehler_curve(ax_local)
        text = '$k$ = '+str(np.round(self.woehler_curve.k, decimals=2)) + '\n'
        text += '$1/T_N$ = ' + str(np.round(self.woehler_curve.TN, decimals=2)) + '\n'
        text += '$1/T_S^*$ = ' + str(np.round(self.woehler_curve.TS, decimals=2))

        ax_local.text(0.01, 0.03, text, verticalalignment='bottom',
                    horizontalalignment='left', transform=ax_local.transAxes,
                    bbox={'facecolor': 'grey', 'alpha': 0.2, 'pad': 10})
        ax_local.legend(loc='upper right', fontsize=11)
        return self

    def plot_whole_woehler_curve_graph(self, k_2, ax=None):
        ax_local = self.__default_ax_config('Woehler curve') if ax is None else ax
        whole_woehler_curve_graph = WholeWoehlerCurveGraph(self.woehler_curve, k_2, self.y_min, self.y_max)
        WL_50 = whole_woehler_curve_graph.graph_50
        WL_10 = whole_woehler_curve_graph.graph_10
        WL_90 = whole_woehler_curve_graph.graph_90
        ax_local.plot(WL_50[:, 1], WL_50[:, 0], 'r', linewidth=2., label=u'WC, $P_A$=50%')
        ax_local.plot(WL_10[:, 1], WL_10[:, 0], 'r', linewidth=1.5, linestyle='--', label=u'WC, $P_A$=10% u. 90%')
        ax_local.plot(WL_90[:, 1], WL_90[:, 0], 'r', linewidth=1.5, linestyle='--')
        ax_local.legend(loc='upper right', fontsize=11)

        text = '$k_1$ = '+str(np.round(self.woehler_curve.k,decimals=2)) + '\n'
        text += '$k_2$ = '+str(np.round(k_2,decimals=2)) + '\n'
        text += '$1/T_N$ = ' + str(np.round(self.woehler_curve.TN,decimals=2)) + '\n'
        text += '$1/T_S^*$ = ' + str(np.round(self.woehler_curve.TN**(1./self.woehler_curve.k), decimals=2)) + '\n'
        text += '$S_{D,50}$ = ' + str(np.round(self.woehler_curve.SD_50,decimals=1)) + '\n'
        text += '$N_{D,50}$ = ' + '{:1.2e}'.format(self.woehler_curve.ND_50) + '\n'
        text += '$1/T_S$ = ' + str(np.round(self.woehler_curve.TS,decimals=2))

        ax_local.text(0.01, 0.03, text,
                 verticalalignment='bottom',horizontalalignment='left',
                 transform=ax_local.transAxes, bbox={'facecolor':'grey', 'alpha':0.2, 'pad':10})
        return self

    def plot_slope(self, ax=None):
        ax_local = self.__default_ax_config('Slope') if ax is None else ax
        text = '$k$ = '+str(np.round(self.woehler_curve.k, decimals=2))
        ax_local.text(0.01, 0.03, text, verticalalignment='bottom',
            horizontalalignment='left', transform=ax_local.transAxes,
            bbox={'facecolor': 'grey', 'alpha': 0.2, 'pad': 10})
        self.plot_basic_fatigue_data(ax_local)
        self.plot_woehler_curve(ax_local)
        return self

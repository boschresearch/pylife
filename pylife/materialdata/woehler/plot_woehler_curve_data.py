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
from scipy import stats
from pylife.materialdata.woehler.woehler_curve_graph import WoehlerCurveGraph
from pylife.materialdata.woehler.whole_woehler_curve import WholeWoehlerCurve

class PlotWoehlerCurveData:
    def __init__(self, woehler_curve, y_min = None, y_max = None, ax = None):
        self.woehler_curve = woehler_curve
        self.y_min = self.woehler_curve.fatigue_data.loads_min*0.8 if y_max == None else y_max
        self.y_max = self.woehler_curve.fatigue_data.loads_max*1.2 if y_min == None else y_min


    def plot_basic_fatigue_data(self, ax):
        ax.plot(self.woehler_curve.fatigue_data.runouts.cycles, self.woehler_curve.fatigue_data.runouts.loads, 'bo', mfc='none', label='Runout')   
        ax.plot(self.woehler_curve.fatigue_data.fractures.cycles, self.woehler_curve.fatigue_data.fractures.loads, 'bo', label='Failure')
        ax.axhline(y=self.woehler_curve.fatigue_data.fatg_lim, linewidth=2, color='r', label='Endurance limit') 
        ax.legend(loc='upper right', fontsize=11)
        return self

    def plot_woehler_curve(self, ax):
        woehler_curve_graph = WoehlerCurveGraph(self.woehler_curve, self.y_min, self.y_max)
        ax.plot(woehler_curve_graph.points[:,1], woehler_curve_graph.points[:,0], color='r', linewidth=2., label=u'WL, $P_A$=50%')
        return self
    
    def plot_pearl_chain_method(self, ax):
        fatigue_data = self.woehler_curve.fatigue_data
        ax.plot(fatigue_data.N_shift, np.ones(len(fatigue_data.N_shift))*fatigue_data.Sa_shift,
                    'go',label='PCM shifted probes', marker="v")
        xlim_WL = (round(min(fatigue_data.data.cycles)*0.4,-1), round(max(fatigue_data.data.cycles)*2,-1))
        ax.plot(xlim_WL, np.ones(len(xlim_WL))*fatigue_data.Sa_shift,'g')
        return self
        
    
    def plot_deviation(self, ax):
        woehler_curve_graph = WoehlerCurveGraph(self.woehler_curve, self.y_min, self.y_max)
        ax.plot(woehler_curve_graph.calc_shifted_woehlercurve_points(0.1)[:, 1],
                woehler_curve_graph.points[:, 0], 'r', linewidth=1.5,
                linestyle='--', label=u'WL, $P_A$=10% u. 90%')
        ax.plot(woehler_curve_graph.calc_shifted_woehlercurve_points(0.9)[:, 1],
                woehler_curve_graph.points[:, 0], 'r', linewidth=1.5, 
                linestyle='--')
        return self

    def plot_whole_woehler_curve_graph(self, k_2, ax):
        whole_woehler_curve_graph = WholeWoehlerCurve(self.woehler_curve, k_2, self.y_min, self.y_max)
        WL_50 = whole_woehler_curve_graph.graph_50
        WL_10 = whole_woehler_curve_graph.graph_10
        WL_90 = whole_woehler_curve_graph.graph_90
        ax.plot(WL_50[:, 1], WL_50[:, 0], 'r', linewidth=2., label=u'WC, $P_A$=50%')
        ax.plot(WL_10[:, 1], WL_10[:, 0], 'r', linewidth=1.5, linestyle='--', label=u'WC, $P_A$=10% u. 90%')
        ax.plot(WL_90[:, 1], WL_90[:, 0], 'r', linewidth=1.5, linestyle='--')
        return self


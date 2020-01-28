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
from scipy import stats
from pylife.materialdata.woehler.woehler_curve_graph import WoehlerCurveGraph
from pylife.materialdata.woehler.woehler_curve import WoehlerCurve

class WholeWoehlerCurve:

    def __init__(self, woehler_curve, k_2, y_min, y_max):
        self.woehler_curve = woehler_curve
        self.k_2 = k_2
        self.y_min = y_min
        self.y_max = y_max

        SD_50 = woehler_curve.SD_50
        SD_10 = SD_50 / (10**(-stats.norm.ppf(0.1)*np.log10(woehler_curve.TS)/2.56))
        SD_90 = SD_50 / (10**(-stats.norm.ppf(0.9)*np.log10(woehler_curve.TS)/2.56))

        #50%
        self.graph_50 = self.__create_whole_woehler_curve_graph(SD_50, 0.5)

        #10%
        self.graph_10 = self.__create_whole_woehler_curve_graph(SD_10, 0.1)

        #90%
        self.graph_90 = self.__create_whole_woehler_curve_graph(SD_90, 0.9)      

        # #50
        # y_lim = woehler_curve.SD_50 
        # graph_50_1 = WoehlerCurveGraph(woehler_curve, y_lim, y_max)
        # graph_50_1.points = graph_50_1.calc_shifted_woehlercurve_points(0.5)
    
        # #WC = WoehlerCurve({'SD_50': woehler_curve.SD_50, '1/TS': woehler_curve.TS,'ND_50': woehler_curve.ND_50, 'k_1': k_2, '1/TN': woehler_curve.TN})      
        # WC = WoehlerCurve({'SD_50': y_lim, '1/TS': woehler_curve.TS,'ND_50': graph_50_1.points[-1, -1], 'k_1': k_2, '1/TN': woehler_curve.TN})  
        # graph_50_2 = WoehlerCurveGraph(WC, y_min, y_lim)
        # self.graph_50 = np.append(graph_50_1.points, graph_50_2.points, axis=0)

        # #10
        # y_lim = SD_10
        # graph_10_1 = WoehlerCurveGraph(woehler_curve, y_lim, y_max)
        # graph_10_1.points = graph_10_1.calc_shifted_woehlercurve_points(0.1)

        # WC = WoehlerCurve({'SD_50': y_lim, '1/TS': woehler_curve.TS,'ND_50': graph_10_1.points[-1, -1], 'k_1': k_2, '1/TN': woehler_curve.TN}) 
        # graph_10_2 = WoehlerCurveGraph(WC, y_min, y_lim)   
        # self.graph_10 = np.append(graph_10_1.points, graph_10_2.points, axis=0)    

        # #90
        # y_lim = SD_90
        # graph_90_1 = WoehlerCurveGraph(woehler_curve, y_lim, y_max)
        # graph_90_1.points = graph_90_1.calc_shifted_woehlercurve_points(0.9) 

        # WC = WoehlerCurve({'SD_50': y_lim, '1/TS': woehler_curve.TS,'ND_50': graph_90_1.points[-1, -1], 'k_1': k_2, '1/TN': woehler_curve.TN}) 
        # graph_90_2 = WoehlerCurveGraph(WC, y_min, y_lim)                 
        # self.graph_90 = np.append(graph_90_1.points, graph_90_2.points, axis=0)

    def __create_whole_woehler_curve_graph(self, y_lim, pa_goal):
        woehler_curve = self.woehler_curve
        graph_1 = WoehlerCurveGraph(woehler_curve, y_lim, self.y_max)
        graph_1.points = graph_1.calc_shifted_woehlercurve_points(pa_goal) 

        woehler_curve_2 = WoehlerCurve({'SD_50': y_lim, '1/TS': woehler_curve.TS, 'ND_50': graph_1.points[-1, -1], 'k_1': self.k_2, '1/TN': woehler_curve.TN})
        graph_2 = WoehlerCurveGraph(woehler_curve_2, self.y_min, y_lim)                 
        return np.append(graph_1.points, graph_2.points, axis=0)        

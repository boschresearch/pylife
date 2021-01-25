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
import pandas as pd
from scipy import stats

from .woehler_curve_graph import WoehlerCurveGraph

class WholeWoehlerCurveGraph:

    def __init__(self, woehler_curve, k_2, y_min, y_max):
        self.woehler_curve = woehler_curve
        self.k_2 = woehler_curve['k_1'] if k_2 is None else k_2
        self.y_min = y_min
        self.y_max = y_max

        SD_50 = woehler_curve['SD_50']
        SD_10 = SD_50 / (10**(-stats.norm.ppf(0.1)*np.log10(woehler_curve['1/TS'])/2.56))
        SD_90 = SD_50 / (10**(-stats.norm.ppf(0.9)*np.log10(woehler_curve['1/TS'])/2.56))

        self.graph_50 = self.__create_whole_woehler_curve_graph(SD_50, 0.5)
        self.graph_10 = self.__create_whole_woehler_curve_graph(SD_10, 0.1)
        self.graph_90 = self.__create_whole_woehler_curve_graph(SD_90, 0.9)


    def __create_whole_woehler_curve_graph(self, y_lim, pa_goal):
        woehler_curve = self.woehler_curve
        graph_1 = WoehlerCurveGraph(woehler_curve, y_lim, self.y_max, pa_goal)

        woehler_curve_2 = pd.Series({
            'SD_50': y_lim, '1/TS': woehler_curve['1/TS'], 'ND_50': graph_1.points[-1, -1],
            'k_1': self.k_2, '1/TN': woehler_curve['1/TN']})
        graph_2 = WoehlerCurveGraph(woehler_curve_2, self.y_min, y_lim)
        return np.append(graph_1.points, graph_2.points, axis=0)

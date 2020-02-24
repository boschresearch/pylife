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


class WoehlerCurveGraph:

    def __init__(self, woehler_curve, y_min, y_max):
        self.woehler_curve = woehler_curve
        self.points = self.calc_woehler_curve_points(y_min, y_max)

    def calc_woehler_curve_points(self, y_min, y_max):
        """ Basquin curve equation

        http://www.ux.uis.no/~hirpa/6KdB/ME/S-N%20diagram.pdf
        """
        woehler_curve_points = np.array([[self.woehler_curve['SD_50'], 1E9]])
        if self.woehler_curve['k_1'] != 0:
            y = np.linspace(y_max, y_min, num=100)
            x = self.woehler_curve['ND_50']*(y/self.woehler_curve['SD_50'])**(-self.woehler_curve['k_1'])
            woehler_curve_points = np.array([y, x]).transpose()
        return woehler_curve_points

    def calc_shifted_woehlercurve_points(self, pa_goal):
        """ Shift the Basquin-curve according to the failure probability value (obtain the 10-90 % curves)"""
        woehler_curve_graph_shift = self.points
        if self.woehler_curve['k_1'] != 0:
            woehler_curve_graph_shift = np.array(woehler_curve_graph_shift)
            woehler_curve_graph_shift[:, 1] /= (10**(-stats.norm.ppf(pa_goal)*np.log10(self.woehler_curve['1/TN'])/2.56))
        return woehler_curve_graph_shift

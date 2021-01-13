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
from scipy import stats

from pylife.utils.functions import scatteringRange2std


class WoehlerCurveGraph:

    def __init__(self, woehler_curve, y_min, y_max, failure_probablilty=0.5):
        self._woehler_curve = woehler_curve
        self.__calc_woehler_curve_points(y_min, y_max)
        self.__shift_woehlercurve_points(failure_probablilty)

    def __calc_woehler_curve_points(self, y_min, y_max):
        """ Basquin curve equation

        http://www.ux.uis.no/~hirpa/6KdB/ME/S-N%20diagram.pdf
        """
        if self._woehler_curve['k_1'] == np.inf:
            self._points = np.array([[self._woehler_curve['SD_50'], 1E9]])
        else:
            y = np.linspace(y_max, y_min, num=100)
            x = self._woehler_curve['ND_50']*(y/self._woehler_curve['SD_50'])**(-self._woehler_curve['k_1'])
            self._points = np.array([y, x]).transpose()

    def __shift_woehlercurve_points(self, pa_goal):
        """ Shift the Basquin-curve according to the failure probability value (obtain the 10-90 % curves)"""
        TN_inv = self._woehler_curve['1/TN']
        self._points[:, 1] /= 10**(-stats.norm.ppf(pa_goal)*scatteringRange2std(TN_inv))

    @property
    def points(self):
        return self._points

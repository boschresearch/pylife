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

__author__ = "Gyöngyvér Kiss"
__maintainer__ = __author__

from ..diagrams.woehler_curve_diagrams import WoehlerCurveDiagrams
from .radio_button_woehler_curve import RadioButtonWoehlerCurve


class WoehlerCurveDataPlotter(RadioButtonWoehlerCurve):
    def __init__(self, woehler_curve, fatigue_data, analyzer):
        self._woehler_curve = woehler_curve
        self._woehler_curve_diagrams = WoehlerCurveDiagrams(self._woehler_curve, fatigue_data, analyzer)
        self._woehler_curve_diagrams.plot_fatigue_data()
        super().__init__(options=['Only initial data',
                                  'Slope',
                                  'Pearl chain',
                                  'Deviation in load-cycle direction'], description='Plot Type')

    def selection_changed_handler(self, change):
        self.clear_selection_change_output()
        self._woehler_curve_diagrams.reset().plot_fatigue_data()
        if change['new'] == change.owner.options[0]:
            pass
        elif change['new'] == change.owner.options[1]:
            self._woehler_curve_diagrams.plot_fatigue_data().plot_slope()
        elif change['new'] == change.owner.options[2]:
            self._woehler_curve_diagrams.plot_pearl_chain_method()
        elif change['new'] == change.owner.options[3]:
            self._woehler_curve_diagrams.plot_deviation()
        else:
            raise AttributeError('Unexpected selection')

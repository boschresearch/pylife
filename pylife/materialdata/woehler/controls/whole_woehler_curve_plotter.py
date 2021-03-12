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


class WholeWoehlerCurvePlotter(RadioButtonWoehlerCurve):
    def __init__(self, woehler_curve, fatigue_data, analyzer):
        self.woehler_curve = woehler_curve
        self.woehler_curve_diagrams = WoehlerCurveDiagrams(self.woehler_curve, fatigue_data, analyzer)
        super().__init__(options=['k_2 = inf', 'k_2 = k_1', 'k_2 = 2 k_1 - 1'], description='Runout zone plot')
        self.woehler_curve_diagrams.plot_whole_woehler_curve_graph(0)

    def selection_changed_handler(self, change):
        self.clear_selection_change_output()
        if change['new'] == change.owner.options[0]:
            self.woehler_curve_diagrams.plot_whole_woehler_curve_graph(None)
        elif change['new'] == change.owner.options[1]:
            self.woehler_curve_diagrams.plot_whole_woehler_curve_graph(self.woehler_curve.k)
        elif change['new'] == change.owner.options[2]:
            self.woehler_curve_diagrams.plot_whole_woehler_curve_graph(2 * self.woehler_curve.k - 1)
        else:
            raise AttributeError('Unexpected selection')

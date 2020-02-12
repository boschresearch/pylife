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

import sys, os

sys.path.insert(0, os.path.abspath('..\\pylife'))

from pylife.materialdata.woehler.diagrams.woehler_curve_diagrams import WoehlerCurveDiagrams
from pylife.materialdata.woehler.controls.radio_button_woehler_curve import RadioButtonWoehlerCurve

class WoehlerCurveDataPlotter(RadioButtonWoehlerCurve):
    def __init__(self, woehler_curve):
        self.woehler_curve = woehler_curve
        self.woehler_curve_diagrams = WoehlerCurveDiagrams(self.woehler_curve)
        self.woehler_curve_diagrams.plot_initial_data()
        super().__init__(options=['Initial data', 'Slope', 'Pearl chain method', 'Deviation in load-cycle direction'], description='Plot Type')
        
    def selection_changed_handler(self, change):
        self.clear_selection_change_output()
        if change['new'] == change.owner.options[0]:
            self.woehler_curve_diagrams.plot_initial_data()
        elif change['new'] == change.owner.options[1]:
            self.woehler_curve_diagrams.plot_slope()
        elif change['new'] == change.owner.options[2]:
            self.woehler_curve_diagrams.plot_pearl_chain_method()
        elif change['new'] == change.owner.options[3]:
            self.woehler_curve_diagrams.plot_deviation()
        else:
            raise AttributeError('Unexpected selection')



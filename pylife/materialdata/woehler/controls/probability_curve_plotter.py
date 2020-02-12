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

from pylife.materialdata.woehler.factories.probability_curve_factory import ProbabilityCurveFactory
from pylife.materialdata.woehler.factories.probability_curve_diagrams_factory import ProbabilityCurveDiagramsFactory
from pylife.materialdata.woehler.controls.radio_button_woehler_curve import RadioButtonWoehlerCurve

class ProbabilityCurvePlotter(RadioButtonWoehlerCurve):
    def __init__(self, fatigue_data):
        super().__init__(options=['Probability plot of the finite zone', 'Probability plot of the infinite zone'], description='Plot type')
        self.probability_curve_factory = ProbabilityCurveFactory(fatigue_data)
        probability_curve = self.probability_curve_factory.create_probability_curve_finite()
        probability__curve_diagrams_factory = ProbabilityCurveDiagramsFactory(probability_curve)
        probability__curve_diagrams_factory.plot_probability_curve_diagram_finite()
        
    def selection_changed_handler(self, change):
        self.clear_selection_change_output()
        if change['new'] == change.owner.options[0]:
            probability_curve = self.probability_curve_factory.create_probability_curve_finite()
            probability__curve_diagrams_factory = ProbabilityCurveDiagramsFactory(probability_curve)
            probability__curve_diagrams_factory.plot_probability_curve_diagram_finite()
        elif change['new'] == change.owner.options[1]:
            probability_curve = self.probability_curve_factory.create_probability_curve_infinite()
            probability__curve_diagrams_factory = ProbabilityCurveDiagramsFactory(probability_curve)
            probability__curve_diagrams_factory.plot_probability_curve_diagram_infinite()
        else:
            raise AttributeError('Unexpected selection')        





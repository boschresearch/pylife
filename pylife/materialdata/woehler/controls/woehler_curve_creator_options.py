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

import ipywidgets as widgets
from IPython.display import display

from pylife.materialdata.woehler.controls.radio_button_woehler_curve import RadioButtonWoehlerCurve


class WoehlerCurveCreatorOptions(RadioButtonWoehlerCurve):
    def __init__(self, fatigue_data):
        self.woehler_curve = None
        self.fatigue_data = fatigue_data
        self.collect_fixed_params()
        super().__init__(['Maximum likelihood 2 params', 'Maximum likelihood 5 params', 'Probit', 'Bayes (SLOW!)'], 'Select method')
        self.calculate_curve_button = widgets.Button(description='Calculate curve')
        self.calculate_curve_button.on_click(self.calculate_curve_button_clicked_handler)
        self.woehler_curve = self.fatigue_data.woehler_max_likelihood_inf_limit()
        print(self.woehler_curve)

    def selection_changed_handler(self, change):
        self.clear_selection_change_output()
        if change['new'] == change.owner.options[0]:
            self.woehler_curve = self.fatigue_data.woehler_max_likelihood_inf_limit()
            print(self.woehler_curve)
        elif change['new'] == change.owner.options[1]:
            display(self.param_fix_tab)
            display(self.calculate_curve_button)
        elif change['new'] == change.owner.options[2]:
            self.woehler_curve = self.fatigue_data.woehler_probit()
            print(self.woehler_curve)
        elif change['new'] == change.owner.options[3]:
            self.woehler_curve = self.fatigue_data.woehler_bayesian()
            print(self.woehler_curve)
        else:
            raise AttributeError('Unexpected selection')

    def tab_content_changed_handler(self, change):
        try:
            self.param_fix.update({change.owner.description: float(change.new)})
        except(ValueError, TypeError):
            self.param_fix.update({change.owner.description: ''})

    def collect_fixed_params(self):
        self.param_fix = {'SD_50': '', '1/TS': '', 'ND_50': '', 'k_1': '', '1/TN': ''}
        items = []
        for k in self.param_fix:
            text = widgets.Text(description=k, value=self.param_fix[k])
            text.observe(self.tab_content_changed_handler, names='value')
            items.append(text)
        self.param_fix_tab = widgets.VBox(items)

    def calculate_curve_button_clicked_handler(self, b):
        param_fix = {k: v for k, v in self.param_fix.items() if v != ''}
        self.woehler_curve = self.fatigue_data.woehler_max_likelihood(param_fix)
        print(self.woehler_curve)

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

import ipywidgets as widgets
from IPython.display import display

from ..analyzers.maxlike import MaxLikeInf, MaxLikeFull
from ..analyzers.probit import Probit
from ..analyzers.bayesian import Bayesian
from .radio_button_woehler_curve import RadioButtonWoehlerCurve


class WoehlerCurveAnalyzerOptions(RadioButtonWoehlerCurve):
    def __init__(self, df):
        self.woehler_curve = None
        self._df = df
        self.collect_fixed_params()
        self._option_dict = {
            'Maximum likelihood 2 params': (MaxLikeInf(self._df), self.__just_analyze),
            'Maximum likelihood 5 params': (MaxLikeFull(self._df), self.__display_param_fix),
            'Probit': (Probit(self._df), self.__just_analyze),
            'Bayes (SLOW!)': (Bayesian(self._df), self.__just_analyze)
        }

        super().__init__(self._option_dict.keys(), 'Select method')
        self.calculate_curve_button = widgets.Button(description='Calculate curve')
        self.calculate_curve_button.on_click(self.calculate_curve_button_clicked_handler)
        self._analyzer, _ = list(self._option_dict.values())[0]
        self.__just_analyze()

    def analyzer(self):
        return self._analyzer

    def __just_analyze(self, **kwargs):
        self.woehler_curve = self._analyzer.analyze(**kwargs)
        print(self.woehler_curve)
        self.bic = self._analyzer.bayesian_information_criterion()
        print("BIC:", self.bic)

    def __display_param_fix(self):
        display(self.param_fix_tab)
        display(self.calculate_curve_button)

    def selection_changed_handler(self, change):
        self.clear_selection_change_output()
        try:
            self._analyzer, analyze_method = self._option_dict[change['new']]
        except KeyError:
            raise AttributeError('Unexpected selection')

        analyze_method()

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
        self.__just_analyze(fixed_parameters=param_fix)

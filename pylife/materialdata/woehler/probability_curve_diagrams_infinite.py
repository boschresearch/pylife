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

import matplotlib
import matplotlib.pyplot as plt
from pylife.materialdata.woehler.probability_curve_diagrams import ProbabilityCurveDiagrams

class ProbabilityCurveDiagramsInfinite(ProbabilityCurveDiagrams):
    def get_scatter_label(self):
        return '$1/T_S$ = '

    def get_xlabel(self):
        return 'Amplitude' + ' ('+ 'Stress' +') in ' + '$N/mm^2$'

    def get_title(self):
        return 'Probability plot of the infinite zone'
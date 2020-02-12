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

from pylife.materialdata.woehler.curves.probability_curve import ProbabilityCurve
from pylife.materialdata.woehler.creators.probability_curve_creator import ProbabilityCurveCreator

class ProbabilityCurveCreatorFinite(ProbabilityCurveCreator):    
    def create_probability_curve(self):
        probability_curve_finite = {'X': self.fatigue_data.N_shift, 'Y': self.fatigue_data.u, 'a': self.fatigue_data.a_pa, 
                                   'b': self.fatigue_data.b_pa, 'T': self.fatigue_data.TN}
        return ProbabilityCurve(self.fatigue_data, probability_curve_finite)
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
from pylife.materialdata.woehler.curves.woehler_curve import WoehlerCurve

class WoehlerCurveWithBIC(WoehlerCurve): 
    def __init__(self, curve_parameters, p_opt, param_fix, fopt, fatigue_data = None):
        super().__init__(curve_parameters, fatigue_data)
        self.p_opt = p_opt
        self.param_fix = param_fix
        self.bic = None if fopt is None else self.bayes_inf_crit(fopt)


    def bayes_inf_crit(self, LLi):
        ''' Bayesian Information Criterion: is a criterion for model selection among a finite set of models;
        the model with the lowest BIC is preferred.
        https://www.statisticshowto.datasciencecentral.com/bayesian-information-criterion/
        '''
        if self.fatigue_data is None: return None
        elif LLi is None: return None
        else:
            param_est = len([*self.p_opt.values()])
            bic = (-2*LLi)+(param_est*np.log(self.fatigue_data.data.shape[0]))
            return bic

    def __str__ (self):
        return str(self.curve_parameters) + ', BIC: ' + str(self.bic)
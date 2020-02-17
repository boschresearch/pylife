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

from pylife.core.data_validator import DataValidator


class WoehlerCurveElementary:
    ''' Contains a Miner Elementary Wöhler curve
    '''
    def __init__(self, curve_parameters, fatigue_data=None):
        self.fatigue_data = fatigue_data
        self.k = DataValidator.fill_member('k_1', curve_parameters)
        self.TS = DataValidator.fill_member('1/TS', curve_parameters)
        self.TN = DataValidator.fill_member('1/TN', curve_parameters)
        self.curve_parameters = curve_parameters

    def __str__(self):
        return str(self.curve_parameters)


class WoehlerCurve(WoehlerCurveElementary):
    ''' Contains a classic Wöhler Curve with endurance limit
    '''
    def __init__(self, curve_parameters, fatigue_data=None, fopt=None):
        super().__init__(curve_parameters, fatigue_data)
        self.ND_50 = DataValidator.fill_member('ND_50', curve_parameters)
        self.SD_50 = DataValidator.fill_member('SD_50', curve_parameters)
        self.bic = None if fopt is None else self.__bayes_inf_crit(fopt)

    def __bayes_inf_crit(self, LLi):
        ''' Bayesian Information Criterion: is a criterion for model selection among a finite set of models;
        the model with the lowest BIC is preferred.
        https://www.statisticshowto.datasciencecentral.com/bayesian-information-criterion/
        '''
        if self.fatigue_data is None:
            return None
        elif LLi is None:
            return None
        else:
            param_est = len([*self.p_opt.values()])
            bic = (-2*LLi)+(param_est*np.log(self.fatigue_data.data.shape[0]))
            return bic

    def __str__(self):
        bic = '' if self.bic is None else ', BIC: ' + str(self.bic)
        return str(self.curve_parameters) + bic

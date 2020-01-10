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

class Analyzer: 
    
    def bayes_inf_crit(self, LLi, dict_opt, n_data_pt):
        ''' Bayesian Information Criterion: is a criterion for model selection among a finite set of models;
        the model with the lowest BIC is preferred.
        https://www.statisticshowto.datasciencecentral.com/bayesian-information-criterion/
        '''

        param_est = len([*dict_opt.values()])
        bic = (-2*LLi)+(param_est*np.log(n_data_pt))
        return bic


    def life_cyc_eval(self, method):
        if method == 'Mali':
            # Allowed load amplitude with a certain load and 50% failure probability
            N_allowed_50 = self.ND50_mali*(self.Load_ampl_goal/
                                                   self.Param_SD50_mali)**(-self.Param_k_mali)
            # Allowed load relative to the goal failure probability
            N_allowed_pa_goal = (N_allowed_50/(10**(-stats.norm.ppf(self.Pa_goal)*
                                                    np.log10(self.Param_TN_mali)/2.56
                                                   )
                                              )
                                )
        else:
            N_allowed_50 = self.ND50_mali*(self.Load_ampl_goal/
                                                                self.SD50_probit)**(-WoehlerCurve.k)

            N_allowed_pa_goal = (N_allowed_50/(10**(-stats.norm.ppf(self.Pa_goal)*
                                                    np.log10(self.TN)/2.56
                                                   )
                                              )
                                )

        return N_allowed_50, N_allowed_pa_goal

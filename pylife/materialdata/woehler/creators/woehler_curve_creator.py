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
import pandas as pd
import numpy.ma as ma
import mystic as my
from scipy import stats, optimize
from pylife.materialdata.woehler.curves.woehler_curve import WoehlerCurve
from pylife.materialdata.woehler.curves.woehler_curve_with_bic import WoehlerCurveWithBIC
from pylife.materialdata.woehler.fatigue_data import FatigueData
from pylife.materialdata.woehler.curves.woehler_curve_pearl_chain import WoehlerCurvePearlChain


class WoehlerCurveCreator:
    def __init__(self, fatigue_data):
        self.fatigue_data = fatigue_data
        #self.optimzer = WoehlerCurveOptimizer()
        
    def pearl_chain_method(self):        
        woehler_curve = {'k_1': self.fatigue_data.k, '1/TN': self.fatigue_data.TN, '1/TS': self.fatigue_data.TS}
        return WoehlerCurvePearlChain(woehler_curve, self.fatigue_data)    
    
    def maximum_like_procedure(self, param_fix):
        """
        Maximum likelihood is a method of estimating the parameters of a distribution model by maximizing
        a likelihood function, so that under the assumed statistical model the observed data is most probable.
        This procedure consists of estimating the Woehler curve parameters, where some of these paramters may
        be fixed by the user. The remaining paramters are then fitted to produce the best possible outcome.
        The procedure uses the function Optimize.fmin
        Optimize.fmin iterates over the function mali_sum_lolli values till it finds the minimum

        https://en.wikipedia.org/wiki/Maximum_likelihood_estimation

        Parameters
        ----------
        self.p_opt: Start values of the Mali estimated parameters if none are fixed by the user.

        self.dict_bound: Boundary values of the Mali estimated parameters if none are fixed by the user.
        This forces the optimizer to search for a minimum solution within a given area.

        Returns
        -------
        self.Mali_5p_result: The estimated parameters computed using the optimizer.

        """
        p_opt = self.fatigue_data.initial_p_opt

        dict_bound = self.fatigue_data.initial_dict_bound
        
        for k in param_fix:
            p_opt.pop(k)
            dict_bound.pop(k)
            
        mali_5p_result = {}
        if dict_bound: 
            var_opt = my.scipy_optimize.fmin(self.mali_sum_lolli_wrapper, [*p_opt.values()],
                                            bounds=[*dict_bound.values()],
                                            args=([*p_opt], param_fix, self.fatigue_data.fractures, self.fatigue_data.zone_inf,
                                                self.fatigue_data.load_cycle_limit
                                                ),
                                            full_output=True,
                                            disp=True,
                                            maxiter=1e4,
                                            maxfun=1e4,
                                            )
            #self.optimzer.OptimizerFunction(self.optimzer.mali_sum_lolli, [*self.p_opt.values()], [*self.dict_bound.values()], )
            mali_5p_result.update(param_fix)
            mali_5p_result.update(zip([*p_opt], var_opt[0]))
        else:
            raise AttributeError('You need to leave at least one parameter empty!')
        
        

        return WoehlerCurveWithBIC(mali_5p_result, p_opt, param_fix, -var_opt[1], self.fatigue_data)
    
    def maximum_like_procedure_2_param(self):
        ''' This maximum likelihood procedure estimates the load endurance limit SD50_mali_2_param and the
        scatter in load direction TS_mali_2_param.
        Moreover, the load cycle endurance is computed by the interesecting endurance limit line with the
        line of slope k_1
        '''
        SD_start = self.fatigue_data.fatg_lim
        TS_start = 1.2

        var_opt = optimize.fmin(self.Mali_SD_TS, [SD_start, TS_start],
                                           args=(self.fatigue_data.zone_inf, self.fatigue_data.load_cycle_limit),
                                           disp=False, full_output=True)

        ND50 = 10**(self.fatigue_data.b_wl + self.fatigue_data.a_wl*np.log10(var_opt[0][0]))
        mali_2p_result = {'SD_50': var_opt[0][0], '1/TS': var_opt[0][1],'ND_50': ND50, 'k_1': self.fatigue_data.k, '1/TN': self.fatigue_data.TN}
        return WoehlerCurveWithBIC(mali_2p_result, {'SD_50': SD_start, '1/TS': TS_start}, {}, -var_opt[1], self.fatigue_data)          
        
    def probit_procedure(self):
        '''
        Evaluation of infinite zone. Probit procedure uses the rossow function for infinite zone to compute the failure probability of the infinite
        zone, in order to estimate the endurance parameters as well as the scatter in load direction
        '''
        probit_data = self.fatigue_data.determine_probit_parameters()
        # Average fatigue strength
        SD50_probit = 10**((0 - probit_data['b']) / probit_data['a'])
        # Average endurance load cycle
        ND50_probit = 10**(self.fatigue_data.b_wl + self.fatigue_data.a_wl * np.log10(SD50_probit))

        probit_result = {'SD_50': SD50_probit, '1/TS': probit_data['T'],'ND_50': ND50_probit, 'k_1': self.fatigue_data.k, '1/TN': self.fatigue_data.TN}
        return WoehlerCurve(probit_result, self.fatigue_data)
    
    def mali_sum_lolli(self, SD, TS, k, N_E, TN, fractures, zone_inf, load_cycle_limit):
        """
        Produces the likelihood functions that are needed to compute the parameters of the woehler curve.
        The likelihood functions are represented by probability and cummalative distribution functions.
        The likelihood function of a runout is 1-Li(fracture). The functions are added together, and the
        negative value is returned to the optimizer.

        Parameters
        ----------
        SD:
            Endurnace limit start value to be optimzed, unless the user fixed it.
        TS:
            The scatter in load direction 1/TS to be optimzed, unless the user fixed it.
        k:
            The slope k_1 to be optimzed, unless the user fixed it.
        N_E:
            Load-cycle endurance start value to be optimzed, unless the user fixed it.
        TN:
            The scatter in load-cycle direction 1/TN to be optimzed, unless the user fixed it.
        fractures:
            The data that our log-likelihood function takes in. This data represents the fractured data.
        zone_inf:
            The data that our log-likelihood function takes in. This data is found in the infinite zone.
        load_cycle_limit:
            The dependent variable that our model requires, in order to seperate the fractures from the
            runouts.

        Returns
        -------
        neg_sum_lolli :
            Sum of the log likelihoods. The negative value is taken since optimizers in statistical
            packages usually work by minimizing the result of a function. Performing the maximum likelihood
            estimate of a function is the same as minimizing the negative log likelihood of the function.

        """
        # Likelihood functions of the fractured data
        x_ZF = np.log10(fractures.cycles * ((fractures.loads/SD)**(k)))
        Mu_ZF = np.log10(N_E)
        Sigma_ZF = np.log10(TN)/2.5631031311
        Li_ZF = stats.norm.pdf(x_ZF, Mu_ZF, abs(Sigma_ZF))
        LLi_ZF = np.log(Li_ZF)

        # Likelihood functions of the data found in the infinite zone
        std_log = np.log10(TS)/2.5631031311
        runouts = ma.masked_where(zone_inf.cycles >= load_cycle_limit, zone_inf.cycles)
        t = runouts.mask.astype(int)
        Li_DF = stats.norm.cdf(np.log10(zone_inf.loads/SD), loc=np.log10(1), scale=abs(std_log))
        LLi_DF = np.log(t+(1-2*t)*Li_DF)

        sum_lolli = LLi_DF.sum() + LLi_ZF.sum()
        neg_sum_lolli = -sum_lolli

        return neg_sum_lolli
    
     
    def mali_sum_lolli_wrapper(self, var_args, var_keys, fix_args, fractures, zone_inf, load_cycle_limit):
        ''' 1) Finds the start values to be optimized. The rest of the paramters are fixed by the user.
            2) Calls function mali_sum_lolli to calculate the maximum likelihood of the current
            variable states.
        '''
        args = {}
        args.update(fix_args)
        args.update(zip(var_keys, var_args))

        return self.mali_sum_lolli(args['SD_50'], args['1/TS'], args['k_1'], args['ND_50'],
                                          args['1/TN'], fractures, zone_inf, load_cycle_limit)   

    def Mali_SD_TS(self, variables, zone_inf, load_cycle_limit):
        """
        Produces the likelihood functions that are needed to compute the endurance limit and the scatter
        in load direction. The likelihood functions are represented by a cummalative distribution function.
        The likelihood function of a runout is 1-Li(fracture).

        Parameters
        ----------
        variables:
            The start values to be optimized. (Endurance limit SD, Scatter in load direction 1/TS)
        zone_inf:
            The data that our log-likelihood function takes in. This data is found in the infinite zone.
        load_cycle_limit:
            The dependent variable that our model requires, in order to seperate the fractures from the
            runouts.

        Returns
        -------
        neg_sum_lolli :
            Sum of the log likelihoods. The negative value is taken since optimizers in statistical
            packages usually work by minimizing the result of a function. Performing the maximum likelihood
            estimate of a function is the same as minimizing the negative log likelihood of the function.

        """

        SD = variables[0]
        TS = variables[1]

        std_log = np.log10(TS)/2.5631031311
        runouts = ma.masked_where(zone_inf.cycles >= load_cycle_limit, zone_inf.cycles)
        t = runouts.mask.astype(int)
        Li_DF = stats.norm.cdf(np.log10(zone_inf.loads/SD), loc=np.log10(1), scale=abs(std_log))
        LLi_DF = np.log(t+(1-2*t)*Li_DF)

        sum_lolli = LLi_DF.sum()
        neg_sum_lolli = -sum_lolli

        return neg_sum_lolli
   
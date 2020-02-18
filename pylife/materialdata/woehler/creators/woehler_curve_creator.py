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
from pylife.materialdata.woehler.curves.woehler_curve import WoehlerCurve, WoehlerCurveElementary


class WoehlerCurveCreator:
    def __init__(self, fatigue_data):
        self.fatigue_data = fatigue_data
        self._bic = None

    def baysian_information_criterion(self):
        return self._bic

    def pearl_chain_method(self):
        woehler_curve = {'k_1': self.fatigue_data.k, '1/TN': self.fatigue_data.TN, '1/TS': self.fatigue_data.TS, 'load_intercept': self.fatigue_data.load_intercept}
        return WoehlerCurveElementary(woehler_curve, self.fatigue_data)

    def max_likelihood_full(self, param_fix):
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

        if not dict_bound:
            raise AttributeError('You need to leave at least one parameter empty!')
        var_opt = my.scipy_optimize.fmin(
            self.__likelihood_wrapper, [*p_opt.values()],
            bounds=[*dict_bound.values()],
            args=([*p_opt], param_fix),
            full_output=True,
            disp=True,
            maxiter=1e4,
            maxfun=1e4,
        )
        res = {}
        res.update(param_fix)
        res.update(zip([*p_opt], var_opt[0]))

        self.__calc_bic(p_opt, self.likelihood_total(res['SD_50'], res['1/TS'], res['k_1'], res['ND_50'], res['1/TN']))
        return WoehlerCurve(res, self.fatigue_data)

    def max_likelihood_inf_limit(self):
        ''' This maximum likelihood procedure estimates the load endurance limit SD50_mali_2_param and the
        scatter in load direction TS_mali_2_param.
        Moreover, the load cycle endurance is computed by the interesecting endurance limit line with the
        line of slope k_1
        '''
        SD_start = self.fatigue_data.fatg_lim
        TS_start = 1.2

        var_opt = optimize.fmin(self.likelihood_infinite, [SD_start, TS_start], disp=False, full_output=True)

        ND50 = 10**(self.fatigue_data.b_wl + self.fatigue_data.a_wl*np.log10(var_opt[0][0]))
        mali_inf_limit_result = {
            'SD_50': var_opt[0][0],
            '1/TS': var_opt[0][1],
            'ND_50': ND50,
            'k_1': self.fatigue_data.k,
            '1/TN': self.fatigue_data.TN
        }
        return WoehlerCurve(mali_inf_limit_result, self.fatigue_data)

    def probit(self):
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

    def likelihood_total(self, SD, TS, k, N_E, TN):
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
        return self.likelihood_finite(SD, k, N_E, TN) + self.likelihood_infinite(SD, TS)

    def likelihood_finite(self, SD, k, N_E, TN):
        # Likelihood functions of the fractured data
        fractures = self.fatigue_data.fractures
        x = np.log10(fractures.cycles * ((fractures.load/SD)**(k)))
        mu = np.log10(N_E)
        sigma = np.log10(TN)/2.5631031311
        log_likelihood = np.log(stats.norm.pdf(x, mu, abs(sigma)))

        return -log_likelihood.sum()

    def likelihood_infinite(self, SD, TS):
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
        zone_inf = self.fatigue_data.zone_inf
        std_log = np.log10(TS)/2.5631031311
        runouts = ma.masked_where(zone_inf.cycles >= self.fatigue_data.load_cycle_limit, zone_inf.cycles)
        t = runouts.mask.astype(int)
        likelihood = stats.norm.cdf(np.log10(zone_inf.load/SD), loc=np.log10(1), scale=abs(std_log))
        log_likelihood = np.log(t+(1-2*t)*likelihood)

        return -log_likelihood.sum()

    def __likelihood_wrapper(self, var_args, var_keys, fix_args):
        ''' 1) Finds the start values to be optimized. The rest of the paramters are fixed by the user.
            2) Calls function mali_sum_lolli to calculate the maximum likelihood of the current
            variable states.
        '''
        args = {}
        args.update(fix_args)
        args.update(zip(var_keys, var_args))

        return self.likelihood_total(args['SD_50'], args['1/TS'], args['k_1'], args['ND_50'], args['1/TN'])

    def __calc_bic(self, p_opt, log_likelihood):
        ''' Bayesian Information Criterion: is a criterion for model selection among a finite set of models;
        the model with the lowest BIC is preferred.
        https://www.statisticshowto.datasciencecentral.com/bayesian-information-criterion/
        '''
        param_est = len([*p_opt.values()])
        self._bic = (-2*log_likelihood)+(param_est*np.log(self.fatigue_data.data.shape[0]))

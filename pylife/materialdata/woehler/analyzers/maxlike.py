# Copyright (c) 2019-2020 - for information on the respective copyright owner
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
from .elementary import Elementary

import pandas as pd
from scipy import optimize
import mystic as my


class MaxLikeInf(Elementary):
    def _specific_analysis(self, wc):
        SD_50, TS_inv = self.__max_likelihood_inf_limit()

        wc['SD_50'] = SD_50
        wc['1/TS'] = TS_inv
        wc['ND_50'] = self._transition_cycles(SD_50)

        return wc

    def __max_likelihood_inf_limit(self):
        ''' This maximum likelihood procedure estimates the load endurance limit SD50_mali_2_param and the
        scatter in load direction TS_mali_2_param.
        Moreover, the load cycle endurance is computed by the interesecting endurance limit line with the
        line of slope k_1
        '''
        SD_start = self._fd.fatigue_limit
        TS_start = 1.2

        var_opt = optimize.fmin(lambda p: -self._lh.likelihood_infinite(p[0], p[1]),
                                [SD_start, TS_start], disp=False, full_output=True)

        SD_50 = var_opt[0][0]
        TS_inv = var_opt[0][1]

        return SD_50, TS_inv


class MaxLikeFull(Elementary):
    def _specific_analysis(self, wc, fixed_parameters={}):
        return pd.Series(self.__max_likelihood_full(wc, fixed_parameters))

    def __max_likelihood_full(self, initial_wcurve, fixed_prms):
        """
        Maximum likelihood is a method of estimating the parameters of a distribution model by maximizing
        a likelihood function, so that under the assumed statistical model the observed data is most probable.
        This procedure consists of estimating the  curve parameters, where some of these paramters may
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

        p_opt = initial_wcurve.to_dict()
        for k in fixed_prms:
            p_opt.pop(k)

        if not p_opt:
            raise AttributeError('You need to leave at least one parameter empty!')
        var_opt = my.scipy_optimize.fmin(
            self.__likelihood_wrapper, [*p_opt.values()],
            args=([*p_opt], fixed_prms),
            full_output=True,
            disp=True,
            maxiter=1e4,
            maxfun=1e4,
        )
        res = {}
        res.update(fixed_prms)
        res.update(zip([*p_opt], var_opt[0]))

        return res

    def __likelihood_wrapper(self, var_args, var_keys, fix_args):
        ''' 1) Finds the start values to be optimized. The rest of the paramters are fixed by the user.
            2) Calls function mali_sum_lolli to calculate the maximum likelihood of the current
            variable states.
        '''
        args = {}
        args.update(fix_args)
        args.update(zip(var_keys, var_args))

        return -self._lh.likelihood_total(args['SD_50'], args['1/TS'], args['k_1'], args['ND_50'], args['1/TN'])

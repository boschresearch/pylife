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
from .elementary import Elementary

import pandas as pd
import numpy as np
from scipy import optimize
import mystic as my
import warnings

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
        def warn_and_fix_if_no_runouts():
            nonlocal fixed_prms
            if self._fd.num_runouts == 0:
                warnings.warn(UserWarning("MaxLikeHood: no runouts are present in fatigue data. "
                                          "Proceeding with SD_50 = 0 and 1/TS = 1 as fixed parameters. "
                                          "This is NOT a standard evaluation!"))
                fixed_prms = fixed_prms.copy()
                fixed_prms.update({'SD_50': 0.0, '1/TS': 1.0})

        def fail_if_less_than_three_fractures():
            if self._fd.num_fractures < 3 or len(self._fd.fractured_loads) < 2:
                raise ValueError("MaxLikeHood: need at least three fractures on two load levels.")

        def warn_and_fix_if_less_than_two_mixed_levels():
            nonlocal fixed_prms
            if len(self._fd.mixed_loads) < 2 and self._fd.num_runouts > 0 and self._fd.num_fractures > 0:
                warnings.warn(UserWarning("MaxLikeHood: less than two mixed load levels in fatigue data."
                                          "Proceeding by setting a predetermined scatter from the standard Wöhler curve."))
                fixed_prms = fixed_prms.copy()
                TN_inv, TS_inv = self._pearl_chain_method()
                fixed_prms.update({'1/TS': TS_inv})

        fail_if_less_than_three_fractures()
        warn_and_fix_if_no_runouts()
        warn_and_fix_if_less_than_two_mixed_levels()

        p_opt = initial_wcurve.to_dict()
        for k in fixed_prms:
            p_opt.pop(k)

        if not p_opt:
            raise AttributeError('You need to leave at least one parameter empty!')
        var_opt = my.scipy_optimize.fmin(
            self.__likelihood_wrapper, [*p_opt.values()],
            args=([*p_opt], fixed_prms),
            full_output=True,
            disp=False,
            maxiter=1e4,
            maxfun=1e4,
        )
        res = {}
        res.update(fixed_prms)
        res.update(zip([*p_opt], var_opt[0]))

        return self.__make_parameters(res)

    def __make_parameters(self, params):
        params['SD_50'] = np.abs(params['SD_50'])
        params['1/TS'] = np.abs(params['1/TS'])
        params['k_1'] = np.abs(params['k_1'])
        params['ND_50'] = np.abs(params['ND_50'])
        params['1/TN'] = np.abs(params['1/TN'])
        return params

    def __likelihood_wrapper(self, var_args, var_keys, fix_args):
        ''' 1) Finds the start values to be optimized. The rest of the paramters are fixed by the user.
            2) Calls function mali_sum_lolli to calculate the maximum likelihood of the current
            variable states.
        '''
        args = {}
        args.update(fix_args)
        args.update(zip(var_keys, var_args))
        args = self.__make_parameters(args)

        return -self._lh.likelihood_total(args['SD_50'], args['1/TS'], args['k_1'], args['ND_50'], args['1/TN'])

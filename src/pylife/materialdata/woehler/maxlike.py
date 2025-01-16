# Copyright (c) 2019-2024 - for information on the respective copyright owner
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
import warnings


class MaxLikeInf(Elementary):
    """Maximum likelihood procedure estimating the ``SD_50`` and ``TS`` and ``ND_50``.

    Only the values describing the infinite lifetime (load endurance limit
    ``SD_50`` and load endurance scatter ``TS``) are calculated by maximum
    likelihood.  The slope ``k_1`` and ``TN`` are taken from the
    :class:`Elementary` calculation.

    The load cycle endurance ``ND_50`` is computed by intersecting load
    endurance limit line with the line of slope ``k_1``.

    """
    def _specific_analysis(self, wc):
        SD, TS = self.__max_likelihood_inf_limit()

        wc['SD'] = SD
        wc['TS'] = TS
        if not np.isnan(wc['ND']):
            wc['ND'] = self._transition_cycles(SD)

        return wc

    def __max_likelihood_inf_limit(self):

        infinite_fractures = self._fd.infinite_zone.loc[self._fd.infinite_zone.fracture]
        infinite_runouts = self._fd.infinite_zone.loc[~self._fd.infinite_zone.fracture]
        infinite_fractured_loads = np.unique(infinite_fractures.load.values)
        infinite_runout_loads = np.unique(infinite_runouts.load.values)
        infinite_mixed_loads = np.intersect1d(infinite_runout_loads, infinite_fractured_loads)

        def fail_if_less_than_two_mixed_levels():
            if len(infinite_mixed_loads) < 2:
                raise ValueError("MaxLikeHood: need at least two mixed load levels.")

        def fail_if_less_than_three_fractures():
            if infinite_fractures.shape[0] < 3 or len(infinite_fractured_loads) < 2:
                raise ValueError("MaxLikeHood: need at least three fractures on two load levels.")


        fail_if_less_than_two_mixed_levels()
        fail_if_less_than_three_fractures()

        SD_start = self._fd.finite_infinite_transition
        TS_start = 1.2

        var_opt = optimize.fmin(lambda p: -self._lh.likelihood_infinite(p[0], p[1]),
                                [SD_start, TS_start], disp=False, full_output=True)

        SD_50 = var_opt[0][0]
        TS = var_opt[0][1]

        return SD_50, TS


class MaxLikeFull(Elementary):
    """Maximum likelihood procedure estimating all parameters.

    Maximum likelihood is a method of estimating the parameters of a
    distribution model by maximizing a likelihood function, so that under the
    assumed statistical model the observed data is most probable.  This
    procedure consists of estimating the curve parameters, where some of these
    parameters may be fixed by the user. The remaining parameters are then
    fitted to produce the best possible outcome.

    https://en.wikipedia.org/wiki/Maximum_likelihood_estimation

    Parameters
    ----------
    fixed_parameters : dict
        Dictionary of fixed parameters

    """

    def _specific_analysis(self, wc, fixed_parameters={}):
        return pd.Series(self.__max_likelihood_full(wc, fixed_parameters))

    def __max_likelihood_full(self, initial_wcurve, fixed_prms):

        def warn_and_fix_if_no_runouts():
            nonlocal fixed_prms
            if self._fd.num_runouts == 0:
                warnings.warn(UserWarning("MaxLikeHood: no runouts are present in fatigue data. "
                                          "Proceeding with SD = 0 and TS = 1 as fixed parameters. "
                                          "This is NOT a standard evaluation!"))
                fixed_prms = fixed_prms.copy()
                fixed_prms.update({'SD': 0.0, 'TS': 1.0})

        def fail_if_less_than_three_fractures():
            if self._fd.num_fractures < 3 or len(self._fd.fractured_loads) < 2:
                raise ValueError("MaxLikeHood: need at least three fractures on two load levels.")

        def warn_and_fix_if_less_than_two_mixed_levels():
            nonlocal fixed_prms
            if len(self._fd.mixed_loads) < 2 and self._fd.num_runouts > 0 and self._fd.num_fractures > 0:
                warnings.warn(UserWarning("MaxLikeHood: less than two mixed load levels in fatigue data."
                                          "Proceeding by setting a predetermined scatter from the standard Wöhler curve."))
                fixed_prms = fixed_prms.copy()
                TN, TS = self._pearl_chain_method()
                fixed_prms.update({'TS': TS})

        fail_if_less_than_three_fractures()
        warn_and_fix_if_no_runouts()
        warn_and_fix_if_less_than_two_mixed_levels()

        prms_to_optimize = initial_wcurve.to_dict()
        for k in fixed_prms:
            prms_to_optimize.pop(k)

        if not prms_to_optimize:
            raise AttributeError('You need to leave at least one parameter empty!')
        optimized_prms = optimize.fmin(
            self.__likelihood_wrapper, [*prms_to_optimize.values()],
            args=([*prms_to_optimize], fixed_prms),
            full_output=True,
            disp=False,
            maxiter=1e4,
            maxfun=1e4,
        )[0]

        result = fixed_prms | dict(zip(prms_to_optimize, optimized_prms))

        return self.__make_parameters(result)

    def __make_parameters(self, params):
        return {k: np.abs(v) for k, v in params.items()}

    def __likelihood_wrapper(self, var_args, var_keys, fix_args):
        args = self.__make_parameters(fix_args | dict(zip(var_keys, var_args)))
        return -self._lh.likelihood_total(**args)

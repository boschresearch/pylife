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

import warnings

from .elementary import Elementary

import numpy as np

import pylife.utils.functions as functions
from pylife.utils.probability_data import ProbabilityFit


class Probit(Elementary):
    """Wöhler analysis according to the Probit method.

    For each load level in the infinite regime a failure probability is
    estimated.  To these failure probability a log norm distribution is fitted,
    whose parameters are then used to calculate the Wöhler curve parameters.
    """

    def _specific_analysis(self, wc):
        wc_init = wc.copy(deep=True)
        self._inf_groups = self._fd.infinite_zone.groupby('load')
        if len(self._inf_groups) < 2:
            warnings.warn(UserWarning("Probit needs at least two – preferably mixed – load levels in the infinite zone. Falling back to Elementary."))
            return wc

        wc['TS'], wc['SD'], wc['ND'] = self.__probit_analysis(wc_init)
        return wc

    def __probit_rossow_estimation(self):
        frac_num = self._inf_groups.fracture.sum().astype(int).to_numpy()
        tot_num = self._inf_groups.fracture.count().to_numpy()

        fprobs = np.empty_like(frac_num, dtype=np.float64)
        no_fractures = frac_num == 0
        w = np.where(no_fractures)
        fprobs[w] = 1. - 0.5**(1./tot_num[w])

        all_fractures = frac_num == tot_num
        w = np.where(all_fractures)
        fprobs[w] = 0.5**(1./tot_num[w])

        some_fractures = np.logical_and(np.logical_not(no_fractures), np.logical_not(all_fractures))
        w = np.where(some_fractures)
        fprobs[w] = (3*frac_num[w] - 1) / (3*tot_num[w] + 1)

        return fprobs, self._inf_groups.load.mean()

    def __probit_analysis(self, wc_init):
        if self._fd.num_runouts == 0:
            return 1., 0., self._transition_cycles(0.0)

        fprobs, load = self.__probit_rossow_estimation()

        self._probability_fit = ProbabilityFit(fprobs, load)

        TS = functions.std_to_scattering_range(1./self._probability_fit.slope)
        SD = 10**(-self._probability_fit.intercept/self._probability_fit.slope)

        ND = np.nan if np.isnan(wc_init['ND']) else self._transition_cycles(SD)

        return TS, SD, ND

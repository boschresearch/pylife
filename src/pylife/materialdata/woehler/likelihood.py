# Copyright (c) 2019-2023 - for information on the respective copyright owner
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
from scipy import stats


from pylife.utils.functions import scattering_range_to_std, std_to_scattering_range


class Likelihood:
    """Calculate the likelihood a fatigue dataset matches with Wöhler curve parameters."""

    def __init__(self, fatigue_data):
        self._fd = fatigue_data

    def likelihood_total(self, SD, TS, k_1, ND, TN):
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
        k_1:
            The slope k_1 to be optimzed, unless the user fixed it.
        ND:
            Load-cycle endurance start value to be optimzed, unless the user fixed it.
        TN:
            The scatter in load-cycle direction 1/TN to be optimzed, unless the user fixed it.

        Returns
        -------
        neg_sum_lolli :
            Sum of the log likelihoods. The negative value is taken since optimizers in statistical
            packages usually work by minimizing the result of a function. Performing the maximum likelihood
            estimate of a function is the same as minimizing the negative log likelihood of the function.

        """
        return self.likelihood_finite(SD, k_1, ND, TN) + self.likelihood_infinite(SD, TS)

    def likelihood_finite(self, SD, k_1, ND, TN):
        if not (SD > 0.0).all():
            return -np.inf
        fractures = self._fd.fractures
        x = np.log10(fractures.cycles * ((fractures.load/SD)**k_1))
        mu = np.log10(ND)
        std_log = scattering_range_to_std(TN)
        log_likelihood = np.log(stats.norm.pdf(x, mu, std_log))

        return log_likelihood.sum()

    def likelihood_infinite(self, SD, TS):
        """
        Produces the likelihood functions that are needed to compute the endurance limit and the scatter
        in load direction. The likelihood functions are represented by a cummalative distribution function.
        The likelihood function of a runout is 1-Li(fracture).

        Parameters
        ----------
        SD:
            Endurnace limit start value to be optimzed, unless the user fixed it.
        TS:
            The scatter in load direction 1/TS to be optimzed, unless the user fixed it.

        Returns
        -------
        neg_sum_lolli :
            Sum of the log likelihoods. The negative value is taken since optimizers in statistical
            packages usually work by minimizing the result of a function. Performing the maximum likelihood
            estimate of a function is the same as minimizing the negative log likelihood of the function.

        """
        infinite_zone = self._fd.infinite_zone
        std_log = scattering_range_to_std(TS)
        t = np.logical_not(self._fd.infinite_zone.fracture).astype(np.float64)
        likelihood = stats.norm.cdf(np.log10(infinite_zone.load/SD),  scale=abs(std_log))
        non_log_likelihood = t+(1.-2.*t)*likelihood
        if non_log_likelihood.eq(0.0).any():
            return -np.inf

        return np.log(non_log_likelihood).sum()

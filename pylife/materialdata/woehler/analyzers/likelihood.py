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
from scipy import stats


class Likelihood:
    def __init__(self, fatigue_data):
        self._fd = fatigue_data

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
        fractures = self._fd.fractures
        x = np.log10(fractures.cycles * ((fractures.load/SD)**(k)))
        mu = np.log10(N_E)
        sigma = np.log10(TN)/2.5631031311
        log_likelihood = np.log(stats.norm.pdf(x, mu, abs(sigma)))

        return log_likelihood.sum()

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
        infinite_zone = self._fd.infinite_zone
        std_log = np.log10(TS)/2.5631031311
        t = np.logical_not(self._fd.infinite_zone.fracture).astype(int)
        likelihood = stats.norm.cdf(np.log10(infinite_zone.load/SD), loc=np.log10(1), scale=abs(std_log))
        log_likelihood = np.log(t+(1-2*t)*likelihood)

        return log_likelihood.sum()

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

from abc import ABC, abstractmethod

import numpy as np
from scipy import stats


from pylife.utils.functions import scattering_range_to_std, std_to_scattering_range


class AbstractLikelihood(ABC):
    """Calculate the likelihood a fatigue dataset matches with Wöhler curve parameters.

    This is an abstract base class that must be subclassed from.
    """

    def __init__(self, fatigue_data):
        self._fd = fatigue_data

    def likelihood_total(self, SD, TS, k_1, ND, TN):
        """Determine the likelihood for a certain Wöhler curve.

        Parameters
        ----------
        SD: float
            The tested endurance infinite limit
        k_1: float
            The tested slope for the finite zone of the Wöhler curve
        TN: float
            The tested scatter of the finite endurance limit
        ND: float
            The testsd finite limit cycle of the Wöhler curve

        Returns
        -------
        likelihood : float
            The likelihood that the parameters are correct.
        """
        return self.likelihood_finite(SD, k_1, ND, TN) + self.likelihood_infinite(SD, TS)

    def likelihood_finite(self, SD, k_1, ND, TN):
        """Determine the likelihood for a certain finite endurance curve.

        Parameters
        ----------
        SD: float
            The tested endurance infinite limit
        k_1: float
            The tested slope for the finite zone of the Wöhler curve
        TN: float
            The tested scatter of the finite endurance limit
        ND: float
            The testsd finite limit cycle of the Wöhler curve

        Returns
        -------
        likelihood : float
            The likelihood that the parameters are correct.
        """
        if SD <= 0.0:
            return -np.inf
        fractures = self._fractures_for_finite_likelihood()
        x = np.log10(fractures.cycles * ((fractures.load/SD)**k_1))
        mu = np.log10(ND)
        std_log = scattering_range_to_std(TN)
        log_likelihood = np.log(stats.norm.pdf(x, mu, std_log))

        return log_likelihood.sum()

    def likelihood_infinite(self, SD, TS):
        """Determine the likelihood for a certain inifinite endurance limit.

        Parameters
        ----------
        SD:
            Endurnace limit start value to be optimzed, unless the user fixed it.
        TS:
            The scatter in load direction TS to be optimzed, unless the user fixed it.

        Returns
        -------
        likelihood : float
            The likelihood that the parameters are correct.

        """
        relevant_zone = self._zone_for_infinite_likelihood()
        std_log = scattering_range_to_std(TS)
        t = np.logical_not(relevant_zone.fracture).astype(np.float64)
        likelihood = stats.norm.cdf(np.log10(relevant_zone.load/SD),  scale=abs(std_log))
        non_log_likelihood = t+(1.-2.*t)*likelihood
        if non_log_likelihood.eq(0.0).any():
            return -np.inf

        return np.log(non_log_likelihood).sum()

    def _zone_for_infinite_likelihood(self):
        """The zone for the infinite likelihood. By default the whole dataset."""
        return self._fd

    @abstractmethod
    def _fractures_for_finite_likelihood(self):
        """The fractures for the finite likelihood. Must be implemented by subclasses."""
        ...


class LikelihoodPureFiniteZone(AbstractLikelihood):
    def _zone_for_infinite_likelihood(self):
        return self._fd

    def _fractures_for_finite_likelihood(self):
        finite_zone = self._fd.finite_zone
        return finite_zone[finite_zone.fracture]


class LikelihoodHighestMixedLevel(AbstractLikelihood):
    def _fractures_for_finite_likelihood(self):
        fractures = self._fd.fractures
        loads = fractures.load
        new_limit = loads[loads < self._fd.finite_infinite_transition].max()

        return fractures[loads >= new_limit]


class LikelihoodAllFractures(AbstractLikelihood):
    def _fractures_for_finite_likelihood(self):
        return self._fd.fractures


class LikelihoodLegacy(AbstractLikelihood):

    def _zone_for_infinite_likelihood(self):
        return self._fd.infinite_zone

    def _fractures_for_finite_likelihood(self):
        return self._fd.fractures

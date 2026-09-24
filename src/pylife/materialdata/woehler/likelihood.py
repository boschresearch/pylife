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

"""Calculate log-likelihoods for Wöhler curve parameter estimation.

The likelihood formulations combine fracture tests in the finite-life region
with fracture and runout outcomes around the endurance limit.
"""

__author__ = "Mustapha Kassem"
__maintainer__ = "Johannes Mueller"

from abc import ABC, abstractmethod

import numpy as np
from scipy import stats


from pylife.utils.functions import scattering_range_to_std, std_to_scattering_range


class AbstractLikelihood(ABC):
    """Calculate log-likelihoods for Wöhler curve parameters.

    Subclasses define which fracture tests contribute to the finite-life term
    and which tests contribute to the endurance-limit term.  The likelihoods
    are used by the maximum-likelihood analyzers to fit ``SD``, ``TS``,
    ``k_1``, ``ND``, and ``TN``.

    Parameters
    ----------
    fatigue_data : FatigueData
        Validated Wöhler fatigue data accessor.

    Notes
    -----
    The finite-life likelihood evaluates shifted cycle numbers

    .. math::

        x_i = \\log_{10}\\left(N_i \\left(\\frac{S_i}{SD}\\right)^{k_1}\\right)

    against a normal distribution with mean ``log10(ND)`` and standard
    deviation derived from ``TN``.  The endurance-limit likelihood evaluates
    the fracture probability at each load level with a log-normal distribution
    around ``SD`` and scatter ``TS``; runouts contribute the survival
    probability.
    """

    def __init__(self, fatigue_data):
        self._fd = fatigue_data

    def likelihood_total(self, SD, TS, k_1, ND, TN):
        """Return the total log-likelihood for Wöhler curve parameters.

        Parameters
        ----------
        SD : float
            Endurance limit load at the knee point.
        TS : float
            Scatter in load direction, expressed as the 10 %/90 % load ratio.
        k_1 : float
            Finite-life Wöhler slope above the endurance limit.
        ND : float
            Cycle number at the knee point.
        TN : float
            Scatter in cycle direction, expressed as the 10 %/90 % cycle
            ratio.

        Returns
        -------
        float
            Sum of the finite-life and endurance-limit log-likelihoods.
        """
        return self.likelihood_finite(SD, k_1, ND, TN) + self.likelihood_infinite(SD, TS)

    def likelihood_finite(self, SD, k_1, ND, TN):
        """Return the finite-life log-likelihood for fractures.

        Parameters
        ----------
        SD : float
            Endurance limit load at the knee point.
        k_1 : float
            Finite-life Wöhler slope above the endurance limit.
        ND : float
            Cycle number at the knee point.
        TN : float
            Scatter in cycle direction, expressed as the 10 %/90 % cycle
            ratio.

        Returns
        -------
        float
            Log-likelihood that the selected fracture tests follow the
            finite-life branch defined by ``SD``, ``k_1``, ``ND``, and ``TN``.
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
        """Return the endurance-limit log-likelihood for outcomes.

        Parameters
        ----------
        SD : float
            Endurance limit load at the knee point.
        TS : float
            Scatter in load direction, expressed as the 10 %/90 % load ratio.

        Returns
        -------
        float
            Log-likelihood that fracture and runout outcomes around the
            endurance limit follow the log-normal distribution defined by
            ``SD`` and ``TS``.
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
        """Return the tests used for the endurance-limit likelihood."""
        return self._fd

    @abstractmethod
    def _fractures_for_finite_likelihood(self):
        """Return the fractures used for the finite-life likelihood."""
        ...


class LikelihoodPureFiniteZone(AbstractLikelihood):
    """Use finite-zone fractures for the finite-life likelihood.

    Parameters
    ----------
    fatigue_data : FatigueData
        Validated Wöhler fatigue data accessor.
    """

    def _zone_for_infinite_likelihood(self):
        return self._fd

    def _fractures_for_finite_likelihood(self):
        finite_zone = self._fd.finite_zone
        return finite_zone[finite_zone.fracture]


class LikelihoodHighestMixedLevel(AbstractLikelihood):
    """Use pure finite fractures and the highest mixed load level.

    Parameters
    ----------
    fatigue_data : FatigueData
        Validated Wöhler fatigue data accessor.
    """

    def _fractures_for_finite_likelihood(self):
        fractures = self._fd.fractures
        loads = fractures.load
        new_limit = loads[loads < self._fd.finite_infinite_transition].max()

        return fractures[loads >= new_limit]


class LikelihoodAllFractures(AbstractLikelihood):
    """Use all fracture tests for the finite-life likelihood.

    Parameters
    ----------
    fatigue_data : FatigueData
        Validated Wöhler fatigue data accessor.
    """

    def _fractures_for_finite_likelihood(self):
        return self._fd.fractures


class LikelihoodLegacy(AbstractLikelihood):
    """Use the likelihood formulation from pyLife 2.1.x and earlier.

    Parameters
    ----------
    fatigue_data : FatigueData
        Validated Wöhler fatigue data accessor.
    """

    def _zone_for_infinite_likelihood(self):
        return self._fd.infinite_zone

    def _fractures_for_finite_likelihood(self):
        return self._fd.fractures

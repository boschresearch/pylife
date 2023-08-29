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

"""
Implementation of the miner rule for fatigue analysis
=====================================================

Currently, the following implementations are part of this module:

* Miner Elementary
* Miner Haibach

References
----------
* M. Wächter, C. Müller and A. Esderts, "Angewandter Festigkeitsnachweis nach {FKM}-Richtlinie"
  Springer Fachmedien Wiesbaden 2017, https://doi.org/10.1007/978-3-658-17459-0

* E. Haibach, "Betriebsfestigkeit", Springer-Verlag 2006, https://doi.org/10.1007/3-540-29364-7
"""

__author__ = "Cedric Philip Wagner"
__maintainer__ = "Johannes Mueller"

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from pylife.strength.fatigue import Fatigue
from pylife.materiallaws.woehlercurve import WoehlerCurve

import pylife.strength.solidity as SOL


class MinerBase(WoehlerCurve, ABC):
    """Basic functions related to miner-rule (original).

    Uses the constructor of :class:`~pylife.materiallaws.WoehlerCurve`.
    """

    def finite_life_factor(self, N):
        """Calculate *finite life factor* according to Waechter2017 (p. 96).

        Parameters
        ----------
        N : float
            Collective range (sum of cycle numbers) of load collective
        """
        return np.power(self.ND/N, 1./self.k_1)

    def effective_damage_sum(self, collective):
        """Compute *effective damage sum* D_m.

        Refers to the formula given in Waechter2017, p. 99

        Parameters
        ----------
        collective : a load collective
            the multiple of the lifetime

        Returns
        -------
        effective_damage_sum : float or :class:`pandas.Series`
            The effective damage sums for the collective
        """
        A = self.lifetime_multiple(collective)
        return effective_damage_sum(A)

    def gassner_cycles(self, collective):
        """Compute the cycles of the Gassner line for a certain load collective.

        Parameters
        ----------
        collective : :class:`~pylife.stress.rainflow.LoadCollective` or similar
            The load collective

        Returns
        -------
        cycles
            The cycles for the collective

        Note
        ----
        The absolute load levels of the collective are important.
        """
        return self.cycles(collective.amplitude.max()) * self.lifetime_multiple(collective)

    @abstractmethod
    def lifetime_multiple(self, collective):
        """Compute the lifetime multiple according to the corresponding Miner rule.

        Needs to be implemented in the class implementing the Miner rule.

        Parameters
        ----------
        collective : :class:`~pylife.stress.rainflow.LoadCollective` or similar
            The load collective

        Returns
        -------
        lifetime_multiple : float > 0
            lifetime multiple
        """

        pass


@pd.api.extensions.register_series_accessor('gassner_miner_elementary')
class MinerElementary(MinerBase):
    """Implementation of Miner Elementary according to Waechter2017."""

    def gassner(self, collective):
        """Calculate the Gaßner shift according to Miner Elementary.

        Parameters
        ----------
        collective : :class:`~pylife.stress.rainflow.LoadCollective` or similar
            The load collective

        Returns
        -------
        gassner : :class:`~pylife.stength.Fatigue`
            The Gaßner shifted fatigue strength object.
        """
        gassner = self.to_pandas().copy()
        gassner['ND'] = self.ND * self.lifetime_multiple(collective)
        return Fatigue(gassner)

    def lifetime_multiple(self, collective):
        """Compute the lifetime multiple according to Miner Elementary.

        Described in Waechter2017 as "Lebensdauervielfaches, A_ele".

        Parameters
        ----------
        collective : :class:`~pylife.stress.rainflow.LoadCollective` or similar
            The load collective

        Returns
        -------
        lifetime_multiple : float > 0
            lifetime multiple
        """
        return 1. / SOL.haibach(collective, self.k_1)


@pd.api.extensions.register_series_accessor('gassner_miner_haibach')
class MinerHaibach(MinerBase):
    """Miner-modified according to Haibach (2006).

    Warnings
    --------
    Contrary to Miner Elementary, the lifetime multiple is not constant but
    dependent on the evaluated load level!  That is why there is no method for
    the Gaßner shift.
    """

    def lifetime_multiple(self, collective):
        """Compute the lifetime multiple for Miner-modified according to Haibach.

        Refer to Haibach (2006), p. 291 (3.21-61). The lifetime multiple can be
        expressed in respect to the maximum amplitude so that
        N_lifetime = N_Smax * A

        Parameters
        ----------
        collective : :class:`~pylife.stress.rainflow.LoadCollective` or similar
            The load collective

        Returns
        -------
        lifetime_multiple : float > 0
            lifetime multiple
            return value is 'inf' if maximum collective amplitude < SD
        """
        s_a = collective.amplitude
        max_amp = s_a.max()

        cycles = collective.cycles

        s_a = s_a / max_amp
        x_D = self.SD / max_amp

        i_full_damage = (s_a >= x_D)
        i_reduced_damage = (s_a < x_D)

        s_full_damage = s_a[i_full_damage]
        s_reduced_damage = s_a[i_reduced_damage]

        n_full_damage = cycles[i_full_damage]
        n_reduced_damage = cycles[i_reduced_damage]

        # first expression of the summation term in the denominator
        sum_1 = np.dot(n_full_damage, (s_full_damage**self.k_1))
        sum_2 = x_D**(1 - self.k_1) * np.dot(n_reduced_damage, (s_reduced_damage**(2 * self.k_1 - 1)))

        return cycles.sum() / (sum_1 + sum_2)


def effective_damage_sum(lifetime_multiple):
    """Compute *effective damage sum*.

    Refers to the formula given in Waechter2017, p. 99

    Parameters
    ----------
    A : float or np.ndarray (with 1 element)
        the multiple of the lifetime

    Returns
    d_m : float
        the effective damage sum
    """
    d_min = 0.3  # minimum as suggested by FKM
    d_max = 1.0

    d_m_no_limits = 2. / (lifetime_multiple**(1./4.))
    d_m = min(
        max(d_min, d_m_no_limits),
        d_max
    )

    return d_m

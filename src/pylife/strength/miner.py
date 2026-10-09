# Copyright (c) 2019-2026 - for information on the respective copyright owner
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

"""Provide Miner damage accumulation rules for fatigue analysis.

The module implements helpers for choosing between Miner original, Miner
elementary, and Miner-Haibach assumptions. The variants differ in how they
extend the S-N curve below the endurance knee: horizontal for Miner original,
unchanged finite-life slope for Miner elementary, and the Haibach slope
``2 * k_1 - 1`` for Miner-Haibach. The terminology follows Wächter et al.
[Waechter-Miner]_ and Haibach [Haibach-Miner]_.

References
----------
.. [Waechter-Miner] M. Wächter, C. Müller, and A. Esderts, "Angewandter
   Festigkeitsnachweis nach FKM-Richtlinie", Springer Fachmedien Wiesbaden,
   2017.
.. [Haibach-Miner] E. Haibach, "Betriebsfestigkeit", Springer-Verlag, 2006.
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
    """Provide common operations for Gassner-Miner calculations.

    ``MinerBase`` extends :class:`pylife.materiallaws.WoehlerCurve` with
    operations that depend on a load collective shape, such as lifetime
    multiples and effective damage sums. Subclasses implement the selected
    Miner hypothesis.

    Parameters
    ----------
    pandas_obj : pandas.Series or pandas.DataFrame
        Pandas object containing Wöhler curve parameters.

    See Also
    --------
    pylife.strength.miner.MinerElementary : Use the elementary Miner hypothesis.
    pylife.strength.miner.MinerHaibach : Use the Miner-Haibach hypothesis.
    """

    def finite_life_factor(self, N):
        r"""Calculate the finite-life factor for a collective cycle count.

        Parameters
        ----------
        N : float or array_like
            Total number of cycles in the load collective.

        Returns
        -------
        float or numpy.ndarray
            Finite-life factor, dimensionless.

        Notes
        -----
        Following Wächter et al. [Waechter-MinerBase]_ the factor is

        .. math::

            f_N = \left(\frac{N_D}{N}\right)^{1/k_1}.

        References
        ----------
        .. [Waechter-MinerBase] M. Wächter, C. Müller, and A. Esderts,
           "Angewandter Festigkeitsnachweis nach FKM-Richtlinie", Springer
           Fachmedien Wiesbaden, 2017, p. 96.
        """
        return np.power(self.ND/N, 1./self.k_1)

    def effective_damage_sum(self, collective):
        """Calculate the effective damage sum for a load collective.

        Parameters
        ----------
        collective : pylife.stress.collective.LoadCollective
            Load collective with stress amplitudes and cycle counts.

        Returns
        -------
        float or pandas.Series
            Effective damage sum, dimensionless. A value of ``1.0`` means
            failure in the unmodified Miner convention.

        See Also
        --------
        pylife.strength.miner.effective_damage_sum : Calculate the value from a lifetime multiple.
        """
        A = self.lifetime_multiple(collective)
        return effective_damage_sum(A)

    def gassner_cycles(self, collective):
        """Calculate the Gassner cycle count for a load collective.

        Parameters
        ----------
        collective : pylife.stress.collective.LoadCollective
            Load collective with stress amplitudes and cycle counts.

        Returns
        -------
        float or pandas.Series
            Number of cycles on the Gassner line for the given collective.

        Notes
        -----
        The absolute load level matters because the maximum collective
        amplitude is used as the reference point on the Wöhler curve.
        """
        return self.cycles(collective.amplitude.max()) * self.lifetime_multiple(collective)

    @abstractmethod
    def lifetime_multiple(self, collective):
        """Calculate the lifetime multiple for the selected Miner rule.

        Parameters
        ----------
        collective : pylife.stress.collective.LoadCollective
            Load collective with stress amplitudes and cycle counts.

        Returns
        -------
        float
            Lifetime multiple ``A``, dimensionless and greater than ``0``.

        Notes
        -----
        Subclasses implement the actual hypothesis. The value scales the life
        at the maximum collective amplitude to the life of the complete
        collective.
        """

        pass


@pd.api.extensions.register_series_accessor('gassner_miner_elementary')
class MinerElementary(MinerBase):
    """Apply the Miner elementary damage accumulation hypothesis.

    Miner elementary extends the S-N curve below the knee point with the same
    slope as the finite-life branch, ``k_2 = k_1``. The resulting collective
    shape factor is independent of the absolute load level, so a
    Gassner-shifted Wöhler curve can be constructed.

    Parameters
    ----------
    pandas_obj : pandas.Series or pandas.DataFrame
        Pandas object containing Wöhler curve parameters.

    See Also
    --------
    pylife.materiallaws.WoehlerCurve.miner_elementary : Select ``k_2 = k_1`` for direct damage calculation.
    pylife.strength.miner.MinerHaibach : Use the Haibach slope below the knee.
    """

    def gassner(self, collective):
        """Calculate the Gassner-shifted Wöhler curve for Miner elementary.

        Parameters
        ----------
        collective : pylife.stress.collective.LoadCollective
            Load collective with stress amplitudes and cycle counts.

        Returns
        -------
        pylife.strength.fatigue.Fatigue
            Fatigue accessor of the Gassner-shifted Wöhler curve. The knee
            cycle count ``ND`` is multiplied by the lifetime multiple of
            ``collective``.
        """
        gassner = self.to_pandas().copy()
        gassner['ND'] = self.ND * self.lifetime_multiple(collective)
        return Fatigue(gassner)

    def lifetime_multiple(self, collective):
        r"""Calculate the Miner-elementary lifetime multiple.

        Parameters
        ----------
        collective : pylife.stress.collective.LoadCollective
            Load collective with stress amplitudes and cycle counts.

        Returns
        -------
        float
            Lifetime multiple ``A_ele``, dimensionless and greater than ``0``.

        Notes
        -----
        Following Wächter et al. [Waechter-MinerElementary]_, the
        Miner-elementary lifetime multiple is the reciprocal of the Haibach
        solidity value ``V`` computed with slope ``k_1``:

        .. math::

            A_{ele} = \frac{1}{V}.

        References
        ----------
        .. [Waechter-MinerElementary] M. Wächter, C. Müller, and A. Esderts,
           "Angewandter Festigkeitsnachweis nach FKM-Richtlinie", Springer
           Fachmedien Wiesbaden, 2017.
        """
        return 1. / SOL.haibach(collective, self.k_1)


@pd.api.extensions.register_series_accessor('gassner_miner_haibach')
class MinerHaibach(MinerBase):
    """Apply the Miner-Haibach damage accumulation hypothesis.

    Miner-Haibach extends the S-N curve below the knee point with the slope
    ``k_2 = 2 * k_1 - 1``. Loads below ``SD`` therefore still contribute
    damage, but less strongly than loads above the knee.

    Parameters
    ----------
    pandas_obj : pandas.Series or pandas.DataFrame
        Pandas object containing Wöhler curve parameters.

    Warnings
    --------
    Unlike Miner elementary, the lifetime multiple depends on the evaluated
    load level. Therefore this accessor does not provide a Gassner-shift
    method.
    """

    def lifetime_multiple(self, collective):
        r"""Calculate the Miner-Haibach lifetime multiple.

        Parameters
        ----------
        collective : pylife.stress.collective.LoadCollective
            Load collective with stress amplitudes and cycle counts.

        Returns
        -------
        float
            Lifetime multiple ``A``, dimensionless and greater than ``0``. The
            return value is ``numpy.inf`` if the maximum collective amplitude
            is below ``SD``.

        Notes
        -----
        With normalized amplitudes ``x_i = S_{a,i} / S_{a,max}`` and
        ``x_D = SD / S_{a,max}``, Haibach damage accumulation
        [Haibach-MinerHaibach]_ uses

        .. math::

            A = \frac{\sum_i n_i}
                     {\sum_{x_i \ge x_D} n_i x_i^{k_1}
                     + x_D^{1-k_1}\sum_{x_i < x_D} n_i x_i^{2k_1-1}}.

        References
        ----------
        .. [Haibach-MinerHaibach] E. Haibach, "Betriebsfestigkeit",
           Springer-Verlag, 2006, p. 291.
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
    r"""Calculate the FKM effective damage sum from a lifetime multiple.

    Parameters
    ----------
    lifetime_multiple : float
        Lifetime multiple ``A`` of a load collective, dimensionless.

    Returns
    -------
    float
        Effective damage sum, dimensionless. The value is limited to the
        interval ``[0.3, 1.0]``.

    Notes
    -----
    The FKM effective damage sum is calculated as

    .. math::

        D_m = \min\left(1, \max\left(0.3, \frac{2}{A^{1/4}}\right)\right).

    A damage sum of ``1.0`` means failure in the unmodified Miner convention.
    """
    d_min = 0.3  # minimum as suggested by FKM
    d_max = 1.0

    d_m_no_limits = 2. / (lifetime_multiple**(1./4.))
    d_m = min(
        max(d_min, d_m_no_limits),
        d_max
    )

    return d_m

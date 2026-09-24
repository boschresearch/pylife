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

"""Provide deprecated finite-life S-N curve helper classes.

The classes in this module predate :class:`pylife.materiallaws.WoehlerCurve`
and :class:`pylife.strength.fatigue.Fatigue`. They remain available for
compatibility and delegate their calculations to the modern accessors.
"""

__author__ = "Cedric Philip Wagner"
__maintainer__ = "Johannes Mueller"

import warnings

import pandas as pd

import pylife.strength.fatigue
import pylife.stress

warnings.warn(
    FutureWarning(
        "The module pylife.strength.helpers is deprecated and no longer under test. "
        "The functionality is now avaliable in the pylife.materiallaws.WoehlerCurve."
    )
)

class FiniteLifeBase:
    """Provide shared state for deprecated finite-life S-N curve helpers.

    Parameters
    ----------
    k_1 : float
        Wöhler slope in the finite-life region, dimensionless.
    SD_50 : float
        Stress or load amplitude at the knee point for ``50 %`` failure
        probability, in MPa or another consistent unit.
    ND_50 : float
        Number of cycles at the knee point for ``50 %`` failure probability.

    Warnings
    --------
    This class is deprecated. Use :class:`pylife.materiallaws.WoehlerCurve`
    and :class:`pylife.strength.fatigue.Fatigue` instead.
    """

    def __init__(self, k_1, SD_50, ND_50):
        warnings.warn(DeprecationWarning("FiniteLifeBase and derived classes are deperecated. "
                                         "Use WoehlerCurve and Fatigue accessors instead."))
        self._wc = pd.Series({
            'k_1': k_1,
            'SD': SD_50,
            'ND': ND_50
        })

    @property
    def k_1(self):
        """Return the finite-life Wöhler slope.

        Returns
        -------
        float
            Wöhler slope ``k_1``, dimensionless.
        """
        return self._wc.k_1


class FiniteLifeLine(FiniteLifeBase):
    r"""Represent a deprecated logarithmic finite-life S-N line.

    Parameters
    ----------
    k : float
        Wöhler slope in the finite-life region, dimensionless.
    SD_50 : float
        Stress or load amplitude at the knee point for ``50 %`` failure
        probability, in MPa or another consistent unit.
    ND_50 : float
        Number of cycles at the knee point for ``50 %`` failure probability.

    Warnings
    --------
    This class is deprecated. Use :class:`pylife.materiallaws.WoehlerCurve`
    and :class:`pylife.strength.fatigue.Fatigue` instead.

    Notes
    -----
    Following Haibach [Haibach-SNCurve]_, the finite-life branch follows the
    Basquin relation

    .. math::

        S_a = S_D \left(\frac{N_D}{N}\right)^{1/k_1}.

    References
    ----------
    .. [Haibach-SNCurve] E. Haibach, "Betriebsfestigkeit", Springer-Verlag,
       2006.
    """

    def __init__(self, k, SD_50, ND_50):
        super().__init__(k, SD_50, ND_50)


class FiniteLifeCurve(FiniteLifeBase):
    r"""Represent a deprecated finite-life S-N curve in linear scale.

    Parameters
    ----------
    k_1 : float
        Wöhler slope in the finite-life region, dimensionless.
    SD_50 : float
        Stress or load amplitude at the knee point for ``50 %`` failure
        probability, in MPa or another consistent unit.
    ND_50 : float
        Number of cycles at the knee point for ``50 %`` failure probability.

    Warnings
    --------
    This class is deprecated. Use :class:`pylife.materiallaws.WoehlerCurve`
    and :class:`pylife.strength.fatigue.Fatigue` instead.

    Notes
    -----
    Use either stress amplitudes consistently or stress ranges consistently for
    the curve and for load collectives. Following Haibach
    [Haibach-FiniteLifeCurve]_, the Basquin relation is

    .. math::

        N = N_D \left(\frac{S_D}{S_a}\right)^{k_1}.

    References
    ----------
    .. [Haibach-FiniteLifeCurve] E. Haibach, "Betriebsfestigkeit",
       Springer-Verlag, 2006.
    """
    def __init__(self, k_1, SD_50, ND_50):
        super().__init__(k_1, SD_50, ND_50)

    def calc_S(self, N, ignore_limits=True):
        r"""Calculate the finite-life stress amplitude for a cycle count.

        Parameters
        ----------
        N : float or array_like
            Number of cycles.
        ignore_limits : bool, optional
            Retained for compatibility. The implementation delegates to
            :meth:`pylife.materiallaws.WoehlerCurve.basquin_load` and does not
            enforce finite-life limits. Default is ``True``.

        Returns
        -------
        float or numpy.ndarray
            Stress or load amplitude corresponding to ``N``, in the same unit
            as ``SD_50``.

        Notes
        -----
        The returned amplitude follows

        .. math::

            S_a = S_D \left(\frac{N_D}{N}\right)^{1/k_1}.
        """
        return self._wc.woehler.basquin_load(N)

    def calc_N(self, S, ignore_limits=False):
        r"""Calculate the finite-life cycle count for a stress amplitude.

        Parameters
        ----------
        S : float or array_like
            Stress or load amplitude in the same unit as ``SD_50``.
        ignore_limits : bool, optional
            Retained for compatibility. The implementation delegates to
            :meth:`pylife.materiallaws.WoehlerCurve.basquin_cycles` and does
            not enforce finite-life limits. Default is ``False``.

        Returns
        -------
        float or numpy.ndarray
            Number of cycles corresponding to ``S``.

        Notes
        -----
        The returned cycle count follows

        .. math::

            N = N_D \left(\frac{S_D}{S_a}\right)^{k_1}.
        """
        return self._wc.woehler.basquin_cycles(S)

    def calc_damage(self, loads, method="elementar", index_name="range"):
        """Calculate Miner damage for a load histogram.

        Parameters
        ----------
        loads : pandas.Series
            Load histogram whose index contains a load level, named ``range``
            by default, and whose values are cycle counts. The load level must
            be a stress range if the S-N curve is defined by ranges, or an
            amplitude if the S-N curve is defined by amplitudes.
        method : {'elementar', 'MinerHaibach', 'original'}, optional
            Damage accumulation hypothesis. ``'elementar'`` sets
            ``k_2 = k_1``, ``'MinerHaibach'`` sets ``k_2 = 2 * k_1 - 1``, and
            ``'original'`` leaves the endurance branch horizontal. Default is
            ``'elementar'``.
        index_name : str, optional
            Name of the load-level index in ``loads``. It is temporarily
            mapped to ``'range'`` for the load collective accessor. Default is
            ``'range'``.

        Returns
        -------
        pandas.Series
            Damage contribution for each load histogram bin, dimensionless.

        Warnings
        --------
        This compatibility method uses the modern ``.fatigue`` and
        ``.load_collective`` accessors internally. New code should call those
        accessors directly.
        """
        estimator = self._wc.fatigue

        if method == 'elementar':
            estimator = estimator.miner_elementary()
        elif method == 'MinerHaibach':
            estimator = estimator.miner_haibach()

        if index_name != "range":
            loads = loads.copy()
            names = ["range" if name == index_name else name for name in loads.index.names]
            loads.index.set_names(names, inplace=True)

        damage = estimator.damage(loads.load_collective)
        if index_name != "range":
            names = [index_name if name == "range" else name for name in loads.index.names]
            damage.index.set_names(names, inplace=True)

        return damage

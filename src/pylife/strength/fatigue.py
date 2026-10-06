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

"""Provide the ``.fatigue`` accessor for Wöhler curve damage calculations.

The module registers :class:`Fatigue` as a pandas Series and DataFrame
accessor. It extends :class:`pylife.materiallaws.WoehlerCurve` with helpers
for evaluating load collectives against S-N curve data.
"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import pandas as pd

from pylife.materiallaws import WoehlerCurve


@pd.api.extensions.register_series_accessor('fatigue')
@pd.api.extensions.register_dataframe_accessor('fatigue')
class Fatigue(WoehlerCurve):
    """Represent a Wöhler curve for fatigue damage calculations.

    ``Fatigue`` is registered as the ``.fatigue`` accessor on
    :class:`pandas.Series` and :class:`pandas.DataFrame` objects. It uses the
    Wöhler curve signal contract of :class:`pylife.materiallaws.WoehlerCurve`
    and therefore expects the following keys:

    * ``k_1``: Mandatory finite-life slope above the knee point, dimensionless.
    * ``ND``: Mandatory number of cycles at the knee point, in cycles.
    * ``SD``: Mandatory stress or load amplitude at the knee point, in a
      consistent unit such as MPa or N.
    * ``k_2``: Optional slope below the knee point, dimensionless. Default is
      ``numpy.inf`` for Miner original behavior.
    * ``TN``: Optional scatter range in cycle direction, ``N_90 / N_10``.
      Default is ``1.0`` or is derived from ``TS ** k_1``.
    * ``TS``: Optional scatter range in load direction, ``SD_90 / SD_10``.
      Default is ``1.0`` or is derived from ``TN ** (1 / k_1)``.
    * ``failure_probability``: Optional failure probability represented by the
      stored curve. Default is ``0.5``.

    The accessor expects load collectives that expose ``amplitude`` and
    ``cycles`` through :mod:`pylife.stress.collective`. Load amplitudes must
    use the same unit as ``SD``.

    Parameters
    ----------
    pandas_obj : pandas.Series or pandas.DataFrame
        Pandas object containing Wöhler curve parameters.

    See Also
    --------
    pylife.materiallaws.WoehlerCurve : Represent and transform Wöhler curves.
    pylife.stress.collective.LoadCollective : Represent counted load collectives.

    Examples
    --------
    >>> import pandas as pd
    >>> import pylife.strength.fatigue
    >>> import pylife.stress.collective
    >>> woehler = pd.Series({"k_1": 7.0, "ND": 2e6, "SD": 300.0,
    ...                      "TN": 3.4, "TS": 1.234})
    >>> collective = pd.DataFrame({"range": [1200.0, 1600.0],
    ...                            "mean": [0.0, 0.0],
    ...                            "cycles": [10.0, 5.0]}).load_collective
    >>> round(float(woehler.fatigue.damage(collective).sum()), 6)
    0.003037
    >>> round(float(woehler.fatigue.miner_elementary().damage(collective).sum()), 6)
    0.003037
    >>> round(float(woehler.fatigue.miner_haibach().damage(collective).sum()), 6)
    0.003037
    """

    def damage(self, load_collective):
        r"""Calculate fatigue damage for a load collective.

        Parameters
        ----------
        load_collective : pandas.DataFrame or pylife.stress.collective.LoadCollective
            Load collective containing cycle counts and stress or load
            amplitudes in the same unit as ``SD``.

        Returns
        -------
        pandas.Series
            Damage contribution for each collective row, dimensionless. The
            index is the broadcast index of the Wöhler curve parameters and
            ``load_collective``.

        Notes
        -----
        Damage is calculated by Palmgren-Miner linear accumulation for each
        load level, using the current Wöhler curve and its selected ``k_2``
        branch:

        .. math::

            D_i = \frac{n_i}{N_i}.

        ``D_i`` is dimensionless; a sum of ``1.0`` conventionally means
        failure.
        """
        cycles = self.cycles(load_collective.amplitude)
        return pd.Series(load_collective.cycles / cycles, name='damage')

    def security_load(self, load_distribution, allowed_failure_probability):
        """Calculate the load-direction safety factor for a load distribution.

        Parameters
        ----------
        load_distribution : pandas.DataFrame or pylife.stress.collective.LoadCollective
            Load distribution containing cycle counts and stress or load
            amplitudes in the same unit as ``SD``.
        allowed_failure_probability : float or array_like
            Target failure probability, dimensionless, used to transform the
            Wöhler curve before calculating the allowed load.

        Returns
        -------
        pandas.Series
            Safety factor in load direction, dimensionless. Values greater than
            ``1.0`` indicate that the allowed amplitude exceeds the applied
            amplitude. The index is the broadcast index of the Wöhler curve and
            load data.
        """
        allowed_load = self.load(load_distribution.cycles, allowed_failure_probability)
        return pd.Series(allowed_load / load_distribution.amplitude, name='security_factor')

    def security_cycles(self, load_distribution, allowed_failure_probability):
        """Calculate the cycle-direction safety factor for a load distribution.

        Parameters
        ----------
        load_distribution : pandas.DataFrame or pylife.stress.collective.LoadCollective
            Load distribution containing cycle counts and stress or load
            amplitudes in the same unit as ``SD``.
        allowed_failure_probability : float or array_like
            Target failure probability, dimensionless, used to transform the
            Wöhler curve before calculating the allowed cycle count.

        Returns
        -------
        pandas.Series
            Safety factor in cycle direction, dimensionless. Values greater
            than ``1.0`` indicate that the allowable number of cycles exceeds
            the applied number of cycles. The index is the broadcast index of
            the Wöhler curve and load data.
        """
        allowed_cycles = self.cycles(load_distribution.amplitude, allowed_failure_probability)
        return pd.Series(allowed_cycles / load_distribution.cycles, name='security_factor')

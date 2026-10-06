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

"""Provide pandas accessors for Wöhler fatigue curves.

The module exposes the ``.woehler`` accessor for :class:`pandas.Series` and
:class:`pandas.DataFrame` objects that describe S-N curves for fatigue-life
calculations.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

from pylife.utils.functions import scattering_range_to_std

from pylife import PylifeSignal


@pd.api.extensions.register_series_accessor('woehler')
@pd.api.extensions.register_dataframe_accessor('woehler')
class WoehlerCurve(PylifeSignal):
    """Represent a Wöhler curve stored in a pandas object.

    A Wöhler curve, also called an S-N curve, relates a load amplitude to the
    number of cycles to failure. The accessor accepts scalar curve parameters
    in a :class:`pandas.Series` or row-wise curve parameters in a
    :class:`pandas.DataFrame`.

    Parameters
    ----------
    pandas_obj : pandas.Series or pandas.DataFrame
        Pandas object containing Wöhler curve parameters.

    Notes
    -----
    The signal contract is validated and completed as follows:

    * ``k_1`` : Mandatory slope in the finite-life range above the knee point,
      dimensionless.
    * ``ND`` : Mandatory number of cycles at the knee point, sometimes called
      endurance cycles, in cycles.
    * ``SD`` : Mandatory load amplitude at the knee point, sometimes called
      endurance limit, in a consistent load unit such as MPa or N.
    * ``k_2`` : Optional slope below the knee point, dimensionless. Default is
      ``numpy.inf``, representing a horizontal endurance branch.
    * ``TN`` : Optional scatter range in cycle direction, defined as
      ``N_90 / N_10``. Default is ``1.0`` if neither ``TN`` nor ``TS`` is
      given; otherwise it is derived from ``TS ** k_1``.
    * ``TS`` : Optional scatter range in load direction, defined as
      ``SD_90 / SD_10``. Default is ``1.0`` if neither ``TN`` nor ``TS`` is
      given; otherwise it is derived from ``TN ** (1 / k_1)``.
    * ``failure_probability`` : Optional failure probability represented by
      the stored curve. Default is ``0.5``.

    The S-N terminology follows fatigue testing practice, for example DIN
    50100. The load unit is not fixed by pyLife; use one consistent unit for
    ``SD`` and all load values passed to the accessor.
    """

    def _validate(self):
        self.fail_if_key_missing(['k_1', 'ND', 'SD'])
        self._k_2 = self._obj.get('k_2', np.inf)

        self._TN = self._obj.get('TN', None)
        self._TS = self._obj.get('TS', None)

        if self._TN is None and self._TS is None:
            self._TN = 1.0
            self._TS = 1.0
        elif self._TS is None:
            self._TS = np.power(self._TN, 1./self._obj.k_1)
        elif self._TN is None:
            self._TN = np.power(self._TS, self._obj.k_1)

        self._failure_probability = self._obj.get('failure_probability', 0.5)

        self._obj['k_2'] = self._k_2
        self._obj['TN'] = self._TN
        self._obj['TS'] = self._TS
        self._obj['failure_probability'] = self._failure_probability

    @property
    def SD(self):
        """Return the load amplitude at the knee point.
        """
        return self._obj.SD

    @property
    def ND(self):
        """Return the number of cycles at the knee point.
        """
        return self._obj.ND

    @property
    def k_1(self):
        """Return the finite-life Wöhler slope.
        """
        return self._obj.k_1

    @property
    def k_2(self):
        """Return the Wöhler slope below the knee point.
        """
        return self._obj.k_2

    @property
    def TN(self):
        """Return the scatter range in cycle direction.
        """
        return self._obj.TN

    @property
    def TS(self):
        """Return the scatter range in load direction.
        """
        return self._obj.TS

    @property
    def failure_probability(self):
        """Return the represented failure probability.
        """
        return self._failure_probability

    def transform_to_failure_probability(self, failure_probability):
        """Transform the Wöhler curve to another failure probability.

        Parameters
        ----------
        failure_probability : float or array_like or None
            Target failure probability. If ``None``, return the accessor
            itself without changing the stored curve.

        Returns
        -------
        WoehlerCurve
            Curve accessor transformed to ``failure_probability`` or ``self``
            if ``failure_probability`` is ``None``.
        """
        if failure_probability is None:
            return self

        failure_probability = np.asarray(failure_probability, dtype=np.float64)

        failure_probability, obj = self.broadcast(failure_probability)

        native_ppf = stats.norm.ppf(obj.failure_probability)
        goal_ppf = stats.norm.ppf(failure_probability)

        SD = np.asarray(obj.SD / 10**((native_ppf-goal_ppf)*scattering_range_to_std(obj.TS)))
        ND = np.asarray(obj.ND / 10**((native_ppf-goal_ppf)*scattering_range_to_std(obj.TN)))
        ND.flags.writeable = True
        ND[SD != 0] *= np.power(SD[SD != 0]/obj.SD, -obj.k_1)

        transformed = obj.copy()
        transformed['SD'] = SD
        transformed['ND'] = ND
        transformed['failure_probability'] = failure_probability

        return WoehlerCurve(transformed)

    def miner_original(self):
        """Set ``k_2`` according to the Miner original method.

        Returns
        -------
        WoehlerCurve
            Copy of the curve with ``k_2`` set to ``numpy.inf``.
        """
        new = self._obj.copy()
        new['k_2'] =  np.inf
        return self.__class__(new)

    def miner_elementary(self):
        """Set ``k_2`` according to the Miner elementary method.

        Returns
        -------
        WoehlerCurve
            Copy of the curve with ``k_2`` set to ``k_1``.
        """
        new = self._obj.copy()
        new['k_2'] =  self._obj.k_1
        return self.__class__(new)

    def miner_haibach(self):
        """Set ``k_2`` according to the Miner-Haibach method.

        Returns
        -------
        WoehlerCurve
            Copy of the curve with ``k_2`` set to ``2 * k_1 - 1``.
        """
        new = self._obj.copy()
        new['k_2'] = 2. * self._obj.k_1 - 1.
        return self.__class__(new)

    def cycles(self, load, failure_probability=None):
        """Calculate cycle numbers from load amplitudes.

        Parameters
        ----------
        load : array_like
            Load amplitudes in the same unit as ``SD``.
        failure_probability : float or array_like or None, optional
            Failure probability for which the cycle numbers are calculated. If
            ``None``, use the accessor's current ``failure_probability``.
            Default is ``None``.

        Returns
        -------
        numpy.ndarray or pandas.Series
            Numbers of cycles to failure for the given ``load`` values. The
            result is a :class:`pandas.Series` if ``load`` is a series.

        Notes
        -----
        By default the calculation is performed according to the Basquin
        equation using :meth:`basquin_cycles`. Derived classes can override
        this method to implement a different fatigue law.
        """
        return self.basquin_cycles(load, failure_probability)

    def load(self, cycles, failure_probability=None):
        """Calculate load amplitudes from cycle numbers.

        Parameters
        ----------
        cycles : array_like
            Numbers of cycles to failure.
        failure_probability : float or array_like or None, optional
            Failure probability for which the load amplitudes are calculated.
            If ``None``, use the accessor's current ``failure_probability``.
            Default is ``None``.

        Returns
        -------
        numpy.ndarray or pandas.Series
            Load amplitudes in the same unit as ``SD``. The result is a
            :class:`pandas.Series` if ``cycles`` is a series.

        Notes
        -----
        By default the calculation is performed according to the Basquin
        equation using :meth:`basquin_load`. Derived classes can override this
        method to implement a different fatigue law.
        """
        return self.basquin_load(cycles, failure_probability)

    def basquin_cycles(self, load, failure_probability=None):
        r"""Calculate cycle numbers from loads using the Basquin equation.

        Parameters
        ----------
        load : array_like
            Load amplitudes in the same unit as ``SD``.
        failure_probability : float or array_like or None, optional
            Failure probability for which the cycle numbers are calculated. If
            ``None``, use the accessor's current ``failure_probability``.
            Default is ``None``.

        Returns
        -------
        numpy.ndarray or pandas.Series
            Numbers of cycles to failure for the given ``load`` values. The
            result is a :class:`pandas.Series` if ``load`` is a series.

        Notes
        -----
        The finite-life branch follows

        .. math::

            N = ND \left(\frac{L}{SD}\right)^{-k}

        with ``k_1`` above the knee point and ``k_2`` below it. If ``k_2`` is
        infinite, loads below ``SD`` lead to infinite life.

        Examples
        --------
        >>> import pandas as pd
        >>> wc = pd.Series({'k_1': 5.0, 'ND': 1e6, 'SD': 100.0}).woehler
        >>> float(wc.basquin_cycles(200.0))
        31250.0
        """
        def ensure_float_to_prevent_int_overflow(load):
            if isinstance(load, pd.Series):
                return pd.Series(load, dtype=np.float64)
            return np.asarray(load, dtype=np.float64)

        transformed = self.transform_to_failure_probability(failure_probability)

        load = ensure_float_to_prevent_int_overflow(load)
        ld, wc = transformed.broadcast(load)
        cycles = np.full_like(ld, np.inf)

        k = self._make_k(ld, wc.SD, wc)
        in_limit = np.isfinite(k)
        cycles[in_limit] = wc.ND[in_limit] * np.power(ld[in_limit]/wc.SD[in_limit], -k[in_limit])

        if not isinstance(load, pd.Series):
            return cycles
        return pd.Series(cycles, index=ld.index)

    def basquin_load(self, cycles, failure_probability=None):
        r"""Calculate loads from cycle numbers using the Basquin equation.

        Parameters
        ----------
        cycles : array_like
            Numbers of cycles to failure.
        failure_probability : float or array_like or None, optional
            Failure probability for which the load amplitudes are calculated.
            If ``None``, use the accessor's current ``failure_probability``.
            Default is ``None``.

        Returns
        -------
        numpy.ndarray or pandas.Series
            Load amplitudes in the same unit as ``SD``. The result is a
            :class:`pandas.Series` if ``cycles`` is a series.

        Notes
        -----
        This method inverts the Basquin relation used by
        :meth:`basquin_cycles`:

        .. math::

            L = SD \left(\frac{N}{ND}\right)^{-1/k}
        """
        transformed = self.transform_to_failure_probability(failure_probability)

        cyc, wc = transformed.broadcast(cycles)
        load = np.asarray(wc.SD).copy()

        k = self._make_k(-cyc, -wc.ND, wc)
        in_limit = np.isfinite(k)
        load[in_limit] = wc.SD[in_limit] * np.power(cyc[in_limit]/wc.ND[in_limit], -1./k[in_limit])

        if not isinstance(cycles, pd.Series):
            return load
        return pd.Series(load, index=cyc.index)

    def _make_k(self, src, ref, wc):
        k = np.asarray(wc.k_1).copy()
        k_2 = np.asarray(wc.k_2)

        below_limit = np.asarray(src < ref)
        if k.shape == ():
            k = np.full_like(src, k, dtype=np.double)
            k_2 = np.full_like(src, k_2, dtype=np.double)

        k[below_limit] = k_2[below_limit]
        return k

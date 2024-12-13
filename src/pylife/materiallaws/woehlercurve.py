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

import pandas as pd
import numpy as np
import scipy.stats as stats

from pylife.utils.functions import scattering_range_to_std

from pylife import PylifeSignal


@pd.api.extensions.register_series_accessor('woehler')
@pd.api.extensions.register_dataframe_accessor('woehler')
class WoehlerCurve(PylifeSignal):
    """A PylifeSignal accessor for Wöhler Curve data.

    Wöhler Curve (aka SN-curve) determines after how many load cycles at a
    certain load amplitude the component is expected to fail.

    The signal has the following mandatory keys:

    * ``k_1`` : The slope of the Wöhler Curve
    * ``ND`` : The cycle number of the endurance limit
    * ``SD`` : The load level of the endurance limit

    The ``_50`` suffixes imply that the values are valid for a 50% probability
    of failure.

    There are the following optional keys:

    * ``k_2`` : The slope of the Wöhler Curve below the endurance limit
                If the key is missing it is assumed to be infinity, i.e. perfect endurance
    * ``TN`` : The scatter in cycle direction, (N_90/N_10)
               If the key is missing it is assumed to be 1.0 – i.e. no scatter –
               or calculated from ``TS`` if given.
    * ``TS`` : The scatter in load direction, (SD_90/SD_10)
               If the key is missing it is assumed to be 1.0 – i.e. no scatter –
               or calculated from ``TN`` if given.
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
        return self._obj.SD

    @property
    def ND(self):
        return self._obj.ND

    @property
    def k_1(self):
        """The second Wöhler slope."""
        return self._obj.k_1

    @property
    def k_2(self):
        """The second Wöhler slope."""
        return self._obj.k_2

    @property
    def TN(self):
        """The load direction scatter value TN."""
        return self._obj.TN

    @property
    def TS(self):
        """The load direction scatter value TS."""
        return self._obj.TS

    @property
    def failure_probability(self):
        return self._failure_probability

    def transform_to_failure_probability(self, failure_probability):
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
        """Set k_2 to inf according Miner Original method (k_2 = inf).

        Returns
        -------
        modified copy of self
        """
        new = self._obj.copy()
        new['k_2'] =  np.inf
        return self.__class__(new)

    def miner_elementary(self):
        """Set k_2 to k_1 according Miner Elementary method (k_2 = k_1).

        Returns
        -------
        modified copy of self
        """
        new = self._obj.copy()
        new['k_2'] =  self._obj.k_1
        return self.__class__(new)

    def miner_haibach(self):
        """Set k_2 to value according Miner Haibach method (k_2 = 2 * k_1 - 1).

        Returns
        -------
        modified copy of self
        """
        new = self._obj.copy()
        new['k_2'] = 2. * self._obj.k_1 - 1.
        return self.__class__(new)

    def cycles(self, load, failure_probability=0.5):
        """Calculate the cycles numbers from loads.

        Parameters
        ----------
        load : array_like
            The load levels for which the corresponding cycle numbers are to be calculated.
        failure_probability : float, optional
            The failure probability with which the component should fail when
            charged with `load` for the calculated cycle numbers. Default 0.5

        Returns
        -------
        cycles : numpy.ndarray
            The cycle numbers at which the component fails for the given `load` values


        Notes
        -----
        By default the calculation is performed according to the Basquin
        equation using :meth:`basquin_cycles`.  Derived classes can choose to
        override this in order to implement a different fatigue law.
        """
        return self.basquin_cycles(load, failure_probability)

    def load(self, cycles, failure_probability=0.5):
        """Calculate the load values from loads.

        Parameters
        ----------
        cycles : array_like
            The cycle numbers for which the corresponding load levels are to be calculated.
        failure_probability : float, optional
            The failure probability with which the component should fail when
            charged with `load` for the calculated cycle numbers. Default 0.5

        Returns
        -------
        cycles : numpy.ndarray
            The cycle numbers at which the component fails for the given `load` values

        Notes
        -----
        By default the calculation is performed according to the Basquin
        equation using :meth:`basquin_cycles`.  Derived classes can choose to
        override this in order to implement a different fatigue law.
        """
        return self.basquin_load(cycles, failure_probability)

    def basquin_cycles(self, load, failure_probability=0.5):
        """Calculate the cycles numbers from loads according to the Basquin equation.

        Parameters
        ----------
        load : array_like
            The load levels for which the corresponding cycle numbers are to be calculated.
        failure_probability : float, optional
            The failure probability with which the component should fail when
            charged with `load` for the calculated cycle numbers. Default 0.5

        Returns
        -------
        cycles : numpy.ndarray
            The cycle numbers at which the component fails for the given `load` values
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

    def basquin_load(self, cycles, failure_probability=0.5):
        """Calculate the load values from loads according to the Basquin equation.

        Parameters
        ----------
        cycles : array_like
            The cycle numbers for which the corresponding load levels are to be calculated.
        failure_probability : float, optional
            The failure probability with which the component should fail when
            charged with `load` for the calculated cycle numbers. Default 0.5

        Returns
        -------
        cycles : numpy.ndarray
            The cycle numbers at which the component fails for the given `load` values
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

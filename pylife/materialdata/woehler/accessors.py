# Copyright (c) 2019-2021 - for information on the respective copyright owner
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

from pylife.utils.functions import scatteringRange2std


from pylife import signal
from pylife import DataValidator


@pd.api.extensions.register_series_accessor('woehler')
@pd.api.extensions.register_dataframe_accessor('woehler')
class WoehlerCurveAccessor(signal.PylifeSignal):
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
    * ``TN`` : The scatter in cycle direction, (N_10/N_90)
               If the key is missing it is assumed to be 1.0 or calculated from ``TS``
               if given.
    * ``TS`` : The scatter in cycle direction, (S_10/S_90)
               If the key is missing it is assumed to be 1.0 or calculated from ``TN``
               if given.
    """

    def _validate(self, obj, validator):
        validator.fail_if_key_missing(obj, ['k_1', 'ND', 'SD'])
        self._k_2 = obj.get('k_2', np.inf)

        self._TN = obj.get('TN', None)
        self._TS = obj.get('TS', None)

        if self._TN is None and self._TS is None:
            self._TN = 1.0
            self._TS = 1.0
        elif self._TS is None:
            self._TS = np.power(self._TN, 1./obj.k_1)
        elif self._TN is None:
            self._TN = np.power(self._TS, obj.k_1)

        self._failure_probability = obj.get('failure_probability', 0.5)

    def to_pandas(self):
        res = self._obj.copy()
        res['k_2'] = self._k_2
        res['failure_probability'] = self._failure_probability
        if 'TS' not in self._obj:
            res['TS'] = self._TS
        if 'TN' not in self._obj:
            res['TN'] = self._TN
        return res

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
        return self._k_2

    @property
    def TN(self):
        """The load direction scatter value TN."""
        return self._TN

    @property
    def TS(self):
        """The load direction scatter value TS."""
        return self._TS

    @property
    def failure_probability(self):
        return self._failure_probability


    def transform_to_failure_probability(self, failure_probability):
        native_ppf = stats.norm.ppf(self._failure_probability)
        goal_ppf = stats.norm.ppf(failure_probability)

        SD = self._obj.SD / 10**((native_ppf-goal_ppf)*scatteringRange2std(self.TS))
        ND = self._obj.ND / 10**((native_ppf-goal_ppf)*scatteringRange2std(self.TN))
        ND *= np.power(SD/self._obj.SD, -self._obj.k_1)

        transformed = self._obj.copy()
        transformed['SD'] = SD
        transformed['ND'] = ND
        transformed['failure_probability'] = failure_probability

        return transformed

    def miner_elementary(self):
        """Set k_2 to k_1 according Miner Elementary method (k_2 = k_1).

        Returns
        -------
        self
        """
        self._k_2 = self._obj.k_1
        return self

    def miner_haibach(self):
        """Set k_2 to value according Miner Haibach method (k_2 = 2 * k_1 - 1).

        Returns
        -------
        self
        """
        self._k_2 = 2. * self._obj.k_1 - 1.
        return self

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

        transformed = self.transform_to_failure_probability(failure_probability)

        load_index = None if not isinstance(load, pd.Series) else load.index
        load = np.asfarray(load)
        load, wc = signal.Broadcaster(transformed).broadcast(load)
        cycles = np.full_like(load, np.inf)

        k = self._make_k(load, wc.SD)
        in_limit = np.isfinite(k)
        cycles[in_limit] = wc.ND[in_limit] * np.power(load[in_limit]/wc.SD[in_limit], -k[in_limit])

        if load_index is None:
            return cycles
        return pd.Series(cycles, index=load_index)

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

        cycles, wc = signal.Broadcaster(transformed).broadcast(cycles)
        load = np.asarray(wc.SD.copy())
        cycles = np.asarray(cycles)

        k = self._make_k(-cycles, -wc.ND)
        in_limit = np.isfinite(k)
        load[in_limit] = wc.SD[in_limit] * np.power(cycles[in_limit]/wc.ND[in_limit], -1./k[in_limit])
        return load

    def _make_k(self, src, ref):
        k = np.asfarray(self._obj.k_1)
        if k.shape == ():
            k = np.full_like(src, k, dtype=np.double)
        k[src < ref] = self._k_2

        return k


@pd.api.extensions.register_dataframe_accessor('fatigue_data')
class FatigueDataAccessor(signal.PylifeSignal):
    '''Accessor class for fatigue data

    Mandatory keys are
        * ``load`` : float, the load level
        * ``cycles`` : float, the cycles of failure or runout
        * ``fracture``: bool, ``True`` iff the test is a runout
     '''

    def _validate(self, obj, validator):
        validator.fail_if_key_missing(obj, ['load', 'cycles', 'fracture'])
        self._fatigue_limit = None

    @property
    def num_tests(self):
        '''The number of tests'''
        return self._obj.shape[0]

    @property
    def num_fractures(self):
        '''The number of fractures'''
        return self.fractures.shape[0]

    @property
    def num_runouts(self):
        '''The number of runouts'''
        return self.runouts.shape[0]

    @property
    def fractures(self):
        '''Only the fracture tests'''
        return self._obj[self._obj.fracture]

    @property
    def runouts(self):
        '''Only the runout tests'''
        return self._obj[~self._obj.fracture]

    @property
    def load(self):
        '''The load levels'''
        return self._obj.load

    @property
    def cycles(self):
        '''the cycle numbers'''
        return self._obj.cycles

    @property
    def fatigue_limit(self):
        '''The start value of the load endurance limit.

        It is determined by searching for the lowest load level before the
        appearance of a runout data point, and the first load level where a
        runout appears.  Then the median of the two load levels is the start
        value.
        '''
        if self._fatigue_limit is None:
            self._calc_fatigue_limit()
        return self._fatigue_limit

    @property
    def finite_zone(self):
        '''All the tests with load levels above ``fatigue_limit``, i.e. the finite zone'''
        if self._fatigue_limit is None:
            self._calc_fatigue_limit()
        return self._finite_zone

    @property
    def infinite_zone(self):
        '''All the tests with load levels below ``fatigue_limit``, i.e. the infinite zone'''
        if self._fatigue_limit is None:
            self._calc_fatigue_limit()
        return self._infinite_zone

    @property
    def fractured_loads(self):
        return np.unique(self.fractures.load.values)

    @property
    def runout_loads(self):
        return np.unique(self.runouts.load.values)

    @property
    def non_fractured_loads(self):
        return np.setdiff1d(self.runout_loads, self.fractured_loads)

    @property
    def mixed_loads(self):
        return np.intersect1d(self.runout_loads, self.fractured_loads)

    def conservative_fatigue_limit(self):
        """
        Sets a lower fatigue limit that what is expected from the algorithm given by Mustafa Kassem.
        For calculating the fatigue limit, all amplitudes where runouts and fractures are present are collected.
        To this group, the maximum amplitude with only runouts present is added.
        Then, the fatigue limit is the mean of all these amplitudes.

        Returns
        -------
        self

        See also
        --------
        Kassem, Mustafa - "Open Source Software Development for Reliability and Lifetime Calculation" pp. 34
        """
        amps_to_consider = self.mixed_loads

        if len(self.non_fractured_loads ) > 0:
            amps_to_consider = np.concatenate((amps_to_consider, [self.non_fractured_loads.max()]))

        if len(amps_to_consider) > 0:
            self._fatigue_limit = amps_to_consider.mean()
            self._calc_finite_zone()

        return self

    @property
    def max_runout_load(self):
        return self.runouts.load.max()

    def _calc_fatigue_limit(self):
        self._calc_finite_zone()
        self._fatigue_limit = 0.0 if len(self.runouts) == 0 else self._half_level_above_highest_runout()

    def _half_level_above_highest_runout(self):
        if len(self._finite_zone) > 0:
            return (self._finite_zone.load.min() + self.max_runout_load) / 2.

        return self._guess_from_second_highest_runout()

    def _guess_from_second_highest_runout(self):
        max_loads = np.sort(self._obj.load.unique())[-2:]
        return max_loads[1] + (max_loads[1]-max_loads[0]) / 2.

    def _calc_finite_zone(self):
        if len(self.runouts) == 0:
            self._infinite_zone = self._obj[:0]
            self._finite_zone = self._obj
            return

        self._finite_zone = self.fractures[self.fractures.load > self.max_runout_load]
        self._infinite_zone = self._obj[self._obj.load <= self.max_runout_load]


def determine_fractures(df, load_cycle_limit=None):
    '''Adds a fracture column according to defined load cycle limit

    Parameters
    ----------
    df : DataFrame
        A ``DataFrame`` containing ``fatigue_data`` without ``fractures`` column
    load_cycle_limit : float, optional
        If given, all the tests of ``df`` with ``cycles`` equal od above
        ``load_cycle_limit`` are considered as runouts. Others as fractures.
        If not given the maximum cycle number in ``df`` is used as load cycle
        limit.

    Returns
    -------
    df : DataFrame
        A ``DataFrame`` with the column ``fracture`` added
    '''
    DataValidator().fail_if_key_missing(df, ['load', 'cycles'])
    if load_cycle_limit is None:
        load_cycle_limit = df.cycles.max()
    ret = df.copy()
    ret['fracture'] = df.cycles < load_cycle_limit
    return ret

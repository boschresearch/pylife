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
from pylife import signal
from pylife import DataValidator


@pd.api.extensions.register_series_accessor('woehler_elementary')
class WoehlerCurveElementaryAccessor(signal.PylifeSignal):
    def _validate(self, obj, validator):
        validator.fail_if_key_missing(obj, ['k_1', '1/TN', 'ND_50', 'SD_50'])


@pd.api.extensions.register_series_accessor('woehler')
class WoehlerCurveAccessor(WoehlerCurveElementaryAccessor):
    def _validate(self, obj, validator):
        super(WoehlerCurveAccessor, self)._validate(obj, validator)
        validator.fail_if_key_missing(obj, ['1/TS'])


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
            self._calc_finite_zone()
        return self._fatigue_limit

    @property
    def finite_zone(self):
        '''All the tests with load levels above ``fatigue_limit``, i.e. the finite zone'''
        if self._fatigue_limit is None:
            self._calc_finite_zone()
        return self._finite_zone

    @property
    def infinite_zone(self):
        '''All the tests with load levels below ``fatigue_limit``, i.e. the infinite zone'''
        if self._fatigue_limit is None:
            self._calc_finite_zone()
        return self._infinite_zone

    def _calc_finite_zone(self):
        if len(self.runouts) == 0:
            self._fatigue_limit = 0
            self._finite_zone = self._obj[:0]
            self._infinite_zone = self._obj
            return

        max_runout_load = self.runouts.load.max()
        self._finite_zone = self.fractures[self.fractures.load > max_runout_load]
        self._fatigue_limit = (self._finite_zone.load.min() + max_runout_load) / 2
        self._infinite_zone = self._obj[self._obj.load <= self._fatigue_limit]


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

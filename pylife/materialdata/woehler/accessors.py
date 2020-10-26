# Copyright (c) 2019-2020 - for information on the respective copyright owner
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

    def _validate(self, obj, validator):
        validator.fail_if_key_missing(obj, ['load', 'cycles', 'fracture'])
        self._fatigue_limit = None

    @property
    def num_tests(self):
        return self._obj.shape[0]

    @property
    def num_fractures(self):
        return self.fractures.shape[0]

    @property
    def num_runouts(self):
        return self.runouts.shape[0]

    @property
    def fractures(self):
        return self._obj[self._obj.fracture == True]

    @property
    def runouts(self):
        return self._obj[self._obj.fracture == False]

    @property
    def load(self):
        return self._obj.load

    @property
    def cycles(self):
        return self._obj.cycles

    @property
    def fatigue_limit(self):
        if self._fatigue_limit is None:
            self._calc_finite_zone()
        return self._fatigue_limit

    @property
    def finite_zone(self):
        if self._fatigue_limit is None:
            self._calc_finite_zone()
        return self._finite_zone

    @property
    def infinite_zone(self):
        if self._fatigue_limit is None:
            self._calc_finite_zone()
        return self._infinite_zone

    def _calc_finite_zone(self):
        '''
        Computes the start value of the load endurance limit. This is done by searching for the lowest load
        level before the appearance of a runout data point, and the first load level where a runout appears.
        Then the median of the two load levels is the start value.
        '''
        if len(self.runouts) == 0:
            self._fatigue_limit = 0
            self._finite_zone = self._obj[:0]
            self._infinte_zone = self._obj
            return

        max_runout_load = self.runouts.load.max()
        self._finite_zone = self.fractures[self.fractures.load > max_runout_load]
        self._fatigue_limit = (self._finite_zone.load.min() + max_runout_load) / 2
        self._infinite_zone = self._obj[self._obj.load <= self._fatigue_limit]


def determine_fractures(df, load_cycle_limit=None):
    DataValidator().fail_if_key_missing(df, ['load', 'cycles'])
    if load_cycle_limit is None:
        load_cycle_limit = df.cycles.max()
    ret = df.copy()
    ret['fracture'] = pd.Series([True] * len(df)).where(df.cycles < load_cycle_limit, False)
    return ret

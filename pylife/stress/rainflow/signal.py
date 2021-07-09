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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import pandas as pd
import numpy as np

from pylife import signal


@pd.api.extensions.register_series_accessor('rainflow')
class RainflowAccessor(signal.PylifeSignal):

    def _validate(self, obj, validator):
        self._class_location = 'mid'
        if 'range' in obj.index.names:
            return
        if 'from' in obj.index.names and 'to' in obj.index.names:
            return
        raise AttributeError("RainflowAccessor needs either 'range'/('mean') or 'from'/'to' in index levels.")

    @property
    def amplitude(self):
        if 'range' in self._obj.index.names:
            rng = getattr(self._obj.index.get_level_values('range'), self._class_location)
        else:
            fr = getattr(self._obj.index.get_level_values('from'), self._class_location).values
            to = getattr(self._obj.index.get_level_values('to'), self._class_location).values
            rng = np.abs(fr-to)

        return pd.Series(rng/2., name='amplitude', index=self._obj.index)

    @property
    def frequency(self):
        freq = self._obj.copy()
        freq.name = 'frequency'
        return freq

    def use_class_right(self):
        self._class_location = 'right'
        return self

    def use_class_left(self):
        self._class_location = 'left'
        return self

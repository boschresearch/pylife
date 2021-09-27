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

from pylife import PylifeSignal


@pd.api.extensions.register_series_accessor('rainflow')
class RainflowMatrix(PylifeSignal):

    def _validate(self):
        self._class_location = 'mid'
        if 'range' in self._obj.index.names:
            self._fail_if_not_multiindex(['range', 'mean'])
            self._impl = _RangeMeanMatrix(self._obj)
            return
        if 'from' in self._obj.index.names and 'to' in self._obj.index.names:
            self._fail_if_not_multiindex(['from', 'to'])
            self._impl = _FromToMatrix(self._obj)
            return

        raise AttributeError("Rainflow needs either 'range'/('mean') or 'from'/'to' in index levels.")

    def _fail_if_not_multiindex(self, index_names):
        for name in index_names:
            if name not in self._obj.index.names:
                continue
            if not isinstance(self._obj.index.get_level_values(name), pd.IntervalIndex):
                raise AttributeError("Index of a rainflow matrix must be pandas.IntervalIndex.")

    @property
    def amplitude(self):
        rng = self._impl.amplitude()
        return pd.Series(rng/2., name='amplitude', index=self._obj.index)

    @property
    def meanstress(self):
        mean = self._impl.meanstress()
        return pd.Series(mean, name='meanstress', index=self._obj.index)

    @property
    def upper(self):
        res = self.meanstress + self.amplitude
        res.name = 'upper'
        return res

    @property
    def lower(self):
        res = self.meanstress - self.amplitude
        res.name = 'lower'
        return res

    @property
    def cycles(self):
        freq = self._obj.copy()
        freq.name = 'cycles'
        return freq

    def use_class_right(self):
        self._impl._class_location = 'right'
        return self

    def use_class_left(self):
        self._impl._class_location = 'left'
        return self

    def scale(self, factors):
        return self._shift_or_scale(lambda x, y: x * y, factors).rainflow

    def shift(self, diffs):
        return self._shift_or_scale(lambda x, y: x + y, diffs, skip=['range']).rainflow

    def _shift_or_scale(self, func, operand, skip=[]):
        def do_transform_interval_index(level_name):
            level = obj.index.get_level_values(level_name)
            if level.name not in self._obj.index.names or level_name in skip:
                return level
            values = level.values
            left = func(values.left, operand_broadcast)
            right = func(values.right, operand_broadcast)

            index = pd.IntervalIndex.from_arrays(left, right)
            return index

        operand_broadcast, obj = self.broadcast(operand)

        levels = [do_transform_interval_index(lv) for lv in obj.index.names]

        new_index = pd.MultiIndex.from_arrays(levels, names=obj.index.names)
        return pd.Series(obj.values, index=new_index, name='cycles')


class _RainflowMatrixImpl:
    def __init__(self, obj):
        self._obj = obj
        self._class_location = 'mid'


class _FromToMatrix(_RainflowMatrixImpl):

    def _from_tos(self):
        fr = getattr(self._obj.index.get_level_values('from'), self._class_location).values
        to = getattr(self._obj.index.get_level_values('to'), self._class_location).values
        return fr, to

    def amplitude(self):
        fr, to = self._from_tos()
        return np.abs(fr-to)

    def meanstress(self):
        fr, to = self._from_tos()
        return (fr+to) / 2.


class _RangeMeanMatrix(_RainflowMatrixImpl):

    def amplitude(self):
        return getattr(self._obj.index.get_level_values('range'), self._class_location)

    def meanstress(self):
        if 'mean' not in self._obj.index.names:
            return np.zeros_like(self._obj)

        return getattr(self._obj.index.get_level_values('mean'), self._class_location)

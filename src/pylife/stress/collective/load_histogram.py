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

__author__ = "Johannes Mueller"
__maintainer__ = __author__


from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from pylife import PylifeSignal

from .abstract_load_collective import AbstractLoadCollective


@pd.api.extensions.register_series_accessor('load_collective')
class LoadHistogram(PylifeSignal, AbstractLoadCollective):

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

        raise AttributeError("Load collective matrix needs either 'range'/('mean') or 'from'/'to' in index levels.")

    def _fail_if_not_multiindex(self, index_names):
        for name in index_names:
            if name not in self._obj.index.names:
                continue
            if not isinstance(self._obj.index.get_level_values(name), pd.IntervalIndex):
                raise AttributeError("Index of a load collective matrix must be pandas.IntervalIndex.")

    @property
    def amplitude(self):
        """Calculate the amplitudes of the load collective.

        Returns
        -------
        amplitude : pd.Series
            The amplitudes of the load collective
        """
        rng = self._impl.amplitude()
        return pd.Series(rng/2., name='amplitude', index=self._obj.index)

    @property
    def amplitude_histogram(self):
        index = self._impl.amplitude_histogram_index()
        index.name = 'amplitude'
        return pd.Series(self._obj.values, index=index, name='cycles')

    @property
    def meanstress(self):
        """Calculate the mean load values of the load collective.

        Returns
        -------
        mean : pd.Series
            The mean load values of the load collective
        """
        mean = self._impl.meanstress()
        return pd.Series(mean, name='meanstress', index=self._obj.index)

    @property
    def R(self):
        """Calculate the R values of the load collective.

        Returns
        -------
        R : pd.Series
            The R values of the load collective
        """
        res = (self.lower / self.upper).fillna(0.0)
        res.name = 'R'
        return res

    @property
    def upper(self):
        """Calculate the upper load values of the load collective.

        Returns
        -------
        upper : pd.Series
            The upper load values of the load collective
        """
        res = self.meanstress + self.amplitude
        res.name = 'upper'
        return res

    @property
    def lower(self):
        """Calculate the lower load values of the load collective.

        Returns
        -------
        lower : pd.Series
            The lower load values of the load collective
        """
        res = self.meanstress - self.amplitude
        res.name = 'lower'
        return res

    @property
    def cycles(self):
        """The cycles of each class of the collective.

        Returns
        -------
        cycles : pd.Series
            The cycles of each class of the collective
        """
        cycles = self._obj.copy()
        cycles.name = 'cycles'
        return cycles

    def use_class_right(self):
        """Use the upper limit of the class bins.


        Returns
        -------
        self
        """
        self._impl._class_location = 'right'
        return self

    def use_class_left(self):
        """Use the lower limit of the class bins.

        Returns
        -------
        self
        """
        self._impl._class_location = 'left'
        return self

    def scale(self, factors):
        """Scale the collective.

        Parameters
        ----------
        factors : scalar or :class:`pandas.Series`
            The factor(s) to scale the collective with.

        Returns
        -------
        scaled : ``LoadCollective``
            The scaled collective.
        """
        return self._shift_or_scale(lambda x, y: x * y, factors).load_collective

    def shift(self, diffs):
        """Shift the collective.

        Parameters
        ----------
        diffs : scalar or :class:`pandas.Series`
            The diff(s) to shift the collective by.

        Returns
        -------
        shifted : ``LoadCollective``
            The shifted collective.
        """
        return self._shift_or_scale(lambda x, y: x + y, diffs, skip=['range']).load_collective

    def _shift_or_scale(self, func, operand, skip=None):
        def do_transform_interval_index(level_name):
            level = obj.index.get_level_values(level_name)
            if level.name not in self._impl.index_names or level_name in skip:
                return level
            values = level.values
            left = func(values.left, operand_broadcast)
            right = func(values.right, operand_broadcast)

            index = pd.IntervalIndex.from_arrays(left, right)
            return index

        skip = skip or []
        operand_broadcast, obj = self.broadcast(operand)

        levels = [do_transform_interval_index(lv) for lv in obj.index.names]

        new_index = pd.MultiIndex.from_arrays(levels, names=obj.index.names)
        return pd.Series(obj.values, index=new_index, name='cycles')

    def cumulated_range(self):
        return pd.Series(self._obj.groupby('range').transform(lambda g: np.cumsum(g)),
                         name='cumulated_cycles')

class _LoadHistogramImpl(ABC):

    @property
    @abstractmethod
    def index_names(self):
        return set([])

    def __init__(self, obj):
        self._obj = obj
        self._class_location = 'mid'


class _FromToMatrix(_LoadHistogramImpl):

    @property
    def index_names(self):
        return set(['from', 'to'])

    def _from_tos(self):
        fr = getattr(self._obj.index.get_level_values('from'), self._class_location).values
        to = getattr(self._obj.index.get_level_values('to'), self._class_location).values
        return fr, to

    def amplitude(self):
        fr, to = self._from_tos()
        return np.abs(fr-to)

    def amplitude_histogram_index(self):
        left = np.zeros(len(self._obj))
        right = np.zeros(len(self._obj))

        fr = self._obj.index.get_level_values('from')
        to = self._obj.index.get_level_values('to')

        hanging = fr.mid > to.mid
        standing = fr.mid <= to.mid

        left[hanging] = fr[hanging].left - to[hanging].right
        right[hanging] = fr[hanging].right - to[hanging].left

        left[standing] = to[standing].left - fr[standing].right
        right[standing] = to[standing].right - fr[standing].left

        left[left < 0.0] = 0.0

        return pd.IntervalIndex.from_arrays(left, right)

    def meanstress(self):
        fr, to = self._from_tos()
        return (fr+to) / 2.


class _RangeMeanMatrix(_LoadHistogramImpl):

    @property
    def index_names(self):
        return set(['range', 'mean'])

    def amplitude(self, location=None):
        return getattr(self._obj.index.get_level_values('range'), location or self._class_location)

    def amplitude_histogram_index(self):
        left = self.amplitude(location='left') / 2.
        right = self.amplitude(location='right') / 2.
        return pd.IntervalIndex.from_arrays(left, right)

    def meanstress(self):
        if 'mean' not in self._obj.index.names:
            return np.zeros_like(self._obj)

        return getattr(self._obj.index.get_level_values('mean'), self._class_location)

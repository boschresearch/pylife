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

"""Provide the binned ``.load_collective`` accessor implementation."""

__author__ = "Johannes Mueller"
__maintainer__ = __author__


from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from pylife import PylifeSignal

from .abstract_load_collective import AbstractLoadCollective


@pd.api.extensions.register_series_accessor('load_collective')
class LoadHistogram(PylifeSignal, AbstractLoadCollective):
    """Represent binned rainflow cycles as a load histogram.

    The accessor is registered as ``Series.load_collective`` for compatibility
    with explicit collectives.  The series values are the number of cycles in
    each bin.  The series index must be a :class:`pandas.MultiIndex` that
    contains one of the following interval-index structures:

    * ``range`` and optionally ``mean``: Load range (peak-to-peak) classes and
      mean-load classes, usually in MPa.
    * ``from`` and ``to``: Classes of the two loop turning loads, usually in
      MPa.

    Every load axis listed above must be a :class:`pandas.IntervalIndex`.
    Derived properties use the interval midpoint by default.  They expose
    ``amplitude = range / 2`` for ``range``/``mean`` histograms or
    ``amplitude = abs(from - to) / 2`` for ``from``/``to`` histograms,
    ``meanstress``, ``upper``, ``lower``, ``R = lower / upper`` with missing
    values filled by ``0.0``, and ``cycles``.

    Parameters
    ----------
    pandas_obj : pandas.Series
        Series containing cycle counts indexed by load intervals.
    """

    def _validate(self):
        self._class_location = 'mid'
        if 'range' in self._obj.index.names:
            self._fail_if_not_multiindex(['range', 'mean'])
            self._impl = _RangeMeanMatrix(self._obj)
            self._axes = ["range", "mean"]
            return
        if 'from' in self._obj.index.names and 'to' in self._obj.index.names:
            self._fail_if_not_multiindex(['from', 'to'])
            self._impl = _FromToMatrix(self._obj)
            self._axes = ["from", "to"]
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
        """Calculate the load amplitude for each histogram bin.

        Returns
        -------
        pandas.Series
            Load amplitude in the same unit as the histogram load axes,
            typically MPa.
        """
        rng = self._impl.amplitude()
        return pd.Series(rng/2., name='amplitude', index=self._obj.index)

    @property
    def amplitude_histogram(self):
        """Return the cycle histogram indexed by load-amplitude intervals.

        Returns
        -------
        pandas.Series
            Number of cycles indexed by load-amplitude intervals in the same
            unit as the histogram load axes, typically MPa.
        """
        index = self._impl.amplitude_histogram_index()
        index.name = 'amplitude'
        return pd.Series(self._obj.values, index=index, name='cycles')

    @property
    def meanstress(self):
        """Calculate the mean load for each histogram bin.

        Returns
        -------
        pandas.Series
            Mean load in the same unit as the histogram load axes, typically
            MPa.
        """
        mean = self._impl.meanstress()
        return pd.Series(mean, name='meanstress', index=self._obj.index)

    @property
    def R(self):
        """Calculate the stress ratio ``R`` for each histogram bin.

        Returns
        -------
        pandas.Series
            Stress ratio ``R = lower / upper``, dimensionless.  Undefined
            ratios are returned as ``0.0``.
        """
        res = (self.lower / self.upper).fillna(0.0)
        res.name = 'R'
        return res

    @property
    def upper(self):
        """Calculate the upper turning load for each histogram bin.

        Returns
        -------
        pandas.Series
            Upper load in the same unit as the histogram load axes, typically
            MPa.
        """
        res = self.meanstress + self.amplitude
        res.name = 'upper'
        return res

    @property
    def lower(self):
        """Calculate the lower turning load for each histogram bin.

        Returns
        -------
        pandas.Series
            Lower load in the same unit as the histogram load axes, typically
            MPa.
        """
        res = self.meanstress - self.amplitude
        res.name = 'lower'
        return res

    @property
    def cycles(self):
        """Return the number of cycles in each histogram bin.

        Returns
        -------
        pandas.Series
            Number of cycles represented by each histogram bin.
        """
        cycles = self._obj.copy()
        cycles.name = 'cycles'
        return cycles

    def use_class_right(self):
        """Use the right interval boundary for derived load values.

        Returns
        -------
        LoadHistogram
            The same histogram accessor, configured to use right interval
            boundaries.
        """
        self._impl._class_location = 'right'
        return self

    def use_class_left(self):
        """Use the left interval boundary for derived load values.

        Returns
        -------
        LoadHistogram
            The same histogram accessor, configured to use left interval
            boundaries.
        """
        self._impl._class_location = 'left'
        return self

    def scale(self, factors):
        """Scale all load-axis intervals of the histogram.

        Parameters
        ----------
        factors : float or pandas.Series
            Factor or row-wise factors used to multiply load interval
            boundaries.  The ``range`` axis is not scaled when shifting but is
            scaled here.

        Returns
        -------
        LoadHistogram
            Scaled histogram accessor.
        """
        return self._shift_or_scale(lambda x, y: x * y, factors).load_collective

    def shift(self, diffs):
        """Shift all load-axis intervals of the histogram.

        Parameters
        ----------
        diffs : float or pandas.Series
            Difference or row-wise differences added to load interval
            boundaries.  The ``range`` axis is not shifted because a constant
            offset changes mean load but not load range.

        Returns
        -------
        LoadHistogram
            Shifted histogram accessor.
        """
        return self._shift_or_scale(lambda x, y: x + y, diffs, skip=['range']).load_collective

    @property
    def index_levels(self) -> list[str]:
        """Return the index levels that define the load axes.

        Returns
        -------
        list of str
            Either ``["range", "mean"]`` or ``["from", "to"]``.
        """
        return self._axes

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
        """Cumulate cycle counts along the load-range classes.

        Returns
        -------
        pandas.Series
            Cumulative number of cycles within each ``range`` interval.
        """
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

        return pd.IntervalIndex.from_arrays(left/2.0, right/2.0)

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

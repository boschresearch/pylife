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

"""Provide the ``.load_collective`` accessor for explicit rainflow loops."""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import warnings

import pandas as pd
import numpy as np

from pylife import PylifeSignal

from .abstract_load_collective import AbstractLoadCollective
from .load_histogram import LoadHistogram


@pd.api.extensions.register_dataframe_accessor('load_collective')
class LoadCollective(PylifeSignal, AbstractLoadCollective):
    r"""Represent explicit rainflow loops as a load collective.

    The accessor is registered as ``DataFrame.load_collective`` and represents
    one hysteresis loop per row.  The input frame must contain either:

    * ``from`` and ``to``: Load values at the two turning points of the loop,
      usually in MPa.
    * ``range`` and ``mean``: Load range (peak-to-peak) and mean load,
      usually in MPa.  The accessor converts these columns internally to
      lower ``from`` and upper ``to`` values.

    An optional ``cycles`` column gives the number of cycles represented by
    each row.  If it is absent, every row represents one cycle.  Derived
    properties expose ``amplitude = abs(from - to) / 2``, ``meanstress =
    (from + to) / 2``, ``upper = max(from, to)``, ``lower = min(from, to)``,
    ``R = lower / upper`` with missing values filled by ``0.0``, and
    ``cycles``.

    Parameters
    ----------
    pandas_obj : pandas.DataFrame
        DataFrame containing explicit load loops as described above.

    Notes
    -----
    The amplitude, range, and mean load are related by

    .. math::

        S_\mathrm{a} = \frac{|S_\mathrm{from} - S_\mathrm{to}|}{2}

        S_\mathrm{range} = 2 S_\mathrm{a}

        S_\mathrm{mean} = \frac{S_\mathrm{from} + S_\mathrm{to}}{2}
    """

    def _validate(self):
        if 'from' in self.keys() and 'to' in self.keys():
            self._axes = ["from", "to"]
            return
        if 'range' in self.keys() and 'mean' in self.keys():
            self._axes = ["range", "mean"]
            fr = self._obj['mean'] - self._obj['range'] / 2.
            to = self._obj['mean'] + self._obj['range'] / 2.

            cycles = self._obj.get('cycles')

            self._obj = pd.DataFrame({
                'from': fr,
                'to': to
            }, index=self._obj.index)

            if cycles is not None:
                self._obj['cycles'] = cycles

            return
        raise AttributeError("Load collective needs either 'range'/'mean' or 'from'/'to' in column names.")

    @property
    def columns(self) -> list[str]:
        """Return the column names that define the load axes.

        Returns
        -------
        list of str
            Either ``["from", "to"]`` or ``["range", "mean"]`` as supplied
            by the input signal.
        """
        return self._axes

    @property
    def amplitude(self):
        """Calculate the load amplitude for each loop.

        Returns
        -------
        pandas.Series
            Load amplitude in the same unit as ``from`` and ``to``, typically
            MPa.
        """
        fr = self._obj['from']
        to = self._obj['to']
        rng = np.abs(fr-to)

        return pd.Series(rng/2., name='amplitude', index=self._obj.index)

    @property
    def meanstress(self):
        """Calculate the mean load for each loop.

        Returns
        -------
        pandas.Series
            Mean load in the same unit as ``from`` and ``to``, typically MPa.
        """
        fr = self._obj['from']
        to = self._obj['to']
        return pd.Series((fr+to)/2., name='meanstress')

    @property
    def R(self):
        """Calculate the stress ratio ``R`` for each loop.

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
        """Calculate the upper turning load for each loop.

        Returns
        -------
        pandas.Series
            Upper load in the same unit as ``from`` and ``to``, typically MPa.
        """
        res = self._obj.loc[:, ['from', 'to']].max(axis=1)
        res.name = 'upper'
        return res

    @property
    def lower(self):
        """Calculate the lower turning load for each loop.

        Returns
        -------
        pandas.Series
            Lower load in the same unit as ``from`` and ``to``, typically MPa.
        """
        res = self._obj.loc[:, ['from', 'to']].min(axis=1)
        res.name = 'lower'
        return res

    @property
    def cycles(self):
        """Return the number of cycles represented by each loop.

        Returns
        -------
        pandas.Series
            Number of cycles.  If the source data has no ``cycles`` column,
            every loop is returned as one cycle.
        """
        if 'cycles' in self._obj.keys():
            return self._obj.cycles

        return pd.Series(1.0, name='cycles', index=self._obj.index)

    def scale(self, factors):
        """Scale all load values of the collective.

        Parameters
        ----------
        factors : float or pandas.Series
            Factor or row-wise factors used to multiply the ``from`` and
            ``to`` load values.

        Returns
        -------
        LoadCollective
            Scaled collective accessor.
        """
        factors, obj = self.broadcast(factors)
        obj[['from', 'to']] = obj[['from', 'to']].multiply(factors, axis=0)
        return obj.load_collective

    def shift(self, diffs):
        """Shift all load values of the collective.

        Parameters
        ----------
        diffs : float or pandas.Series
            Difference or row-wise differences added to the ``from`` and
            ``to`` load values.

        Returns
        -------
        LoadCollective
            Shifted collective accessor.
        """
        diffs, obj = self.broadcast(diffs)
        obj[['from', 'to']] = obj[['from', 'to']].add(diffs, axis=0)
        return obj.load_collective

    def range_histogram(self, bins, axis=None):
        """Calculate a load-range histogram of the collective.

        Parameters
        ----------
        bins : int, array_like or pandas.IntervalIndex
            Bin specification for the load range (peak-to-peak), in the same
            unit as the source load values.
        axis : str, optional
            Index level that identifies individual loops within each group.
            If omitted, calculate one histogram over the whole collective.
            Default is ``None``.

        Returns
        -------
        LoadHistogram
            Histogram accessor with a ``range`` interval index and cycle
            counts as values.

        See Also
        --------
        histogram : Calculate a two-dimensional range-mean histogram.

        Notes
        -----
        The resulting histogram does not contain mean-load information and
        does not apply a mean-stress transformation.  The ``range`` axis uses
        load range (peak-to-peak), not load amplitude.

        Examples
        --------
        Calculate a range histogram of a simple load collective

        >>> df = pd.DataFrame(
        ...     {'range': [1.0, 2.0, 1.0, 2.0, 1.0], 'mean': [0, 0, 0, 0, 0]},
        ...     columns=['range', 'mean'],
        ... )
        >>> df.load_collective.range_histogram([0, 1, 2, 3]).to_pandas()
        range
        (0, 1]    0
        (1, 2]    3
        (2, 3]    2
        Name: cycles, dtype: int64

        Calculate a range histogram of a load collective collection for
        multiple nodes.  The axis along which to aggregate the histogram is
        given as ``cycle_number``.

        >>> element_idx = pd.Index([10, 20, 30], name='element_id')
        >>> cycle_idx = pd.Index([0, 1, 2], name='cycle_number')
        >>> index = pd.MultiIndex.from_product((element_idx, cycle_idx))

        >>> df = pd.DataFrame({
        ...     'range': [1., 2., 2., 0., 1., 2., 1., 1., 2.],
        ...     'mean': [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ... }, columns=['range', 'mean'], index=index)

        >>> h = df.load_collective.range_histogram([0, 1, 2, 3], 'cycle_number')
        >>> h.to_pandas()
        element_id  range
        10          (0, 1]    0
                    (1, 2]    1
                    (2, 3]    2
        20          (0, 1]    1
                    (1, 2]    1
                    (2, 3]    1
        30          (0, 1]    0
                    (1, 2]    2
                    (2, 3]    1
        Name: cycles, dtype: int64
        """
        def make_histogram(group):
            weights = self.cycles.loc[group.index].to_numpy().astype(np.int64)
            cycles, intervals = np.histogram(group * 2., bins, weights=weights)
            idx = pd.IntervalIndex.from_breaks(intervals, name='range')
            return pd.Series(cycles, index=idx, name='cycles')

        if isinstance(bins, pd.IntervalIndex) or isinstance(bins, pd.arrays.IntervalArray):
            bins = np.append(bins.left[0], bins.right)

        if axis is None:
            return LoadHistogram(make_histogram(self.amplitude))

        result = pd.Series(
            self.amplitude.groupby(self._levels_from_axis(axis)).apply(
                make_histogram
            ),
            name='cycles',
        )

        return LoadHistogram(result)

    def histogram(self, bins, axis=None):
        """Calculate a range-mean histogram of the collective.

        Parameters
        ----------
        bins : int, array_like or pandas.IntervalIndex
            Bin specification for both load range (peak-to-peak) and mean
            load, in the same unit as the source load values.
        axis : str, optional
            Index level that identifies individual loops within each group.
            If omitted, calculate one histogram over the whole collective.
            Default is ``None``.

        Returns
        -------
        LoadHistogram
            Histogram accessor with ``range`` and ``mean`` interval index
            levels and cycle counts as values.

        See Also
        --------
        range_histogram : Calculate a one-dimensional load-range histogram.

        Notes
        -----
        The ``range`` axis stores load range (peak-to-peak), not load
        amplitude.  The ``mean`` axis stores mean load.

        Examples
        --------
        Calculate a range histogram of a simple load collective

        >>> df = pd.DataFrame(
        ...     {'range': [1.0, 2.0, 1.0, 2.0, 1.0], 'mean': [0.5, 1.5, 1.0, 1.5, 0.5]},
        ...     columns=['range', 'mean'],
        ... )
        >>> df.load_collective.histogram([0, 1, 2, 3]).to_pandas()
        range   mean
        (0, 1]  (0, 1]    0.0
                (1, 2]    0.0
                (2, 3]    0.0
        (1, 2]  (0, 1]    2.0
                (1, 2]    1.0
                (2, 3]    0.0
        (2, 3]  (0, 1]    0.0
                (1, 2]    2.0
                (2, 3]    0.0
        Name: cycles, dtype: float64

        Calculate a range histogram of a load collective collection for
        multiple nodes.  The axis along which to aggregate the histogram is
        given as ``cycle_number``.

        >>> element_idx = pd.Index([10, 20], name='element_id')
        >>> cycle_idx = pd.Index([0, 1, 2], name='cycle_number')
        >>> index = pd.MultiIndex.from_product((element_idx, cycle_idx))

        >>> df = pd.DataFrame({
        ...     'range': [1., 2., 2., 0., 1., 2.],
        ...     'mean': [0.5, 1.0, 1.0, 0.0, 1.0, 1.5]
        ... }, columns=['range', 'mean'], index=index)

        >>> h = df.load_collective.histogram([0, 1, 2, 3], 'cycle_number')
        >>> h.to_pandas()
        element_id  range   mean
        10          (0, 1]  (0, 1]    0.0
                            (1, 2]    0.0
                            (2, 3]    0.0
                    (1, 2]  (0, 1]    1.0
                            (1, 2]    0.0
                            (2, 3]    0.0
                    (2, 3]  (0, 1]    0.0
                            (1, 2]    2.0
                            (2, 3]    0.0
        20          (0, 1]  (0, 1]    1.0
                            (1, 2]    0.0
                            (2, 3]    0.0
                    (1, 2]  (0, 1]    0.0
                            (1, 2]    1.0
                            (2, 3]    0.0
                    (2, 3]  (0, 1]    0.0
                            (1, 2]    1.0
                            (2, 3]    0.0
        Name: cycles, dtype: float64
        """
        def make_histogram(group):
            weights = self.cycles.loc[group.index].to_numpy()
            cycles, range_bins, mean_bins = np.histogram2d(group["range"], group["meanstress"], bins, weights=weights)

            return pd.Series(
                cycles.ravel(),
                name="cycles",
                index=pd.MultiIndex.from_product(
                    [
                        pd.IntervalIndex.from_breaks(range_bins),
                        pd.IntervalIndex.from_breaks(mean_bins),
                    ],
                    names=["range", "mean"],
                ),
            )

        range_mean = pd.DataFrame(
            {'range': self.amplitude * 2, 'meanstress': self.meanstress},
            index=self._obj.index,
        )

        if isinstance(bins, pd.IntervalIndex) or isinstance(bins, pd.arrays.IntervalArray):
            bins = np.append(bins.left[0], bins.right)

        if axis is None:
            return LoadHistogram(make_histogram(range_mean))

        result = pd.Series(
            range_mean.groupby(self._levels_from_axis(axis))
            .apply(make_histogram)
            .stack(['range', 'mean'], future_stack=True),
            name="cycles",
        )

        return LoadHistogram(result)

    def _levels_from_axis(self, axis):
        return [lv for lv in self._obj.index.names if lv not in [axis] and lv is not None]

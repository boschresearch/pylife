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

import warnings

import pandas as pd
import numpy as np

from pylife import PylifeSignal

from .abstract_load_collective import AbstractLoadCollective
from .load_histogram import LoadHistogram


@pd.api.extensions.register_dataframe_accessor('load_collective')
class LoadCollective(PylifeSignal, AbstractLoadCollective):
    """A Load collective.

    The usual use of this signal is to process hysteresis loop data from a
    rainflow recording.  Usually the keys ``from`` and ``to`` are used to
    describe the hysteresis loops.  Alternatively also the keys ``range`` and
    ``mean`` can be given.  In that case the frame is internally converted to
    ``from`` and ``to`` where the ``from`` values are the lower ones.
    """

    def _validate(self):
        if 'from' in self.keys() and 'to' in self.keys():
            return
        if 'range' in self.keys() and 'mean' in self.keys():
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
    def amplitude(self):
        """Calculate the amplitudes of the load collective.

        Returns
        -------
        amplitude : pd.Series
            The amplitudes of the load collective
        """
        fr = self._obj['from']
        to = self._obj['to']
        rng = np.abs(fr-to)

        return pd.Series(rng/2., name='amplitude', index=self._obj.index)

    @property
    def meanstress(self):
        """Calculate the mean load values of the load collective.

        Returns
        -------
        mean : pd.Series
            The mean load values of the load collective
        """
        fr = self._obj['from']
        to = self._obj['to']
        return pd.Series((fr+to)/2., name='meanstress')

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
        res = self._obj.loc[:, ['from', 'to']].max(axis=1)
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
        res = self._obj.loc[:, ['from', 'to']].min(axis=1)
        res.name = 'lower'
        return res

    @property
    def cycles(self):
        """The cycles of each member of the collective is 1.0. when no cycles are given

        This is for compatibility with :class:`~pylife.stress.pylife.stress.LoadHistogram`
        """
        if 'cycles' in self._obj.keys():
            return self._obj.cycles

        return pd.Series(1.0, name='cycles', index=self._obj.index)

    def scale(self, factors):
        """Scale the collective.

        Parameters
        ----------
        factors : scalar or :class:`pandas.Series`
            The factor(s) to scale the collective 'from' and 'to' with.

        Returns
        -------
        scaled : ``LoadHistogram``
            The scaled histogram.
        """
        factors, obj = self.broadcast(factors)
        obj[['from', 'to']] = obj[['from', 'to']].multiply(factors, axis=0)
        return obj.load_collective

    def shift(self, diffs):
        """Shift the collective.

        Parameters
        ----------
        diffs : scalar or :class:`pandas.Series`
            The diff(s) to shift the collective by.

        Returns
        -------
        shifted : ``LoadHistogram``
            The shifted histogram.
        """
        diffs, obj = self.broadcast(diffs)
        obj[['from', 'to']] = obj[['from', 'to']].add(diffs, axis=0)
        return obj.load_collective

    def range_histogram(self, bins, axis=None):
        """Calculate the histogram of cycles for range intervals along a given axis.

        Parameters
        ----------
        bins : int, sequence of scalars or pd.IntervalIndex
            The bins of the histogram to be calculated

        axis : str, optional
            The index axis along which the histogram is calculated. If missing
            the histogram is calculated over the whole collective.


        Returns
        -------
        range histogram : :class:`~pylife.pylife.stress.LoadHistogram`


        Note
        ----
        This resulting histogram does not contain any information on the mean
        stress. Neither does it perform any kind of mean stress transformation

        See also
        --------
        histogram

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
            cycles, intervals = np.histogram(group * 2., bins)
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
        """Calculate the histogram of cycles along a given axis.

        Parameters
        ----------
        bins : int, sequence of scalars or pd.IntervalIndex
            The bins of the histogram to be calculated

        axis : str, optional
            The index axis along which the histogram is calculated. If missing
            the histogram is calculated over the whole collective.

        Returns
        -------
        range histogram : :class:`~pylife.pylife.stress.LoadHistogram`

        See also
        --------
        range_histogram

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
            cycles, range_bins, mean_bins = np.histogram2d(
                group["range"], group["meanstress"], bins
            )

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

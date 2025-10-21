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


import numpy as np
import pandas as pd

from .general import AbstractRecorder


class LoopValueRecorder(AbstractRecorder):
    """Rainflow recorder that collects the loop values."""

    def __init__(self):
        """Instantiate a LoopRecorder."""
        super().__init__()
        self._values_from = np.zeros((0,))
        self._values_to = np.zeros((0,))

    @property
    def values_from(self):
        """1-D float array containing the values from which the loops start."""
        return self._values_from

    @property
    def values_to(self):
        """1-D float array containing the values the loops go to before turning back."""
        return self._values_to

    @property
    def collective(self):
        """The overall collective recorded as :class:`pandas.DataFrame`.

        The columns are named ``from``, ``to``.
        """
        return pd.DataFrame({'from': self._values_from, 'to': self._values_to})

    def record_values(self, values_from, values_to):
        """Record the loop values."""
        self._values_from = np.append(self._values_from, values_from)
        self._values_to = np.append(self._values_to, values_to)

    def histogram_numpy(self, bins=10):
        """Calculate a histogram of the recorded values into a plain numpy.histogram2d.

        Parameters
        ----------
        bins : int or array_like or [int, int] or [array, array], optional
            The bin specification (see numpy.histogram2d)

        Returns
        -------
        H : ndarray, shape(nx, ny)
            The bi-dimensional histogram of samples (see numpy.histogram2d)
        xedges : ndarray, shape(nx+1,)
            The bin edges along the first dimension.
        yedges : ndarray, shape(ny+1,)
            The bin edges along the second dimension.
        """
        def is_non_continous(intervals):
            lefts = intervals.left
            rights = intervals.right
            return np.any(lefts[1:] != rights[:-1])

        if isinstance(bins, pd.IntervalIndex) or isinstance(bins, pd.arrays.IntervalArray):
            if not bins.is_non_overlapping_monotonic or is_non_continous(bins):
                raise ValueError("Intervals must not overlap and must be continuous and monotonic.")
            new_bins = np.empty(len(bins) + 1)
            new_bins[:-1] = bins.left
            new_bins[-1] = bins.right[-1]
            bins = new_bins

        return np.histogram2d(self._values_from, self._values_to, bins)

    def histogram(self, bins=10):
        """Calculate a histogram of the recorded values into a :class:`pandas.Series`.

        An interval index is used to index the bins.

        Parameters
        ----------
        bins : int or array_like or [int, int] or [array, array], optional
            The bin specification (see numpy.histogram2d)

        Returns
        -------
        pandas.Series
            A pandas.Series using a multi interval index in order to
            index data point for a given from/to value pair.
        """
        hist, fr, to = self.histogram_numpy(bins)
        index_fr = pd.IntervalIndex.from_breaks(fr)
        index_to = pd.IntervalIndex.from_breaks(to)

        mult_idx = pd.MultiIndex.from_product([index_fr, index_to], names=['from', 'to'])
        return pd.Series(data=hist.flatten(), index=mult_idx)


class FullRecorder(LoopValueRecorder):
    """Rainflow recorder that collects the loop values and the loop index.

    Same functionality like :class:`.LoopValueRecorder` but additionally
    collects the loop index.
    """

    def __init__(self):
        """Instantiate a FullRecorder."""
        super().__init__()
        self._index_from = np.array([], dtype=np.uintp)
        self._index_to = np.array([], dtype=np.uintp)

    @property
    def index_from(self):
        """1-D int array containing the index to the samples from which the loops start."""
        return self._index_from

    @property
    def index_to(self):
        """1-D int array containing the index to the samples the loops go to before turning back."""
        return self._index_to

    @property
    def collective(self):
        """The overall collective recorded as :class:`pandas.DataFrame`.

        The columns are named ``from``, ``to``, ``index_from``, ``index_to``.
        """
        return pd.DataFrame({
            'from': self._values_from,
            'to': self._values_to,
            'index_from': self._index_from,
            'index_to': self._index_to
        })

    def record_index(self, index_from, index_to):
        """Record the index."""
        self._index_from = np.concatenate(
            (self._index_from, np.asarray(index_from, dtype=np.uintp))
        )
        self._index_to = np.concatenate(
            (self._index_to, np.asarray(index_to, dtype=np.uintp))
        )

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


import numpy as np
import pandas as pd


class AbstractRecorder:

    def __init__(self):
        self._chunks = np.array([], dtype=np.int64)

    @property
    def chunks(self):
        return self._chunks

    def report_chunk(self, chunksize):
        self._chunks = np.append(self._chunks, chunksize)

    def chunk_local_index(self, global_index):
        chunk_index = np.insert(np.cumsum(self._chunks), 0, 0)
        chunk_num = np.searchsorted(chunk_index, global_index, side='right') - 1

        return chunk_num, global_index - chunk_index[chunk_num]


class GenericRainflowRecorder(AbstractRecorder):

    def __init__(self):
        super().__init__()
        self._values_from = []
        self._values_to = []
        self._index_from = []
        self._index_to = []

    @property
    def values_from(self):
        return self._values_from

    @property
    def values_to(self):
        return self._values_to

    @property
    def index_from(self):
        return self._index_from

    @property
    def index_to(self):
        return self._index_to

    def record_values(self, value_from, value_to):
        self._values_from.append(value_from)
        self._values_to.append(value_to)

    def record_index(self, index_from, index_to):
        self._index_from.append(index_from)
        self._index_to.append(index_to)

    def matrix(self, bins=10):
        """Calculate a histogram of the recorded values.

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
        return np.histogram2d(self._values_from, self._values_to, bins)

    def matrix_frame(self, bins=10):
        """Calculate a histogram of the recorded values into a pandas.DataFrame.

        An interval index is used to index the bins.

        Parameters
        ----------
        bins : int or array_like or [int, int] or [array, array], optional
            The bin specification: see numpy.histogram2d

        Returns
        -------
        pandas.DataFrame
            A pandas.DataFrame using a multi interval index in order to
            index data point for a given from/to value pair.
        """
        hist, fr, to = self.matrix(bins)
        index_fr = pd.IntervalIndex.from_breaks(fr)
        index_to = pd.IntervalIndex.from_breaks(to)

        mult_idx = pd.MultiIndex.from_product([index_fr, index_to], names=['from', 'to'])
        return pd.DataFrame(data=hist.flatten(), index=mult_idx)


import numpy as np
import pandas as pd


class GenericRainflowRecorder:

    def __init__(self):
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

    def record(self, index_from, index_to, loop_from, loop_to):
        self._values_from.append(loop_from)
        self._values_to.append(loop_to)
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

import numpy as np
import pandas as pd


class GenericRainflowRecorder:

    def __init__(self):
        self._loops_from = []
        self._loops_to = []
        self._index_from = []
        self._index_to = []

    @property
    def loops_from(self):
        return self._loops_from

    @property
    def loops_to(self):
        return self._loops_to

    @property
    def index_from(self):
        return self._index_from

    @property
    def index_to(self):
        return self._index_to

    def record(self, index_from, index_to, loop_from, loop_to):
        self._loops_from.append(loop_from)
        self._loops_to.append(loop_to)
        self._index_from.append(index_from)
        self._index_to.append(index_to)

    def matrix(self, bins=10):
        return np.histogram2d(self._loops_from, self._loops_to, bins)

    def matrix_frame(self, bins=10):
        hist, fr, to = self.matrix(bins)
        index_fr = pd.IntervalIndex.from_breaks(fr)
        index_to = pd.IntervalIndex.from_breaks(to)

        mult_idx = pd.MultiIndex.from_product([index_fr, index_to], names=['from', 'to'])
        return pd.DataFrame(data=hist.flatten(), index=mult_idx)

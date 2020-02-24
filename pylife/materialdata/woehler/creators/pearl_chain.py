

import numpy as np
import scipy.stats as stats

import pylife.utils.functions as functions
from pylife.utils.probability_data import ProbabilityFit


class PearlChainProbability(ProbabilityFit):
    def __init__(self, fractures, slope):
        self._normed_load = fractures.load.mean()
        self._normed_cycles = np.sort(fractures.cycles * ((self._normed_load/fractures.load)**(slope)))

        fp = functions.rossow_cumfreqs(len(self._normed_cycles))
        super().__init__(fp, self._normed_cycles)

    @property
    def normed_load(self):
        return self._normed_load

    @property
    def normed_cycles(self):
        return self._normed_cycles



import numpy as np
import scipy.stats as stats

import pylife.utils.functions as functions


class PearlChainProbability():
    def __init__(self, fractures, slope):
        self._normed_load = fractures.load.mean()
        self._normed_cycles = np.sort(fractures.cycles * ((self._normed_load/fractures.load)**(slope)))

        fp = functions.rossow_cumfreqs(len(self._normed_cycles))
        self._percentiles = stats.norm.ppf(fp)
        self._prob_slope, self._prob_intercept, _, _, _ = stats.linregress(np.log10(self._normed_cycles),
                                                                           self._percentiles)

    @property
    def normed_load(self):
        return self._normed_load

    @property
    def normed_cycles(self):
        return self._normed_cycles

    @property
    def probability_slope(self):
        return self._prob_slope

    @property
    def probability_intercept(self):
        return self._prob_intercept

    @property
    def percentiles(self):
        return self._percentiles

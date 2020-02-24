

import numpy as np
import scipy.stats as stats


class ProbabilityFit:
    def __init__(self, fprobs, occurences):
        ppf = stats.norm.ppf(fprobs)
        self._slope, self._intercept, _, _, _ = stats.linregress(np.log10(occurences), ppf)
        self._ppf = ppf
        self._occurences = occurences

    @property
    def slope(self):
        return self._slope

    @property
    def intercept(self):
        return self._intercept

    @property
    def occurences(self):
        return self._occurences

    @property
    def precentiles(self):
        return self._ppf

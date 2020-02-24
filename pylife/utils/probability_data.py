

import numpy as np
import scipy.stats as stats


class ProbabilityFit:
    def __init__(self, probs, occurrences):
        ppf = stats.norm.ppf(probs)
        self._slope, self._intercept, _, _, _ = stats.linregress(np.log10(occurrences), ppf)
        self._ppf = ppf
        self._occurrences = occurrences

    @property
    def slope(self):
        return self._slope

    @property
    def intercept(self):
        return self._intercept

    @property
    def occurrences(self):
        return self._occurrences

    @property
    def precentiles(self):
        return self._ppf



import numpy as np
import scipy.stats as stats


class ProbabilityFit:
    def __init__(self, probs, occurrences):
        ''' Fit samples and their estimated occurences to a lognorm distribution

        Parameters:
        -----------
        probs : array_like
            The estimated cumulated probabilities of the sample values
            (i.e. estimated by func:`pylife.utils.functions.rossow_cumfreqs`)
        occurences : array_like
            the values of the samples
        '''
        if len(probs) != len(occurrences):
            raise ValueError("probs and occurence arrays must have the same 1D shape.")
        ppf = stats.norm.ppf(probs)
        self._occurrences = np.array(occurrences, dtype=np.float)
        self._slope, self._intercept, _, _, _ = stats.linregress(np.log10(self._occurrences), ppf)
        self._ppf = ppf


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
    def percentiles(self):
        return self._ppf

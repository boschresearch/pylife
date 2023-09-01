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


import numpy as np
import scipy.stats as stats


class ProbabilityFit:
    """Fit samples and their estimated occurrences to a lognorm distribution.

    Parameters:
    -----------
    probs : array_like
        The estimated cumulated probabilities of the sample values
        (i.e. estimated by func:`pylife.utils.functions.rossow_cumfreqs`)
    occurences : array_like
        the values of the samples
    """

    def __init__(self, probs, occurrences):
        if len(probs) != len(occurrences):
            raise ValueError("probs and occurrence arrays must have the same 1D shape.")
        if len(probs) < 2:
            raise ValueError("Need at least two datapoints for probabilities and occurrences.")
        ppf = stats.norm.ppf(probs)
        self._occurrences = np.array(occurrences, dtype=np.float64)
        self._slope, self._intercept, _, _, _ = stats.linregress(np.log10(self._occurrences), ppf)
        self._ppf = ppf


    @property
    def slope(self):
        """The slope of the probability curve"""
        return self._slope

    @property
    def intercept(self):
        """The intercept of the probability curve"""
        return self._intercept

    @property
    def occurrences(self):
        """The occurrences (sample values)"""
        return self._occurrences

    @property
    def percentiles(self):
        """The calculated percentiles of the cumulated probabilities"""
        return self._ppf

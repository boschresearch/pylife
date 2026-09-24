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

"""Provide probability-paper fitting helpers for fatigue samples."""


import numpy as np
import scipy.stats as stats


class ProbabilityFit:
    """Fit occurrence samples for a normal probability-paper plot.

    ``ProbabilityFit`` transforms cumulative probabilities with the standard
    normal percent point function and fits a straight line against the
    base-10 logarithm of the occurrence values.  Use it for fatigue test data
    after sorting the measured occurrences and estimating their cumulative
    frequencies, for example with
    :func:`pylife.utils.functions.rossow_cumfreqs`.

    Parameters
    ----------
    probs : array_like
        Estimated cumulative probabilities of the sorted sample values.
        Values should lie between ``0`` and ``1``.
    occurrences : array_like
        Positive sample values, usually fatigue lives or strengths, in the
        same order as ``probs``.

    Raises
    ------
    ValueError
        Raised if ``probs`` and ``occurrences`` have different lengths or if
        fewer than two data points are supplied.

    Notes
    -----
    The fitted probability-paper line is
    ``percentile = slope * log10(occurrence) + intercept``.  Plot
    ``percentiles`` versus ``log10(occurrences)`` together with this line
    to inspect whether the data are compatible with a log-normal model.

    Examples
    --------
    >>> from pylife.utils.functions import rossow_cumfreqs
    >>> occurrences = np.array([1.0e4, 2.0e4, 4.0e4])
    >>> fit = ProbabilityFit(rossow_cumfreqs(len(occurrences)), occurrences)
    >>> round(float(fit.slope), 6)
    2.795805
    >>> [round(float(value), 6) for value in fit.percentiles]
    [-0.841621, 0.0, 0.841621]
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
        """Return the slope of the fitted probability-paper line.

        Returns
        -------
        float
            Change in standard-normal percentile per decade of occurrence.
        """
        return self._slope

    @property
    def intercept(self):
        """Return the intercept of the fitted probability-paper line.

        Returns
        -------
        float
            Standard-normal percentile at ``log10(occurrence) == 0``.
        """
        return self._intercept

    @property
    def occurrences(self):
        """Return the occurrence sample values used for the fit.

        Returns
        -------
        numpy.ndarray
            Sample values passed to the fit, usually fatigue lives or
            strengths.
        """
        return self._occurrences

    @property
    def percentiles(self):
        """Return standard-normal percentiles of the cumulative probabilities.

        Returns
        -------
        numpy.ndarray
            Values computed as ``scipy.stats.norm.ppf(probs)`` for the
            probabilities supplied to the fit.
        """
        return self._ppf

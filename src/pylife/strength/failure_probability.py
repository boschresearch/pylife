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

"""Provide failure probability calculations for log-normal strength data.

The module contains :class:`FailureProbability`, a small helper that combines
a log-normal strength distribution with deterministic, log-normal, or
arbitrary load distributions.
"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate


class FailureProbability:
    r"""Represent a log-normal strength distribution for failure probability.

    Failure is assumed to occur when load exceeds strength. All deterministic
    load and strength medians use the same physical unit, for example MPa.

    Parameters
    ----------
    strength_median : array_like
        Median strength value in MPa or another consistent load unit.
    strength_std : array_like
        Standard deviation of the base-10 logarithm of strength,
        dimensionless.

    Notes
    -----
    For a load probability density ``f_L`` and a strength cumulative
    distribution ``F_S``, the total failure probability is

    .. math::

        P_f = \int_{-\infty}^{\infty} f_L(x) F_S(x)\,dx.

    The implementation stores strengths in base-10 logarithmic space.
    """

    def __init__(self, strength_median, strength_std):
        self.s_50 = np.log10(strength_median)
        self.s_std = strength_std

    def pf_simple_load(self, load):
        r"""Calculate failure probability for a deterministic load.

        Parameters
        ----------
        load : array_like
            Deterministic load value in the same unit as ``strength_median``.

        Returns
        -------
        numpy.ndarray or float
            Failure probability, dimensionless.

        Notes
        -----
        For a non-random load ``L`` the probability of failure is the strength
        cumulative distribution evaluated at that load:

        .. math::

            P_f = F_S(\log_{10}(L)).
        """
        return norm.cdf(np.log10(load), loc=self.s_50, scale=self.s_std)

    def pf_norm_load(self, load_median, load_std, lower_limit=None, upper_limit=None):
        """Calculate failure probability for a log-normal load distribution.

        Parameters
        ----------
        load_median : array_like
            Median load value in the same unit as ``strength_median``.
        load_std : array_like
            Standard deviation of the base-10 logarithm of load, dimensionless.
        lower_limit : float or None, optional
            Lower integration limit in load units. If ``None``, use a
            logarithmic bound of ``-16 * load_std`` around the median.
            Default is ``None``.
        upper_limit : float or None, optional
            Upper integration limit in load units. If ``None``, use a
            logarithmic bound of ``16 * load_std`` around the median. Default
            is ``None``.

        Returns
        -------
        numpy.ndarray or float
            Failure probability, dimensionless.

        Notes
        -----
        The load and strength distributions are shifted into logarithmic space
        with the load median at zero before numerical integration. For very
        small ``load_std`` the result approaches :meth:`pf_simple_load`.
        """
        lm = np.log10(load_median)

        sc = load_std

        if lower_limit is None:
            lower_limit = -16.*sc
        else:
            lower_limit -= lm
        if upper_limit is None:
            upper_limit = +16.*sc
        else:
            upper_limit -= lm

        q1, err_est = integrate.quad(
            lambda x: norm.pdf(x, loc=0.0, scale=sc) * norm.cdf(x, loc=self.s_50-lm, scale=self.s_std),
            lower_limit, upper_limit)

        return q1

    def pf_arbitrary_load(self, load_values, load_pdf):
        """Calculate failure probability for an arbitrary load distribution.

        Parameters
        ----------
        load_values : numpy.ndarray
            Load support points in base-10 logarithmic units.
        load_pdf : numpy.ndarray
            Probability density values corresponding to ``load_values``.

        Returns
        -------
        numpy.ndarray or float
            Failure probability, dimensionless.

        Raises
        ------
        ValueError
            Raised if ``load_values`` and ``load_pdf`` do not have the same
            shape.

        Notes
        -----
        The integral of load density times strength cumulative distribution is
        approximated with the trapezoidal rule.
        """
        if load_values.shape != load_pdf.shape:
            raise ValueError("Load values and pdf must have same dimensions.")

        strength_cdf = norm.cdf(load_values, loc=self.s_50, scale=self.s_std)

        return np.trapezoid(load_pdf * strength_cdf, x = load_values)

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

"""Provide fatigue statistics helper functions."""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import numpy as np


def scattering_range_to_std(T):
    r"""Convert a fatigue scattering range to a log-standard deviation.

    The scattering range ``T`` is the ratio of the 90 % quantile to the
    10 % quantile of a log-normal distribution, as used for ``TS`` and
    ``TN`` in DIN 50100:2016-12.  The returned standard deviation is the
    normal standard deviation of the base-10 logarithm of the quantity.

    Parameters
    ----------
    T : float
        Scattering range as the 90 %/10 % quantile ratio of a log-normal
        distribution.

    Returns
    -------
    float
        Standard deviation in log10 units corresponding to ``T``.

    See Also
    --------
    pylife.utils.functions.std_to_scattering_range : Convert a log10 standard
        deviation to a scattering range.

    Notes
    -----
    The conversion follows from the symmetric 10 % and 90 % quantiles of a
    normal distribution in log10 space:

    .. math::

        \sigma_{\log_{10}} =
        \frac{\log_{10}(T)}{2 \Phi^{-1}(0.9)}

    where :math:`\Phi^{-1}` is the inverse standard normal cumulative
    distribution function.

    Examples
    --------
    >>> from pylife.utils.functions import scattering_range_to_std, std_to_scattering_range
    >>> round(float(scattering_range_to_std(10.0)), 6)
    0.390152
    >>> round(float(std_to_scattering_range(scattering_range_to_std(1.25))), 6)
    1.25
    """
    return 0.39015207303618954*np.log10(T)


def std_to_scattering_range(std):
    r"""Convert a log-standard deviation to a fatigue scattering range.

    The scattering range ``T`` is the ratio of the 90 % quantile to the
    10 % quantile of a log-normal distribution, as used for ``TS`` and
    ``TN`` in DIN 50100:2016-12.  The input ``std`` is the standard
    deviation of the base-10 logarithm of the quantity.

    Parameters
    ----------
    std : float
        Standard deviation in log10 units.

    Returns
    -------
    float
        Scattering range as the 90 %/10 % quantile ratio of a log-normal
        distribution.

    See Also
    --------
    pylife.utils.functions.scattering_range_to_std : Convert a scattering
        range to a log10 standard deviation.

    Notes
    -----
    The conversion is the inverse of
    :func:`pylife.utils.functions.scattering_range_to_std`:

    .. math::

        T = 10^{2 \Phi^{-1}(0.9) \sigma_{\log_{10}}}

    where :math:`\Phi^{-1}` is the inverse standard normal cumulative
    distribution function.

    Examples
    --------
    >>> from pylife.utils.functions import scattering_range_to_std, std_to_scattering_range
    >>> round(float(std_to_scattering_range(0.39015207303618954)), 6)
    10.0
    >>> round(float(scattering_range_to_std(std_to_scattering_range(0.2))), 6)
    0.2
    """
    return 10**(2.5631031310892007*std)


def rossow_cumfreqs(N):
    r"""Estimate cumulative frequencies according to Rossow.

    Use this estimator to assign plotting positions to sorted fatigue test
    samples before fitting them in probability paper.  The estimate gives the
    probability that the next observation is below the ``i``-th value of
    ``N`` sorted samples [Rossow1964]_.

    Parameters
    ----------
    N : int
        Sample size of the statistical population.

    Returns
    -------
    numpy.ndarray
        Estimated cumulative frequencies for the ``N`` sorted samples.

    Notes
    -----
    For one-based sample ranks :math:`i = 1, \ldots, N`, Rossow's estimator
    computes the cumulative frequency as

    .. math::

        P_i = \frac{3 i - 1}{3 N + 1}

    The estimates are symmetric around 0.5 and sum to ``N / 2``.

    References
    ----------
    .. [Rossow1964] Rossow, E., "Statistics of Metal Fatigue in Engineering",
       page 16.

    Examples
    --------
    >>> from pylife.utils.functions import rossow_cumfreqs
    >>> rossow_cumfreqs(1)
    array([0.5])
    >>> rossow_cumfreqs(3)
    array([0.2, 0.5, 0.8])
    >>> round(float(rossow_cumfreqs(4).sum()), 6)
    2.0
    """
    i = np.arange(1, N+1)
    return (3.*i-1.)/(3.*N+1)

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

"""
Utility Functions
=================

A collection of functions frequently used in lifetime estimation
business.
"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import numpy as np


def scattering_range_to_std(T):
    """Convert a scattering range (``TS`` or ``TN`` in DIN 50100:2016-12) into standard deviation.

    Parameters
    ----------
    T : float
        inverted scattering range

    Returns
    -------
    std : float
        standard deviation corresponding to TS or TN assuming a normal distribution

    Notes
    -----
    Actually ``1/(2*norm.ppf(0.9))*np.log10(T)``

    Inverse of ``std_to_scattering_range()``
    """
    return 0.39015207303618954*np.log10(T)


def std_to_scattering_range(std):
    """Convert a standard deviation into scattering range (``TS`` or ``TN`` in DIN 50100:2016-12).

    Parameters
    ----------
    std : float
        standard deviation

    Returns
    -------
    T : float
        inverted scattering range corresponding to ``std`` assuming a normal distribution

    Notes
    -----
    Actually ``10**(2*norm.ppf(0.9)*std``

    Inverse of ``scattering_range_to_std()``
    """
    return 10**(2.5631031310892007*std)


def rossow_cumfreqs(N):
    """Cumulative frequency estimator according to Rossow.

    Parameters
    ----------
    N : int
        The sample size of the statistical population

    Returns
    -------
    cumfreqs : numpy.ndarray
        The estimated cumulated frequencies of the N samples

    Notes
    -----
    The returned value is the probability that the next taken sample
    is below the value of the i-th sample of n sorted samples.

    Examples
    --------
    >>> rossow_cumfreqs(1)
    array([0.5])

    If we have one sample, the probability that the next sample will
    be below it is 0.5.

    >>> rossow_cumfreqs(3)
    array([0.2, 0.5, 0.8])

    If we have three sorted samples, the probability that the next
    sample will be
    * below the first is 0.2
    * below the second is 0.5
    * below the third is 0.8


    References
    ----------
    *Statistics of Metal Fatigue in Engineering' page 16*

    https://books.google.de/books?isbn=3752857722

    """
    i = np.arange(1, N+1)
    return (3.*i-1.)/(3.*N+1)

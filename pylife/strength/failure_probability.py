# Copyright (c) 2019-2021 - for information on the respective copyright owner
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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate


class FailureProbability:
    '''Strength representation to calculate failure probabilities

    The strength is represented as a log normal distribution of
    strength_median and strength_std.

    Failure probabilities can be calculated for a given load or load
    distribution.

    Parameters
    ----------
    strength_median : array_like, shape (N, )
        The median value of the strength
    strength_std : array_like, shape (N, )
        The standard deviation of the strength

    Note
    ----
    We assume that the load and the strength are statistically
    distributed values. In case the load is higher than the strength
    we get failure. So if we consider a quantile of our load
    distribution of a probability p_load, the probability of failure
    due to a load of this quantile is p_load times the probability
    that the strength lies within this quantile or below.

    So in order to calculate the total failure probability, we need to
    integrate the load's pdf times the strength' cdf from -inf to +inf.

    '''

    def __init__(self, strength_median, strength_std):
        self.s_50 = np.log10(strength_median)
        self.s_std = strength_std

    def pf_simple_load(self, load):
        '''Failure probability for a simple load value

        Parameters
        ----------
        load : array_like, shape (N,) consistent with class parameters
            The load of for which the failure probability is
            calculated.

        Returns
        -------
        failure probability : numpy.ndarray or float

        Notes
        -----
        This is the case of a non statistical load. So failure occurs
        if the strength is below the given load, i.e. the strength'
        cdf at the load.
        '''
        return norm.cdf(np.log10(load), loc=self.s_50, scale=self.s_std)

    def pf_norm_load(self, load_median, load_std, lower_limit=None, upper_limit=None):
        '''Failure probability for a log normal distributed load

        Parameters
        ----------
        load_median : array_like, shape (N,) consistent with class parameters
            The median of the load distribution for which the failure
            probability is calculated.
        load_std : array_like, shape (N,) consistent with class parameters
            The standard deviation of the load distribution
        lower_limit : float, optional
            The lower limit of the integration, default None
        upper_limit : float, optional
            The upper limit of the integration, default None

        Returns
        -------
        failure probability : numpy.ndarray or float

        Notes
        -----

        The log normal distribution of the load is determined by the
        load parameters. Only load distribution between
        ``lower_limit`` and ``upper_limit`` is considered.

        For small values for ``load_std`` this function gives the same
        result as ``pf_simple_load``.

        Note
        ----
        The load and strength distributions are transformed in a way,
        that the median of the load distribution is zero. This
        guarantees that in any case we can provide a set of relevant
        points to take into account for the integration.

        '''
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
        ''' Calculates the failure probability for an arbitrary load

        Parameters
        ----------
        load_values : array_like, shape (N,)
            The load values of the load distribution
        load_pdf : array_like, shape (N, )
            The probability density values for the ``load_value`` values to
            occur

        Returns
        -------
        failure probability : numpy.ndarray or float
        '''
        if load_values.shape != load_pdf.shape:
            raise Exception("Load values and pdf must have same dimensions.")

        strength_cdf = norm.cdf(load_values, loc=self.s_50, scale=self.s_std)

        return np.trapz(load_pdf * strength_cdf, x = load_values)

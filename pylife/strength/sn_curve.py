# Copyright (c) 2019 - for information on the respective copyright owner
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

__author__ = "Cedric Philip Wagner"
__maintainer__ = "Johannes Mueller"

import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n')
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


class FiniteLifeBase:
    """Base class for SN curve calculations - either in logarithmic or regular scale"""

    _k_initial = None  # maintain the initial slope even when k is updated later on
    _k = None
    S_d = None
    N_e = None

    def __init__(self, k, S_d, N_e):
        self._k_initial = k
        self._k = self._k_initial
        self.S_d = S_d
        self.N_e = N_e
        self._check_inputs_valid(k=self.k,
                                 S_d=self.S_d,
                                 N_e=self.N_e)

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k):
        if self._k is not None:
            logger.info("The slope 'k' of the SN curve is updated from "
                        "'{}' to '{}'".format(self._k, k))
        self._k = k

    def _check_inputs_valid(self, **kwargs):
        try:
            for param, value in kwargs.items():
                if not float(value) > 0:
                    logger.error("Parameter '{}' is not a number greater than zero: {}".format(
                            param,
                            value,
                            ))
        except Exception as e:
            raise ValueError("Not all inputs are valid numbers: {}".format(e))


class FiniteLifeLine(FiniteLifeBase):
    """Sample points on the finite life line - either N or S (LOGARITHMIC SCALE)

    The formula for calculation is taken from
    "Betriebsfestigkeit", Haibach, 3. Auflage 2006

    Notes
    -----
    In contrast to the case

    Parameters
    ----------
    k : float
        slope of the SN-curve
    S_d : float
        lower stress limit in the finite life region
    N_e : float
        number of cycles at stress S_d
    """
    S_d_log = None
    N_e_log = None

    def __init__(self, k, S_d, N_e):
        super(FiniteLifeLine, self).__init__(k, S_d, N_e)
        self.S_d_log = np.log10(self.S_d)
        self.N_e_log = np.log10(self.N_e)
        if self.k < 0:
            logger.warning("Parameter k should be positive. Given k ('{}') has been "
                           "assigned as absolute value.".format(self.k))
            self.k = abs(self.k)

    def calc_S_log(self, N_log, ignore_limits=False):
        """Calculate stress logarithmic S_log for a given number of cycles N_log

        Parameters
        ----------
        N_log : float
            logarithmic number of cycles
        ignore_limits : boolean
            ignores the upper limit of the number of cycles
            generally it should be smaller than N_e (=the limit of the finite life region)
            but some special evaluation methods (e.g. according to marquardt2004) require
            extrapolation to estimate an equivalent stress

        Returns
        -------
        S_log : float
            logarithmic stress corresponding to the given number of cycles (point on the SN-curve)
        """
        if N_log > self.N_e_log:
            if ignore_limits:
                logger.warning("The limits to the interpolation of the finite life region "
                               "have explicitly been ignored.")
            else:
                raise ValueError("The given N_log ('{}') must be smaller than "
                                 "the lifetime ('{}').".format(
                                     N_log,
                                     self.N_e_log,
                                     ))
        elif N_log < 1:
            raise ValueError("Invalid input for parameter N_log: {}".format(N_log))
        return self.S_d_log + (self.N_e_log - N_log) / self.k

    def calc_N_log(self, S_log, ignore_limits=False):
        """Calculate number of cycles N_log for a given stress S_log

        Parameters
        ----------
        S_log : float
            logarithmic stress (point on the SN-curve)
        ignore_limits : boolean
            ignores the upper limit of the number of cycles
            generally it should be smaller than N_e_log (=the limit of the finite life region)
            but some special evaluation methods (e.g. according to marquardt2004) require
            extrapolation to estimate an equivalent stress

        Returns
        -------
        N_log : float
            logarithmic number of cycles corresponding to the given stress value (point on the SN-curve)
        """
        if S_log < self.S_d_log:
            if ignore_limits:
                logger.warning("The limits to the interpolation of the finite life region "
                               "have explicitly been ignored.")
            else:
                raise ValueError("The given stress S_log ('{}') must be larger than "
                                 "S_d_log ('{}').".format(
                                     S_log,
                                     self.S_d_log,
                                     ))
        return self.N_e_log + (self.S_d_log - S_log) * self.k


class FiniteLifeCurve(FiniteLifeBase):
    """Sample points on the finite life curve - either N or S (NOT logarithmic scale)

    The formula for calculation is taken from
    "Betriebsfestigkeit", Haibach, 3. Auflage 2006

    Parameters
    ----------
    k : float
        slope of the SN-curve
    S_d : float
        lower stress limit in the finite life region
    N_e : float
        number of cycles at stress S_d
    """
    def __init__(self, k, S_d, N_e):
        super(FiniteLifeCurve, self).__init__(k, S_d, N_e)

    def calc_S(self, N, ignore_limits=False):
        """Calculate stress S for a given number of cycles N

        Parameters
        ----------
        N : float
            number of cycles
        ignore_limits : boolean
            ignores the upper limit of the number of cycles
            generally it should be smaller than N_e (=the limit of the finite life region)
            but some special evaluation methods (e.g. according to marquardt2004) require
            extrapolation to estimate an equivalent stress

        Returns
        -------
        S : float
            stress corresponding to the given number of cycles (point on the SN-curve)
        """
        if N > self.N_e:
            if ignore_limits:
                logger.warning("The limits to the interpolation of the finite life region "
                               "have explicitly been ignored.")
            else:
                raise ValueError("The given N ('{}') must be smaller than "
                                 "the lifetime ('{}').".format(
                                     N,
                                     self.N_e,
                                     ))
        elif N < 1:
            raise ValueError("Invalid input for parameter N: {}".format(N))
        return self.S_d * (N / self.N_e)**(- 1/self.k)

    def calc_N(self, S, ignore_limits=False):
        """Calculate number of cycles N for a given stress S

        Parameters
        ----------
        S : float
            Stress (point on the SN-curve)
        ignore_limits : boolean
            ignores the upper limit of the number of cycles
            generally it should be smaller than N_e (=the limit of the finite life region)
            but some special evaluation methods (e.g. according to marquardt2004) require
            extrapolation to estimate an equivalent stress

        Returns
        -------
        N : float
            number of cycles corresponding to the given stress value (point on the SN-curve)
        """
        if S < self.S_d:
            if ignore_limits:
                logger.warning("The limits to the interpolation of the finite life region "
                               "have explicitly been ignored.")
            else:
                raise ValueError("The given stress S ('{}') must be larger than "
                                 "S_d ('{}').".format(
                                     S,
                                     self.S_d,
                                     ))
        return self.N_e * (S / self.S_d)**(-self.k)

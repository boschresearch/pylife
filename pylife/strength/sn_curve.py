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

__author__ = "Cedric Philip Wagner"
__maintainer__ = "Johannes Mueller"

import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n')
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


class FiniteLifeBase:
    """Base class for SN curve calculations - either in logarithmic or regular scale"""

    def __init__(self, k_1, SD_50, ND_50):
        self._k = k_1
        self.SD_50 = SD_50
        self.ND_50 = ND_50
        self._check_inputs_valid(k_1=self.k_1,
                                 SD_50=self.SD_50,
                                 ND_50=self.ND_50)

    @property
    def k_1(self):
        return self._k

    @k_1.setter
    def k_1(self, k_1):
        self._k = k_1

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
    SD_50 : float
        lower stress limit in the finite life region
    ND_50 : float
        number of cycles at stress SD_50
    """

    def __init__(self, k, SD_50, ND_50):
        super(FiniteLifeLine, self).__init__(k, SD_50, ND_50)
        self.SD_50_log = np.log10(self.SD_50)
        self.ND_50_log = np.log10(self.ND_50)
        if self.k_1 < 0:
            logger.warning("Parameter k should be positive. Given k ('{}') has been "
                           "assigned as absolute value.".format(self.k_1))
            self.k_1 = abs(self.k_1)

    def calc_S_log(self, N_log, ignore_limits=False):
        """Calculate stress logarithmic S_log for a given number of cycles N_log

        Parameters
        ----------
        N_log : float
            logarithmic number of cycles
        ignore_limits : boolean
            ignores the upper limit of the number of cycles
            generally it should be smaller than ND_50 (=the limit of the finite life region)
            but some special evaluation methods (e.g. according to marquardt2004) require
            extrapolation to estimate an equivalent stress

        Returns
        -------
        S_log : float
            logarithmic stress corresponding to the given number of cycles (point on the SN-curve)
        """
        if N_log > self.ND_50_log:
            if ignore_limits:
                logger.warning("The limits to the interpolation of the finite life region "
                               "have explicitly been ignored.")
            else:
                raise ValueError("The given N_log ('{}') must be smaller than "
                                 "the lifetime ('{}').".format(
                                     N_log,
                                     self.ND_50_log,
                                     ))
        elif N_log < 1:
            raise ValueError("Invalid input for parameter N_log: {}".format(N_log))
        return self.SD_50_log + (self.ND_50_log - N_log) / self.k_1

    def calc_N_log(self, S_log, ignore_limits=False):
        """Calculate number of cycles N_log for a given stress S_log

        Parameters
        ----------
        S_log : float
            logarithmic stress (point on the SN-curve)
        ignore_limits : boolean
            ignores the upper limit of the number of cycles
            generally it should be smaller than ND_50_log (=the limit of the finite life region)
            but some special evaluation methods (e.g. according to marquardt2004) require
            extrapolation to estimate an equivalent stress

        Returns
        -------
        N_log : float
            logarithmic number of cycles corresponding to the given stress value (point on the SN-curve)
        """
        if S_log < self.SD_50_log:
            if ignore_limits:
                logger.warning("The limits to the interpolation of the finite life region "
                               "have explicitly been ignored.")
            else:
                raise ValueError("The given stress S_log ('{}') must be larger than "
                                 "SD_50_log ('{}').".format(
                                     S_log,
                                     self.SD_50_log,
                                     ))
        return self.ND_50_log + (self.SD_50_log - S_log) * self.k_1


class FiniteLifeCurve(FiniteLifeBase):
    """Sample points on the finite life curve - either N or S (NOT logarithmic scale)

    The formula for calculation is taken from
    "Betriebsfestigkeit", Haibach, 3. Auflage 2006

    **Consider:** load collective and life curve have to be consistent:

        * range vs range
        * amplitude vs amplitude

    Parameters
    ----------
    k : float
        slope of the SN-curve
    SD_50 : float
        lower stress limit in the finite life region
    ND_50 : float
        number of cycles at stress SD_50
    """
    def __init__(self, k_1, SD_50, ND_50):
        super(FiniteLifeCurve, self).__init__(k_1, SD_50, ND_50)

    def calc_S(self, N, ignore_limits=True):
        """Calculate stress S for a given number of cycles N

        Parameters
        ----------
        N : float
            number of cycles
        ignore_limits : boolean
            ignores the upper limit of the number of cycles
            generally it should be smaller than ND_50 (=the limit of the finite life region)
            but some special evaluation methods (e.g. according to marquardt2004) require
            extrapolation to estimate an equivalent stress

        Returns
        -------
        S : float
            stress corresponding to the given number of cycles (point on the SN-curve)
        """
        if N > self.ND_50:
            if ignore_limits:
                logger.warning("The limits to the interpolation of the finite life region "
                               "have explicitly been ignored.")
            else:
                raise ValueError("The given N ('{}') must be smaller than "
                                 "the lifetime ('{}').".format(
                                     N,
                                     self.ND_50,
                                     ))
        elif N < 1:
            raise ValueError("Invalid input for parameter N: {}".format(N))
        return self.SD_50 * (N / self.ND_50)**(- 1/self.k_1)

    def calc_N(self, S, ignore_limits=False):
        """Calculate number of cycles N for a given stress S

        Parameters
        ----------
        S : array like
            Stress (point(s) on the SN-curve)
        ignore_limits : boolean
            ignores the upper limit of the number of cycles
            generally it should be smaller than ND_50 (=the limit of the finite life region)
            but some special evaluation methods (e.g. according to marquardt2004) require
            extrapolation to estimate an equivalent stress

        Returns
        -------
        N : array like
            number of cycles corresponding to the given stress value (point on the SN-curve)
        """
        if np.any(S < self.SD_50):
            if ignore_limits:
                logger.warning("Using all cycles")
            else:
                raise ValueError("The given stress S ('{}') must be larger than "
                                 "SD_50 ('{}').".format(
                                     S,
                                     self.SD_50,
                                     ))
        return self.ND_50 * (S / self.SD_50)**(-self.k_1)

    def calc_damage(self, loads, method="elementar", index_name="range"):
        """Calculate the damage based on the methods
         * Miner elementar (k_2 = k)
         * Miner Haibach (k_2 = 2k-1)
         * Miner original (k_2 = -\inf)

        **Consider:** load collective and life curve have to be consistent:

        * range vs range
        * amplitude vs amplitude

        Parameters
        ----------
        loads : pandas series histogram
            loads (index is the load, column the cycles)
        method : str
         * 'elementar': Miner elementar (k_2 = k)
         * 'MinerHaibach': Miner Haibach (k_2 = 2k-1)
         * 'original': Miner original (k_2 = -\inf)

        Returns
        -------
        damage : pd.DataFrame
            damage for every load horizont based on the load collective and the method
        """
        damage = pd.DataFrame(index=loads.index,columns=loads.columns, data=0)
        load_values = loads.index.get_level_values(index_name).mid.values
        # Miner elementar
        cycles_SN = self.calc_N(load_values, ignore_limits=True)
        if method == 'original':
            damage = loads.divide(cycles_SN,axis = 0)
            damage[load_values <= self.SD_50] = 0
        elif method == 'elementar':
            damage = loads.divide(cycles_SN,axis = 0)
        elif method == 'MinerHaibach':
            # k2
            cycles_SN[load_values <= self.SD_50] = self.ND_50 * (load_values[load_values <= self.SD_50] / self.SD_50)**(-(2*self.k_1-1))
            damage = loads.divide(cycles_SN, axis=0)
        return damage

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

__author__ = "Cedric Philip Wagner"
__maintainer__ = "Johannes Mueller"

import warnings

import pandas as pd

import pylife.strength.fatigue
import pylife.stress

warnings.warn(
    FutureWarning(
        "The module pylife.strength.helpers is deprecated and no longer under test. "
        "The functionality is now avaliable in the pylife.materiallaws.WoehlerCurve."
    )
)

class FiniteLifeBase:
    """Base class for SN curve calculations - either in logarithmic or regular scale"""

    def __init__(self, k_1, SD_50, ND_50):
        warnings.warn(DeprecationWarning("FiniteLifeBase and derived classes are deperecated. "
                                         "Use WoehlerCurve and Fatigue accessors instead."))
        self._wc = pd.Series({
            'k_1': k_1,
            'SD': SD_50,
            'ND': ND_50
        })

    @property
    def k_1(self):
        return self._wc.k_1


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
        super().__init__(k, SD_50, ND_50)


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
        super().__init__(k_1, SD_50, ND_50)

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
        return self._wc.woehler.basquin_load(N)

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
        return self._wc.woehler.basquin_cycles(S)

    def calc_damage(self, loads, method="elementar", index_name="range"):
        """Calculate the damage based on the methods
         * Miner elementar (k_2 = k)
         * Miner Haibach (k_2 = 2k-1)
         * Miner original (k_2 = -inf)

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
         * 'original': Miner original (k_2 = -inf)

        Returns
        -------
        damage : pd.DataFrame
            damage for every load horizont based on the load collective and the method
        """
        estimator = self._wc.fatigue

        if method == 'elementar':
            estimator = estimator.miner_elementary()
        elif method == 'MinerHaibach':
            estimator = estimator.miner_haibach()

        if index_name != "range":
            loads = loads.copy()
            names = ["range" if name == index_name else name for name in loads.index.names]
            loads.index.set_names(names, inplace=True)

        damage = estimator.damage(loads.load_collective)
        if index_name != "range":
            names = [index_name if name == "range" else name for name in loads.index.names]
            damage.index.set_names(names, inplace=True)

        return damage

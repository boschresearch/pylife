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

"""

Implementation of the miner rule for fatigue analysis
=====================================================

Currently, the following implementations are part of this module:

* Miner-elementary
* Miner-haibach

The source will be given in the function/class

References
----------
M. Wächter, C. Müller and A. Esderts, "Angewandter Festigkeitsnachweis nach {FKM}-Richtlinie"
Springer Fachmedien Wiesbaden 2017, https://doi.org/10.1007/978-3-658-17459-0

E. Haibach, "Betriebsfestigkeit", Springer-Verlag 2006, https://doi.org/10.1007/3-540-29364-7
"""

__author__ = "Cedric Philip Wagner"
__maintainer__ = "Johannes Mueller"


import numpy as np
import pandas as pd

from pylife.strength.fatigue import Fatigue
from pylife.materiallaws.woehlercurve import WoehlerCurve

import pylife.strength.solidity



class MinerBase:
    """Basic functions related to miner-rule (original)

    Definitions will be based on the given references.
    Therefore, the original names are used so that they can
    be looked up easily.

    Parameters
    ----------
    ND : float
        number of cycles of the fatigue strength of the S/N curve [number of cycles]
    k_1 : float
        slope of the S/N curve [unitless]
    SD : float
        fatigue strength of the S/N curve [MPa]
    """

    def calc_zeitfestigkeitsfaktor(self, N):
        """Calculate "Zeitfestigkeitsfaktor" according to Waechter2017 (p. 96)"""
        return np.power(self.ND/N, 1./self.k_1)

    def effective_damage_sum(self, collective):
        """Compute 'effective damage sum' D_m

        Refers to the formula given in Waechter2017, p. 99

        Parameters
        ----------
        A : float or np.ndarray (with 1 element)
            the multiple of the lifetime
        """

        A = self.lifetime_multiple(collective)
        return effective_damage_sum(A)

    def gassner_cycles(self, collective):
        print(self.lifetime_multiple(collective))
        return self.cycles(collective.rainflow.amplitude.max()) * self.lifetime_multiple(collective)


def effective_damage_sum(lifetime_multiple):
    """Compute 'effective damage sum' D_m

    Refers to the formula given in Waechter2017, p. 99

    Parameters
    ----------
    A : float or np.ndarray (with 1 element)
        the multiple of the lifetime
    """

    d_min = 0.3  # minimum as suggested by FKM
    d_max = 1.0

    d_m_no_limits = 2. / (lifetime_multiple**(1./4.))
    d_m = min(
        max(d_min, d_m_no_limits),
        d_max
    )

    return d_m


@pd.api.extensions.register_series_accessor('gassner_miner_elementary')
class MinerElementary(WoehlerCurve, MinerBase):
    """Implementation of Miner-elementary according to Waechter2017

    """

    def gassner(self, collective):
        gassner = self.to_pandas().copy()
        gassner['ND'] = self.ND * self.lifetime_multiple(collective)
        return Fatigue(gassner)

    def lifetime_multiple(self, collective):
        """Compute the lifetime multiple according to miner-elementary

        Described in Waechter2017 as "Lebensdauervielfaches, A_ele".

        Parameters
        ----------
        collective : np.ndarray
            numpy array of shape (:, 2)
            where ":" depends on the number of classes defined
            for the rainflow counting
            * column: class values in ascending order
            * column: accumulated number of cycles
            first entry is the total number of cycles
            then in a descending manner till the
            number of cycles of the highest stress class
        """
        return 1. / collective.solidity.haibach(self.k_1)


@pd.api.extensions.register_series_accessor('gassner_miner_haibach')
class MinerHaibach(WoehlerCurve, MinerBase):
    """Miner-modified according to Haibach (2006)

    WARNING: Contrary to Miner-elementary, the lifetime multiple A
             is not constant but dependent on the evaluated load level!

    Parameters
    ----------
    see MinerBase

    Attributes
    ----------
    A : dict
        the multiple of the life time initiated as dict
        Since A is different for each load level, the
        load level is taken as dict key (values are rounded to 0 decimals)
    """

    def lifetime_multiple(self, collective):
        """Compute the lifetime multiple for Miner-modified according to Haibach

        Refer to Haibach (2006), p. 291 (3.21-61). The lifetime multiple can be
        expressed in respect to the maximum amplitude so that
        N_lifetime = N_Smax * A

        Parameters
        ----------
        collective : np.ndarray (optional)
            the collective can optionally be input to this function
            if it is not specified, then the attribute is used.
            If no collective exists as attribute (is set during setup)
            then an error is thrown

        Returns
        -------
        lifetime_multiple  : float > 0
            lifetime multiple
            return value is 'inf' if maximum collective amplitude < SD
        """

        rf = collective.rainflow
        s_a = rf.amplitude
        max_amp = s_a.max()

        cycles = rf.cycles

        s_a = s_a / max_amp
        x_D = self.SD / max_amp

        i_full_damage = (s_a >= x_D)
        i_reduced_damage = (s_a < x_D)

        s_full_damage = s_a[i_full_damage]
        s_reduced_damage = s_a[i_reduced_damage]

        n_full_damage = cycles[i_full_damage]
        n_reduced_damage = cycles[i_reduced_damage]

        # first expression of the summation term in the denominator
        sum_1 = np.dot(n_full_damage, (s_full_damage**self.k_1))
        sum_2 = x_D**(1 - self.k_1) * np.dot(n_reduced_damage, (s_reduced_damage**(2 * self.k_1 - 1)))

        return cycles.sum() / (sum_1 + sum_2)

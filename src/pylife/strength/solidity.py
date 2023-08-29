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

"""Small helper functions for fatigue analysis

"""

__author__ = "Cedric Philip Wagner"
__maintainer__ = "Johannes Mueller"


import pandas as pd
import numpy as np

import pylife.stress.collective as CL

@pd.api.extensions.register_series_accessor('solidity')
class SolidityAccessor(CL.LoadHistogram):

    def haibach(self, k):
        return haibach(self, k)

    def fkm(self, k):
        return fkm(self, k)


def haibach(collective, k):
    """Compute solidity according to Haibach

    Refer to:
    Haibach - Betriebsfestigkeit - 3. Auflage (2005) - S.271

    Parameters
    ----------
    collective : np.ndarray
        numpy array of shape (:, 2) where ":" depends on the number of classes
        defined for the rainflow counting

            1. column: class values in ascending order

            2. column: accumulated number of cycles first entry is the total
               number of cycles then in a descending manner till the number of
               cycles of the highest stress class

    k : float
        slope of the S/N curve

    Returns
    -------
    V : np.ndarray (1,)
        Völligkeitswert (solidity)

    """

    S = collective.amplitude
    hi = collective.cycles

    xi = S / S[hi > 0].max()
    V = np.sum((hi * (xi**k)) / hi.sum())

    return V


def fkm(collective, k):
    """Compute solidity according to the FKM guideline (2012)

    Refer to:
    FKM-Richtlinie - 6. Auflage (2012) - S.58 - Gl. (2.4.55) +  Gl. (2.4.55)

    Parameters
    ----------
    collective : np.ndarray
        numpy array of shape (:, 2) where ":" depends on the number of classes
        defined for the rainflow counting

            1. column: class values in ascending order

            2. column: accumulated number of cycles first entry is the total
               number of cycles then in a descending manner till the number of
               cycles of the highest stress class k : float slope of the S/N
               curve

    Returns
    -------
    V : np.ndarray
        Völligkeitswert (solidity)

    """

    V_haibach = haibach(collective, k)
    V = V_haibach**(1./k)

    return V

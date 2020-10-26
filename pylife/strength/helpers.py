# Copyright (c) 2019-2020 - for information on the respective copyright owner
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

import numpy as np

def solidity_haibach(collective, k):
    """Compute solidity according to Haibach

    Refer to:
    Haibach - Betriebsfestigkeit - 3. Auflage (2005) - S.271

    Parameters
    ----------
    collective : np.ndarray
        numpy array of shape (:, 2)
        where ":" depends on the number of classes defined
        for the rainflow counting
            1. column: class values in ascending order
            2. column: accumulated number of cycles
                first entry is the total number of cycles
                then in a descending manner till the
                number of cycles of the highest stress class
    k : float
        slope of the S/N curve

    Returns
    -------
    V : np.ndarray (1,)
        Völligkeitswert (solidity)
    """

    S = collective[:, 0]
    # the accumulated number of cycles
    N_acc = collective[:, 1]

    # the number of cycles for each class
    hi = np.zeros_like(N_acc)
    # get the number of cycles for each class
    for i in range(len(hi)):
        if i == (len(hi) - 1):
            # the last entry is the accumulation of only the last class
            # so it is already the number of cycles of the highest class
            hi[i] = N_acc[i]
        else:
            hi[i] = N_acc[i] - N_acc[i + 1]

    # the selection of S is required so that the highest class
    # with actual counts (hi > 0) is taken as reference for all stress values
    xi = S / S[hi > 0].max()

    V = np.sum((hi * (xi**k)) / hi.sum())

    return V


def solidity_fkm(collective, k):
    """Compute solidity according to the FKM guideline (2012)

    Refer to:
    FKM-Richtlinie - 6. Auflage (2012) - S.58 - Gl. (2.4.55) +  Gl. (2.4.55)

    Parameters
    ----------
    collective : np.ndarray
        numpy array of shape (:, 2)
        where ":" depends on the number of classes defined
        for the rainflow counting
            1. column: class values in ascending order
            2. column: accumulated number of cycles
                first entry is the total number of cycles
                then in a descending manner till the
                number of cycles of the highest stress class
    k : float
        slope of the S/N curve

    Returns
    -------
    V : np.ndarray
        Völligkeitswert (solidity)
    """

    V_haibach = solidity_haibach(collective, k)
    V = V_haibach**(1./k)

    return V


class StressRelations:
    """Namespace for simple relations of stress / amplitude / R-ratio

    Refer to:
    Haibach (2006), p. 21
    """

    @staticmethod
    def get_max_stress_from_amplitude(amplitude, R):
        return 2 * amplitude / (1 - R)

    @staticmethod
    def get_mean_stress_from_amplitude(amplitude, R):
        return amplitude * (1 + R) / (1 - R)

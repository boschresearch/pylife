# Copyright (c) 2019-2024 - for information on the respective copyright owner
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

"""Evaluate finite-life scatter with the pearl chain method.

The module shifts fracture tests to one normalized load level and fits their
failure probabilities in log-cycle direction.
"""

import numpy as np

import pylife.utils.functions as functions
from pylife.utils.probability_data import ProbabilityFit


class PearlChainProbability(ProbabilityFit):
    """Shift fracture data to a normalized load level.

    The pearl chain method moves fracture test results along a Wöhler slope to
    a common load level.  Rossow cumulative failure probabilities are then
    assigned to the sorted shifted cycle numbers and fitted in a probability
    net.  The fit is used to derive the scatter in cycle direction ``TN``.

    Parameters
    ----------
    fractures : pandas.DataFrame
        Fracture test data with ``load`` and ``cycles`` columns.
    slope : float
        Slope used to shift the fracture tests in double-logarithmic Wöhler
        space.  The elementary analyzer passes the fitted regression slope.

    Notes
    -----
    The method is commonly used for DIN 50100-style Wöhler evaluations when
    fracture points at several load levels need to be represented by one
    probability distribution in cycle direction.
    """

    def __init__(self, fractures, slope):
        self._normed_load = fractures.load.mean()
        self._normed_cycles = np.sort(fractures.cycles * ((self._normed_load/fractures.load)**(slope)))

        fp = functions.rossow_cumfreqs(len(self._normed_cycles))
        super().__init__(fp, self._normed_cycles)

    @property
    def normed_load(self):
        """Return the normalized load level."""
        return self._normed_load

    @property
    def normed_cycles(self):
        """Return the cycle numbers shifted to ``normed_load``."""
        return self._normed_cycles

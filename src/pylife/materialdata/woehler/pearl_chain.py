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


import numpy as np

import pylife.utils.functions as functions
from pylife.utils.probability_data import ProbabilityFit


class PearlChainProbability(ProbabilityFit):
    """Shift all the data point to a normalized load level.

    Pearl chain method: consists of shifting the fractured data to a median
    load level.  The shifted data points are assigned to a Rossow failure
    probability.  The scatter in load-cycle direction can be computed from the
    probability net.

    Parameters
    ----------
    fracutres: pd.DataFrame consisting `load` and `cycles`
        The data point of the fractures to be shifted.

    slope: float
        The ``k_1`` slope the data is to be shifted along.

    """

    def __init__(self, fractures, slope):
        self._normed_load = fractures.load.mean()
        self._normed_cycles = np.sort(fractures.cycles * ((self._normed_load/fractures.load)**(slope)))

        fp = functions.rossow_cumfreqs(len(self._normed_cycles))
        super().__init__(fp, self._normed_cycles)

    @property
    def normed_load(self):
        """The normalized (shifted) load level."""
        return self._normed_load

    @property
    def normed_cycles(self):
        """The cycles shifted to the normalized load level."""
        return self._normed_cycles

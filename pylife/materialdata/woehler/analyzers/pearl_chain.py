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


import numpy as np
import scipy.stats as stats

import pylife.utils.functions as functions
from pylife.utils.probability_data import ProbabilityFit


class PearlChainProbability(ProbabilityFit):
    def __init__(self, fractures, slope):
        self._normed_load = fractures.load.mean()
        self._normed_cycles = np.sort(fractures.cycles * ((self._normed_load/fractures.load)**(slope)))

        fp = functions.rossow_cumfreqs(len(self._normed_cycles))
        super().__init__(fp, self._normed_cycles)

    @property
    def normed_load(self):
        return self._normed_load

    @property
    def normed_cycles(self):
        return self._normed_cycles

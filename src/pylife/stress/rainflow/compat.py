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

"""The old pylife-1.x rainflow counting API

In order to not to break existing code, the old pylife-1.x API is still in
place as wrappers around the new API. Using it is strongly discouraged. It will
be deprecated and eventually removed.
"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import pandas as pd

from .threepoint import ThreePointDetector
from .fkm import FKMDetector
from .recorders import FullRecorder


class AbstractRainflowCounter:
    def __init__(self):
        self._recorder = FullRecorder()

    @property
    def loops_from(self):
        return self._recorder.values_from

    @property
    def loops_to(self):
        return self._recorder.values_to

    def residuals(self):
        return self._detector.residuals

    def get_rainflow_matrix(self, bins):
        return self._recorder.histogram_numpy(bins)

    def get_rainflow_matrix_frame(self, bins):
        return pd.DataFrame(self._recorder.histogram(bins))


class RainflowCounterThreePoint(AbstractRainflowCounter):
    def __init__(self):
        super().__init__()
        self._detector = ThreePointDetector(recorder=self._recorder)

    def process(self, samples):
        self._detector.process(samples)
        return self


class RainflowCounterFKM(AbstractRainflowCounter):
    def __init__(self):
        super().__init__()
        self._detector = FKMDetector(recorder=self._recorder)

    def process(self, samples):
        self._detector.process(samples)
        return self

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

"""A module performing rainflow counting

Overview over pyLife's rainflow counting module
-----------------------------------------------

From pyLife-2.0.0 on rainflow counting has been split into two different subtasks:

* hysteresis loop detection, done by a subclass of :class:`.AbstractDetector`.
* hysteresis loop recording, done by a subclass of :class:`.AbstractRecorder`.

That means you can combine detectors and recorders freely. You can choose
recorders and detectors that come with pyLife but also write your own custom
detectors and custom recorders.

Detectors
^^^^^^^^^

Detectors process a one dimensional time signal and detect hysteresis loops in
them. A hysteresis loop consists of the sample point where the hysteresis
starts, and the sample of the turning point where the hysteresis loop starts to
turn back towards the load level of the starting point.

Once the detector has detected such a sample pair that makes a closed
hysteresis loop it reports it to the recorder. All detectors report the load
levels, some detectors also the index to the samples defining the loop limits.

pyLife's detectors are implemented in a way that the samples are
chunkable. That means that you don't need to feed them the complete signal at
once, but you can resume the rainflow analysis later when you have the next
sample chunk.

As of now, pyLife comes with the following detectors:

* :class:`.ThreePointDetector`, classic three point algorithm, reports sample index

* :class:`.FourPointDetector`, recent four point algorithm, reports sample index

* :class:`.FKMDetector`, algorithm described by Clormann & Seeger, recommended by FKM,
  does not report sample index.


Warning
.......

Make sure you don't have any ``NaN`` values in your input signal.  They are
dropped in order to make sure not to miss any hysteresis loops and thus will
render the index invalid.  A warning is issued by :func:`~.find_turns` if
``NaN`` values are dropped.

Recorders
^^^^^^^^^

Recorders are notified by detectors about loops and will process the loop
information as they wish.

As of now, pyLife comes with the following recorders:

* :class:`.LoopValueRecorder`, only records the `from` and `to` values of all
  the closed hysteresis loops.`

* :class:`.FullRecorder`, records additionally to the `from` and `to` values
  also the indices of the loop turning points in the original time series, so
  that additional data like temperature during the loop or dwell times can be
  looked up in the original time series data.

"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

from .general import find_turns, AbstractDetector, AbstractRecorder
from .threepoint import ThreePointDetector
from .fourpoint import FourPointDetector
from .fkm import FKMDetector
from .recorders import LoopValueRecorder, FullRecorder

from .compat import RainflowCounterThreePoint, RainflowCounterFKM

__all__ = [
    'find_turns',
    'AbstractDetector',
    'AbstractRecorder',
    'ThreePointDetector',
    'FourPointDetector',
    'FKMDetector',
    'LoopValueRecorder',
    'FullRecorder',
    'RainflowCounterThreePoint',
    'RainflowCounterFKM',
]

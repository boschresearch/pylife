# Copyright (c) 2019-2026 - for information on the respective copyright owner
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

"""Count rainflow cycles from load-time signals.

The rainflow package separates hysteresis-loop detection from recording.  A
detector implements a counting rule, for example
:class:`pylife.stress.rainflow.FourPointDetector`,
:class:`pylife.stress.rainflow.ThreePointDetector`, or
:class:`pylife.stress.rainflow.FKMDetector`.  A recorder stores the detected
loops, for example :class:`pylife.stress.rainflow.LoopValueRecorder` for loop
loads or :class:`pylife.stress.rainflow.FullRecorder` for loop loads and
sample indices.

The user-facing workflow is:

* Create a recorder.
* Create a detector with that recorder.
* Call ``detector.process(samples)`` once or repeatedly for streamed chunks.
* Read ``recorder.collective`` and pass the resulting load collective to
  fatigue-strength routines such as those in ``pylife.strength.fatigue``.

This separation lets engineers combine different counting standards with the
amount of recorded data they need.  Detectors report closed hysteresis loops to
their recorder.  All detectors report load values; detectors that retain sample
indices also call ``record_index()`` and ``report_chunk()`` so additional time
series information can be recovered from the original signal.

Warnings
--------
Remove ``NaN`` values before rainflow counting.  :func:`find_turns` drops
``NaN`` values to avoid missing loops and issues a warning, but dropping them
can make reported sample indices differ from the original input.

Notes
-----
The available detectors cover common three-point and four-point rainflow
counting variants and the FKM detector after Clormann and Seeger.  Repeated
calls to ``process()`` continue a count, so large signals can be processed in
chunks without changing the final result.
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

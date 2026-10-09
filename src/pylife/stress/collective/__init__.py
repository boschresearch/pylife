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

"""Represent load collectives as explicit loops or binned histograms.

Use :class:`pylife.stress.collective.LoadCollective` when every rainflow loop
shall remain available as an individual row.  Use
:class:`pylife.stress.collective.LoadHistogram` when cycles are already binned
in load range and mean load classes or in ``from`` and ``to`` classes.

Both accessors expose the same engineering quantities: load amplitude, load
range, mean load, upper and lower turning load, stress ratio ``R``, and number
of cycles.  The tutorial at ``docs/tutorials/load_collective.rst`` explains the
trade-off between retaining every loop and aggregating cycles into histogram
classes.
"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

from .load_collective import LoadCollective
from .load_histogram import LoadHistogram

__all__ = [
    "LoadCollective",
    "LoadHistogram"
]

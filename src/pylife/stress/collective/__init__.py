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

"""
There are two ways to deal with a load collective.

* :class:`~pylife.stress.LoadCollective` lets you keep every load hysteresis in
  and then calculate the amplitude, meanstress and damage for each and every
  hyteresis indivudually.

* :class:`~pylife.stress.LoadHistogram` keeps the load information in a binned
  histogram.  That means that not each and every hystresis is stored
  individually but there are bin classes for the load levels the hysteresis is
  originating from and one for the levels the hysteresis is open.

This :doc:`tutorial </tutorials/load_collective>` shows the difference and how
to use the two.

"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

from .load_collective import LoadCollective
from .load_histogram import LoadHistogram

__all__ = [
    "LoadCollective",
    "LoadHistogram"
]

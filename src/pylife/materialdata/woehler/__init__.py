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

"""A module for Wöhler curve fatigue data analysis

Overview
========

:class:`FatigueData` is a signal accessor class to handle fatigue data from a
Wöhler test.  They can be analyzed by several analyzers according to your choice

* :class:`Elementary` only treats the finite zone of the fatigue data and
  calculates the slope and the scatter in lifetime direction.  It is the base
  class for all other analyzers

* :class:`Probit` calculates parameters not calculated by :class:`Elementary`
  using the Probit method.

* :class:`MaxLikeInf` calculates parameters not calculated by :class:`Elementary`
  using the maximum likelihood method.

* :class:`MaxLikeFull` calculates all parameters using the maximum likelihood
  method.  The result from :class:`Elementary` is used as start values.

"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

from .fatigue_data import \
    FatigueData, \
    determine_fractures

from .elementary import Elementary
from .probit import Probit
from .maxlike import MaxLikeInf, MaxLikeFull

__all__ = [
    'FatigueData',
    'determine_fractures',
    'Elementary',
    'Probit',
    'MaxLikeInf',
    'MaxLikeFull',
]

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

"""Provide strength assessment tools for fatigue engineering.

The :mod:`pylife.strength` package collects user-facing accessors and helper
classes for S-N curve based fatigue assessment. It covers Wöhler curve damage
calculation, Miner damage accumulation, mean stress correction, failure
probability evaluation, and the FKM linear and nonlinear assessment
procedures.
"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

from .fatigue import Fatigue
from .failure_probability import FailureProbability

__all__ = [
    'Fatigue',
    'FailureProbability'
]

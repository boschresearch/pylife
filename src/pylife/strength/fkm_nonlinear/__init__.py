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

r"""Provide the FKM nonlinear strength assessment workflow.

The package supports the user workflow from an assessment-parameter series to
an FKM nonlinear lifetime result.  Start with material and component inputs,
derive guideline parameters with ``parameter_calculations``, apply nonlinear
notch approximation laws from ``pylife.materiallaws``, count hysteresis loops
with ``pylife.stress.rainflow.fkm_nonlinear.FKMNonlinearDetector``, evaluate
``P_RAM`` or ``P_RAJ`` damage parameters, and use the damage calculators to
obtain damage sums and lifetimes.
"""

__author__ = "Benjamin Maier"
__maintainer__ = __author__

import pylife.strength.fkm_nonlinear.constants
import pylife.strength.fkm_nonlinear.parameter_calculations
import pylife.strength.fkm_nonlinear.assessment_nonlinear_standard

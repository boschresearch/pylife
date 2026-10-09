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

"""Evaluate Wöhler fatigue tests and return Wöhler curve parameters.

Load fatigue test data into the :attr:`pandas.DataFrame.fatigue_data` accessor,
which validates the mandatory ``load``, ``cycles``, and ``fracture`` columns.
Then evaluate the data with one of the analyzers exported by this package:
``Elementary``, ``Probit``, ``MaxLikeInf``, or ``MaxLikeFull``.  The analyzers
return a :class:`pandas.Series` with Wöhler curve parameters such as ``SD``,
``ND``, ``k_1``, ``TN``, and ``TS``.  That series can be used directly with
:class:`pylife.materiallaws.WoehlerCurve`.

See Also
--------
pylife.materialdata.woehler.FatigueData : Validate measured Wöhler test data.
pylife.materialdata.woehler.Elementary : Estimate the finite-life parameters.
pylife.materialdata.woehler.Probit : Estimate endurance-limit parameters with the Probit method.
pylife.materialdata.woehler.MaxLikeInf : Estimate endurance-limit parameters by maximum likelihood.
pylife.materialdata.woehler.MaxLikeFull : Estimate all Wöhler parameters by maximum likelihood.
pylife.materiallaws.WoehlerCurve : Use evaluated parameters for fatigue assessment.

Notes
-----
Wöhler tests, also called SN tests, record whether a specimen fractured or
survived as a runout at a prescribed load level and cycle count.  The
parameters follow DIN 50100 terminology: ``SD`` is the endurance limit load at
the knee point, ``ND`` is the cycle number at the knee point, ``k_1`` is the
finite-life slope, ``TN`` is the scatter in cycle direction, and ``TS`` is the
scatter in load direction.
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

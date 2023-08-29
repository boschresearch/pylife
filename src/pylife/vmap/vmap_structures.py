# Copyright (c) 2020-2023 - for information on the respective copyright owner
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

__author__ = "Gyöngyvér Kiss"
__maintainer__ = __author__

from enum import Enum


class VariableLocations(Enum):
    NODE = 2
    ELEMENT_NODAL = 6


column_names = {
    'DISPLACEMENT': [['dx', 'dy', 'dz'], VariableLocations.NODE],
    'STRESS_CAUCHY': [['S11', 'S22', 'S33', 'S12', 'S13', 'S23'], VariableLocations.ELEMENT_NODAL],
    'E': [['E11', 'E22', 'E33', 'E12', 'E13', 'E23'], VariableLocations.ELEMENT_NODAL],
}

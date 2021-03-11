# Copyright (c) 2020-2021 - for information on the respective copyright owner
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

"""VMAPImport interface for pyLife
============================

`VMAPImport <https://www.vmap.eu.com/>`_ *is a vendor-neutral standard
for CAE data storage to enhance interoperability in virtual
engineering workflows.*

pyLife supports a growing subset of the VMAPImport standard. That means that
only features relevant for pyLife's addressed real life use cases are
or will be implemented. Probably there are features missing, that are
important for some valid use cases. In that case please file a feature
request at https://github.com/boschresearch/pylife/issues


Reading a VMAPImport file
-------------------

"""
__author__ = "Gyöngyvér Kiss"
__maintainer__ = __author__

import numpy as np
from .exceptions import *


class VMAPIntegrationType:
    def __init__(self, type_name, number_of_points, dimension, offset,
                 abscissas=None, weights=None, subtypes=None):
        self._identifier = None
        self._type_name = type_name
        self._number_of_points = number_of_points
        self._dimension = dimension
        self._offset = offset

        self._abscissas = []
        if abscissas is not None:
            self._abscissas = abscissas

        self._weights = []
        if weights is not None:
            self._weights = weights

        self._subtypes = []
        if subtypes is not None:
            self._subtypes = subtypes

    def set_identifier(self, identifier):
        self._identifier = identifier

    @property
    def attributes(self):
        if self._identifier is None:
            raise (APIUseError("Need to set_identifier() before requesting the attributes."))
        return (self._identifier, self._type_name, self._number_of_points, self._dimension, self._offset,
                np.array(self._abscissas), np.array(self._weights), np.array(self._subtypes))

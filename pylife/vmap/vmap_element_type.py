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


class VMAPElementType:
    def __init__(self, type_name, type_description, number_of_nodes, dimensions, shape_type, interpolation_type,
                 integration_type, number_of_normal_components, number_of_shear_components,
                 connectivity=None, face_connectivity=None):
        self._identifier = None
        self._type_name = type_name
        self._type_description = type_description
        self._number_of_nodes = number_of_nodes
        self._dimension = dimensions
        self._shape_type = shape_type
        self._interpolation_type = interpolation_type
        self._integration_type = integration_type
        self._number_of_normal_components = number_of_normal_components
        self._number_of_shear_components = number_of_shear_components
        self._connectivity = connectivity
        self._face_connectivity = face_connectivity

    def set_identifier(self, identifier):
        self._identifier = identifier

    @property
    def attributes(self):
        if self._identifier is None:
            raise (APIUseError("Need to set_identifier() before requesting the attributes."))
        return (self._identifier, self._type_name, self._type_description, self._number_of_nodes, self._dimension,
                self._shape_type, self._interpolation_type, self._integration_type, self._number_of_normal_components,
                self._number_of_shear_components, np.array(self._connectivity), np.array(self._face_connectivity))

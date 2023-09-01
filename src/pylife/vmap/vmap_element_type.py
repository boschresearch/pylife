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

import numpy as np
import h5py
from h5py.h5t import string_dtype

from .exceptions import *
from .vmap_dataset import VMAPDataset


class VMAPElementType(VMAPDataset):
    def __init__(self, identifier, type_name, type_description, number_of_nodes, dimensions, shape_type, interpolation_type,
                 integration_type, number_of_normal_components, number_of_shear_components,
                 connectivity=None, face_connectivity=None):
        super().__init__(identifier)
        self._type_name = type_name
        self._type_description = type_description
        self._number_of_nodes = number_of_nodes
        self._dimension = dimensions
        self._shape_type = shape_type
        self._interpolation_type = interpolation_type
        self._integration_type = integration_type
        self._number_of_normal_components = number_of_normal_components
        self._number_of_shear_components = number_of_shear_components

        self._connectivity = []
        if connectivity is not None:
            self._connectivity = connectivity

        self._face_connectivity = []
        if face_connectivity is not None:
            self._face_connectivity = face_connectivity

    @property
    def attributes(self):
        if self._identifier is None:
            raise (APIUseError("Need to set_identifier() before requesting the attributes."))
        return (self._identifier, self._type_name, self._type_description, self._number_of_nodes, self._dimension,
                self._shape_type, self._interpolation_type, self._integration_type, self._number_of_normal_components,
                self._number_of_shear_components, np.array(self._connectivity), np.array(self._face_connectivity))

    @property
    def dtype(self):
        dt_type = np.dtype({"names": ["myIdentifier", "myTypeName", "myTypeDescription", "myNumberOfNodes",
                                      "myDimension", "myShapeType", "myInterpolationType", "myIntegrationType",
                                      "myNumberOfNormalComponents", "myNumberOfShearComponents", "myConnectivity",
                                      "myFaceConnectivity"],
                            "formats": ['<i4', string_dtype(), string_dtype(), '<i4', '<i4', '<i4', '<i4', '<i4',
                                        '<i4', '<i4', h5py.special_dtype(vlen=np.dtype('int32')),
                                        h5py.special_dtype(vlen=np.dtype('int32'))]})
        return dt_type

    @property
    def dataset_name(self):
        return 'ELEMENTTYPES'

    @property
    def type_name(self):
        return self._type_name

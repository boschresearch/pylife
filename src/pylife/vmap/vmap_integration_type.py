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

import h5py
from h5py.h5t import string_dtype
import numpy as np

from .exceptions import *
from .vmap_dataset import VMAPDataset


class VMAPIntegrationType(VMAPDataset):
    def __init__(self, identifier, type_name, number_of_points, dimension, offset,
                 abscissas=None, weights=None, subtypes=None):
        super().__init__(identifier)
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

    @property
    def attributes(self):
        if self._identifier is None:
            raise (APIUseError("Need to set_identifier() before requesting the attributes."))
        return (self._identifier, self._type_name, self._number_of_points, self._dimension, self._offset,
                np.array(self._abscissas), np.array(self._weights), np.array(self._subtypes))

    @property
    def dtype(self):
        dt_type = np.dtype({"names": ["myIdentifier", "myTypeName", "myNumberOfPoints", "myDimension",
                                      "myOffset", "myAbscissas", "myWeights", "mySubTypes"],
                            "formats": ['<i4', string_dtype(), '<i4', '<i4', '<f8',
                                        h5py.special_dtype(vlen=np.dtype('float64')),
                                        h5py.special_dtype(vlen=np.dtype('float64')),
                                        h5py.special_dtype(vlen=np.dtype('float32'))]})
        return dt_type

    @property
    def dataset_name(self):
        return 'INTEGRATIONTYPES'

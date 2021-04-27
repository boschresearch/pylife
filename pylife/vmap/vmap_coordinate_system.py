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

__author__ = "Gyöngyvér Kiss"
__maintainer__ = __author__

import numpy as np

from .exceptions import *
from .vmap_dataset import VMAPDataset


class VMAPCoordinateSystem(VMAPDataset):
    def __init__(self, identifier, type_id, reference_points, axis_vectors):
        super().__init__(identifier)
        self._type_id = type_id
        self._reference_points = reference_points
        self._axis_vectors = axis_vectors

    @property
    def attributes(self):
        if self._identifier is None:
            raise (APIUseError("Need to set_identifier() before requesting the attributes."))
        return self._identifier, self._type_id, self._reference_points, self._axis_vectors

    @property
    def dtype(self):
        dt_type = np.dtype({"names": ["myIdentifier", "myType", "myReferencePoint", "myAxisVectors"],
                            "formats": ['<i4', '<i4', ('<f8', (3,)), ('<f8', (9,))]})
        return dt_type

    @property
    def dataset_name(self):
        return 'COORDINATESYSTEM'


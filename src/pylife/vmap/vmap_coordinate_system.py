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

"""Describe coordinate system rows in the VMAP system group."""

__author__ = "Gyöngyvér Kiss"
__maintainer__ = __author__

import numpy as np

from .exceptions import *
from .vmap_dataset import VMAPDataset


class VMAPCoordinateSystem(VMAPDataset):
    """Represent one VMAP coordinate system definition.

    Coordinate systems define reference points and axis vectors that VMAP
    sections or result data can refer to when data is exchanged between CAE
    tools.

    Parameters
    ----------
    identifier : int or None
        VMAP coordinate system identifier.
    type_id : int
        VMAP type code describing the coordinate system kind.
    reference_points : numpy.ndarray
        Three reference point coordinates stored in ``myReferencePoint``.
    axis_vectors : numpy.ndarray
        Nine axis-vector components stored in ``myAxisVectors``.
    """
    def __init__(self, identifier, type_id, reference_points, axis_vectors):
        super().__init__(identifier)
        self._type_id = type_id
        self._reference_points = reference_points
        self._axis_vectors = axis_vectors

    @property
    def attributes(self):
        """Return the coordinate system fields for HDF5 storage.

        Returns
        -------
        tuple
            Identifier, type code, reference point, and axis vectors.

        Raises
        ------
        APIUseError
            If no identifier has been assigned before exporting.
        """
        if self._identifier is None:
            raise (APIUseError("Need to set_identifier() before requesting the attributes."))
        return self._identifier, self._type_id, self._reference_points, self._axis_vectors

    @property
    def dtype(self):
        """Return the compound dtype for the VMAP coordinate system table.

        Returns
        -------
        numpy.dtype
            HDF5 compound dtype with identifier, type, reference point, and
            axis-vector fields.
        """
        dt_type = np.dtype({"names": ["myIdentifier", "myType", "myReferencePoint", "myAxisVectors"],
                            "formats": ['<i4', '<i4', ('<f8', (3,)), ('<f8', (9,))]})
        return dt_type

    @property
    def dataset_name(self):
        """Return the VMAP coordinate system dataset name.

        Returns
        -------
        str
            Name ``COORDINATESYSTEM`` below ``/VMAP/SYSTEM``.
        """
        return 'COORDINATESYSTEM'

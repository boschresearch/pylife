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

"""Describe finite-element type rows in the VMAP system group."""

__author__ = "Gyöngyvér Kiss"
__maintainer__ = __author__

import numpy as np
import h5py
from h5py.h5t import string_dtype

from .exceptions import *
from .vmap_dataset import VMAPDataset


class VMAPElementType(VMAPDataset):
    """Represent one VMAP element type definition.

    VMAP element types describe the topology and interpolation of finite
    elements used by meshes and result fields.  Users meet these definitions
    indirectly when VMAP import or export maps solver-specific elements to the
    neutral VMAP system tables.

    Parameters
    ----------
    identifier : int or None
        VMAP element type identifier.
    type_name : str
        Short VMAP element type name.
    type_description : str
        Human-readable description of the element type.
    number_of_nodes : int
        Number of nodes in one element of this type.
    dimensions : int
        Spatial dimension stored in ``myDimension``.
    shape_type : int
        VMAP shape-type code.
    interpolation_type : int
        VMAP interpolation-type code.
    integration_type : int
        Identifier of the VMAP integration type used by this element type.
    number_of_normal_components : int
        Number of normal tensor components associated with the element type.
    number_of_shear_components : int
        Number of shear tensor components associated with the element type.
    connectivity : list of int, optional
        Node-ordering information stored in ``myConnectivity``.  Default is
        ``None``, which stores an empty list.
    face_connectivity : list of int, optional
        Face node-ordering information stored in ``myFaceConnectivity``.
        Default is ``None``, which stores an empty list.
    """
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
        """Return the element type fields for HDF5 storage.

        Returns
        -------
        tuple
            Values for the VMAP ``ELEMENTTYPES`` row.

        Raises
        ------
        APIUseError
            If no identifier has been assigned before exporting.
        """
        if self._identifier is None:
            raise (APIUseError("Need to set_identifier() before requesting the attributes."))
        return (self._identifier, self._type_name, self._type_description, self._number_of_nodes, self._dimension,
                self._shape_type, self._interpolation_type, self._integration_type, self._number_of_normal_components,
                self._number_of_shear_components, np.array(self._connectivity), np.array(self._face_connectivity))

    @property
    def dtype(self):
        """Return the compound dtype for the VMAP element type table.

        Returns
        -------
        numpy.dtype
            HDF5 compound dtype matching the ``ELEMENTTYPES`` fields.
        """
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
        """Return the VMAP element type dataset name.

        Returns
        -------
        str
            Name ``ELEMENTTYPES`` below ``/VMAP/SYSTEM``.
        """
        return 'ELEMENTTYPES'

    @property
    def type_name(self):
        """Return the short VMAP element type name.

        Returns
        -------
        str
            Element type name passed as ``type_name``.
        """
        return self._type_name

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

"""Describe section rows in the VMAP system group."""

__author__ = "Gyöngyvér Kiss"
__maintainer__ = __author__

from h5py.h5t import string_dtype
import numpy as np

from .exceptions import *
from .vmap_dataset import VMAPDataset


class VMAPSection(VMAPDataset):
    """Represent one VMAP section definition.

    Sections connect mesh parts to material, coordinate-system, integration,
    and thickness definitions in a VMAP file.  They are part of the system
    tables used when exporting solver-independent model metadata.

    Parameters
    ----------
    identifier : int or None
        VMAP section identifier.
    name : str
        Section name stored in ``myName``.
    type_id : int
        VMAP section type code.
    material : int
        Identifier of the referenced VMAP material entry.
    coordinate_system : int
        Identifier of the referenced VMAP coordinate system.
    integration_type : int
        Identifier of the referenced VMAP integration type.
    thickness_type : int
        VMAP thickness type code.
    """
    def __init__(self, identifier, name, type_id, material, coordinate_system, integration_type, thickness_type):
        super().__init__(identifier)
        self._name = name
        self._type_id = type_id
        self._material = material
        self._coordinate_system = coordinate_system
        self._integration_type = integration_type
        self._thickness_type = thickness_type

    @property
    def attributes(self):
        """Return the section fields for HDF5 storage.

        Returns
        -------
        tuple
            Values for the VMAP ``SECTION`` row.

        Raises
        ------
        APIUseError
            If no identifier has been assigned before exporting.
        """
        if self._identifier is None:
            raise (APIUseError("Need to set_identifier() before requesting the attributes."))
        return (self._identifier, self._name, self._type_id, self._material, self._coordinate_system,
                self._integration_type, self._thickness_type)

    @property
    def dtype(self):
        """Return the compound dtype for the VMAP section table.

        Returns
        -------
        numpy.dtype
            HDF5 compound dtype matching the ``SECTION`` fields.
        """
        dt_type = np.dtype({"names": ["myIdentifier", "myName", "myType", "myMaterial", "myCoordinateSystem",
                                      "myIntegrationType", "myThicknessType"],
                            "formats": ['<i4', string_dtype(), '<i4', '<i4', '<i4', '<i4', '<i4']})
        return dt_type

    @property
    def dataset_name(self):
        """Return the VMAP section dataset name.

        Returns
        -------
        str
            Name ``SECTION`` below ``/VMAP/SYSTEM``.
        """
        return 'SECTION'

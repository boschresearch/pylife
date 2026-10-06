# Copyright (c) 2020-2026 - for information on the respective copyright owner
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

"""Describe unit-system rows in the VMAP system group."""

__author__ = "Gyöngyvér Kiss"
__maintainer__ = __author__

import numpy as np
from h5py.h5t import string_dtype

from .exceptions import *
from .vmap_dataset import VMAPDataset


class VMAPUnit(VMAPDataset):
    """Represent one VMAP unit-system definition.

    Unit-system entries describe how quantities in a VMAP file map to SI units.
    pyLife writes and reads them as part of the standard ``UNITSYSTEM`` table.

    Parameters
    ----------
    identifier : int or None
        VMAP unit identifier.
    si_scale : float
        Multiplicative scale factor to convert values to SI units.
    si_shift : float
        Additive shift used during conversion to SI units.
    unit_symbol : str
        Unit symbol, for example ``MPa``.
    unit_quantity : str
        Physical quantity represented by the unit entry.
    """
    def __init__(self, identifier, si_scale, si_shift, unit_symbol, unit_quantity):
        super().__init__(identifier)
        self._si_scale = si_scale
        self._si_shift = si_shift
        self._unit_symbol = unit_symbol
        self._unit_quantity = unit_quantity

    @property
    def attributes(self):
        """Return the unit-system fields for HDF5 storage.

        Returns
        -------
        tuple
            Values for the VMAP ``UNITSYSTEM`` row.

        Raises
        ------
        APIUseError
            If no identifier has been assigned before exporting.
        """
        if self._identifier is None:
            raise (APIUseError("Need to set_identifier() before requesting the attributes."))
        return self._identifier, self._si_scale, self._si_shift, self._unit_symbol, self._unit_quantity

    @property
    def dtype(self):
        """Return the compound dtype for the VMAP unit-system table.

        Returns
        -------
        numpy.dtype
            HDF5 compound dtype matching the ``UNITSYSTEM`` fields.
        """
        dt_type = np.dtype({"names": ["myIdentifier", "mySIScale", "mySIShift", "myUnitSymbol", "myUnitQuantity"],
                            "formats": ['<i4', '<f8', '<f8', string_dtype(), string_dtype()]})
        return dt_type

    @property
    def dataset_name(self):
        """Return the VMAP unit-system dataset name.

        Returns
        -------
        str
            Name ``UNITSYSTEM`` below ``/VMAP/SYSTEM``.
        """
        return 'UNITSYSTEM'

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
from h5py.h5t import string_dtype

from .exceptions import *
from .vmap_dataset import VMAPDataset


class VMAPUnitSystem(VMAPDataset):
    def __init__(self, si_scale, si_shift, unit_symbol, unit_quantity):
        super().__init__()
        self._si_scale = si_scale
        self._si_shift = si_shift
        self._unit_symbol = unit_symbol
        self._unit_quantity = unit_quantity

    @property
    def attributes(self):
        if self._identifier is None:
            raise (APIUseError("Need to set_identifier() before requesting the attributes."))
        return self._identifier, self._si_scale, self._si_shift, self._unit_symbol, self._unit_quantity

    @property
    def dtype(self):
        dt_type = np.dtype({"names": ["myIdentifier", "mySIScale", "mySIShift", "myUnitSymbol", "myUnitQuantity"],
                            "formats": ['<i4', '<f4', '<f4', string_dtype(), string_dtype()]})
        return dt_type

    @property
    def dataset_name(self):
        return 'UNITSYSTEM'

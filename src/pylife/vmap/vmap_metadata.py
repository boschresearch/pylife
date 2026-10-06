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

"""Describe metadata rows in the VMAP system group."""

__author__ = "Gyöngyvér Kiss"
__maintainer__ = __author__

from h5py.h5t import string_dtype
from .vmap_dataset import VMAPDataset


class VMAPMetadata(VMAPDataset):
    """Represent one textual VMAP metadata key-value entry.

    VMAP metadata stores descriptive information about the neutral CAE result
    file.  This helper represents one two-column row used by the exporter.

    Parameters
    ----------
    key : str
        Metadata key stored in the first column.
    value : str
        Metadata value stored in the second column.
    """
    def __init__(self, key, value):
        self._column_0 = key
        self._column_1 = value

    @property
    def attributes(self):
        """Return the metadata key and value for HDF5 storage.

        Returns
        -------
        tuple
            Metadata key and metadata value.
        """
        return self._column_0, self._column_1

    @property
    def dtype(self):
        """Return the string dtype for the metadata dataset.

        Returns
        -------
        h5py.Datatype
            Variable-length HDF5 string dtype.
        """
        return string_dtype()

    @property
    def dataset_name(self):
        """Return the VMAP metadata dataset name.

        Returns
        -------
        str
            Name ``METADATA`` below ``/VMAP/SYSTEM``.
        """
        return 'METADATA'

    @property
    def compound_dataset(self):
        """Return whether metadata is stored as a compound dataset.

        Returns
        -------
        bool
            ``False`` because metadata uses a plain string dataset.
        """
        return False

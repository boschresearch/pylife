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

"""Provide the common interface for VMAP system-table dataset rows."""

__author__ = "Gyöngyvér Kiss"
__maintainer__ = __author__

from abc import ABC, abstractmethod


class VMAPDataset(ABC):
    """Define shared behavior for VMAP system dataset entries.

    VMAP stores many standard building blocks, such as element types and unit
    systems, as HDF5 datasets below ``/VMAP/SYSTEM``.  Subclasses provide the
    dataset-specific fields and HDF5 dtype used by the exporter.

    Parameters
    ----------
    identifier : int or None
        VMAP identifier of the row.  Use ``None`` for objects that receive
        their identifier later via :meth:`set_identifier`.
    """
    def __init__(self, identifier):
        self._identifier = identifier

    def set_identifier(self, identifier):
        """Set the VMAP identifier used in the exported system table.

        Parameters
        ----------
        identifier : int
            Identifier assigned to this VMAP dataset entry.
        """
        self._identifier = identifier

    @property
    @abstractmethod
    def attributes(self):
        """Return the values written as one row of the VMAP dataset.

        Returns
        -------
        tuple
            Field values in the order described by :attr:`dtype`.
        """
        pass

    @property
    @abstractmethod
    def dtype(self):
        """Return the NumPy dtype used for the HDF5 dataset.

        Returns
        -------
        numpy.dtype
            Compound or scalar dtype expected by the VMAP system table.
        """
        pass

    @property
    @abstractmethod
    def dataset_name(self):
        """Return the VMAP system dataset name.

        Returns
        -------
        str
            Dataset name below :attr:`group_path`.
        """
        pass

    @property
    def group_path(self):
        """Return the standard VMAP group that contains this dataset.

        Returns
        -------
        str
            HDF5 group path ``/VMAP/SYSTEM``.
        """
        return '/VMAP/SYSTEM'

    @property
    def compound_dataset(self):
        """Return whether the VMAP entry is stored as a compound dataset.

        Returns
        -------
        bool
            ``True`` for the standard structured datasets.
        """
        return True

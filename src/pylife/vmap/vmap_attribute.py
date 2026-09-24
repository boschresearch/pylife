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

"""Represent HDF5 attributes attached to VMAP groups and datasets."""

__author__ = "Gyöngyvér Kiss"
__maintainer__ = __author__


class VMAPAttribute:
    """Store one VMAP HDF5 attribute name-value pair.

    VMAP files use HDF5 attributes for small pieces of metadata on the
    standard groups and datasets.  pyLife uses this helper while creating or
    inspecting those attributes.

    Parameters
    ----------
    attribute_name : str
        Name of the HDF5 attribute.
    attribute_value : object
        Value stored under ``attribute_name``.
    """
    def __init__(self, attribute_name, attribute_value):
        self._name = attribute_name
        self._value = attribute_value

    @property
    def name(self):
        """Return the HDF5 attribute name.

        Returns
        -------
        str
            Name passed as ``attribute_name``.
        """
        return self._name

    @property
    def value(self):
        """Return the HDF5 attribute value.

        Returns
        -------
        object
            Value passed as ``attribute_value``.
        """
        return self._value

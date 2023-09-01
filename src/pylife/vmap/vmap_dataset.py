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

from abc import ABC, abstractmethod


class VMAPDataset(ABC):
    """
    Base class of VMAP dataset types
    """
    def __init__(self, identifier):
        self._identifier = identifier

    def set_identifier(self, identifier):
        self._identifier = identifier

    @property
    @abstractmethod
    def attributes(self):
        pass

    @property
    @abstractmethod
    def dtype(self):
        pass

    @property
    @abstractmethod
    def dataset_name(self):
        pass

    @property
    def group_path(self):
        return '/VMAP/SYSTEM'

    @property
    def compound_dataset(self):
        return True

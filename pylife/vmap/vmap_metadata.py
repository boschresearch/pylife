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

from h5py.h5t import string_dtype
from .vmap_dataset import VMAPDataset


class VMAPMetadata(VMAPDataset):
    def __init__(self, key, value):
        self._column_0 = key
        self._column_1 = value

    @property
    def attributes(self):
        return self._column_0, self._column_1

    @property
    def dtype(self):
        return string_dtype()

    @property
    def dataset_name(self):
        return 'METADATA'

    @property
    def compound_dataset(self):
        return False

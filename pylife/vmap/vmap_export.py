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

'''VMAPImport interface for pyLife
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

'''
__author__ = "Gyöngyvér Kiss"
__maintainer__ = __author__

import numpy as np
import pandas as pd

import h5py

from .exceptions import *


class VMAPExport:

    def __init__(self, filename):
        self._file = h5py.File(filename, 'w')

    def create_vmap_groups(self):
        vmap_group = self._file.create_group('VMAP')
        vmap_group.create_group('GEOMETRY')
        vmap_group.create_group('MATERIAL')
        vmap_group.create_group('SYSTEM')
        vmap_group.create_group('VARIABLES')
        # d1 = np.random.random(size=(1000, 20))
        # d2 = np.random.random(size=(1000, 200))
        # self._file.create_dataset('dataset_3', data=d1)
        # g1 = self._file.create_group('group1')
        # g1.create_dataset('dataset_2', data=d2)
        # self._file.close()

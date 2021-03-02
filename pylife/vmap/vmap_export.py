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
        self._create_groups()

    def _create_groups(self):
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

    def create_geometry(self, geometry_name, mesh):
        geometry = self.create_geometry_groups(geometry_name)
        points_group = geometry.get('POINTS')
        node_ids_info = mesh.groupby('node_id').first()
        self.create_points_ids_dataset(node_ids_info, points_group)
        self.create_points_conn_dataset(mesh, points_group)
        self._file.close()
        return self

    def create_geometry_groups(self, geometry_name):
        geometry_group = self._file["/VMAP/GEOMETRY"]
        geometry = geometry_group.create_group(geometry_name)
        geometry.create_group('ELEMENTS')
        geometry.create_group('GEOMETRYSETS')
        geometry.create_group('POINTS')
        return geometry

    def create_points_ids_dataset(self, node_ids_info, point_group):
        # element_ids = mesh_index.get_level_values('element_id')
        # mesh_index = mesh.index
        # node_ids = mesh_index.get_level_values('node_id').drop_duplicates()
        # mesh_columns = mesh.columns
        # node_ids_info = mesh.groupby('node_id').first()
        point_group.create_dataset('MYIDENTIFIERS', data=node_ids_info.index)
        point_group.create_dataset('MYCOORDINATES', data=node_ids_info[['x', 'y', 'z']].values)
        return self

    def create_points_conn_dataset(self, mesh, point_group):
        coordinates = mesh['x']
        return self

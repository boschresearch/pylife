# Copyright (c) 2019-2020 - for information on the respective copyright owner
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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import numpy as np
import pandas as pd

import h5py

from .exceptions import *


class VMAP:
    _column_names = {
        'DISPLACEMENT': ['dx', 'dy', 'dz'],
        'STRESS_CAUCHY': ['S11', 'S22', 'S33', 'S12', 'S13', 'S23'],
        'EVOL': ['V_e']
    }

    def __init__(self, filename):
        self._file = h5py.File(filename, 'r')

    def nodes(self, geometry):
        return pd.DataFrame(
            self._file["/VMAP/GEOMETRY/%s/POINTS/MYCOORDINATES" % geometry][()],
            columns=['x', 'y', 'z'],
            index=self._node_index(geometry)
        )

    def mesh_index(self, geometry):
        connectivity = self._element_connectivity(geometry).connectivity
        length = sum([el.shape[0] for el in connectivity])
        index_np = np.empty((2, length), dtype=np.int64)

        i = 0
        for eid, nds in connectivity.iteritems():
            i_next = i + nds.shape[0]
            index_np[0, i:i_next] = eid
            index_np[1, i:i_next] = nds
            i = i_next

        return pd.MultiIndex.from_arrays(index_np, names=['element_id', 'node_id'])

    def mesh_coords(self, geometry):
        return pd.DataFrame(index=self.mesh_index(geometry)).join(self.nodes(geometry))

    def variable(self, geometry, state, varname):
        var_tree = self._file["/VMAP/VARIABLES/%s/%s/%s" % (state, geometry, varname)]
        return pd.DataFrame(
            data=var_tree['MYVALUES'][()],
            columns=self._column_names[varname],
            index=self._make_index(var_tree, geometry)
        )

    def _element_connectivity(self, geometry):
        elements = self._file['/VMAP/GEOMETRY/'+geometry+'/ELEMENTS/MYELEMENTS']
        element_connectivity = elements['myIdentifier', 'myConnectivity'][:, 0]
        element_ids = [elid for elid, _ in element_connectivity]
        connectivity = [conn for _, conn in element_connectivity]
        return pd.DataFrame(data={'element_id': element_ids, 'connectivity': connectivity}).set_index('element_id')

    def _node_index(self, geometry):
        return pd.Index(
            self._file["/VMAP/GEOMETRY/%s/POINTS/MYIDENTIFIERS" % geometry][:, 0],
            name='node_id'
        )

    def _make_index(self, var_tree, geometry):
        location = var_tree.attrs['MYLOCATION']
        if location == 2:
            return self._var_node_index(var_tree)
        if location == 3:
            return self._var_element_index(var_tree)
        if location == 6:
            return self._var_element_nodal_index(var_tree, geometry)
        raise FeatureNotSupportedError("Unsupported value location, sorry\nSupported: NODE, ELEMENT, ELEMENT NODAL")

    def _var_node_index(self, var_tree):
        return pd.Index(var_tree['MYGEOMETRYIDS'][:, 0], name='node_id')

    def _var_element_index(self, var_tree):
        return pd.Index(var_tree['MYGEOMETRYIDS'][:, 0], name='element_id')

    def _var_element_nodal_index(self, var_tree, geometry):
        mesh_index_frame = self.mesh_index(geometry).to_frame(index=False)
        index_frame = pd.DataFrame(var_tree['MYGEOMETRYIDS'], columns=['element_id'])

        return (index_frame
                .merge(mesh_index_frame)
                .set_index(['element_id', 'node_id'])
                .index)

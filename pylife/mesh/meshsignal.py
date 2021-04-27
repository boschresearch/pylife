# Copyright (c) 2019-2021 - for information on the respective copyright owner
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
'''
Helper to process mesh based data
=================================

Data that is distributed over a geometrical body, e.g. a stress tensor
distribution on a component, is usually transported via a mesh. The
meshes are a list of items (e.g. nodes or elements of a FEM mesh),
each being described by the geometrical coordinates and the local data
values, like for example the local stress tensor data.

In a plain mesh (see :class:`PlainMeshAccessor`) there is no further
relation between the items is known, whereas a complete FEM mesh (see
:class:`MeshAccessor`) there is also information on the connectivity
of the nodes and elements.

Examples
--------
Read in a mesh from a vmap file:


>>> df = (vm = pylife.vmap.VMAPImport('demos/plate_with_hole.vmap')
             .make_mesh('1', 'STATE-2')
             .join_variable('STRESS_CAUCHY')
             .join_variable('DISPLACEMENT')
             .to_frame())
>>> df.head()
                            x         y    z        S11       S22  S33        S12  S13  S23        dx        dy   dz
element_id node_id
1          1734     14.897208  5.269875  0.0  27.080811  6.927080  0.0 -13.687358  0.0  0.0  0.005345  0.000015  0.0
           1582     14.555333  5.355806  0.0  28.319006  1.178649  0.0 -10.732705  0.0  0.0  0.005285  0.000003  0.0
           1596     14.630658  4.908741  0.0  47.701195  5.512213  0.0 -17.866833  0.0  0.0  0.005376  0.000019  0.0
           4923     14.726271  5.312840  0.0  27.699907  4.052865  0.0 -12.210032  0.0  0.0  0.005315  0.000009  0.0
           4924     14.592996  5.132274  0.0  38.010101  3.345431  0.0 -14.299768  0.0  0.0  0.005326  0.000013  0.0

Get the coordinates of the mesh.

>>> df.plain_mesh.coordinates.head()
                            x         y    z
element_id node_id
1          1734     14.897208  5.269875  0.0
           1582     14.555333  5.355806  0.0
           1596     14.630658  4.908741  0.0
           4923     14.726271  5.312840  0.0
           4924     14.592996  5.132274  0.0

Now the same with a 2D mesh:

>>> df.drop(columns=['z']).plain_mesh.coordinates.head()
                            x         y
element_id node_id
1          1734     14.897208  5.269875
           1582     14.555333  5.355806
           1596     14.630658  4.908741
           4923     14.726271  5.312840
           4924     14.592996  5.132274
'''

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import pandas as pd
from pylife import signal


@pd.api.extensions.register_dataframe_accessor("plain_mesh")
class PlainMeshAccessor(signal.PylifeSignal):
    '''DataFrame accessor to access plain 2D and 3D mesh data, i.e. without connectivity

    Raises
    ------
    AttributeError
        if at least one of the columns `x`, `y` is missing

    Notes
    -----
    The PlainMeshAccessor describes meshes whose only geometrical
    information is the coordinates of the nodes or elements. Unlike
    :class:`MeshAccessor` they don't know about connectivity, not even
    about elements and nodes.

    See also
    --------
    :class:`MeshAccessor`: accesses meshes with connectivity information
    :func:`pandas.api.extensions.register_dataframe_accessor()`: concept of DataFrame accessors
    '''
    def _validate(self, obj, validator):
        self._coord_keys = ['x', 'y']
        validator.fail_if_key_missing(obj, self._coord_keys)
        if 'z' in obj.columns:
            self._coord_keys.append('z')

    @property
    def coordinates(self):
        '''Returns the coordinate colums of the accessed DataFrame

        Returns
        -------
        coordinates : pandas.DataFrame
            The coordinates `x`, `y` and if 3D `z` of the accessed mesh
        '''
        return self._obj[self._coord_keys]


@pd.api.extensions.register_dataframe_accessor("mesh")
class MeshAccessor(PlainMeshAccessor):

    '''DataFrame accessor to access FEM mesh data (2D and 3D)

    Raises
    ------
    AttributeError
        if at least one of the columns `x`, `y` is missing
    AttributeError
        if the index of the DataFrame is not a two level MultiIndex
        with the names `node_id` and `element_id`

    Notes
    -----
    The MeshAccessor describes how we expect FEM data to look like. It
    consists of nodes identified by `node_id` and elements identified
    by `element_id`. A node playing a role in several elements and an
    element consists of several nodes. So in the DataFrame a `node_id`
    can appear multiple times (for each element, the node is playing a
    role in). Likewise each `element_id` appears multiple times (for
    each node the element consists of).

    The combination `node_id`:`element_id` however, is unique. So the
    table is indexed by a :class:`pandas.MultiIndex` with the level
    names `node_id`, `element_id`.

    See also
    --------
    :class:`PlainMeshAccessor`: accesses meshes without connectivity information
    :func:`pandas.api.extensions.register_dataframe_accessor()`: concept of DataFrame accessors

    Examples
    --------
    For an example see :mod:`meshplot`.
    '''
    def _validate(self, obj, validator):
        super(MeshAccessor, self)._validate(obj, validator)
        if set(obj.index.names) != set(['element_id', 'node_id']):
            raise AttributeError("A mesh needs a pd.MultiIndex with the names `element_id` and `node_id`")

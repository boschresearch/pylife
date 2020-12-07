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
'''VMAP interface for pyLife
============================

`VMAP <https://www.vmap.eu.com/>`_ *is a vendor-neutral standard
for CAE data storage to enhance interoperability in virtual
engineering workflows.*

pyLife supports a growing subset of the VMAP standard. That means that
only features relevant for pyLife's addressed real life use cases are
or will be implemented. Probably there are features missing, that are
important for some valid use cases. In that case please file a feature
request at https://github.com/boschresearch/pylife/issues


Reading a VMAP file
-------------------

The most common use case is to get the element nodal stress tensor for
a certain geometry ``1`` and a certain load state ``STATE-2`` out of the
vmap file. The vmap interface provides you the nodal geometry (node
coordinates), the mesh connectivity index and the field variables.

Steps
.....

Open the vmap file

>>> import pylife.vmap
>>>
>>> vm = pylife.vmap.VMAP('demos/plate_with_hole.vmap')

Read in the geometry along with the mesh connectivity index for geometry ``1``

>>> mesh_coords = vm.mesh_coords('1')
>>> mesh_coords.head(12)
                            x         y    z
element_id node_id
1          1734     14.897208  5.269875  0.0
           1582     14.555333  5.355806  0.0
           1596     14.630658  4.908741  0.0
           4923     14.726271  5.312840  0.0
           4924     14.592996  5.132274  0.0
           4925     14.763933  5.089308  0.0
2          1730     14.184048  4.278657  0.0
           1601     13.863862  4.341518  0.0
           576      14.066248  4.005180  0.0
           4926     14.023954  4.310088  0.0
           4927     13.965055  4.173349  0.0
           4928     14.125148  4.141919  0.0

Read in and the variable ``STRESS_CAUCHY`` of geometry ``1`` and load state
``STATE-2``.

>>> stress = vm.variable('1', 'STATE-2', 'STRESS_CAUCHY')
>>> stress.head(12)
                          S11       S22  S33        S12  S13  S23
element_id node_id
1          1734     27.080811  6.927080  0.0 -13.687358  0.0  0.0
           1582     28.319006  1.178649  0.0 -10.732705  0.0  0.0
           1596     47.701195  5.512213  0.0 -17.866833  0.0  0.0
           4923     27.699907  4.052865  0.0 -12.210032  0.0  0.0
           4924     38.010101  3.345431  0.0 -14.299768  0.0  0.0
           4925     37.391003  6.219646  0.0 -15.777096  0.0  0.0
2          1730     26.149452  9.321044  0.0 -16.487396  0.0  0.0
           1601     23.324717  1.107580  0.0 -13.920287  0.0  0.0
           576      35.546097  4.856305  0.0 -14.435610  0.0  0.0
           4926     24.737087  5.214312  0.0 -15.203842  0.0  0.0
           4927     29.435410  2.981942  0.0 -14.177948  0.0  0.0
           4928     30.847776  7.088674  0.0 -15.461502  0.0  0.0

Next you join those together to one `pandas.DataFrame`

>>> df = mesh_coords.join(stress)
>>> df.head(12)
                            x         y    z        S11       S22  S33        S12  S13  S23
element_id node_id
1          1734     14.897208  5.269875  0.0  27.080811  6.927080  0.0 -13.687358  0.0  0.0
           1582     14.555333  5.355806  0.0  28.319006  1.178649  0.0 -10.732705  0.0  0.0
           1596     14.630658  4.908741  0.0  47.701195  5.512213  0.0 -17.866833  0.0  0.0
           4923     14.726271  5.312840  0.0  27.699907  4.052865  0.0 -12.210032  0.0  0.0
           4924     14.592996  5.132274  0.0  38.010101  3.345431  0.0 -14.299768  0.0  0.0
           4925     14.763933  5.089308  0.0  37.391003  6.219646  0.0 -15.777096  0.0  0.0
2          1730     14.184048  4.278657  0.0  26.149452  9.321044  0.0 -16.487396  0.0  0.0
           1601     13.863862  4.341518  0.0  23.324717  1.107580  0.0 -13.920287  0.0  0.0
           576      14.066248  4.005180  0.0  35.546097  4.856305  0.0 -14.435610  0.0  0.0
           4926     14.023954  4.310088  0.0  24.737087  5.214312  0.0 -15.203842  0.0  0.0
           4927     13.965055  4.173349  0.0  29.435410  2.981942  0.0 -14.177948  0.0  0.0
           4928     14.125148  4.141919  0.0  30.847776  7.088674  0.0 -15.461502  0.0  0.0


Of course you can also do this in one step

>>> vm = pylife.vmap.VMAP('demos/plate_with_hole.vmap')
>>> df = (vm.mesh_coords('1')
>>>       .join(vm.variable('1', 'STATE-2', 'STRESS_CAUCHY'))
>>>       .join(vm.variable('1', 'STATE-2', 'DISPLACEMENT')))
>>> df.head(12)
                            x         y    z        S11       S22  S33        S12  S13  S23        dx        dy   dz
element_id node_id
1          1734     14.897208  5.269875  0.0  27.080811  6.927080  0.0 -13.687358  0.0  0.0  0.005345  0.000015  0.0
           1582     14.555333  5.355806  0.0  28.319006  1.178649  0.0 -10.732705  0.0  0.0  0.005285  0.000003  0.0
           1596     14.630658  4.908741  0.0  47.701195  5.512213  0.0 -17.866833  0.0  0.0  0.005376  0.000019  0.0
           4923     14.726271  5.312840  0.0  27.699907  4.052865  0.0 -12.210032  0.0  0.0  0.005315  0.000009  0.0
           4924     14.592996  5.132274  0.0  38.010101  3.345431  0.0 -14.299768  0.0  0.0  0.005326  0.000013  0.0
...                       ...       ...  ...        ...       ...  ...        ...  ...  ...       ...       ...  ...
4770       3812    -13.189782 -5.691876  0.0  36.527439  2.470588  0.0 -14.706686  0.0  0.0 -0.005300  0.000027  0.0
           12418   -13.560289 -5.278386  0.0  32.868889  3.320898  0.0 -14.260107  0.0  0.0 -0.005444  0.000002  0.0
           14446   -13.673285 -5.569107  0.0  34.291058  3.642457  0.0 -13.836027  0.0  0.0 -0.005404  0.000009  0.0
           14614   -13.389065 -5.709927  0.0  36.063541  2.828889  0.0 -13.774759  0.0  0.0 -0.005330  0.000022  0.0
           14534   -13.276068 -5.419206  0.0  33.804211  2.829817  0.0 -14.580153  0.0  0.0 -0.005371  0.000014  0.0


Supported features
------------------

So far the following data can be read from a vmap file

Geometry
........
* node positions
* node element index

Field variables
...............
Any field variables can be read and joined to the node element index
from the following locations:

* element
* node
* element nodal

In particular, field variables at intergration point location *cannot*
cannot be read, as that would require extrapolating them to the node
positions. This functionality is not available in pyLife.

'''
__author__ = "Johannes Mueller"
__amintainer__ = __author__

import numpy as np
import pandas as pd

import h5py

from .exceptions import *


class VMAP:
    '''The interface class to access a vmap file

    Parameters
    ----------
    filename : string
        The path to the vmap file to be read

    Raises
    ------
    Exception
        If the file cannot be read an exception is raised.
        So far any exception from the ``h5py`` module is passed through.
    '''

    _column_names = {
        'DISPLACEMENT': ['dx', 'dy', 'dz'],
        'STRESS_CAUCHY': ['S11', 'S22', 'S33', 'S12', 'S13', 'S23'],
        'E': ['E11', 'E22', 'E33', 'E12', 'E13', 'E23'],
    }

    def __init__(self, filename):
        self._file = h5py.File(filename, 'r')

    def geometries(self):
        '''Retuns a list of geometry strings of geometries present in the vmap data
        '''
        return self._file["/VMAP/GEOMETRY"].keys()

    def states(self):
        '''Retuns a list of state strings of states present in the vmap data
        '''
        return self._file["/VMAP/VARIABLES/"].keys()

    def nodes(self, geometry):
        '''Retrieves the node positions

        Parameters
        ----------
        geometry : string
            The geometry defined in the vmap file

        Returns
        -------
        node_positions : DataFrame
            a DataFrame with the node numbers as index and the columns 'x', 'y' and 'z' for the
            node coordinates.

        Raises
        ------
        KeyError
            if the geometry is not found of if the vmap file is corrupted
        '''
        return pd.DataFrame(
            self._file["/VMAP/GEOMETRY/%s/POINTS/MYCOORDINATES" % geometry][()],
            columns=['x', 'y', 'z'],
            index=self._node_index(geometry)
        )

    def mesh_index(self, geometry):
        '''Retrieves the node element index

        Parameters
        ----------
        geometry : string
            The geometry defined in the vmap file

        Returns
        -------
        node_element_index : MultiIndex
            a MultiIndex with the node ids and element ids

        Raises
        ------
        KeyError
            if the geometry is not found of if the vmap file is corrupted
        '''
        connectivity = self._element_connectivity(geometry).connectivity
        length = sum([el.shape[0] for el in connectivity])
        index_np = np.empty((2, length), dtype=np.int64)

        i = 0
        for element_id, node_ids in connectivity.iteritems():
            i_next = i + node_ids.shape[0]
            index_np[0, i:i_next] = element_id
            index_np[1, i:i_next] = node_ids
            i = i_next

        return pd.MultiIndex.from_arrays(index_np, names=['element_id', 'node_id'])

    def mesh_coords(self, geometry):
        '''Retrieves the mesh with its coordinates

        Parameters
        ----------
        geometry : string
            The geometry defined in the vmap file

        Returns
        -------
        mesh_data : DataFrame
            a DataFrame with the element nodal index and the columns 'x', 'y' and 'z' for the
            node coordinates.

        Raises
        ------
        KeyError
            if the geometry is not found of if the vmap file is corrupted
        '''
        return pd.DataFrame(index=self.mesh_index(geometry)).join(self.nodes(geometry))

    def variable(self, geometry, state, varname, column_names=None):
        '''Retrieves a field output variable

        Parameters
        ----------
        geometry : string
            The geometry defined in the vmap file
        state : string
            The load state of which the field variable is to be read
        varname : string
            The name of the field variables
        column_names : list of string, optional
            The names of the columns names to be used in the DataFrame
            If not provided, it will be chosen according to the list shown below.
            The length of the list must match the dimension of the variable.

        Returns
        -------
        variable_values : DataFrame
            a DataFrame with the value of the field variable.
            The column names are given by ``column_names`` or by the list below.
            The index is depending on the variable's location either the element id,
            the node id or the element node index.

        Raises
        ------
        KeyError
            if the geometry, state or varname is not found of if the vmap file is corrupted
        KeyError
            if there are no column names given and known for the variable.
        ValueError
            if the length of the column_names does not match the dimension of the variable

        Notes
        -----
        If the ``column_names`` argument is not provided the following column names are chosen

        * 'DISPLACEMENT': ``['dx', 'dy', 'dz']``
        * 'STRESS_CAUCHY': ``['S11', 'S22', 'S33', 'S12', 'S13', 'S23']``
        * 'E': ``['E11', 'E22', 'E33', 'E12', 'E13', 'E23']``

        If that fails a ``KeyError`` exception is risen.

        TODO
        ----
        Write a more central document about pyLife's column names.
        '''
        if column_names is None:
            try:
                column_names = self._column_names[varname]
            except KeyError:
                raise KeyError("No column name for variable %s. Please povide with column_names parameter." % varname)

        var_tree = self._file["/VMAP/VARIABLES/%s/%s/%s" % (state, geometry, varname)]
        var_dimension = var_tree.attrs['MYDIMENSION']
        if len(column_names) != var_dimension:
            raise ValueError("Length of column name list (%d) does not match variable dimension (%d)."
                             % (len(column_names), var_dimension))

        return pd.DataFrame(
            data=var_tree['MYVALUES'][()],
            columns=column_names,
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

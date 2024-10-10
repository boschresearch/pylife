# Copyright (c) 2019-2023 - for information on the respective copyright owner
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
'''Helper to process mesh based data

Data that is distributed over a geometrical body, e.g. a stress tensor
distribution on a component, is usually transported via a mesh. The
meshes are a list of items (e.g. nodes or elements of a FEM mesh),
each being described by the geometrical coordinates and the local data
values, like for example the local stress tensor data.

In a plain mesh (see :class:`PlainMesh`) there is no further
relation between the items is known, whereas a complete FEM mesh (see
:class:`Mesh`) there is also information on the connectivity
of the nodes and elements.

Examples
--------
Read in a mesh from a vmap file:

>>> from pylife.vmap import VMAPImport
>>> df = (
...     VMAPImport('demos/plate_with_hole.vmap')
...     .make_mesh('1', 'STATE-2')
...     .join_coordinates()
...     .join_variable('STRESS_CAUCHY')
...     .join_variable('DISPLACEMENT')
...     .to_frame()
... )
>>> df.head()
                            x         y    z  ...        dx        dy   dz
element_id node_id                            ...
1          1734     14.897208  5.269875  0.0  ...  0.005345  0.000015  0.0
           1582     14.555333  5.355806  0.0  ...  0.005285  0.000003  0.0
           1596     14.630658  4.908741  0.0  ...  0.005376  0.000019  0.0
           4923     14.726271  5.312840  0.0  ...  0.005315  0.000009  0.0
           4924     14.592996  5.132274  0.0  ...  0.005326  0.000013  0.0
<BLANKLINE>
[5 rows x 12 columns]

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

import numpy as np
import pandas as pd
from pylife import PylifeSignal


@pd.api.extensions.register_dataframe_accessor("plain_mesh")
class PlainMesh(PylifeSignal):
    '''DataFrame accessor to access plain 2D and 3D mesh data, i.e. without connectivity

    Raises
    ------
    AttributeError
        if at least one of the columns `x`, `y` is missing

    Notes
    -----
    The PlainMesh describes meshes whose only geometrical
    information is the coordinates of the nodes or elements. Unlike
    :class:`Mesh` they don't know about connectivity, not even
    about elements and nodes.

    See also
    --------
    :class:`Mesh`: accesses meshes with connectivity information
    :func:`pandas.api.extensions.register_dataframe_accessor()`: concept of DataFrame accessors
    '''
    def _validate(self):
        self._coord_keys = ['x', 'y']
        self.fail_if_key_missing(self._coord_keys)
        if 'z' in self._obj.columns:
            self._coord_keys.append('z')
        self._cached_dimensions = None

    @property
    def dimensions(self):
        """The dimensions of the mesh (2 for 2D and 3 for 3D)

        Note
        ----
        If all the coordinates in z-direction are equal the mesh is considered 2D.
        """
        if self._cached_dimensions is not None:
            return self._cached_dimensions

        if len(self._coord_keys) == 2 or (self._obj.z == self._obj.z.iloc[0]).all():
            self._cached_dimensions = 2
        else:
            self._cached_dimensions = 3

        return self._cached_dimensions

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
class Mesh(PlainMesh):

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
    The Mesh describes how we expect FEM data to look like. It
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
    :class:`PlainMesh`: accesses meshes without connectivity information
    :func:`pandas.api.extensions.register_dataframe_accessor()`: concept of DataFrame accessors

    Examples
    --------
    For an example see :mod:`hotspot`.
    '''
    def _validate(self):
        super()._validate()
        self._cached_element_groups = None
        if not set(self._obj.index.names).issuperset(['element_id', 'node_id']):
            raise AttributeError(
                "A mesh needs a pd.MultiIndex with the names `element_id` and `node_id`"
            )


    @property
    def connectivity(self):
        """The connectivity of the mesh."""
        return self._element_groups['node_id'].apply(np.hstack)

    def vtk_data(self):
        """Make VTK data structure easily plot the mesh with pyVista.

        Returns
        -------
        offsets : ndarray
            An empty numpy array as ``pyVista.UnstructuredGrid()`` still
            demands the argument for the offsets, even though VTK>9 does not
            accept it.
        cells : ndarray
            The location of the cells describing the points in a way
            ``pyVista.UnstructuredGrid()`` needs it
        cell_types : ndarray
            The VTK code for the cell types (see https://github.com/Kitware/VTK/blob/master/Common/DataModel/vtkCellType.h)
        points : ndarray
            The coordinates of the cell points

        Notes
        -----
        This is a convenience function to easily plot a 3D mesh with
        pyVista. It prepares a data structure which can be passed to
        ``pyVista.UnstructuredGrid()``

        Example
        -------
        >>> import pyvista as pv
        >>> from pylife.vmap import VMAPImport
        >>> df = (
        ...     VMAPImport('demos/plate_with_hole.vmap')
        ...     .make_mesh('1', 'STATE-2')
        ...     .join_coordinates()
        ...     .join_variable('STRESS_CAUCHY')
        ...     .to_frame()
        ... )

        >>> grid = pv.UnstructuredGrid(*df.mesh.vtk_data())
        >>> plotter = pv.Plotter(window_size=[1920, 1080])
        >>> plotter.add_mesh(grid, scalars=df.groupby('element_id')['S11'].mean().to_numpy())  # doctest: +SKIP
        >>> plotter.show()  # doctest: +SKIP

        Note the `*` that needs to be added when calling ``pv.UnstructuredGrid()``.
        """
        def choose_element_types_dict():
            return self._element_types_3d if self.dimensions == 3 else self._element_types_2d

        def cells_with_lengths(index, connectivity):
            def locs(nodes):
                return np.array(list(map(index.get_loc, nodes)))

            cells = connectivity.apply(locs)
            return np.array([nd for cell in cells.values for nd in np.insert(cell, 0, cell.shape[0])])

        def calc_cells():
            element_types_dict = choose_element_types_dict()

            groups = self._element_groups['node_id']
            connectivity = groups.apply(np.hstack)
            count = groups.count()

            for total_num, (first_order_num, _) in element_types_dict.items():
                choice = count == total_num
                connectivity[choice] = connectivity[choice].apply(lambda nds: nds[:first_order_num])

            return connectivity, count.apply(lambda c: element_types_dict[c][1]).to_numpy()

        def first_order_points(connectivity):
            points = self._obj.groupby('node_id', sort=True).first()[self._coord_keys]
            nodes = pd.Series([nd for element in connectivity.values for nd in element], name='node_id').unique()
            selection = points.index.isin(nodes)
            return points[selection]

        connectivity, cell_types = calc_cells()
        points = first_order_points(connectivity)
        cells = cells_with_lengths(points.index, connectivity)

        return cells, cell_types, points.to_numpy()

    _element_types_2d = {
        # Resolve number of nodes of element to number of first order nodes and vtk element type
        # see https://kitware.github.io/vtk-examples/site/VTKFileFormats/
        # and https://github.com/Kitware/VTK/blob/master/Common/DataModel/vtkCellType.h
        # number_of_nodes: (number_of_first_order_nodes, vtk_element_type)
        3: (3, 5),  # tri lin
        6: (3, 5),  # tri quad
        4: (4, 9),  # squ lin
        8: (4, 9),  # squ quad
    }
    _element_types_3d = {
        # Resolve number of nodes of element to number of first order nodes and vtk element type
        # see https://kitware.github.io/vtk-examples/site/VTKFileFormats/
        # and https://github.com/Kitware/VTK/blob/master/Common/DataModel/vtkCellType.h
        # number_of_nodes: (number_of_first_order_nodes, vtk_element_type)
        4: (4, 10),   # tet lin
        6: (6, 13),   # wedge lin
        8: (8, 12),   # hex lin
        10: (4, 10),  # tet quad
        15: (6, 26),  # tet wedge
        20: (8, 12),  # hex quad
    }

    @property
    def _element_groups(self):
        if self._cached_element_groups is None:
            self._cached_element_groups = self._obj.reset_index().groupby('element_id')
        return self._cached_element_groups

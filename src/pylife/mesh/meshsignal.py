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
"""Process coordinate-based mesh data stored in pandas objects.

Mesh data describe quantities distributed over a geometrical body, for
example stress in MPa on finite-element nodes.  A plain mesh is any
:class:`pandas.DataFrame` with coordinate columns ``x`` and ``y`` and
optionally ``z``.  A full finite-element mesh additionally carries a
:class:`pandas.MultiIndex` with the levels ``element_id`` and ``node_id`` so
that node-element connectivity is known.

The module registers the ``plain_mesh`` and ``mesh`` DataFrame accessors used
by mesh algorithms throughout pyLife.  Coordinates are expected in mm.

See Also
--------
pylife.mesh.meshsignal.PlainMesh : Access point-cloud coordinates without
    connectivity.
pylife.mesh.meshsignal.Mesh : Access node-element connectivity in finite-
    element meshes.

Examples
--------
Create a connected triangular mesh and read its coordinate columns.

>>> import pandas as pd
>>> index = pd.MultiIndex.from_tuples(
...     [(1, 10), (1, 11), (1, 12)], names=["element_id", "node_id"]
... )
>>> mesh = pd.DataFrame({"x": [0.0, 1.0, 0.0], "y": [0.0, 0.0, 1.0]}, index=index)
>>> mesh.mesh.coordinates
                       x    y
element_id node_id
1          10       0.0  0.0
           11       1.0  0.0
           12       0.0  1.0
"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import numpy as np
import pandas as pd
from pylife import PylifeSignal


@pd.api.extensions.register_dataframe_accessor("plain_mesh")
class PlainMesh(PylifeSignal):
    """Access plain 2D and 3D point-cloud mesh data.

    A plain mesh represents independent points with coordinates in mm.  It
    does not encode element connectivity; therefore the DataFrame index is
    preserved but not interpreted by the accessor.

    Signal contract:

    * ``x`` : Coordinate in mm along the global x-axis.
    * ``y`` : Coordinate in mm along the global y-axis.
    * ``z`` : Optional coordinate in mm along the global z-axis.  If missing,
      the mesh is treated as two-dimensional.

    Parameters
    ----------
    pandas_obj : pandas.DataFrame
        DataFrame carrying the coordinate columns.  The index may be any
        pandas index and is preserved by returned objects.

    Raises
    ------
    AttributeError
        If at least one of the columns ``x`` or ``y`` is missing.

    See Also
    --------
    pylife.mesh.meshsignal.Mesh : Access meshes with node-element
        connectivity.
    pandas.api.extensions.register_dataframe_accessor : Register pandas
        DataFrame accessors.

    Notes
    -----
    If column ``z`` exists but all values are equal, the mesh is considered
    two-dimensional for :attr:`dimensions`.
    """
    def _validate(self):
        self._coord_keys = ['x', 'y']
        self.fail_if_key_missing(self._coord_keys)
        if 'z' in self._obj.columns:
            self._coord_keys.append('z')
        self._cached_dimensions = None

    @property
    def dimensions(self):
        """Return the spatial dimension of the mesh.

        Returns
        -------
        int
            Spatial dimension, either ``2`` for a planar mesh or ``3`` for a
            mesh with varying ``z`` coordinates.  Coordinates are interpreted
            in mm.

        Notes
        -----
        If column ``z`` is missing or all values in column ``z`` are equal,
        the mesh is considered two-dimensional.
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
        """Return the coordinate columns of the accessed DataFrame.

        Returns
        -------
        pandas.DataFrame
            Coordinate columns ``x`` and ``y`` and, for three-dimensional
            meshes, ``z``.  Values are coordinates in mm and the returned
            DataFrame carries the same index as the accessed object.
        """
        return self._obj[self._coord_keys]


@pd.api.extensions.register_dataframe_accessor("mesh")
class Mesh(PlainMesh):

    """Access connected finite-element mesh data.

    A connected mesh stores one row per node occurrence in an element.  The
    DataFrame must have coordinate columns ``x`` and ``y`` in mm and may have
    ``z`` for three-dimensional meshes.  Its index must contain the levels
    ``element_id`` and ``node_id``; together they identify a unique row.

    Signal contract:

    * ``element_id`` : Index level identifying the finite element.
    * ``node_id`` : Index level identifying the node used by the element.
    * ``x`` : Coordinate in mm along the global x-axis.
    * ``y`` : Coordinate in mm along the global y-axis.
    * ``z`` : Optional coordinate in mm along the global z-axis.

    Parameters
    ----------
    pandas_obj : pandas.DataFrame
        DataFrame with a :class:`pandas.MultiIndex` containing ``element_id``
        and ``node_id`` and with coordinate columns in mm.

    Raises
    ------
    AttributeError
        If at least one of the columns ``x`` or ``y`` is missing.
    AttributeError
        If the index of the DataFrame does not contain the levels ``node_id``
        and ``element_id``.

    See Also
    --------
    pylife.mesh.meshsignal.PlainMesh : Access meshes without connectivity
        information.
    pandas.api.extensions.register_dataframe_accessor : Register pandas
        DataFrame accessors.

    Notes
    -----
    A node can occur in several elements, and each element contains several
    nodes.  The combination of ``element_id`` and ``node_id`` is expected to
    be unique.  pyLife functions preserve this full mesh index unless a
    method explicitly returns node-averaged data indexed only by ``node_id``.

    Examples
    --------
    >>> import pandas as pd
    >>> index = pd.MultiIndex.from_tuples(
    ...     [(1, 10), (1, 11), (1, 12)], names=["element_id", "node_id"]
    ... )
    >>> mesh = pd.DataFrame({"x": [0.0, 1.0, 0.0], "y": [0.0, 0.0, 1.0]}, index=index)
    >>> mesh.mesh.connectivity.loc[1].tolist()
    [10, 11, 12]
    """
    def _validate(self):
        super()._validate()
        self._cached_element_groups = None
        if not set(self._obj.index.names).issuperset(['element_id', 'node_id']):
            raise AttributeError(
                "A mesh needs a pd.MultiIndex with the names `element_id` and `node_id`"
            )


    @property
    def connectivity(self):
        """Return the node connectivity of each element.

        Returns
        -------
        pandas.Series
            Series indexed by ``element_id``.  Each value is a
            :class:`numpy.ndarray` with the ``node_id`` values that define the
            element connectivity.
        """
        return self._element_groups['node_id'].apply(np.hstack)

    def vtk_data(self):
        """Create VTK arrays for plotting the mesh with pyVista.

        Returns
        -------
        cells : numpy.ndarray
            The location of the cells describing the points in a way
            ``pyVista.UnstructuredGrid`` expects it.
        cell_types : numpy.ndarray
            The VTK element type codes for the cells.
        points : numpy.ndarray
            Coordinates in mm of the cell points.  Rows are sorted by
            ``node_id`` and columns are ``x``, ``y`` and, for 3D meshes,
            ``z``.

        Notes
        -----
        This is a convenience method for visualization.  It prepares data that
        can be passed as ``pv.UnstructuredGrid(*mesh.mesh.vtk_data())``.  For
        quadratic elements, only the first-order corner nodes are used because
        the VTK element codes selected here describe first-order geometry.

        Examples
        --------
        >>> import pandas as pd
        >>> index = pd.MultiIndex.from_tuples(
        ...     [(1, 10), (1, 11), (1, 12)], names=["element_id", "node_id"]
        ... )
        >>> mesh = pd.DataFrame({"x": [0.0, 1.0, 0.0], "y": [0.0, 0.0, 1.0]}, index=index)
        >>> cells, cell_types, points = mesh.mesh.vtk_data()
        >>> cells.tolist()
        [3, 0, 1, 2]
        >>> cell_types.tolist()
        [5]
        >>> points.tolist()
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
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

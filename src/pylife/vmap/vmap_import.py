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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import numpy as np
import pandas as pd

import h5py

from .exceptions import *
from . import vmap_structures


class VMAPImport:
    """The interface class to import a vmap file

    Parameters
    ----------
    filename : string
        The path to the vmap file to be read

    Raises
    ------
    Exception
        if the file cannot be read an exception is raised.
        So far any exception from the ``h5py`` module is passed through.
    """

    def __init__(self, filename):
        self._file = h5py.File(filename, 'r')
        self._mesh = None
        self._geometry = None
        self._state = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def geometries(self):
        """Returns a list of geometry strings of geometries present in the vmap data
        """
        return self._file["/VMAP/GEOMETRY"].keys()

    def states(self):
        """Returns a list of state strings of states present in the vmap data
        """
        return self._file["/VMAP/VARIABLES/"].keys()

    def node_sets(self, geometry):
        """Returns a list of the node_sets present in the vmap file
        """
        return self._geometry_sets(geometry, 'nsets').keys()

    def element_sets(self, geometry):
        """Returns a list of the element_sets present in the vmap file
        """
        return self._geometry_sets(geometry, 'elsets').keys()

    def nodes(self, geometry):
        """Retrieves the node positions

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
        """
        return pd.DataFrame(
            self._file["/VMAP/GEOMETRY/%s/POINTS/MYCOORDINATES" % geometry][()],
            columns = ['x', 'y', 'z'],
            index = self._node_index(geometry)
        )

    def make_mesh(self, geometry, state=None):
        """Makes the initial mesh

        Parameters
        ----------
        geometry : string
            The geometry defined in the vmap file
        state : string, optional
            The load state of which the field variable is to be read.
            If not given, the state must be defined in ``join_variable()``.

        Returns
        -------
        self

        Raises
        ------
        KeyError
            if the ``geometry`` is not found of if the vmap file is corrupted
        KeyError
            if the ``node_set`` or ``element_set`` is not found in the geometry.
        APIUseError
            if both, a ``node_set`` and an ``element_set`` are given

        Notes
        -----
        This methods defines the initial mesh to which coordinate data can be joined by ``join_coordinates()``
        and field variables can be joined by ``join_variable()``

        Examples
        --------
        Get the mesh data with the coordinates of geometry '1' and the stress tensor of 'STATE-2'

        >>> (
        ...     VMAPImport('demos/plate_with_hole.vmap')
        ...     .make_mesh('1', 'STATE-2')
        ...     .join_coordinates()
        ...     .join_variable('STRESS_CAUCHY')
        ...     .to_frame()
        ... )
                                    x         y    z  ...        S12  S13  S23
        element_id node_id                            ...
        1          1734     14.897208  5.269875  0.0  ... -13.687358  0.0  0.0
                   1582     14.555333  5.355806  0.0  ... -10.732705  0.0  0.0
                   1596     14.630658  4.908741  0.0  ... -17.866833  0.0  0.0
                   4923     14.726271  5.312840  0.0  ... -12.210032  0.0  0.0
                   4924     14.592996  5.132274  0.0  ... -14.299768  0.0  0.0
        ...                       ...       ...  ...  ...        ...  ...  ...
        4770       3812    -13.189782 -5.691876  0.0  ... -14.706686  0.0  0.0
                   12418   -13.560289 -5.278386  0.0  ... -14.260107  0.0  0.0
                   14446   -13.673285 -5.569107  0.0  ... -13.836027  0.0  0.0
                   14614   -13.389065 -5.709927  0.0  ... -13.774759  0.0  0.0
                   14534   -13.276068 -5.419206  0.0  ... -14.580153  0.0  0.0
        <BLANKLINE>
        [37884 rows x 9 columns]

        """
        self._mesh = pd.DataFrame(index=self._mesh_index(geometry))
        self._geometry = geometry
        self._state = state
        return self

    def filter_node_set(self, node_set):
        """Filters a node set out of the current mesh

        Parameters
        ----------
        node_set : string
            The node set defined in the vmap file as geometry set

        Returns
        -------
        self

        Raises
        ------
        APIUseError
            if the mesh has not been initialized using ``make_mesh()``
        """
        self._check_mesh_for_filtering()
        node_set_ids = self._node_set_ids(self._geometry, node_set)
        self._mesh = self._mesh[self._mesh.index.isin(node_set_ids, level='node_id')]
        return self

    def filter_element_set(self, element_set):
        """Filters a node set out of the current mesh

        Parameters
        ----------
        element_set : string, optional
            The element set defined in the vmap file as geometry set

        Returns
        -------
        self

        Raises
        ------
        APIUseError
            if the mesh has not been initialized using ``make_mesh()``
        """
        self._check_mesh_for_filtering()
        element_set_ids = self._element_set_ids(self._geometry, element_set)
        self._mesh = self._mesh[self._mesh.index.isin(element_set_ids, level='element_id')]
        return self

    def _check_mesh_for_filtering(self):
        if self._mesh is None:
            raise APIUseError("Need to make_mesh() before filtering node or element sets.")

    def join_coordinates(self):
        """Join the coordinates of the predefined geometry in the mesh

        Returns
        -------
        self

        Raises
        ------
        APIUseError
            if the mesh has not been initialized using ``make_mesh()``

        Examples
        --------
        Receive the mesh with the node coordinates

        >>> VMAPImport('demos/plate_with_hole.vmap').make_mesh('1').join_coordinates().to_frame()
                                    x         y    z
        element_id node_id
        1          1734     14.897208  5.269875  0.0
                   1582     14.555333  5.355806  0.0
                   1596     14.630658  4.908741  0.0
                   4923     14.726271  5.312840  0.0
                   4924     14.592996  5.132274  0.0
        ...                       ...       ...  ...
        4770       3812    -13.189782 -5.691876  0.0
                   12418   -13.560289 -5.278386  0.0
                   14446   -13.673285 -5.569107  0.0
                   14614   -13.389065 -5.709927  0.0
                   14534   -13.276068 -5.419206  0.0
        <BLANKLINE>
        [37884 rows x 3 columns]
        """
        if self._mesh is None:
            raise APIUseError("Need to make_mesh() before joining the coordinates.")
        self._mesh = self._mesh.join(self.nodes(self._geometry))
        return self

    def to_frame(self):
        """Returns the mesh and resets the mesh

        Returns
        -------
        mesh : DataFrame
            The mesh data joined so far

        Raises
        ------
        APIUseError
            if there is no mesh present, i.e. make_mesh() has not been called yet
            or the mesh has been reset in the meantime.

        Notes
        -----
        This method resets the mesh, i.e. ``make_mesh()`` must be called again in order to
        fetch more mesh data in another mesh.
        """
        if self._mesh is None:
            raise(APIUseError("Need to make_mesh() before requesting a resulting frame."))
        ret = self._mesh
        self._mesh = None
        return ret

    def variables(self, geometry, state):
        """Ask for available variables for a certain geometry and state.

        Parameters
        ----------
        geometry : string
            Name of the geometry
        state : string
            Name of the state

        Returns
        -------
        variables : list
            List of available variable names for the geometry state combination

        Raises
        ------
        KeyError
            if the geometry state combination is not available.
        """
        self._fail_if_unknown_geometry(geometry)
        self._fail_if_unknown_state(state)
        if geometry not in self._file['/VMAP/VARIABLES/%s' % state].keys():
            raise KeyError("Geometry '%s' not available in state '%s'." % (geometry, state))
        return list(self._file['/VMAP/VARIABLES/%s/%s' % (state, geometry)].keys())

    def join_variable(self, var_name, state=None, column_names=None):
        """Joins a field output variable to the mesh

        Parameters
        ----------
        var_name : string
            The name of the field variables
        state : string, opional
            The load state of which the field variable is to be read
            If not given, the last defined state, either defined in ``make_mesh()``
            or defeined in ``join_variable()`` is used.
        column_names : list of string, optional
            The names of the columns names to be used in the DataFrame
            If not provided, it will be chosen according to the list shown below.
            The length of the list must match the dimension of the variable.

        Returns
        -------
        self

        Raises
        ------
        APIUseError
            if the mesh has not been initialized using ``make_mesh()``
        KeyError
            if the geometry, state or varname is not found of if the vmap file is corrupted
        KeyError
            if there are no column names given and known for the variable.
        ValueError
            if the length of the column_names does not match the dimension of the variable

        Notes
        -----
        The mesh must be initialized with ``make_mesh()``. The final DataFrame can be retrieved with ``to_frame()``.

        If the ``column_names`` argument is not provided the following column names are chosen

        * 'DISPLACEMENT': ``['dx', 'dy', 'dz']``
        * 'STRESS_CAUCHY': ``['S11', 'S22', 'S33', 'S12', 'S13', 'S23']``
        * 'E': ``['E11', 'E22', 'E33', 'E12', 'E13', 'E23']``

        If that fails a ``KeyError`` exception is risen.

        Examples
        --------
        Receiving the 'DISPLACEMENT' of 'STATE-1' , the stress and strain tensors of 'STATE-2'

        >>> (
        ...     VMAPImport('demos/plate_with_hole.vmap')
        ...     .make_mesh('1')
        ...     .join_variable('DISPLACEMENT', 'STATE-1')
        ...     .join_variable('STRESS_CAUCHY', 'STATE-2')
        ...     .join_variable('E').to_frame()
        ... )
                             dx   dy   dz        S11  ...  E33       E12  E13  E23
        element_id node_id                            ...
        1          1734     0.0  0.0  0.0  27.080811  ...  0.0 -0.000169  0.0  0.0
                   1582     0.0  0.0  0.0  28.319006  ...  0.0 -0.000133  0.0  0.0
                   1596     0.0  0.0  0.0  47.701195  ...  0.0 -0.000221  0.0  0.0
                   4923     0.0  0.0  0.0  27.699907  ...  0.0 -0.000151  0.0  0.0
                   4924     0.0  0.0  0.0  38.010101  ...  0.0 -0.000177  0.0  0.0
        ...                 ...  ...  ...        ...  ...  ...       ...  ...  ...
        4770       3812     0.0  0.0  0.0  36.527439  ...  0.0 -0.000182  0.0  0.0
                   12418    0.0  0.0  0.0  32.868889  ...  0.0 -0.000177  0.0  0.0
                   14446    0.0  0.0  0.0  34.291058  ...  0.0 -0.000171  0.0  0.0
                   14614    0.0  0.0  0.0  36.063541  ...  0.0 -0.000171  0.0  0.0
                   14534    0.0  0.0  0.0  33.804211  ...  0.0 -0.000181  0.0  0.0
        <BLANKLINE>
        [37884 rows x 15 columns]

        TODO
        ----
        Write a more central document about pyLife's column names.
        """
        if self._mesh is None:
            raise APIUseError("Need to make_mesh() before joining a variable.")
        state = self._update_state(state)
        self._fail_if_geometry_unknown_in_state(self._geometry, state)
        self._state = state
        variable_data = (pd.DataFrame(index=self._mesh.index)
                         .join(self._variable(self._geometry, self._state, var_name, column_names)))
        self._mesh = self._mesh.join(variable_data.loc[self._mesh.index])
        return self

    def _update_state(self, state):
        if state is None:
            state = self._state
        if state is None:
            raise APIUseError("No state name given.\n"
                              "Must be either given in make_mesh() or in join_variable() as optional state argument.")
        return state

    def _fail_if_unknown_geometry(self, geometry):
        if geometry not in self.geometries():
            raise KeyError("Geometry '%s' not found. Available geometries: [%s]."
                           % (geometry, ', '.join(["'"+g+"'" for g in self.geometries()])))

    def _fail_if_unknown_state(self, state):
        if state not in self.states():
            raise KeyError("State '%s' not found. Available states: [%s]."
                           % (state, ', '.join(["'"+s+"'" for s in self.states()])))

    def _fail_if_geometry_unknown_in_state(self, geometry, state):
        self._fail_if_unknown_geometry(geometry)
        self._fail_if_unknown_state(state)
        if geometry not in self._file['/VMAP/VARIABLES/%s' % state].keys():
            raise KeyError("Geometry '%s' not available in state '%s'." % (geometry, state))

    def _mesh_index(self, geometry):
        self._fail_if_unknown_geometry(geometry)
        connectivity = self._element_connectivity(geometry).connectivity
        length = sum([el.shape[0] for el in connectivity])
        index_np = np.empty((2, length), dtype=np.int64)

        i = 0
        for element_id, node_ids in connectivity.items():
            i_next = i + node_ids.shape[0]
            index_np[0, i:i_next] = element_id
            index_np[1, i:i_next] = node_ids
            i = i_next

        return pd.MultiIndex.from_arrays(index_np, names=['element_id', 'node_id'])

    def _variable(self, geometry, state, var_name, column_names):
        if column_names is None:
            try:
                column_names = vmap_structures.column_names[var_name][0]
            except KeyError:
                raise KeyError("No column name for variable %s. Please provide with column_names parameter." % var_name)

        state_group = self._file["/VMAP/VARIABLES/%s/%s" % (state, geometry)]
        if var_name not in state_group.keys():
            raise KeyError("Variable '%s' not found in geometry '%s', '%s'."
                           % (var_name, geometry, state))
        var_tree = state_group[var_name]
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
        elements = self._file['/VMAP/GEOMETRY/' + geometry + '/ELEMENTS/MYELEMENTS']
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
        mesh_index_frame = self._mesh_index(geometry).to_frame(index=False)
        index_frame = pd.DataFrame(var_tree['MYGEOMETRYIDS'], columns=['element_id'])

        return (index_frame
                .merge(mesh_index_frame)
                .set_index(['element_id', 'node_id'])
                .index)

    def _geometry_sets(self, geometry, set_type):
        s_type = 0 if set_type == 'nsets' else 1
        geometry_sets = self._file["/VMAP/GEOMETRY/%s/GEOMETRYSETS" % geometry]
        return {
            gset.attrs['MYSETNAME'].decode('UTF-8'): gset['MYGEOMETRYSETDATA'][()]
            for (_, gset) in geometry_sets.items() if gset.attrs['MYSETTYPE'] == s_type
        }

    def try_get_geometry_set(self, geometry_name, geometry_set_name):
        try:
            geometry_set = self._file["/VMAP/GEOMETRY/%s/GEOMETRYSETS/%s/MYGEOMETRYSETDATA"
                                      % (geometry_name, geometry_set_name)]
            return pd.Index(geometry_set[()].flatten())
        except KeyError:
            return None

    def try_get_vmap_object(self, group_full_path):
        try:
            return self._file[group_full_path]
        except KeyError:
            return None

    def _node_set_ids(self, geometry, node_set):
        try:
            return self._geometry_sets(geometry, 'nsets')[node_set].T[0]
        except KeyError:
            raise KeyError("Node set '%s' not found in geometry '%s'" % (node_set, geometry))

    def _element_set_ids(self, geometry, element_set):
        try:
            return self._geometry_sets(geometry, 'elsets')[element_set].T[0]
        except KeyError:
            raise KeyError("Element set '%s' not found in geometry '%s'" % (element_set, geometry))

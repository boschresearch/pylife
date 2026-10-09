# Copyright (c) 2020-2026 - for information on the respective copyright owner
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

"""Read VMAP files into pandas meshes.

VMAP stores finite-element geometry and result data in HDF5 groups.  This
module maps VMAP node and element blocks, load states, and variable fields to
the pandas ``DataFrame`` representation used by pyLife mesh accessors.  The
resulting index uses ``element_id`` and ``node_id`` so that coordinates,
stresses, displacements, and other fields can be processed by pyLife.
"""
__author__ = "Johannes Mueller"
__maintainer__ = __author__

import numpy as np
import pandas as pd

import h5py

from .exceptions import *
from . import vmap_structures


class VMAPImport:
    """Read VMAP geometry and result variables into a pandas mesh.

    Use this class to bring finite-element results from a ``.vmap`` file into
    pyLife.  A VMAP file contains one or more geometries, load states, and
    variables.  A geometry defines node coordinates and element connectivity, a
    state represents a load step or increment, and variables are result fields such
    as ``DISPLACEMENT``, ``STRESS_CAUCHY``, or ``E``.

    The usual workflow is to open the file, inspect available geometries and
    states, create a mesh with :meth:`make_mesh`, add coordinates with
    :meth:`join_coordinates`, add one or more variables with :meth:`join_variable`,
    and finish with :meth:`to_frame`.  The returned ``DataFrame`` is indexed by
    ``element_id`` and ``node_id`` where element-nodal data is present and can be
    used by the ``pylife.mesh`` accessors.  Coordinate units, stress units, and all
    other physical units follow the originating finite-element model; pyLife does
    not convert them during import.

    Parameters
    ----------
    filename : str
        Path to the VMAP file to read.

    Raises
    ------
    Exception
        Raised by :mod:`h5py` when the file cannot be opened or read.

    See Also
    --------
    pylife.vmap.VMAPExport : Write pyLife mesh data and variables to VMAP.

    Examples
    --------
    .. code-block:: python

        import pylife.vmap as vmap

        mesh = (
            vmap.VMAPImport("demos/plate_with_hole.vmap")
            .make_mesh("1", "STATE-2")
            .join_coordinates()
            .join_variable("STRESS_CAUCHY")
            .to_frame()
        )
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
        """List geometry names stored in the VMAP file.

        Returns
        -------
        KeysView
            View of the VMAP geometry names.  Pass one of these names to
            :meth:`make_mesh`, :meth:`nodes`, :meth:`node_sets`, or
            :meth:`element_sets`.
        """
        return list(self._file["/VMAP/GEOMETRY"].keys())

    def states(self):
        """List state names stored in the VMAP file.

        Returns
        -------
        KeysView
            View of the VMAP state names.  A state represents a load step or increment
            and can be passed to :meth:`make_mesh`, :meth:`variables`, or
            :meth:`join_variable`.
        """
        return list(self._file["/VMAP/VARIABLES/"].keys())

    def node_sets(self, geometry):
        """List node set names for a geometry.

        Parameters
        ----------
        geometry : str
            Name of the VMAP geometry whose node sets are requested.

        Returns
        -------
        dict_keys
            View of node set names defined for ``geometry``.  Use these names with
            :meth:`filter_node_set` to restrict the current mesh to selected nodes.

        Raises
        ------
        KeyError
            Raised when ``geometry`` is not present or the VMAP geometry set data is
            malformed.
        """
        return self._geometry_sets(geometry, 'nsets').keys()

    def element_sets(self, geometry):
        """List element set names for a geometry.

        Parameters
        ----------
        geometry : str
            Name of the VMAP geometry whose element sets are requested.

        Returns
        -------
        dict_keys
            View of element set names defined for ``geometry``.  Use these names with
            :meth:`filter_element_set` to restrict the current mesh to selected
            elements.

        Raises
        ------
        KeyError
            Raised when ``geometry`` is not present or the VMAP geometry set data is
            malformed.
        """
        return self._geometry_sets(geometry, 'elsets').keys()

    def nodes(self, geometry):
        """Retrieve the node positions.

        Parameters
        ----------
        geometry : str
            Name of the VMAP geometry whose node coordinates are requested.

        Returns
        -------
        pandas.DataFrame
            Node coordinate table indexed by ``node_id`` with columns ``x``, ``y``,
            and ``z``.  Coordinate values use the length unit of the source finite-
            element model, commonly millimetres.

        Raises
        ------
        KeyError
            Raised when ``geometry`` is not present or the VMAP file does not contain
            the expected coordinate datasets.
        """
        return pd.DataFrame(
            self._file["/VMAP/GEOMETRY/%s/POINTS/MYCOORDINATES" % geometry][()],
            columns = ['x', 'y', 'z'],
            index = self._node_index(geometry)
        )

    def make_mesh(self, geometry, state=None):
        """Create the working mesh for a geometry and optional state.

        The working mesh contains the element-to-node connectivity as a pandas
        ``MultiIndex`` with levels ``element_id`` and ``node_id``.  Subsequent calls to
        :meth:`filter_node_set`, :meth:`filter_element_set`, :meth:`join_coordinates`,
        and :meth:`join_variable` operate on this mesh until :meth:`to_frame` returns
        it and resets the importer.

        Parameters
        ----------
        geometry : str
            Name of the VMAP geometry to use.
        state : str, optional
            Name of the VMAP state to use for following :meth:`join_variable` calls.
            If omitted, pass a state to the first :meth:`join_variable` call.  Default
            is ``None``.

        Returns
        -------
        VMAPImport
            The importer itself to allow fluent method chaining.

        Raises
        ------
        KeyError
            Raised when ``geometry`` is not present or the VMAP connectivity data is
            malformed.

        See Also
        --------
        pylife.vmap.VMAPImport.join_coordinates : Add node coordinates to the mesh.
        pylife.vmap.VMAPImport.join_variable : Add a VMAP result variable to the mesh.
        pylife.vmap.VMAPImport.to_frame : Return the finished pandas mesh.

        Examples
        --------
        .. code-block:: python

            mesh = (
                VMAPImport("demos/plate_with_hole.vmap")
                .make_mesh("1", "STATE-2")
                .join_coordinates()
                .join_variable("STRESS_CAUCHY")
                .to_frame()
            )
        """
        self._mesh = pd.DataFrame(index=self._mesh_index(geometry))
        self._geometry = geometry
        self._state = state
        return self

    def filter_node_set(self, node_set):
        """Restrict the working mesh to a node set.

        Parameters
        ----------
        node_set : str
            Name of the VMAP node set in the current geometry.

        Returns
        -------
        VMAPImport
            The importer itself to allow fluent method chaining.

        Raises
        ------
        APIUseError
            Raised when :meth:`make_mesh` has not been called before filtering.
        KeyError
            Raised when ``node_set`` is not defined for the current geometry.
        """
        self._check_mesh_for_filtering()
        node_set_ids = self._node_set_ids(self._geometry, node_set)
        self._mesh = self._mesh[self._mesh.index.isin(node_set_ids, level='node_id')]
        return self

    def filter_element_set(self, element_set):
        """Restrict the working mesh to an element set.

        Parameters
        ----------
        element_set : str
            Name of the VMAP element set in the current geometry.

        Returns
        -------
        VMAPImport
            The importer itself to allow fluent method chaining.

        Raises
        ------
        APIUseError
            Raised when :meth:`make_mesh` has not been called before filtering.
        KeyError
            Raised when ``element_set`` is not defined for the current geometry.
        """
        self._check_mesh_for_filtering()
        element_set_ids = self._element_set_ids(self._geometry, element_set)
        self._mesh = self._mesh[self._mesh.index.isin(element_set_ids, level='element_id')]
        return self

    def _check_mesh_for_filtering(self):
        if self._mesh is None:
            raise APIUseError("Need to make_mesh() before filtering node or element sets.")

    def join_coordinates(self):
        """Join node coordinates to the working mesh.

        The added columns are ``x``, ``y``, and ``z``.  Values use the length unit of
        the source finite-element model, commonly millimetres.

        Returns
        -------
        VMAPImport
            The importer itself to allow fluent method chaining.

        Raises
        ------
        APIUseError
            Raised when :meth:`make_mesh` has not been called before joining
            coordinates.
        KeyError
            Raised when the current geometry does not contain the expected coordinate
            datasets.

        Examples
        --------
        .. code-block:: python

            mesh = (
                VMAPImport("demos/plate_with_hole.vmap")
                .make_mesh("1")
                .join_coordinates()
                .to_frame()
            )
        """
        if self._mesh is None:
            raise APIUseError("Need to make_mesh() before joining the coordinates.")
        self._mesh = self._mesh.join(self.nodes(self._geometry))
        return self

    def to_frame(self):
        """Return the completed mesh and reset the importer.

        Returns
        -------
        pandas.DataFrame
            Mesh data joined so far.  The frame is usually indexed by ``element_id``
            and ``node_id`` and contains any coordinates or variable columns added to
            the working mesh.

        Raises
        ------
        APIUseError
            Raised when :meth:`make_mesh` has not been called or the working mesh has
            already been returned.

        Notes
        -----
        Calling this method clears the working mesh.  Call :meth:`make_mesh` again to
        start importing another geometry, state, or filtered subset.
        """
        if self._mesh is None:
            raise(APIUseError("Need to make_mesh() before requesting a resulting frame."))
        ret = self._mesh
        self._mesh = None
        return ret

    def variables(self, geometry, state):
        """List variable names for a geometry and state.

        Parameters
        ----------
        geometry : str
            Name of the VMAP geometry to inspect.
        state : str
            Name of the VMAP state to inspect.

        Returns
        -------
        list of str
            Names of result variables available for the ``geometry`` and ``state``
            combination, for example ``DISPLACEMENT`` or ``STRESS_CAUCHY``.

        Raises
        ------
        KeyError
            Raised when ``geometry`` or ``state`` is unknown, or when the state does
            not contain data for the geometry.
        """
        self._fail_if_unknown_geometry(geometry)
        self._fail_if_unknown_state(state)
        if geometry not in self._file['/VMAP/VARIABLES/%s' % state].keys():
            raise KeyError("Geometry '%s' not available in state '%s'." % (geometry, state))
        return list(self._file['/VMAP/VARIABLES/%s/%s' % (state, geometry)].keys())

    def join_variable(self, var_name, state=None, column_names=None):
        """Join a VMAP result variable to the working mesh.

        Variables are result fields stored for a geometry and state.  Typical VMAP
        variables are ``DISPLACEMENT`` for nodal displacement, ``STRESS_CAUCHY`` for
        Cauchy stress, and ``E`` for strain.  Their numeric units follow the source
        finite-element model; stresses are commonly stored in MPa.

        Parameters
        ----------
        var_name : str
            Name of the VMAP variable to join.
        state : str, optional
            Name of the state from which to read ``var_name``.  If omitted, the state
            set by :meth:`make_mesh` or by the previous :meth:`join_variable` call is
            used.  Default is ``None``.
        column_names : list of str, optional
            Column names to use in the returned ``DataFrame``.  The list length must
            match the VMAP variable dimension.  If omitted, pyLife uses predefined
            names for known variables.  Default is ``None``.

        Returns
        -------
        VMAPImport
            The importer itself to allow fluent method chaining.

        Raises
        ------
        APIUseError
            Raised when :meth:`make_mesh` has not been called or no state is known.
        KeyError
            Raised when the geometry, state, or variable is not present, or when no
            predefined column names exist for ``var_name`` and ``column_names`` is not
            supplied.
        ValueError
            Raised when ``column_names`` does not match the variable dimension.
        FeatureNotSupportedError
            Raised when the VMAP variable location is not node, element, or element-
            nodal data.

        Notes
        -----
        When ``column_names`` is omitted, pyLife uses these predefined names:

        * ``DISPLACEMENT``: ``dx``, ``dy``, ``dz``.
        * ``STRESS_CAUCHY``: ``S11``, ``S22``, ``S33``, ``S12``, ``S13``, ``S23``.
        * ``E``: ``E11``, ``E22``, ``E33``, ``E12``, ``E13``, ``E23``.

        .. todo:: Move the central definition of pyLife VMAP column names into user
           documentation.

        Examples
        --------
        .. code-block:: python

            mesh = (
                VMAPImport("demos/plate_with_hole.vmap")
                .make_mesh("1")
                .join_variable("DISPLACEMENT", "STATE-1")
                .join_variable("STRESS_CAUCHY", "STATE-2")
                .join_variable("E")
                .to_frame()
            )
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
        """Read a geometry set without raising if it is absent.

        Parameters
        ----------
        geometry_name : str
            Name of the geometry the set belongs to.
        geometry_set_name : str
            Name of the geometry set to read.

        Returns
        -------
        pandas.Index or None
            The node or element IDs of the geometry set, or ``None`` if the
            set does not exist in the VMAP file.

        See Also
        --------
        try_get_vmap_object : Read an arbitrary VMAP group without raising.
        """
        try:
            geometry_set = self._file["/VMAP/GEOMETRY/%s/GEOMETRYSETS/%s/MYGEOMETRYSETDATA"
                                      % (geometry_name, geometry_set_name)]
            return pd.Index(geometry_set[()].flatten())
        except KeyError:
            return None

    def try_get_vmap_object(self, group_full_path):
        """Read a VMAP group or dataset without raising if it is absent.

        Parameters
        ----------
        group_full_path : str
            Full path of the group inside the VMAP file, e.g.
            ``'/VMAP/GEOMETRY/1'``.

        Returns
        -------
        h5py.Group or h5py.Dataset or None
            The requested HDF5 object, or ``None`` if the path does not exist.

        See Also
        --------
        try_get_geometry_set : Read a geometry set without raising.
        """
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

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

import datetime
import getpass

import numpy as np
import pandas as pd
import h5py
import os

from .exceptions import *
from . import vmap_structures
from .vmap_unit_system import VMAPUnit
from .vmap_element_type import VMAPElementType
from .vmap_attribute import VMAPAttribute
from .vmap_coordinate_system import VMAPCoordinateSystem
from .vmap_section import VMAPSection
from .vmap_metadata import VMAPMetadata
from .vmap_integration_type import VMAPIntegrationType


class VMAPExport:
    """
    The interface class to export a vmap file

    Parameters
    ----------
    file_name : string
        The path to the vmap file to be read

    Raises
    ------
    Exception
        If the file cannot be read an exception is raised.
        So far any exception from the ``h5py`` module is passed through.
    """

    """
    These dictionaries are to provide the data to the SYSTEM datasets. This data is going to come from the import
    """
    _element_types = {
        (2, 3): [0, 'VMAP_ELEM_2D_TRIANGLE_3', 'pyLife 2D 3', 3, 2, -1, -1, -1, -1, -1, [], []],
        (2, 6): [1, 'VMAP_ELEM_2D_TRIANGLE_6', 'pyLife 2D 6', 6, 2, -1, -1, -1, -1, -1, [], []],
        (2, 4): [2, 'VMAP_ELEM_2D_QUAD_4', 'pyLife 2D 4', 4, 2, -1, -1, -1, -1, -1, [], []],
        (2, 8): [3, 'VMAP_ELEM_2D_QUAD_8', 'pyLife 2D 8', 8, 2, -1, -1, -1, -1, -1, [], []],
        (3, 4): [4, 'VMAP_ELEMENT_3D_TETRAHEDRON_4', 'pyLife 3D 4', 4, 3, -1, -1, -1, -1, -1, [], []],
        (3, 10): [5, 'VMAP_ELEMENT_3D_TETRAHEDRON_10', 'pyLife 3D 10', 10, 3, -1, -1, -1, -1, -1, [], []],
        (3, 6): [6, 'VMAP_ELEMENT_3D_WEDGE_6', 'pyLife 3D 6', 6, 3, -1, -1, -1, -1, -1, [], []],
        (3, 15): [7, 'VMAP_ELEMENT_3D_WEDGE_15', 'pyLife 3D 15', 15, 3, -1, -1, -1, -1, -1, [], []],
        (3, 8): [8, 'VMAP_ELEMENT_3D_HEXAHEDRON_8', 'pyLife 3D 8', 8, 3, -1, -1, -1, -1, -1, [], []],
        (3, 20): [9, 'VMAP_ELEMENT_3D_HEXAHEDRON_20', 'pyLife 3D 20', 20, 3, -1, -1, -1, -1, -1, [], []]
    }

    _coordinate_systems = {
        'CARTESIAN': [1, 2, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
    }

    _sections = {
        'DEFAULT': [0, 'Section-ASSEMBLY_DEFAULT', 1, 0, _coordinate_systems['CARTESIAN'][0], -1, -1]
    }

    _unit_system = {
        'LENGTH': [1, 1.0, 0.0, 'm', 'LENGTH'],
        'MASS': [2, 1.0, 0.0, 'kg', 'MASS'],
        'TIME': [3, 1.0, 0.0, 's', 'TIME'],
        'ELECTRIC_CURRENT': [4, 1.0, 0.0, 'A', 'ELECTRIC CURRENT'],
        'TEMPERATURE': [5, 1.0, 0.0, 'K', 'TEMPERATURE'],
        'AMOUNT_OF_SUBSTANCE': [6, 1.0, 0.0, 'mol', 'AMOUNT OF SUBSTANCE'],
        'LUMINOUS_INTENSITY': [7, 1.0, 0.0, 'cd', 'LUMINOUS INTENSITY'],
    }

    _metadata = {
        'EXPORTER_NAME': ['ExporterName', 'pyLife'],
        'FILE_DATE': ['FileDate', datetime.datetime.now().date().strftime("%Y-%m-%d")],
        'FILE_TIME': ['FileTime', datetime.datetime.now().time().strftime("%H:%M:%S.%f")],
        'DESCRIPTION': ['Description', ''],
        'ANALYSIS_TYPE': ['Analysis Type', ''],
        'USERID': ['User Id', getpass.getuser()]
    }

    def __init__(self, file_name):
        self._file_name = file_name
        try:
            with h5py.File(file_name, 'w') as file:
                self._create_fundamental_groups(file)
            self._dimension = 2
        except:
            if os.path.exists(self._file_name):
                os.remove(self._file_name)
            raise Exception('An error occurred while creating file %s' % self._file_name)

    @property
    def file_name(self):
        """
        Gets the name of the VMAP file that we are exporting
        """
        return self._file_name

    def variable_column_names(self, parameter_name):
        """
        Gets the column names that the given parameter consists of

        Parameters
        ----------
        parameter_name: string
            The name of the parameter

        Returns
        -------
        The column names of the given parameter in the mesh

        """
        return vmap_structures.column_names[parameter_name][0]

    def variable_location(self, parameter_name):
        """
        Gets the location of the given parameter

        Parameters
        ----------
        parameter_name: string
            The name of the parameter

        Returns
        -------
        The location of the given parameter

        """
        return vmap_structures.column_names[parameter_name][1]

    def set_group_attribute(self, object_path, key, value):
        """
        Sets the 'MYNAME' attribute of the VMAP objects

        Parameters
        ----------
        object_path: string
            The full path to the object that we want to rename
        key: string
            The key of the attribute that we want to set
        value: np.dtype
            The value that we want to set to the attribute

        Returns
        -------
        -
        """
        with h5py.File(self._file_name, 'a') as file:
            try:
                vmap_object = file[object_path]
            except KeyError:
                raise KeyError('VMAP object %s does not exist.' % object_path)
            vmap_object.attrs.create(key, value)

    def add_geometry(self, geometry_name, mesh):
        """
        Exports geometry with given name and mesh data

        Parameters
        ----------
        geometry_name: string
            Name of the geometry to add
        mesh: Pandas DataFrame
            The Data Frame that holds the data of the mesh to export
        Returns
        -------
        self

        """
        with h5py.File(self._file_name, 'a') as file:
            geometry_group = file["/VMAP/GEOMETRY"]
            if geometry_name in geometry_group:
                raise KeyError('Geometry %s already exists' % geometry_name)
            try:
                geometry = self._create_geometry_groups(file, geometry_group, geometry_name)
                self._create_points_datasets(geometry, mesh)
                self._create_elements_dataset(geometry, mesh)
            except:
                del geometry_group[geometry_name]
                raise Exception('An error occurred while creating geometry %s' % geometry_name)
        return self

    def add_node_set(self, geometry_name, indices, mesh, name=None):
        """
        Exports node-type geometry set into given geometry

        Parameters
        ----------
        geometry_name: string`
            The geometry to where we want to export the geometry set
        indices: Pandas Index
            List of node indices that we want to export
        mesh: Pandas DataFrame
            The Data Frame that holds the data of the mesh to export
        name: value of attribute MYSETNAME

        Returns
        -------
        self

        """
        node_id_set = set(mesh.index.get_level_values('node_id'))
        index_set = set(indices)
        if not index_set.issubset(node_id_set):
            raise KeyError('Provided index set is not a subset of the node indices.')
        self._create_geometry_set(geometry_name, 0, indices, name)
        return self

    def add_element_set(self, geometry_name, indices, mesh, name=None):
        """
        Exports element-type geometry set into given geometry

        Parameters
        ----------
        geometry_name: string
            The geometry to where we want to export the geometry set
        indices: Pandas Index
            List of node indices that we want to export
        mesh: Pandas DataFrame
            The Data Frame that holds the data of the mesh to export
        name: value of attribute MYSETNAME

        Returns
        -------
        self

        """
        element_id_set = set(mesh.index.get_level_values('element_id'))
        index_set = set(indices)
        if not index_set.issubset(element_id_set):
            raise KeyError('Provided index set is not a subset of the element indices.')
        self._create_geometry_set(geometry_name, 1, indices, name)
        return self

    def add_integration_types(self, content):
        """
        Creates system dataset IntegrationTypes with the given content

        Parameters
        ----------
        content: the content of the dataset

        Returns
        -------
        self

        """
        self._create_system_dataset(VMAPIntegrationType, content)
        return self

    def add_variable(self, state_name, geometry_name, variable_name, mesh, column_names=None, location=None):
        """
        Exports variable into given state and geometry
        Parameters
        ----------
        state_name: string
            State where we want to export the parameter
        geometry_name: string
            Geometry where we want to export the parameter
        variable_name: string
            The name of the variable to export
        mesh: Pandas DataFrame
            The Data Frame that holds the data of the mesh to export
        column_names: List, optional
            The columns that the parameter consists of
        location: Enum, optional
            The location of the parameter
                2 - node
                3 - element - not supported yet
                6 - element nodal
        Returns
        -------
        self

        """
        with h5py.File(self._file_name, 'a') as file:
            try:
                file["VMAP/GEOMETRY/%s" % geometry_name]
            except KeyError:
                raise KeyError("No geometry with the name %s" % geometry_name)

            try:
                state_group = file["/VMAP/VARIABLES/%s" % state_name]
            except KeyError:
                state_group = self._create_group_with_attributes(file['VMAP/VARIABLES'], state_name,
                                                                 VMAPAttribute('MYSTATEINCREMENT', 0),
                                                                 VMAPAttribute('MYSTATENAME', str.encode(state_name,
                                                                                                         'UTF8')),
                                                                 VMAPAttribute('MYSTEPTIME', 0.0),
                                                                 VMAPAttribute('MYTOTALTIME', 0.0))
            try:
                geometry_group = state_group[geometry_name]
            except KeyError:
                geometry_group = self._create_group_with_attributes(state_group, geometry_name,
                                                                    VMAPAttribute('MYSIZE', 0))

            if variable_name in geometry_group:
                raise KeyError("Variable already exists in state %s and geometry %s" % (state_name, geometry_name))

            if column_names is None:
                try:
                    column_names = vmap_structures.column_names[variable_name][0]
                except KeyError:
                    raise KeyError("No column name for variable %s." % variable_name)

            if location is None:
                if variable_name not in vmap_structures.column_names:
                    raise APIUseError(
                        "Need location for unknown variable %s. Please provide one using 'location' parameter."
                        % variable_name)
                location = vmap_structures.column_names[variable_name][1]

            try:
                variable_dataset = self._create_group_with_attributes(geometry_group, variable_name,
                                                                      VMAPAttribute('MYCOORDINATESYSTEM', -1),
                                                                      VMAPAttribute('MYDIMENSION', len(column_names)),
                                                                      VMAPAttribute('MYENTITY', 1),
                                                                      VMAPAttribute('MYIDENTIFIER',
                                                                                    len(geometry_group)),
                                                                      VMAPAttribute('MYINCREMENTVALUE', 1),
                                                                      VMAPAttribute('MYLOCATION', location.value),
                                                                      VMAPAttribute('MYMULTIPLICITY', 1),
                                                                      VMAPAttribute('MYTIMEVALUE', 0.0),
                                                                      VMAPAttribute('MYUNIT', -1),
                                                                      VMAPAttribute('MYVARIABLEDEPENDENCY', b' '),
                                                                      VMAPAttribute('MYVARIABLEDESCRIPTION',
                                                                                    str.encode(
                                                                                        'pyLife: %s' % variable_name,
                                                                                        'UTF8')),
                                                                      VMAPAttribute('MYVARIABLENAME',
                                                                                    str.encode(variable_name, 'UTF8')))
                if location == vmap_structures.VariableLocations.NODE:
                    node_ids_info = mesh.groupby('node_id').first()
                    variable_dataset.create_dataset('MYGEOMETRYIDS', data=np.array([node_ids_info.index]).T,
                                                    dtype=np.int32, chunks=True)
                    variable_dataset.create_dataset('MYVALUES', data=node_ids_info[column_names].values, chunks=True)
                elif location == vmap_structures.VariableLocations.ELEMENT_NODAL:
                    element_ids = mesh.index.get_level_values('element_id').drop_duplicates().values
                    variable_dataset.create_dataset('MYGEOMETRYIDS', data=np.array([element_ids]).T, dtype=np.int32,
                                                    chunks=True)
                    variable_dataset.create_dataset('MYVALUES', data=mesh[column_names], chunks=True)
                else:
                    raise ValueError('Unknown location')

                geometry_group.attrs['MYSIZE'] = geometry_group.attrs['MYSIZE'] + 1
            except:
                del geometry_group[variable_name]
                raise Exception('An error occurred while creating variable %s' % variable_name)
        return self

    def _create_group_with_attributes(self, parent_group, group_name, *args):
        group = parent_group.create_group(group_name)
        if args is not None:
            for attr in args:
                group.attrs.create(attr.name, attr.value)
        return group

    def _create_compound_attribute(self, parent, attr_name, field_names, field_types, field_values):
        dt = np.dtype({"names": field_names, "formats": field_types})
        compound_attribute = np.array([field_values], dt)
        parent.attrs.create(attr_name, compound_attribute)

    def _create_fundamental_groups(self, file):
        vmap_group = file.create_group('VMAP')
        self._create_compound_attribute(vmap_group, 'VERSION', ["myMajor", "myMinor", "myPatch"],
                                        ['<i4', '<i4', '<i4'], ('0', '5', '2'))

        self._create_group_with_attributes(vmap_group, 'GEOMETRY')
        self._create_group_with_attributes(vmap_group, 'MATERIAL')
        self._create_group_with_attributes(vmap_group, 'SYSTEM')

        self._create_system_dataset(VMAPCoordinateSystem, self._coordinate_systems)
        self._create_system_dataset(VMAPElementType, self._element_types)
        self._create_system_dataset(VMAPMetadata, self._metadata)
        self._create_system_dataset(VMAPSection, self._sections)
        self._create_system_dataset(VMAPUnit, self._unit_system)

        self._create_group_with_attributes(vmap_group, 'VARIABLES')

    def _create_system_dataset(self, class_name, dataset_content):
        attribute_list = []
        for content_key in dataset_content:
            dataset = class_name(*dataset_content[content_key])
            attribute_list.append(dataset.attributes)
        name = dataset.dataset_name
        dt_type = dataset.dtype
        path = dataset.group_path
        compound_dataset = dataset.compound_dataset
        if compound_dataset:
            d = np.array([attribute_list], dtype=dt_type).T
            chunked = True
        else:
            d = np.array(attribute_list, dtype=dt_type)
            chunked = None
        with h5py.File(self._file_name, 'a') as file:
            system_group = file[path]
            if name in system_group:
                raise KeyError('Dataset %s already exists in SYSTEM')

            try:
                system_group.create_dataset(name, dtype=dt_type, data=d, chunks=chunked)
            except:
                del system_group[name]
                raise Exception('An error occurred while creating dataset %s' % name)
        return self

    def _create_geometry_groups(self, file, geometry_group, geometry_name):
        geometry = self._create_group_with_attributes(geometry_group, geometry_name)
        self._create_group_with_attributes(geometry, 'ELEMENTS', VMAPAttribute('MYSIZE', np.int64(0)))
        self._create_group_with_attributes(geometry, 'GEOMETRYSETS', VMAPAttribute('MYSIZE', 0))
        self._create_group_with_attributes(geometry, 'POINTS',
                                           VMAPAttribute('MYSIZE', np.int64(0)),
                                           VMAPAttribute('MYCOORDINATESYSTEM',
                                                         self._coordinate_systems['CARTESIAN'][0]))
        return geometry

    def _create_elements_dataset(self, geometry, mesh):
        dt_type = np.dtype({"names": ["myIdentifier", "myElementType", "myCoordinateSystem",
                                      "myMaterialType", "mySectionType", "myConnectivity"],
                            "formats": ['<i4', '<i4', '<i4', '<i4', '<i4', h5py.special_dtype(vlen=np.dtype('int32'))]})
        element_connectivities = mesh.groupby('element_id')
        element_ids = []
        node_ids_list = []
        element_types_list = []
        coordinate_system = np.empty(len(element_connectivities), dtype=np.int32)
        coordinate_system.fill(1)
        material_type = np.empty(len(element_connectivities), dtype=np.int32)
        material_type.fill(0)
        section_type = np.empty(len(element_connectivities), dtype=np.int32)
        section_type.fill(self._sections['DEFAULT'][0])

        for element_connectivity in element_connectivities:
            c = element_connectivity[1].index.get_level_values('node_id').values
            node_ids_list.append(c)
            element_ids.append(element_connectivity[0])
            element_type = self._element_types[self._dimension, c.size][0]
            element_types_list.append(element_type)

        element_types = np.asarray(element_types_list)
        connectivity = np.asarray(node_ids_list)
        d = np.array([list(zip(element_ids, element_types, coordinate_system,
                               material_type, section_type, connectivity))], dtype=dt_type).T
        elements_group = geometry['ELEMENTS']
        elements_group.create_dataset("MYELEMENTS", dtype=dt_type, data=d, chunks=True)
        elements_group.attrs['MYSIZE'] = np.int64(len(element_ids))

    def _create_points_datasets(self, geometry, mesh):
        node_ids_info = mesh.groupby('node_id').first()
        points_group = geometry['POINTS']
        points_group.create_dataset('MYIDENTIFIERS', data=np.reshape(node_ids_info.index, (-1, 1)), dtype=np.int32,
                                    chunks=True)
        if 'z' in node_ids_info:
            z = node_ids_info['z'].to_numpy()
            if not (z[0] == z).all():
                self._dimension = 3
            points_group.create_dataset('MYCOORDINATES', data=node_ids_info[['x', 'y', 'z']].values, chunks=True)
        else:
            points_group.create_dataset('MYCOORDINATES', data=node_ids_info[['x', 'y']].values, chunks=True)
        points_group.attrs['MYSIZE'] = np.int64(node_ids_info.index.size)

    def _create_geometry_set(self, geometry_name, object_type, indices, name=None):
        if name is None:
            name = ''
        with h5py.File(self._file_name, 'a') as file:
            try:
                geometry_set_group = file["VMAP/GEOMETRY/%s/GEOMETRYSETS" % geometry_name]
            except KeyError:
                raise KeyError("No geometry with the name %s" % geometry_name)
            try:
                set_size = geometry_set_group.attrs['MYSIZE']
                geometry_set_name = str(f"{set_size:06d}")
                geometry_set = self._create_group_with_attributes(geometry_set_group, geometry_set_name,
                                                                  VMAPAttribute('MYIDENTIFIER', set_size),
                                                                  VMAPAttribute('MYSETINDEXTYPE', 1),
                                                                  VMAPAttribute('MYSETNAME', str.encode(name, 'UTF-8')),
                                                                  VMAPAttribute('MYSETTYPE', object_type))
                geometry_set.create_dataset('MYGEOMETRYSETDATA', data=pd.DataFrame(indices),
                                            dtype=np.int32, chunks=True)
                geometry_set_group.attrs['MYSIZE'] = set_size + 1
            except:
                del geometry_set_group[geometry_set_name]
                raise Exception(
                    'An error occurred while creating geometry set %s in geometry %s'
                    % (geometry_set_name, geometry_name))

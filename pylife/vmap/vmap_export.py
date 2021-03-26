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
        (3, 4): [4, 'VMAP_ELEMENT_3D_TETRA_4', 'pyLife 3D 4', 4, 3, -1, -1, -1, -1, -1, [], []],
        (3, 10): [5, 'VMAP_ELEMENT_3D_TETRA_10', 'pyLife 3D 10', 10, 3, -1, -1, -1, -1, -1, [], []],
        (3, 6): [6, 'VMAP_ELEMENT_3D_WEDGE_6', 'pyLife 3D 6', 6, 3, -1, -1, -1, -1, -1, [], []],
        (3, 15): [7, 'VMAP_ELEMENT_3D_WEDGE_15', 'pyLife 3D 15', 15, 3, -1, -1, -1, -1, -1, [], []],
        (3, 8): [8, 'VMAP_ELEMENT_3D_HEX_8', 'pyLife 3D 8', 8, 3, -1, -1, -1, -1, -1, [], []],
        (3, 20): [9, 'VMAP_ELEMENT_3D_HEX_20', 'pyLife 3D 20', 20, 3, -1, -1, -1, -1, -1, [], []]
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
        with h5py.File(file_name, 'w') as file:
            self._create_fundamental_groups(file)
        self._dimension = 2

    @property
    def file_name(self):
        """
        Gets the name of the VMAP file that we are exporting
        """
        return self._file_name

    def parameter_column_names(self, parameter_name):
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

    def rename_vmap_object(self, object_path, name):
        """
        Sets the 'MYNAME' attribute of the VMAP objects that has such attribute

        Parameters
        ----------
        object_path: string
            The full path to the object that we want to rename
        name: string
            The name that we want to provide

        Returns
        -------
        -
        """
        with h5py.File(self._file_name, 'a') as file:
            try:
                vmap_object = file[object_path]
            except KeyError:
                raise KeyError('VMAP object %s does not exist.' % object_path)

            try:
                vmap_object.attrs['MYNAME'] = name
            except KeyError:
                raise KeyError('VMAP object %s does not have attribute name')

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
            geometry = self._create_geometry_groups(file, geometry_name)
            self._create_elements_dataset(geometry, mesh)
            self._create_points_datasets(geometry, mesh)
        return self

    def add_node_set(self, geometry_name, indices, mesh):
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

        Returns
        -------
        self

        """
        node_id_set = set(mesh.index.get_level_values('node_id'))
        index_set = set(indices)
        if not index_set.issubset(node_id_set):
            raise KeyError('Provided index set is not a subset of the node indices.')
        self._add_geometry_set(geometry_name, 0, indices)
        return self

    def add_element_set(self, geometry_name, indices, mesh):
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

        Returns
        -------
        self

        """
        element_id_set = set(mesh.index.get_level_values('element_id'))
        index_set = set(indices)
        if not index_set.issubset(element_id_set):
            raise KeyError('Provided index set is not a subset of the element indices.')
        self._add_geometry_set(geometry_name, 1, indices)
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
        self._create_dataset(VMAPIntegrationType, content)
        return self

    def add_variable(self, state, geometry_name, variable_name, mesh, column_names=None, location=None):
        """
        Exports variable into given state and geometry
        Parameters
        ----------
        state: string
            State where we want to export the parameter
        geometry_name: string
            Geometry where we want to export the parameter
        variable_name: string
            The name of the variable to export
        mesh: Pandas DataFrame
            The Data Frame that holds the data of the mesh to export
        column_names: List, optional
            The columns that the parameter consists of
        location: integer, optional
            The location of the parameter
                2 - node
                3 - element
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
                state_group = file["/VMAP/VARIABLES/%s" % state]
            except KeyError:
                state_group = self._create_group_with_attributes(file['VMAP/VARIABLES'], state,
                                                                 VMAPAttribute('MYSTATEINCREMENT', 0),
                                                                 VMAPAttribute('MYSTATENAME', state),
                                                                 VMAPAttribute('MYSTEPTIME', 0.0),
                                                                 VMAPAttribute('MYTOTALTIME', 0.0))
            try:
                geometry_group = state_group[geometry_name]
            except KeyError:
                geometry_group = self._create_group_with_attributes(state_group, geometry_name,
                                                                    VMAPAttribute('MYSIZE', 0))

            if column_names is None:
                try:
                    column_names = vmap_structures.column_names[variable_name][0]
                except KeyError:
                    raise KeyError("No column name for variable %s." % variable_name)

            if location is None:
                location = vmap_structures.column_names[variable_name][1]

            variable_dataset = self._create_group_with_attributes(geometry_group, variable_name,
                                                                  VMAPAttribute('MYCOORDINATESYSTEM', -1),
                                                                  VMAPAttribute('MYDIMENSION', len(column_names)),
                                                                  VMAPAttribute('MYENTITY', 1),
                                                                  VMAPAttribute('MYIDENTIFIER', len(geometry_group)),
                                                                  VMAPAttribute('MYINCREMENTVALUE', 1),
                                                                  VMAPAttribute('MYLOCATION', location),
                                                                  VMAPAttribute('MYMULTIPLICITY', 1),
                                                                  VMAPAttribute('MYTIMEVALUE', 0.0),
                                                                  VMAPAttribute('MYUNIT', -1),
                                                                  VMAPAttribute('MYVARIABLEDEPENDENCY', ''),
                                                                  VMAPAttribute('MYVARIABLEDESCRIPITON',
                                                                                'pyLife: %s' % variable_name),
                                                                  VMAPAttribute('MYVARIABLENAME', variable_name))
            if location == 2:
                node_ids_info = mesh.groupby('node_id').first()
                variable_dataset.create_dataset('MYGEOMETRYIDS', data=np.array([node_ids_info.index]).T, chunks=True)
                variable_dataset.create_dataset('MYVALUES', data=node_ids_info[column_names].values, chunks=True)
            if location == 6:
                element_ids = mesh.index.get_level_values('element_id').drop_duplicates().values
                variable_dataset.create_dataset('MYGEOMETRYIDS', data=np.array([element_ids]).T, chunks=True)
                variable_dataset.create_dataset('MYVALUES', data=mesh[column_names], chunks=True)
            geometry_group.attrs['MYSIZE'] = geometry_group.attrs['MYSIZE'] + 1
        return self

    def _create_group_with_attributes(self, parent_group, group_name, *args):
        group = parent_group.create_group(group_name)
        if args is not None:
            for attr in args:
                group.attrs[attr.name] = attr.value
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

        self._create_dataset(VMAPCoordinateSystem, self._coordinate_systems)
        self._create_dataset(VMAPElementType, self._element_types)
        self._create_dataset(VMAPMetadata, self._metadata)
        self._create_dataset(VMAPSection, self._sections)
        self._create_dataset(VMAPUnit, self._unit_system)

        self._create_group_with_attributes(vmap_group, 'VARIABLES')

    def _create_dataset(self, class_name, dataset_content):
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
        else:
            d = np.array(attribute_list, dtype=dt_type)

        with h5py.File(self._file_name, 'a') as file:
            system_group = file[path]
            system_group.create_dataset(name, dtype=dt_type, data=d, chunks=True)
        return self

    def _create_geometry_groups(self, file, geometry_name):
        geometry_group = file["/VMAP/GEOMETRY"]
        geometry = self._create_group_with_attributes(geometry_group, geometry_name)
        size_attribute = VMAPAttribute('MYSIZE', 0)
        self._create_group_with_attributes(geometry, 'ELEMENTS', size_attribute)
        self._create_group_with_attributes(geometry, 'GEOMETRYSETS', size_attribute)
        self._create_group_with_attributes(geometry, 'POINTS', size_attribute,
                                           VMAPAttribute('MYCOORDINATESYSTEM',
                                                         self._coordinate_systems['CARTESIAN'][0]))
        return geometry

    def _create_elements_dataset(self, geometry, mesh):
        dt_type = np.dtype({"names": ["myIdentifier", "myElementType", "myCoordinateSystem",
                                      "myMaterialType", "mySectionType", "myConnectivity"],
                            "formats": ['<i4', '<i4', '<i4', '<i4', '<i4', h5py.special_dtype(vlen=np.dtype('int32'))]})
        element_ids = mesh.index.get_level_values('element_id').drop_duplicates().values
        node_ids_list = []
        element_types_list = []
        coordinate_system = np.empty(element_ids.size, dtype=np.int)
        coordinate_system.fill(1)
        material_type = np.empty(element_ids.size, dtype=np.int)
        material_type.fill(0)
        section_type = np.empty(element_ids.size, dtype=np.int)
        section_type.fill(self._sections['DEFAULT'][0])

        for element_id in element_ids:
            node_ids_for_element = mesh.loc[element_id, :].index.values
            node_ids_list.append(node_ids_for_element)
            element_type = self._element_types[self._dimension, node_ids_for_element.size][0]
            element_types_list.append(element_type)
        element_types = np.asarray(element_types_list)
        connectivity = np.asarray(node_ids_list)
        d = np.array([list(zip(element_ids, element_types, coordinate_system,
                               material_type, section_type, connectivity))], dtype=dt_type).T
        elements_group = geometry['ELEMENTS']
        elements_group.create_dataset("MYELEMENTS", dtype=dt_type, data=d, chunks=True)
        elements_group.attrs['MYSIZE'] = element_ids.size

    def _create_points_datasets(self, geometry, mesh):
        node_ids_info = mesh.groupby('node_id').first()
        points_group = geometry['POINTS']
        points_group.create_dataset('MYIDENTIFIERS', data=np.reshape(node_ids_info.index, (-1, 1)), chunks=True)
        if 'z' in node_ids_info:
            z = node_ids_info['z'].to_numpy()
            if not (z[0] == z).all():
                self._dimension = 3
            points_group.create_dataset('MYCOORDINATES', data=node_ids_info[['x', 'y', 'z']].values, chunks=True)
        else:
            points_group.create_dataset('MYCOORDINATES', data=node_ids_info[['x', 'y']].values, chunks=True)
        points_group.attrs['MYSIZE'] = node_ids_info.index.size

    def _add_geometry_set(self, geometry_name, object_type, indices):
        with h5py.File(self._file_name, 'a') as file:
            try:
                geometry_set_group = file["VMAP/GEOMETRY/%s/GEOMETRYSETS" % geometry_name]
            except KeyError:
                raise KeyError("No geometry with the name %s" % geometry_name)

            geometry_set_name = geometry_set_group.attrs['MYSIZE']
            geometry_set = self._create_group_with_attributes(geometry_set_group, str(f"{geometry_set_name:06d}"),
                                                              VMAPAttribute('MYIDENTIFIER', geometry_set_name),
                                                              VMAPAttribute('MYSETINDEXTYPE', 1),
                                                              VMAPAttribute('MYSETNAME', ''),
                                                              VMAPAttribute('MYSETTYPE', object_type))
            geometry_set.create_dataset('MYGEOMETRYSETDATA', data=pd.DataFrame(indices), chunks=True)
            geometry_set_group.attrs['MYSIZE'] = geometry_set_name + 1

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

"""VMAPImport interface for pyLife
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

"""
__author__ = "Gyöngyvér Kiss"
__maintainer__ = __author__

import os
import datetime

import numpy as np
import pandas as pd
import h5py
from h5py.h5t import string_dtype
from h5py.h5t import vlen_dtype

from .exceptions import *


class VMAPExport:

    def __init__(self, filename):
        self._file = h5py.File(filename, 'w')
        self._create_fundamental_groups()
        self._dimension = 2

    def create_geometry(self, geometry_name, mesh):
        geometry = self._create_geometry_groups(geometry_name)
        points_group = geometry.get('POINTS')
        node_ids_info = mesh.groupby('node_id').first()
        self._create_points_datasets(node_ids_info, points_group)
        self._create_elements_dataset(mesh, geometry_name)
        self._file.close()
        return self

    def _create_fundamental_groups(self):
        self._vmap_group = self._file.create_group('VMAP')
        self._create_compound_attribute('VERSION', ["myMajor", "myMinor", "myPatch"],
                                        ['<i4', '<i4', '<i4'], ('0', '5', '2'))
        self._vmap_group.create_group('GEOMETRY')
        self._vmap_group.create_group('MATERIAL')
        self._vmap_group.create_group('SYSTEM')
        self._create_system_metadata()
        self._create_unit_system()
        self._vmap_group.create_group('VARIABLES')

    '''
    def create_integration_types(self, type_names, number_of_points, dimensions, offsets,
                                 abscissas=None, weights=None, subtypes=None):
        if abscissas is None:
            abscissas = []
            for type_name in type_names:
                abscissas.append([])
        if weights is None:
            weights = []
            for type_name in type_names:
                weights.append([])
        if subtypes is None:
            subtypes = []
            for type_name in type_names:
                subtypes.append([])

        dt_type = np.dtype({"names": ["myIdentifier", "myTypeName", "myNumberOfPoints", "myDimensions",
                                      "myOffset", "myAbscissas", "myWeights", "mySubTypes"],
                            "formats": ['<i4', string_dtype(), '<i4', '<i4', '<f4',
                                        h5py.special_dtype(vlen=np.dtype('float64')),
                                        h5py.special_dtype(vlen=np.dtype('float64')),
                                        h5py.special_dtype(vlen=np.dtype('float64'))]})

        identifiers = np.arange(len(type_names))
        abscissas_array = self._convert_2d_list_to_array(abscissas)
        weights_array = self._convert_2d_list_to_array(weights)
        subtypes_array = self._convert_2d_list_to_array(subtypes)
        d = np.array(list(zip(identifiers, np.array(type_names), np.array(number_of_points), np.array(dimensions),
                              np.array(offsets), abscissas_array, weights_array, subtypes_array)), dtype=dt_type)
        system_group = self._file["/VMAP/SYSTEM"]
        system_group.create_dataset("INTEGRATIONTYPES", dtype=dt_type, data=d)
        self._file.close()
        return self
        '''

    def create_integration_types_dataset(self, *args):
        dt_type = np.dtype({"names": ["myIdentifier", "myTypeName", "myNumberOfPoints", "myDimensions",
                                      "myOffset", "myAbscissas", "myWeights", "mySubTypes"],
                            "formats": ['<i4', string_dtype(), '<i4', '<i4', '<f4',
                                        h5py.special_dtype(vlen=np.dtype('float64')),
                                        h5py.special_dtype(vlen=np.dtype('float64')),
                                        h5py.special_dtype(vlen=np.dtype('float64'))]})
        it_attribute_list = []
        i = 0
        for integration_type in args:
            integration_type.set_identifier(i)
            it_attribute_list.append(integration_type.attributes)
            i = i + 1
        d = np.array(it_attribute_list, dtype=dt_type)
        system_group = self._file["/VMAP/SYSTEM"]
        system_group.create_dataset("INTEGRATIONTYPES", dtype=dt_type, data=d)
        self._file.close()
        return self

    def create_element_types_dataset(self, *args):
        dt_type = np.dtype({"names": ["myIdentifier", "myTypeName", "myTypeDescription", "myNumberOfNodes",
                                      "myDimensions", "myShapeType", "myInterpolationType", "myIntegrationType",
                                      "myNumberOfNormalComponents", "myNumberOfShearComponents", "myConnectivity",
                                      "myFaceConnectivity"],
                            "formats": ['<i4', string_dtype(), string_dtype(), '<i4', '<i4', '<i4', '<i4', '<i4',
                                        '<i4', '<i4', h5py.special_dtype(vlen=np.dtype('int32')),
                                        h5py.special_dtype(vlen=np.dtype('int32'))]})
        et_attribute_list = []
        i = 0
        for integration_type in args:
            integration_type.set_identifier(i)
            et_attribute_list.append(integration_type.attributes)
            i = i + 1
        d = np.array(et_attribute_list, dtype=dt_type)
        system_group = self._file["/VMAP/SYSTEM"]
        system_group.create_dataset("ELEMENTTYPES", dtype=dt_type, data=d)
        self._file.close()
        return self

    def _convert_2d_list_to_array(self, list_to_convert):
        result_array = []
        for element in list_to_convert:
            result_array.append(np.array(element))
        return np.array(result_array)

    def _create_geometry_groups(self, geometry_name):
        geometry_group = self._file["/VMAP/GEOMETRY"]
        geometry = geometry_group.create_group(geometry_name)
        geometry.create_group('ELEMENTS')
        geometry.create_group('GEOMETRYSETS')
        geometry.create_group('POINTS')
        return geometry

    def _create_points_datasets(self, node_ids_info, point_group):
        # element_ids = mesh_index.get_level_values('element_id')
        # mesh_index = mesh.index
        # node_ids = mesh_index.get_level_values('node_id').drop_duplicates()
        # mesh_columns = mesh.columns
        # node_ids_info = mesh.groupby('node_id').first()
        point_group.create_dataset('MYIDENTIFIERS', data=node_ids_info.index)
        if 'z' in node_ids_info:
            z = node_ids_info['z'].to_numpy()
            if not (z[0] == z).all():
                self._dimension = 3
            point_group.create_dataset('MYCOORDINATES', data=node_ids_info[['x', 'y', 'z']].values)
        else:
            point_group.create_dataset('MYCOORDINATES', data=node_ids_info[['x', 'y']].values)
        return self

    def _create_compound_attribute(self, attr_name, field_names, field_types, field_values):
        # dt2 = np.dtype({"names": ["myMajor", "myMinor", "myPatch"], "formats": ['<i4', '<i4', '<i4']})
        # arr2 = np.array([('0', '5', '2')], dt2)
        dt = np.dtype({"names": field_names, "formats": field_types})
        compound_attribute = np.array([field_values], dt)
        self._vmap_group.attrs.create(attr_name, compound_attribute)

    def _create_attribute(self, attr_name, attr_value):
        self._vmap_group.attrs[attr_name] = attr_value

    def _create_system_metadata(self):
        analysis_type = None
        user_id = os.getlogin()
        current_date = datetime.datetime.now().date()
        current_time = datetime.datetime.now().time()
        metadata_d = {'0': ['ExporterName', 'FileDate', 'FileTime', 'Description', 'Analysis Type', 'User Id'],
                      '1': ['pyLife', current_date, current_time, 'Test description', analysis_type, user_id]}
        metadata_df = pd.DataFrame(data=metadata_d)
        system_group = self._file["/VMAP/SYSTEM"]
        system_group.create_dataset('METADATA', data=metadata_df, dtype=string_dtype())

    def _create_unit_system(self):
        """
        unit_system_dtype = np.dtype([('myIdentifier', 'i'),
                                              ('mySISCALE', 'f8'),
                                              ('mySIShift', 'f8'),
                                              ('myUnitSymbol', string_dtype()),
                                              ('myUnitQuantity', string_dtype())])
        """
        unit_system_dtype = np.dtype({"names": ["myIdentifier", "mySISCALE", "mySIShift",
                                                "myUnitSymbol", "myUnitQuantity"],
                                      "formats": ['<i4', '<f4', '<f4', string_dtype(), string_dtype()]})
        unit_system_d = np.array([(1, 1.0, 0.0, 'm', 'LENGTH'),
                                  (2, 1.0, 0.0, 'kg', 'MASS'),
                                  (3, 1.0, 0.0, 's', 'TIME'),
                                  (4, 1.0, 0.0, 'A', 'ELECTRIC CURRENT'),
                                  (5, 1.0, 0.0, 'K', 'TEMPERATURE'),
                                  (6, 1.0, 0.0, 'mol', 'AMOUNT OF SUBSTANCE'),
                                  (7, 1.0, 0.0, 'cd', 'LUMINOUS INTENSITY')], dtype=unit_system_dtype)

        system_group = self._file["/VMAP/SYSTEM"]
        system_group.create_dataset('UNITSYSTEM', (7,), unit_system_dtype, unit_system_d)

    def _determine_element_type(self, node_ids_for_element):
        element_type = None
        if self._dimension == 2:
            if node_ids_for_element.size == 3:
                element_type = 'VMAP_ELEM_2D_TRIANGLE_3'
            elif node_ids_for_element.size == 6:
                # element_type = 'VMAP_ELEM_2D_TRIANGLE_6'
                element_type = 0
            elif node_ids_for_element.size == 4:
                element_type = 'VMAP_ELEM_2D_QUAD_4'
            elif node_ids_for_element.size == 8:
                # element_type = 'VMAP_ELEM_2D_QUAD_8'
                element_type = 1
        elif self._dimension == 3:
            if node_ids_for_element.size == 4:
                element_type = 'VMAP_ELEMENT_3D_TETRA_4'
            elif node_ids_for_element.size == 10:
                element_type = 'VMAP_ELEMENT_3D_TETRA_10'
            elif node_ids_for_element.size == 6:
                element_type = 'VMAP_ELEMENT_3D_WEDGE_6'
            elif node_ids_for_element.size == 15:
                element_type = 'VMAP_ELEMENT_3D_WEDGE_15'
            elif node_ids_for_element.size == 8:
                element_type = 'VMAP_ELEMENT_3D_HEX_8'
            elif node_ids_for_element.size == 20:
                element_type = 'VMAP_ELEMENT_3D_HEX_20'
        else:
            raise KeyError("Unknown element type")

        return element_type

    def _create_element_types_dataset(self, all_element_types):
        dt_type = np.dtype({"names": ["myIdentifier", "myElementType", "myCoordinateSystem",
                                      "myMaterialType", "mySectionType", "myConnectivity"],
                            "formats": ['<i4', '<i4', '<i4', '<i4', '<i4', h5py.special_dtype(vlen=np.dtype('int32'))]})
        identifiers = np.arange(all_element_types.size - 1)
        type_names = np.array(all_element_types)
        type_descriptions = np.full((all_element_types.size - 1), 'Description')

    def _create_elements_dataset(self, mesh, geometry_name):
        dt_type = np.dtype({"names": ["myIdentifier", "myElementType", "myCoordinateSystem",
                                      "myMaterialType", "mySectionType", "myConnectivity"],
                            "formats": ['<i4', '<i4', '<i4', '<i4', '<i4', h5py.special_dtype(vlen=np.dtype('int32'))]})
        element_ids = mesh.index.get_level_values('element_id').drop_duplicates().values
        node_ids_list = []
        element_types_list = []
        all_element_types = []
        coordinate_system = np.empty(element_ids.size, dtype=np.int)
        coordinate_system.fill(1)
        material_type = np.empty(element_ids.size, dtype=np.int)
        material_type.fill(0)
        section_type = np.empty(element_ids.size, dtype=np.int)
        section_type.fill(0)

        for element_id in element_ids:
            node_ids_for_element = mesh.loc[element_id, :].index.values
            node_ids_list.append(node_ids_for_element)
            element_type = self._determine_element_type(node_ids_for_element)
            element_types_list.append(element_type)
            if element_type not in all_element_types:
                all_element_types.append(element_type)
        element_types = np.asarray(element_types_list)
        connectivity = np.asarray(node_ids_list)
        d = np.array(list(zip(element_ids, element_types, coordinate_system,
                              material_type, section_type, connectivity)), dtype=dt_type)
        elements_group = self._file["/VMAP/GEOMETRY/%s/ELEMENTS" % geometry_name]
        elements_group.create_dataset("MYELEMENTS", dtype=dt_type, data=d)
        system_group = self._file["/VMAP/SYSTEM"]
        system_group.create_dataset("ELEMENTTYPES", data=all_element_types)
        return self

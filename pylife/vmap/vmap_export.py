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
from .vmap_unit_system import VMAPUnitSystem


class VMAPExport:

    _column_names = {
        'DISPLACEMENT': [['dx', 'dy', 'dz'], 2],
        'STRESS_CAUCHY': [['S11', 'S22', 'S33', 'S12', 'S13', 'S23'], 6],
        'E': [['E11', 'E22', 'E33', 'E12', 'E13', 'E23'], 6],
    }

    def __init__(self, file_name):
        self._file_name = file_name
        self._file = h5py.File(file_name, 'w')
        # self._file = h5py.File(filename, 'a')
        self._create_fundamental_groups()
        self._dimension = 2
        self._file.close()

    def create_geometry(self, geometry_name, mesh):
        self._file = h5py.File(self._file_name, 'a')
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

    def create_dataset(self, *args):
        self._file = h5py.File(self._file_name, 'a')
        if args is None or len(args) == 0:
            raise ValueError(
                "You have to provide at least one VMAP Dataset object")
        attribute_list = []
        i = 0
        for dataset in args:
            dataset.set_identifier(i)
            attribute_list.append(dataset.attributes)
            i = i + 1
        name = args[0].dataset_name
        dt_type = args[0].dtype
        path = args[0].group_path
        d = np.array(attribute_list, dtype=dt_type)
        system_group = self._file[path]
        system_group.create_dataset(name, dtype=dt_type, data=d)
        self._file.close()
        return self

    def _create_geometry_groups(self, geometry_name):
        geometry_group = self._file["/VMAP/GEOMETRY"]
        geometry = geometry_group.create_group(geometry_name)
        geometry.create_group('ELEMENTS')
        geometry.create_group('GEOMETRYSETS')
        geometry.create_group('POINTS')
        return geometry

    def _create_points_datasets(self, node_ids_info, point_group):
        point_group.create_dataset('MYIDENTIFIERS', data=node_ids_info.index)
        if 'z' in node_ids_info:
            z = node_ids_info['z'].to_numpy()
            if not (z[0] == z).all():
                self._dimension = 3
            point_group.create_dataset('MYCOORDINATES', data=node_ids_info[['x', 'y', 'z']].values)
        else:
            point_group.create_dataset('MYCOORDINATES', data=node_ids_info[['x', 'y']].values)
        return self

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

    def _create_compound_attribute(self, attr_name, field_names, field_types, field_values):
        dt = np.dtype({"names": field_names, "formats": field_types})
        compound_attribute = np.array([field_values], dt)
        self._vmap_group.attrs.create(attr_name, compound_attribute)

    def _create_attribute(self, attr_name, attr_value):
        self._vmap_group.attrs[attr_name] = attr_value

    def _create_unit_system(self):
        length = VMAPUnitSystem(1.0, 0.0, 'm', 'LENGTH')
        mass = VMAPUnitSystem(1.0, 0.0, 'kg', 'MASS')
        time = VMAPUnitSystem(1.0, 0.0, 's', 'TIME')
        electric_current = VMAPUnitSystem(1.0, 0.0, 'A', 'ELECTRIC CURRENT')
        temperature = VMAPUnitSystem(1.0, 0.0, 'K', 'TEMPERATURE')
        amount_of_substance = VMAPUnitSystem(1.0, 0.0, 'mol', 'AMOUNT OF SUBSTANCE')
        luminous_intensity = VMAPUnitSystem(1.0, 0.0, 'cd', 'LUMINOUS INTENSITY')
        self.create_dataset(length, mass, time, electric_current, temperature, amount_of_substance, luminous_intensity)

    def _determine_element_type(self, node_ids_for_element):
        type_name = None
        if self._dimension == 2:
            if node_ids_for_element.size == 3:
                type_name = 'VMAP_ELEM_2D_TRIANGLE_3'
            elif node_ids_for_element.size == 6:
                type_name = 'VMAP_ELEM_2D_TRIANGLE_6'
            elif node_ids_for_element.size == 4:
                type_name = 'VMAP_ELEM_2D_QUAD_4'
            elif node_ids_for_element.size == 8:
                type_name = 'VMAP_ELEM_2D_QUAD_8'
        elif self._dimension == 3:
            if node_ids_for_element.size == 4:
                type_name = 'VMAP_ELEMENT_3D_TETRA_4'
            elif node_ids_for_element.size == 10:
                type_name = 'VMAP_ELEMENT_3D_TETRA_10'
            elif node_ids_for_element.size == 6:
                type_name = 'VMAP_ELEMENT_3D_WEDGE_6'
            elif node_ids_for_element.size == 15:
                type_name = 'VMAP_ELEMENT_3D_WEDGE_15'
            elif node_ids_for_element.size == 8:
                type_name = 'VMAP_ELEMENT_3D_HEX_8'
            elif node_ids_for_element.size == 20:
                type_name = 'VMAP_ELEMENT_3D_HEX_20'
        else:
            raise KeyError("Unknown element type")

        element_type = self._find_element_type_name(type_name)
        return element_type

    def _find_element_type_name(self, type_name):
        if type_name is None:
            return None
        element_types_dataset = self._file["/VMAP/SYSTEM/ELEMENTTYPES"]
        for element_type in element_types_dataset:
            if element_type[1] == type_name:
                return element_type[0]
        return None

    def _create_elements_dataset(self, mesh, geometry_name):
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
        section_type.fill(0)

        for element_id in element_ids:
            node_ids_for_element = mesh.loc[element_id, :].index.values
            node_ids_list.append(node_ids_for_element)
            element_type = self._determine_element_type(node_ids_for_element)
            element_types_list.append(element_type)
        element_types = np.asarray(element_types_list)
        connectivity = np.asarray(node_ids_list)
        d = np.array(list(zip(element_ids, element_types, coordinate_system,
                              material_type, section_type, connectivity)), dtype=dt_type)
        elements_group = self._file["/VMAP/GEOMETRY/%s/ELEMENTS" % geometry_name]
        elements_group.create_dataset("MYELEMENTS", dtype=dt_type, data=d)
        return self

    def add_variable(self, state, geometry_name, variable_name, mesh, column_names=None, location=None):
        self._file = h5py.File(self._file_name, 'a')
        try:
            state_group = self._file["/VMAP/VARIABLES/%s" % state]
        except:
            try:
                geometry_group = self._file["VMAP/GEOMETRY/%s" % geometry_name]
                state_group = self._file["/VMAP/VARIABLES"].create_group(state)
            except:
                raise KeyError("No geometry with the name %s" % geometry_name)

        try:
            geometry_group = state_group[geometry_name]
        except:
            geometry_group = state_group.create_group(geometry_name)

        if column_names is None:
            try:
                column_names = self._column_names[variable_name][0]
            except KeyError:
                raise KeyError("No column name for variable %s. Please provide with column_names parameter."
                               % variable_name)

        variable_dataset = geometry_group.create_group(variable_name)
        location = self._column_names[variable_name][1]
        dimension = len(column_names)
        if location == 2:
            node_ids_info = mesh.groupby('node_id').first()
            variable_dataset.create_dataset('MYGEOMETRYIDS', data=node_ids_info.index)
            variable_dataset.create_dataset('MYVALUES', data=node_ids_info[column_names].values)
        if location == 6:
            element_ids = mesh.index.get_level_values('element_id').drop_duplicates().values
            variable_dataset.create_dataset('MYGEOMETRYIDS', data=element_ids)
            variable_dataset.create_dataset('MYVALUES', data=mesh[column_names])
        self._file.close()




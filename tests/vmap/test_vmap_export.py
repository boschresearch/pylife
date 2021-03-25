import pandas as pd
import unittest

import pylife.vmap as vmap
import h5py
import os
import reference_data as RD


class TestExport(unittest.TestCase):
    def setUp(self):
        self._export = vmap.VMAPExport('tests/vmap/testfiles/test.vmap')
        self._import_expected = vmap.VMAPImport('tests/vmap/testfiles/beam_2d_squ_lin.vmap')
        self._mesh = (self._import_expected.make_mesh('1', 'STATE-2')
                      .join_coordinates()
                      .join_variable('STRESS_CAUCHY')
                      .join_variable('DISPLACEMENT')
                      .join_variable('E')
                      .to_frame())
        self._export.add_geometry('1', self._mesh)

    def tearDown(self):
        os.remove(self._export.file_name)

    def test_fundamental_groups(self):
        with h5py.File(self._export.file_name, 'r') as file:
            vmap_group = self.try_get_vmap_structure(file, 'VMAP')
            assert vmap_group is not None
            assert len(vmap_group) == 4

            geometry_group = self.try_get_vmap_structure(vmap_group, 'GEOMETRY')
            assert geometry_group is not None
            assert len(geometry_group) == 1

            material_group = self.try_get_vmap_structure(vmap_group, 'MATERIAL')
            assert material_group is not None
            assert len(material_group) == 0

            system_group = self.try_get_vmap_structure(vmap_group, 'SYSTEM')
            assert system_group is not None
            assert len(system_group) == 5
            self.assert_dataset_correct(system_group, 'ELEMENTTYPES', self._export._element_types)
            self.assert_dataset_correct(system_group, 'METADATA', self._export._metadata, False)
            self.assert_dataset_correct(system_group, 'SECTION', self._export._sections)
            self.assert_dataset_correct(system_group, 'UNITSYSTEM', self._export._unit_system)
            self.assert_dataset_correct(system_group, 'COORDINATESYSTEM', self._export._coordinate_systems)

            variables_group = self.try_get_vmap_structure(vmap_group, 'VARIABLES')
            assert variables_group is not None
            assert len(variables_group) == 0

    def test_set_object_name(self):
        geometry_path = 'VMAP/GEOMETRY/1'
        name = 'PART-1-1'
        self._export.set_object_name(geometry_path, name)
        with h5py.File(self._export.file_name, 'r') as file:
            assert file[geometry_path].attrs['MYNAME'] == name

    def test_add_dataset(self):
        self._export._add_dataset(vmap.VMAPIntegrationType, RD.integration_type_content)
        dataset_name = 'INTEGRATIONTYPES'
        with h5py.File(self._export.file_name, 'r') as file:
            system_group = self.try_get_vmap_structure(file, 'VMAP/SYSTEM')
            assert system_group is not None
            dataset = self.try_get_vmap_structure(system_group, dataset_name)
            assert dataset is not None
            self.assert_dataset_correct(system_group, dataset_name, RD.integration_type_content)

    def test_add_geometry(self):
        geometry_name = '2'
        self._export.add_geometry(geometry_name, self._mesh)
        mesh_expected = self._mesh[['x', 'y', 'z']]
        with vmap.VMAPImport(self._export.file_name) as import_actual:
            mesh_actual = (import_actual.make_mesh(geometry_name)
                           .join_coordinates()
                           .to_frame())
            self.assert_dfs_equal(mesh_expected, mesh_actual)

    def test_add_geometry_set(self):
        geometry_name = '1'
        geometry_set_name = '000000'
        geometry_set_expected = self._import_expected.get_geometry_set(geometry_name, geometry_set_name)
        ind_set = pd.Index([])
        for ind in geometry_set_expected:
            ind_set = ind_set.append(pd.Index(ind))
        self._export.add_node_set(geometry_name, ind_set, self._mesh)
        with vmap.VMAPImport(self._export.file_name) as import_actual:
            geometry_set_actual = import_actual.get_geometry_set(geometry_name, geometry_set_name)
            assert geometry_set_expected.shape == geometry_set_actual.shape
            assert geometry_set_expected.size == geometry_set_actual.size
            # assert geometry_set_expected.dtype == geometry_set_actual.dtype
            assert (geometry_set_expected == geometry_set_actual).all()

    def test_add_variable(self):
        state_name = 'STATE-2'
        geometry_name = '1'
        parameter_name = 'E'
        self._export.add_variable(state_name, geometry_name, parameter_name, self._mesh)
        with vmap.VMAPImport(self._export.file_name) as import_actual:
            mesh_actual = (import_actual.make_mesh(geometry_name, state_name)
                           .join_variable(parameter_name)
                           .to_frame())
        mesh_expected = self._mesh[self._export.get_parameter_column_names(parameter_name)]
        self.assert_dfs_equal(mesh_expected, mesh_actual)

    def try_get_vmap_structure(self, parent, structure_name):
        try:
            structure = parent[structure_name]
            return structure
        except KeyError:
            return None

    def assert_dataset_correct(self, parent, dataset_name, expected_values, is_compound=True):
        actual_values = parent[dataset_name]
        assert len(actual_values) == len(expected_values)
        for e_key, a_0 in zip(expected_values, actual_values):
            e_0 = expected_values[e_key]
            if is_compound:
                a_0 = a_0[0]
            assert len(e_0) == len(a_0)
            for e_1, a_1 in zip(e_0, a_0):
                try:
                    for e_2, a_2 in zip(e_1, a_1):
                        assert e_2 == a_2
                except TypeError:
                    assert e_1 == a_1

    def assert_dfs_equal(self, df_expected, df_actual):
        assert df_expected.shape == df_actual.shape
        assert df_expected.size == df_actual.size
        index_equal = df_expected.index == df_actual.index
        assert index_equal.all()
        values_equal = df_expected == df_actual
        for column_name in values_equal.keys():
            assert values_equal[column_name].all()

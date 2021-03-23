import pandas as pd
import unittest

import pylife.vmap as vmap
import h5py
import os
import reference_data as RD


class TestExport(unittest.TestCase):
    def setUp(self):
        self._export = vmap.VMAPExport('tests/vmap/testfiles/test.vmap')
        self._import = vmap.VMAPImport('tests/vmap/testfiles/beam_2d_squ_lin.vmap')
        self._mesh = (self._import.make_mesh('1', 'STATE-2')
                      .join_coordinates()
                      .join_variable('STRESS_CAUCHY')
                      .join_variable('DISPLACEMENT')
                      .join_variable('E')
                      .to_frame())

    def tearDown(self):
        # os.remove(self._export.file_name)
        x = 5

    def test_fundamental_groups(self):
        with h5py.File(self._export.file_name, 'r') as file:
            vmap_group = self.try_get_vmap_structure(file, 'VMAP')
            assert vmap_group is not None
            assert len(vmap_group) == 4

            geometry_group = self.try_get_vmap_structure(vmap_group, 'GEOMETRY')
            assert geometry_group is not None
            assert len(geometry_group) == 0

            material_group = self.try_get_vmap_structure(vmap_group, 'MATERIAL')
            assert material_group is not None
            assert len(material_group) == 0

            system_group = self.try_get_vmap_structure(vmap_group, 'SYSTEM')
            assert system_group is not None
            assert len(system_group) == 5
            self.assert_dataset_correct(system_group, 'ELEMENTTYPES', self._export._element_types)
            self.assert_dataset_correct(system_group, 'METADATA', self._export._metadata)
            self.assert_dataset_correct(system_group, 'SECTION', self._export._sections)
            self.assert_dataset_correct(system_group, 'UNITSYSTEM', self._export._unit_system)
            self.assert_dataset_correct(system_group, 'COORDINATESYSTEM', self._export._coordinate_systems)

            variables_group = self.try_get_vmap_structure(vmap_group, 'VARIABLES')
            assert variables_group is not None
            assert len(variables_group) == 0

    def test_add_dataset(self):
        self._export._add_dataset(vmap.VMAPIntegrationType, RD.integration_type_content)
        dataset_name = 'INTEGRATIONTYPES'
        with h5py.File(self._export.file_name, 'r') as file:
            system_group = self.try_get_vmap_structure(file, 'VMAP/SYSTEM')
            assert system_group is not None
            dataset = self.try_get_vmap_structure(system_group, dataset_name)
            assert dataset is not None
            self.assert_dataset_correct(system_group, dataset_name, RD.integration_type_content)

    def test_export(self):
        self._export.add_geometry('1', self._mesh) \
                    .add_geometry('2', self._mesh)
        id_set = pd.Index([1, 2, 3, 4])
        self._export.add_node_set('1', id_set, self._mesh) \
            .add_element_set('1', id_set, self._mesh)
        self._export.add_variable('STATE-2', '1', 'E', self._mesh) \
            .add_variable('STATE-2', '1', 'DISPLACEMENT', self._mesh)

    def try_get_vmap_structure(self, parent, structure_name):
        try:
            structure = parent[structure_name]
            return structure
        except KeyError:
            return None

    def assert_dataset_correct(self, parent, dataset_name, expected_values):
        actual_values = parent[dataset_name]
        assert len(actual_values) == len(expected_values)
        for e_key, a_0 in zip(expected_values, actual_values):
            e_0 = expected_values[e_key]
            assert len(e_0) == len(a_0)
            for e_1, a_1 in zip(e_0, a_0):
                try:
                    for e_2, a_2 in zip(e_1, a_1):
                        assert e_2 == a_2
                except TypeError:
                    assert e_1 == a_1

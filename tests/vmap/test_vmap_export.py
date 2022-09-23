import re
import pandas as pd
import pytest

import pylife.vmap as vmap
import h5py
import os
from . import reference_data as RD
import pylife.vmap.vmap_structures as structures


class TestExport:
    @pytest.fixture(scope='function', autouse=True)
    def prepare_data(self, tmp_path_factory):
        print("prepare data")
        tmpdir = tmp_path_factory.mktemp('vmap-export').as_posix()

        self._export = vmap.VMAPExport(os.path.join(tmpdir, 'test.vmap'))
        self._import_expected = vmap.VMAPImport('tests/vmap/testfiles/beam_2d_squ_lin.vmap')
        self._mesh = (self._import_expected.make_mesh('1', 'STATE-2')
                      .join_coordinates()
                      .join_variable('DISPLACEMENT')
                      .to_frame())
        self._export.add_geometry('1', self._mesh)

    def test_fundamental_groups(self):
        with vmap.VMAPImport(self._export.file_name) as import_actual:
            vmap_group = import_actual.try_get_vmap_object('VMAP')
            assert vmap_group is not None
            assert len(vmap_group) == 4

            geometry_group = import_actual.try_get_vmap_object('VMAP/GEOMETRY')
            assert geometry_group is not None
            assert len(geometry_group) == 1

            material_group = import_actual.try_get_vmap_object('VMAP/MATERIAL')
            assert material_group is not None
            assert len(material_group) == 0

            system_group = import_actual.try_get_vmap_object('VMAP/SYSTEM')
            assert system_group is not None
            assert len(system_group) == 5
            self.assert_dataset_correct(system_group['ELEMENTTYPES'], self._export._element_types)
            self.assert_dataset_correct(system_group['METADATA'], self._export._metadata, False)
            self.assert_dataset_correct(system_group['SECTION'], self._export._sections)
            self.assert_dataset_correct(system_group['UNITSYSTEM'], self._export._unit_system)
            self.assert_dataset_correct(system_group['COORDINATESYSTEM'], self._export._coordinate_systems)

            variables_group = import_actual.try_get_vmap_object('VMAP/VARIABLES')
            assert variables_group is not None
            assert len(variables_group) == 0

    def test_set_geometry_name(self):
        geometry_path = 'VMAP/GEOMETRY/1'
        name = 'PART-1-1'
        self._export.set_group_attribute(geometry_path, 'MYNAME', name)
        with h5py.File(self._export.file_name, 'r') as file:
            assert file[geometry_path].attrs['MYNAME'] == name

    def test_set_invalid_geometry_name(self):
        geometry_path = 'INVALID'
        name = 'PART-1-1'
        with pytest.raises(KeyError, match='VMAP object INVALID does not exist.'):
            self._export.set_group_attribute(geometry_path, 'MYNAME', name)

    def test_add_dataset(self):
        self._export.add_integration_types(RD.integration_type_content)
        dataset_name = 'INTEGRATIONTYPES'
        with vmap.VMAPImport(self._export.file_name) as import_actual:
            dataset = import_actual.try_get_vmap_object('VMAP/SYSTEM/%s' % dataset_name)
            assert dataset is not None
            self.assert_dataset_correct(dataset, RD.integration_type_content)

    def test_add_dataset_already_exists(self):
        self._export.add_integration_types(RD.integration_type_content)
        with pytest.raises(KeyError):
            self._export.add_integration_types(RD.integration_type_content)

    def test_geometry(self):
        geometry_full_path = "VMAP/GEOMETRY/1"
        with vmap.VMAPImport(self._export.file_name) as import_actual:
            geometry_actual = import_actual.try_get_vmap_object(geometry_full_path)
            assert geometry_actual is not None
            geometry_expected = self._import_expected.try_get_vmap_object(geometry_full_path)
            assert geometry_expected is not None

            elements_actual = import_actual.try_get_vmap_object("%s/ELEMENTS" % geometry_full_path)
            assert elements_actual is not None
            elements_expected = self._import_expected.try_get_vmap_object("%s/ELEMENTS" % geometry_full_path)
            assert elements_expected is not None

            self.assert_group_attrs_equal(elements_expected, elements_actual)

            points_actual = import_actual.try_get_vmap_object("%s/POINTS" % geometry_full_path)
            assert points_actual is not None
            points_expected = self._import_expected.try_get_vmap_object("%s/POINTS" % geometry_full_path)
            assert points_expected is not None

            self.assert_group_attrs_equal(points_expected, points_actual)

    def test_add_geometry(self):
        geometry_name = '2'
        self._export.add_geometry(geometry_name, self._mesh)
        mesh_expected = self._mesh[['x', 'y', 'z']]
        with vmap.VMAPImport(self._export.file_name) as import_actual:
            mesh_actual = (import_actual.make_mesh(geometry_name)
                           .join_coordinates()
                           .to_frame())
            pd.testing.assert_frame_equal(mesh_expected.sort_index(), mesh_actual.sort_index())

    def test_add_geometry_invalid(self):
        geometry_name = '2'
        with pytest.raises(vmap.VMAPExportError):
            self._export.add_geometry(geometry_name, 5)

        with vmap.VMAPImport(self._export.file_name) as import_actual:
            geometry = import_actual.try_get_vmap_object('VMAP/GEOMETRY/%s' % geometry_name)
            assert geometry is None

    def test_add_geometry_already_exists(self):
        geometry_name = '1'
        with pytest.raises(KeyError):
            self._export.add_geometry(geometry_name, self._mesh)

    def test_add_node_set(self):
        geometry_name = '1'
        geometry_set_name = '000000'
        geometry_set_full_path = 'VMAP/GEOMETRY/%s/GEOMETRYSETS/%s' % (geometry_name, geometry_set_name)
        geometry_set_expected = self._import_expected.try_get_geometry_set(geometry_name, geometry_set_name)
        assert geometry_set_expected is not None
        self._export.add_node_set(geometry_name, geometry_set_expected, self._mesh, 'ALL')
        with vmap.VMAPImport(self._export.file_name) as import_actual:
            geometry_set_group_expected = self._import_expected.try_get_vmap_object(geometry_set_full_path)
            assert geometry_set_group_expected is not None
            geometry_set_group_actual = import_actual.try_get_vmap_object(geometry_set_full_path)
            assert geometry_set_group_actual is not None
            self.assert_group_attrs_equal(geometry_set_group_expected, geometry_set_group_actual)

            geometry_set_actual = import_actual.try_get_geometry_set(geometry_name, geometry_set_name)
            assert geometry_set_actual is not None
            assert geometry_set_expected.shape == geometry_set_actual.shape
            assert geometry_set_expected.size == geometry_set_actual.size
            assert geometry_set_expected.dtype == geometry_set_actual.dtype
            assert (geometry_set_expected == geometry_set_actual).all()

    def test_add_element_set(self):
        geometry_name = '1'
        geometry_set_name_expected = '000003'
        geometry_set_name_actual = '000000'
        geometry_set_full_path_expected = 'VMAP/GEOMETRY/%s/GEOMETRYSETS/%s' % (geometry_name,
                                                                                geometry_set_name_expected)
        geometry_set_full_path_actual = 'VMAP/GEOMETRY/%s/GEOMETRYSETS/%s' % (geometry_name,
                                                                              geometry_set_name_actual)
        geometry_set_expected = self._import_expected.try_get_geometry_set(geometry_name, geometry_set_name_expected)
        assert geometry_set_expected is not None
        self._export.add_element_set(geometry_name, geometry_set_expected, self._mesh, 'ALL')
        with vmap.VMAPImport(self._export.file_name) as import_actual:
            geometry_set_group_expected = self._import_expected.try_get_vmap_object(geometry_set_full_path_expected)
            assert geometry_set_group_expected is not None
            geometry_set_group_actual = import_actual.try_get_vmap_object(geometry_set_full_path_actual)
            assert geometry_set_group_actual is not None
            self.assert_group_attrs_equal(geometry_set_group_expected, geometry_set_group_actual, 'MYIDENTIFIER')

            geometry_set_actual = import_actual.try_get_geometry_set(geometry_name, geometry_set_name_actual)
            assert geometry_set_actual is not None
            assert geometry_set_expected.shape == geometry_set_actual.shape
            assert geometry_set_expected.size == geometry_set_actual.size
            assert geometry_set_expected.dtype == geometry_set_actual.dtype
            assert (geometry_set_expected == geometry_set_actual).all()

    def test_add_node_set_invalid(self):
        geometry_name = '1'
        invalid_node_set = pd.Index([17, 18, 19])
        with pytest.raises(KeyError, match='Provided index set is not a subset of the node indices.'):
            self._export.add_node_set(geometry_name, invalid_node_set, self._mesh, 'ALL')

    def test_add_element_set_invalid(self):
        geometry_name = '1'
        invalid_elemment_set = pd.Index([17, 18, 19])
        with pytest.raises(KeyError, match='Provided index set is not a subset of the element indices.'):
            self._export.add_element_set(geometry_name, invalid_elemment_set, self._mesh, 'ALL')

    def test_add_element_set_invalid_name(self):
        geometry_name = '1'
        elemment_set = pd.Index([1, 2, 3])
        invalid_name = 123
        with pytest.raises(TypeError, match=re.escape('Invalid set name (must be a string).')):
            self._export.add_element_set(geometry_name, elemment_set, self._mesh, invalid_name)

    def test_add_element_set_invalid_geometry_name(self):
        geometry_name = 'foo'
        elemment_set = pd.Index([1, 2, 3])
        with pytest.raises(KeyError, match=re.escape('No geometry with the name foo')):
            self._export.add_element_set(geometry_name, elemment_set, self._mesh, 'ALL')

    def test_add_variable(self):
        state_name = 'STATE-2'
        state_full_path = 'VMAP/VARIABLES/%s' % state_name
        geometry_name = '1'
        variable_name = 'DISPLACEMENT'
        variable_full_path = '%s/%s/%s' % (state_full_path, geometry_name, variable_name)
        self._export.add_variable(state_name, geometry_name, variable_name, self._mesh)
        with vmap.VMAPImport(self._export.file_name) as import_actual:
            variable_group_expected = self._import_expected.try_get_vmap_object(variable_full_path)
            assert variable_group_expected is not None
            variable_group_actual = import_actual.try_get_vmap_object(variable_full_path)
            assert variable_group_expected is not None
            self.assert_group_attrs_equal(variable_group_expected, variable_group_actual,
                                          'MYIDENTIFIER', 'MYTIMEVALUE', 'MYVARIABLEDESCRIPTION')
            mesh_actual = (import_actual.make_mesh(geometry_name, state_name)
                           .join_variable(variable_name)
                           .to_frame())
        mesh_expected = self._mesh[self._export.variable_column_names(variable_name)]
        pd.testing.assert_frame_equal(mesh_expected.sort_index(), mesh_actual.sort_index())

    def test_add_variable_group_invalid(self):
        state_name = 'STATE-2'
        geometry_name = '2'
        variable_name = 'DISPLACEMENT'
        with pytest.raises(KeyError):
            self._export.add_variable(state_name, geometry_name, variable_name, self._mesh)

    def test_add_variable_name_invalid(self):
        state_name = 'STATE-2'
        geometry_name = '1'
        variable_name = 'DISPLACEMENT2'
        with pytest.raises(KeyError):
            self._export.add_variable(state_name, geometry_name, variable_name, self._mesh)

    def test_add_variable_unknown_location(self):
        state_name = 'STATE-2'
        geometry_name = '1'
        variable_name = 'FORCE_REACTION'
        mesh = (self._import_expected.make_mesh(geometry_name, state_name)
                .join_variable(variable_name, column_names=['RF1', 'RF2', 'RF3'])
                .to_frame())
        with pytest.raises(vmap.APIUseError,
                           match=re.escape(
                               "Need location for unknown variable RF1. "
                               "Please provide one using 'location' parameter.")):
            self._export.add_variable(state_name, geometry_name, 'RF1', mesh, column_names=['RF1'])

    def test_add_variable_unknown_column_name(self):
        state_name = 'STATE-2'
        geometry_name = '1'
        variable_name = 'FORCE_REACTION'
        mesh_expected = (self._import_expected.make_mesh(geometry_name, state_name)
                         .join_variable(variable_name, column_names=['RF1', 'RF2', 'RF3'])
                         .to_frame())
        self._export.add_variable(state_name, geometry_name, 'RF1', mesh_expected, column_names=['RF1'],
                                  location=structures.VariableLocations.NODE)
        with vmap.VMAPImport(self._export.file_name) as import_actual:
            mesh_actual = (import_actual.make_mesh(geometry_name, state_name)
                           .join_variable('RF1', column_names=['RF1'])
                           .to_frame())
            pd.testing.assert_series_equal(mesh_actual['RF1'], mesh_expected['RF1'])

    def test_add_variable_not_present_in_mesh(self):
        state_name = 'STATE-2'
        geometry_name = '1'
        variable_name = 'FORCE_REACTION'
        with pytest.raises(vmap.VMAPExportError):
            self._export.add_variable(state_name, geometry_name, variable_name, self._mesh,
                                      column_names=['RF1'], location=structures.VariableLocations.NODE)
        with vmap.VMAPImport(self._export.file_name) as import_actual:
            variable = import_actual.try_get_vmap_object(
                'VMAP/VARIABLES/%s/%s/%s' % (state_name, geometry_name, variable_name))
            assert variable is None

    def test_add_variable_invalid_location(self):
        state_name = 'STATE-2'
        geometry_name = '1'
        variable_name = 'DISPLACEMENT'
        with pytest.raises(vmap.APIUseError,
                           match=re.escape(
                               "location parameter needs to be of type VariableLocations.")):
            self._export.add_variable(state_name, geometry_name, variable_name, self._mesh,
                                      column_names=['dx', 'dy', 'dz'], location=4)
        with vmap.VMAPImport(self._export.file_name) as import_actual:
            variable = import_actual.try_get_vmap_object(
                'VMAP/VARIABLES/%s/%s/%s' % (state_name, geometry_name, variable_name))
            assert variable is None

    def test_variable_location_displacement(self):
        assert self._export.variable_location('DISPLACEMENT') == structures.VariableLocations.NODE

    def test_variable_location_stress_cauchy(self):
        assert self._export.variable_location('STRESS_CAUCHY') == structures.VariableLocations.ELEMENT_NODAL

    def test_add_variable_already_exists(self):
        state_name = 'STATE-2'
        geometry_name = '1'
        variable_name = 'DISPLACEMENT'
        self._export.add_variable(state_name, geometry_name, variable_name, self._mesh)
        with pytest.raises(KeyError):
            self._export.add_variable(state_name, geometry_name, variable_name, self._mesh)

    """
    def test_all(self):
        self.test_add_dataset()
        self.test_add_geometry()
        self.test_add_node_set()
        self.test_add_variable()
    """

    def assert_dataset_correct(self, dataset, expected_values, is_compound=True):
        assert len(dataset) == len(expected_values)
        for e_key, a_0 in zip(expected_values, dataset):
            e_0 = expected_values[e_key]
            if is_compound:
                a_0 = a_0[0]
            assert len(e_0) == len(a_0)
            for e_1, a_1 in zip(e_0, a_0):
                if isinstance(e_1, list):
                    for e_2, a_2 in zip(e_1, a_1):
                        assert e_2 == a_2
                    continue
                e_1 = self.make_bytearray_if_str(e_1)
                a_1 = self.make_bytearray_if_str(a_1)
                assert e_1 == a_1

    def assert_group_attrs_equal(self, group_expected, group_actual, *args):
        attributes_expected = list(group_expected.attrs.items())
        for attr_expected in attributes_expected:
            assert attr_expected[0] in group_actual.attrs
            if attr_expected[0] not in args:
                testval = self.make_bytearray_if_str(group_actual.attrs[attr_expected[0]])
                assert testval == attr_expected[1]

    def make_bytearray_if_str(self, value):
        if isinstance(value, str):
            return bytearray(value, 'utf-8')
        return value


@pytest.mark.parametrize('filename', [
    'beam_2d_tri_lin.vmap',
    'beam_2d_tri_quad.vmap',
    'beam_2d_squ_lin.vmap',
    'beam_2d_squ_quad.vmap',
    'beam_3d_tet_lin.vmap',
    'beam_3d_tet_quad.vmap',
    'beam_3d_wedge_lin.vmap',
    'beam_3d_wedge_quad.vmap',
    'beam_3d_hex_lin.vmap',
    'beam_3d_hex_quad.vmap',
])
def test_export_import_round_robin(tmpdir, filename):
    filename = os.path.join('tests/vmap/testfiles/', filename)
    import_expected = vmap.VMAPImport(filename)
    mesh = (import_expected.make_mesh('1', 'STATE-2')
            .join_coordinates()
            .join_variable('STRESS_CAUCHY')
            .join_variable('DISPLACEMENT')
            .join_variable('E')
            .to_frame())

    export_filename = os.path.join(tmpdir, 'export.vmap')
    (vmap.VMAPExport(export_filename)
     .add_geometry('1', mesh)
     .add_variable('STATE-2', '1', 'STRESS_CAUCHY', mesh)
     .add_variable('STATE-2', '1', 'DISPLACEMENT', mesh)
     .add_variable('STATE-2', '1', 'E', mesh))

    reimport = vmap.VMAPImport(export_filename)
    reimported_mesh = (reimport.make_mesh('1', 'STATE-2')
                       .join_coordinates()
                       .join_variable('STRESS_CAUCHY')
                       .join_variable('DISPLACEMENT')
                       .join_variable('E')
                       .to_frame())

    pd.testing.assert_frame_equal(mesh, reimported_mesh)


def test_os_error_on_open_file():
    with pytest.raises(FileNotFoundError):
        self._export = vmap.VMAPExport(os.path.join("/some/most/probably/not/existing/path", "test.vmap"))

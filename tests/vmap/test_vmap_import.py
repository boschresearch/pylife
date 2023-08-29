import re

import numpy as np
import pandas as pd
import pytest

import pylife.vmap as vmap

from . import reference_data as RD


@pytest.fixture
def beam_2d_squ():
    return vmap.VMAPImport('tests/vmap/testfiles/beam_2d_squ_lin.vmap')


@pytest.fixture
def beam_3d_hex():
    return vmap.VMAPImport('tests/vmap/testfiles/beam_3d_hex_lin.vmap')


@pytest.fixture
def rotsym_quad():
    return vmap.VMAPImport('tests/vmap/testfiles/rotsym_quad_lin.vmap')


@pytest.fixture
def beam_2d_squ_lin_and_quad():
    return vmap.VMAPImport('tests/vmap/testfiles/beam_2d_squ_lin_and_quad.vmap')


def assert_list_equal(l1, l2):
    assert len(l1) == len(l2)
    for e in l1:
        assert e in l2


def test_get_geometries(beam_2d_squ_lin_and_quad):
    assert_list_equal(beam_2d_squ_lin_and_quad.geometries(), ['1', '2'])


def test_get_states(beam_2d_squ):
    assert_list_equal(beam_2d_squ.states(), ['STATE-1', 'STATE-2'])


def test_get_nodes_2d(beam_2d_squ):
    pd.testing.assert_frame_equal(beam_2d_squ.nodes('1'), RD.beam_2d_squ_nodes)


def test_make_mesh_no_set(beam_2d_squ):
    df = beam_2d_squ.make_mesh('1').to_frame()
    assert df.shape[1] == 0
    pd.testing.assert_index_equal(df.index, RD.beam_2d_squ_mesh_coords.index)


def test_make_mesh_unknown_geometry_one_geometry(beam_2d_squ):
    with pytest.raises(KeyError, match=re.escape("Geometry 'foo' not found. "
                                                   "Available geometries: ['1'].")):
        beam_2d_squ.make_mesh('foo')


def test_make_mesh_unknown_geometry_two_geometries(beam_2d_squ_lin_and_quad):
    with pytest.raises(KeyError, match=re.escape("Geometry 'foo' not found. "
                                                   "Available geometries: ['1', '2'].")):
        beam_2d_squ_lin_and_quad.make_mesh('foo')


def test_get_make_mesh_beam_2d_sq_node_set_load(beam_2d_squ):
    pd.testing.assert_index_equal(beam_2d_squ.make_mesh('1').filter_node_set('LOAD').to_frame().index,
                                  RD.beam_2d_squ_mesh_index_load)


def test_get_make_mesh_beam_2d_sq_node_set_fix(beam_2d_squ):
    pd.testing.assert_index_equal(beam_2d_squ.make_mesh('1').filter_node_set('FIX').to_frame().index,
                                  RD.beam_2d_squ_mesh_index_fix)


def test_get_make_mesh_rotsym_quad_element_set_ysym(rotsym_quad):
    pd.testing.assert_index_equal(rotsym_quad.make_mesh('1').filter_element_set('YSYM').to_frame().index,
                                  RD.rotsym_quad_mesh_index_ysym)


def test_make_mesh_join_coordinates_filtered_node_set(beam_2d_squ):
    pd.testing.assert_frame_equal(beam_2d_squ.make_mesh('1')
                                  .filter_node_set('ALL')
                                  .join_coordinates().to_frame(),
                                  RD.beam_2d_squ_mesh_coords)


def test_make_mesh_join_coordinates_filtered_element_set(beam_2d_squ):
    pd.testing.assert_frame_equal(beam_2d_squ.make_mesh('1')
                                  .filter_element_set('ALL')
                                  .join_coordinates().to_frame(),
                                  RD.beam_2d_squ_mesh_coords)


def test_make_mesh_join_coordinates_unfiltered(beam_2d_squ):
    pd.testing.assert_frame_equal(beam_2d_squ.make_mesh('1').join_coordinates().to_frame(),
                                  RD.beam_2d_squ_mesh_coords)


def test_make_mesh_join_coordinates_no_mesh(beam_2d_squ):
    with pytest.raises(vmap.APIUseError, match=re.escape("Need to make_mesh() before joining the coordinates.")):
        beam_2d_squ.join_coordinates()


def test_make_mesh_join_coordinates_node_set_load(rotsym_quad):
    pd.testing.assert_frame_equal(rotsym_quad.make_mesh('1').filter_element_set('YSYM').join_coordinates().to_frame(),
                                  RD.rotsym_quad_mesh_coords_ysym.loc[RD.rotsym_quad_mesh_index_ysym])


def test_make_mesh_join_coordinates_element_set_ysym(beam_2d_squ):
    pd.testing.assert_frame_equal(beam_2d_squ.make_mesh('1').filter_node_set('LOAD').join_coordinates().to_frame(),
                                  RD.beam_2d_squ_mesh_coords.loc[RD.beam_2d_squ_mesh_index_load])


def test_get_make_mesh_fail_nonexistant_node_set(beam_2d_squ):
    with pytest.raises(KeyError, match="Node set 'foo' not found in geometry '1'"):
        beam_2d_squ.make_mesh('1').filter_node_set('foo')


def test_get_make_mesh_fail_nonexistant_element_set(beam_2d_squ):
    with pytest.raises(KeyError, match="Element set 'foo' not found in geometry '1'"):
        beam_2d_squ.make_mesh('1').filter_element_set('foo')


def test_filter_node_set_no_mesh(beam_2d_squ):
    with pytest.raises(vmap.APIUseError, match=re.escape("Need to make_mesh() before filtering node or element sets.")):
        beam_2d_squ.filter_node_set('FIX')


def test_filter_element_set_no_mesh(beam_2d_squ):
    with pytest.raises(vmap.APIUseError, match=re.escape("Need to make_mesh() before filtering node or element sets.")):
        beam_2d_squ.filter_element_set('YSYM')


def test_to_frame_no_mesh(beam_2d_squ):
    with pytest.raises(vmap.APIUseError, match=re.escape("Need to make_mesh() before requesting a resulting frame.")):
        beam_2d_squ.to_frame()


def test_to_frame_twice(beam_2d_squ):
    vm = beam_2d_squ.make_mesh('1')
    vm.to_frame()
    with pytest.raises(vmap.APIUseError, match=re.escape("Need to make_mesh() before requesting a resulting frame.")):
        vm.to_frame()


def test_join_variable_no_mesh(beam_2d_squ):
    with pytest.raises(vmap.APIUseError, match=re.escape("Need to make_mesh() before joining a variable.")):
        beam_2d_squ.join_variable('STATE-1', 'STRESS_CAUCHY')


def test_available_variables(beam_2d_squ_lin_and_quad):
    np.testing.assert_array_equal(beam_2d_squ_lin_and_quad.variables('1', 'STATE-1'),
                                  ['FORCE_CONCENTRATED', 'E', 'FORCE_REACTION', 'STRESS_CAUCHY', 'DISPLACEMENT'])
    np.testing.assert_array_equal(beam_2d_squ_lin_and_quad.variables('1', 'STATE-2'),
                                  ['FORCE_CONCENTRATED', 'E', 'STRESS_CAUCHY', 'DISPLACEMENT'])
    np.testing.assert_array_equal(beam_2d_squ_lin_and_quad.variables('2', 'STATE-1'),
                                  ['E', 'FORCE_REACTION', 'STRESS_CAUCHY', 'DISPLACEMENT'])


def test_available_variables_unknown_geometry(beam_2d_squ_lin_and_quad):
    with pytest.raises(KeyError, match=re.escape("Geometry")):
        beam_2d_squ_lin_and_quad.variables('foo', 'STATE-1')


def test_available_variables_unknown_state(beam_2d_squ_lin_and_quad):
    with pytest.raises(KeyError, match=re.escape("State")):
        beam_2d_squ_lin_and_quad.variables('1', 'state-foo')


def test_available_variables_unknown_geometry_in_state(beam_2d_squ_lin_and_quad):
    with pytest.raises(KeyError, match=re.escape("Geometry '2' not available in state 'STATE-2'.")):
        beam_2d_squ_lin_and_quad.variables('2', 'STATE-2')


def test_join_node_variable_displacement(beam_2d_squ):
    var_frame = beam_2d_squ.make_mesh('1').join_variable('DISPLACEMENT', 'STATE-2').to_frame()
    groups = var_frame.groupby('node_id')
    pd.testing.assert_frame_equal(groups.mean(), RD.beam_2d_squ_node_displacement, check_index_type=False)
    pd.testing.assert_frame_equal(groups.min(), RD.beam_2d_squ_node_displacement, check_index_type=False)
    pd.testing.assert_frame_equal(groups.max(), RD.beam_2d_squ_node_displacement, check_index_type=False)


def test_join_variable_unknown_state(beam_2d_squ):
    with pytest.raises(KeyError, match=re.escape("State 'foo' not found. "
                                                 "Available states: ['STATE-1', 'STATE-2'].")):
        beam_2d_squ.make_mesh('1').join_variable('DISPLACEMENT', 'foo')


def test_join_variable_unknown_variable(beam_2d_squ):
    with pytest.raises(KeyError, match=re.escape("Variable 'foo' not found in geometry '1', 'STATE-1'.")):
        beam_2d_squ.make_mesh('1').join_variable('foo', 'STATE-1', column_names=['foo'])


def test_join_element_variable_evol_no_column_name(beam_3d_hex):
    with pytest.raises(KeyError, match="No column name for variable EVOL. Please provide with column_names parameter"):
        beam_3d_hex.make_mesh('1').join_variable('EVOL', 'STATE-2')


def test_join_element_variable_evol(beam_3d_hex):
    var_frame = beam_3d_hex.make_mesh('1').join_variable('EVOL', 'STATE-2', column_names=['Ve']).to_frame()
    groups = var_frame.groupby('element_id')
    pd.testing.assert_frame_equal(groups.mean(), RD.beam_3d_hex_element_volume)
    pd.testing.assert_frame_equal(groups.min(), RD.beam_3d_hex_element_volume)
    pd.testing.assert_frame_equal(groups.max(), RD.beam_3d_hex_element_volume)


def test_join_element_nodal_variable_stress(beam_2d_squ):
    var_frame = beam_2d_squ.make_mesh('1').join_variable('STRESS_CAUCHY', 'STATE-2').to_frame()
    pd.testing.assert_frame_equal(var_frame, RD.beam_2d_squ_element_nodal_stress)


def test_join_element_nodal_variable_stress_override_column_names(beam_2d_squ):
    var_frame = (beam_2d_squ.make_mesh('1').join_variable('STRESS_CAUCHY', 'STATE-2',
                                                          column_names=['s11', 's22', 's33', 's12', 's13', 's23'])
                 .to_frame())
    reference = RD.beam_2d_squ_element_nodal_stress.copy()
    reference.columns = ['s11', 's22', 's33', 's12', 's13', 's23']
    pd.testing.assert_frame_equal(var_frame, reference)


def test_join_element_nodal_variable_stress_column_name_list_unmatch(beam_2d_squ):
    with pytest.raises(ValueError,
                       match=re.escape("Length of column name list (3) does not match variable dimension (6).")):
        beam_2d_squ.make_mesh('1').join_variable('STRESS_CAUCHY', 'STATE-2', column_names=['s11', 's22', 's33'])


def test_join_element_nodal_variable_strain(beam_2d_squ):
    var_frame = beam_2d_squ.make_mesh('1').join_variable('E', 'STATE-2').to_frame()
    pd.testing.assert_frame_equal(var_frame, RD.beam_2d_squ_element_nodal_strain)


def test_join_element_nodal_variable_node_set(beam_2d_squ):
    pd.testing.assert_frame_equal(beam_2d_squ
                                  .make_mesh('1')
                                  .filter_node_set('FIX')
                                  .join_variable('STRESS_CAUCHY', 'STATE-2')
                                  .to_frame(),
                                  RD.beam_2d_squ_element_nodal_stress.loc[RD.beam_2d_squ_mesh_index_fix])


def test_join_element_nodal_variable_element_set(rotsym_quad):
    pd.testing.assert_frame_equal(rotsym_quad.make_mesh('1')
                                  .filter_element_set('YSYM')
                                  .join_variable('STRESS_CAUCHY', 'STATE-2')
                                  .to_frame(),
                                  RD.rotsym_quad_stress_cauchy.loc[RD.rotsym_quad_mesh_index_ysym])


def test_join_element_nodal_variable_stress_element_variable_evol(beam_3d_hex):
    var_frame = (beam_3d_hex
                 .make_mesh('1')
                 .join_variable('STRESS_CAUCHY', 'STATE-2')
                 .join_variable('EVOL', column_names=['V_e'])
                 .to_frame())
    pd.testing.assert_frame_equal(var_frame, RD.beam_3d_hex_stress_element_volume, rtol=1e-4, check_index_type=False)


def test_join_variable_unsupported_location():
    vm = vmap.VMAPImport('tests/vmap/testfiles/beam_at_integration_points.vmap').make_mesh('1')
    with pytest.raises(vmap.FeatureNotSupportedError,
                       match="Unsupported value location, sorry\nSupported: NODE, ELEMENT, ELEMENT NODAL"):
        vm.join_variable('STRESS_CAUCHY', 'STATE-2')


def test_get_element_nodal_variable_strain_predefined_state(beam_2d_squ):
    var_frame = beam_2d_squ.make_mesh('1', 'STATE-2').join_variable('E').to_frame()
    pd.testing.assert_frame_equal(var_frame, RD.beam_2d_squ_element_nodal_strain)


def test_get_variables_multiple_states(beam_2d_squ):
    var_frame = (beam_2d_squ.make_mesh('1')
                 .join_variable('STRESS_CAUCHY', 'STATE-1', column_names=['S11_0', 'S22_0', 'S33_0', 'S12_0', 'S13_0', 'S23_0'])
                 .join_variable('E', 'STATE-2')
                 .join_variable('STRESS_CAUCHY')
                 .to_frame())
    strain = RD.beam_2d_squ_element_nodal_strain
    stress = RD.beam_2d_squ_element_nodal_stress
    zero_stress = pd.DataFrame(np.zeros((strain.shape[0], 6)),
                               columns=['S11_0', 'S22_0', 'S33_0', 'S12_0', 'S13_0', 'S23_0'],
                               index=strain.index)
    pd.testing.assert_frame_equal(var_frame, zero_stress.join(strain).join(stress))


def test_join_variable_no_given_state(beam_2d_squ):
    with pytest.raises(vmap.APIUseError,
                       match=re.escape("No state name given.\n"
                                       "Must be either given in make_mesh() or in join_variable() as optional state argument.")):
        beam_2d_squ.make_mesh('1').join_variable('STRESS_CAUCHY')


def test_get_node_sets_beam(beam_2d_squ):
    assert_list_equal(beam_2d_squ.node_sets('1'), ['ALL', 'FIX', 'LOAD'])


def test_get_node_sets_rotsym(rotsym_quad):
    assert_list_equal(rotsym_quad.node_sets('1'), ['ALL', 'LOAD', 'YSYM', 'ROTSYM', 'SLOPEDSURFACE'])


def test_get_element_sets_beam(beam_2d_squ):
    assert_list_equal(beam_2d_squ.element_sets('1'), ['ALL'])


def test_get_element_sets_rotsym(rotsym_quad):
    assert_list_equal(rotsym_quad.element_sets('1'), ['ALL', 'LOAD', 'YSYM', 'ROTSYM', 'SLOPEDSURFACE', 'LOADSURFACE'])


def test_get_mesh_index_fail_nonexistant_node_set(beam_2d_squ):
    with pytest.raises(KeyError, match="Node set 'foo' not found in geometry '1'"):
        beam_2d_squ.make_mesh('1').filter_node_set('foo')


def test_get_mesh_index_fail_nonexistant_element_set(beam_2d_squ):
    with pytest.raises(KeyError, match="Element set 'foo' not found in geometry '1'"):
        beam_2d_squ.make_mesh('1').filter_element_set('foo')

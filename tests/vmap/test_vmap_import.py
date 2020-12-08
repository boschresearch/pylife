import re

import pandas as pd
import pytest

import pylife.vmap as vmap

import reference_data as RD


@pytest.fixture
def beam_2d_squ():
    return vmap.VMAP('tests/vmap/testfiles/beam_2d_squ_lin.vmap')


@pytest.fixture
def beam_3d_hex():
    return vmap.VMAP('tests/vmap/testfiles/beam_3d_hex_lin.vmap')


@pytest.fixture
def rotsym_quad():
    return vmap.VMAP('tests/vmap/testfiles/rotsym_quad_lin.vmap')


def assert_list_equal(l1, l2):
    assert len(l1) == len(l2)
    for e in l1:
        assert e in l2


def test_get_geometries(beam_2d_squ):
    assert_list_equal(beam_2d_squ.geometries(), ['1'])


def test_get_states(beam_2d_squ):
    assert_list_equal(beam_2d_squ.states(), ['STATE-1', 'STATE-2'])


def test_get_nodes_2d(beam_2d_squ):
    pd.testing.assert_frame_equal(beam_2d_squ.nodes('1'), RD.beam_2d_squ_nodes)


def test_get_mesh_index(beam_2d_squ):
    pd.testing.assert_index_equal(beam_2d_squ.mesh_index('1'), RD.beam_2d_squ_mesh_index)


def test_get_mesh_coords(beam_2d_squ):
    pd.testing.assert_frame_equal(beam_2d_squ.mesh_coords('1'), RD.beam_2d_squ_mesh_coords)


def test_get_node_variable_displacement(beam_2d_squ):
    var_frame = beam_2d_squ.variable('1', 'STATE-2', 'DISPLACEMENT')
    pd.testing.assert_frame_equal(var_frame, RD.beam_2d_squ_node_displacement)


def test_get_element_variable_evol_no_column_name(beam_3d_hex):
    with pytest.raises(KeyError, match="No column name for variable EVOL. Please povide with column_names parameter"):
        beam_3d_hex.variable('1', 'STATE-2', 'EVOL')


def test_get_element_variable_evol(beam_3d_hex):
    var_frame = beam_3d_hex.variable('1', 'STATE-2', 'EVOL', column_names=['Ve'])
    pd.testing.assert_frame_equal(var_frame, RD.beam_3d_hex_node_displacement)


def test_get_element_nodal_variable_stress(beam_2d_squ):
    var_frame = beam_2d_squ.variable('1', 'STATE-2', 'STRESS_CAUCHY')
    pd.testing.assert_frame_equal(var_frame, RD.beam_2d_squ_element_nodal_stress)


def test_get_element_nodal_variable_stress_override_column_names(beam_2d_squ):
    var_frame = beam_2d_squ.variable('1', 'STATE-2', 'STRESS_CAUCHY',
                                     column_names=['s11', 's22', 's33', 's12', 's13', 's23'])
    reference = RD.beam_2d_squ_element_nodal_stress
    reference.columns = ['s11', 's22', 's33', 's12', 's13', 's23']
    pd.testing.assert_frame_equal(var_frame, reference)


def test_get_element_nodal_variable_stress_column_name_list_unmatch(beam_2d_squ):
    with pytest.raises(ValueError,
                       match=re.escape("Length of column name list (3) does not match variable dimension (6).")):
        beam_2d_squ.variable('1', 'STATE-2', 'STRESS_CAUCHY', column_names=['s11', 's22', 's33'])


def test_get_element_nodal_variable_strain(beam_2d_squ):
    var_frame = beam_2d_squ.variable('1', 'STATE-2', 'E')
    pd.testing.assert_frame_equal(var_frame, RD.beam_2d_squ_element_nodal_strain)


def test_unsupported_location():
    vm = vmap.VMAP('tests/vmap/testfiles/beam_at_integration_points.vmap')
    with pytest.raises(vmap.FeatureNotSupportedError,
                       match="Unsupported value location, sorry\nSupported: NODE, ELEMENT, ELEMENT NODAL"):
        vm.variable('1', 'STATE-2', 'STRESS_CAUCHY')


def test_get_node_sets_beam(beam_2d_squ):
    assert_list_equal(beam_2d_squ.node_sets('1'), ['ALL', 'FIX', 'LOAD'])


def test_get_node_sets_rotsym(rotsym_quad):
    assert_list_equal(rotsym_quad.node_sets('1'), ['ALL', 'LOAD', 'YSYM', 'ROTSYM', 'SLOPEDSURFACE'])


def test_get_element_sets_beam(beam_2d_squ):
    assert_list_equal(beam_2d_squ.element_sets('1'), ['ALL'])


def test_get_element_sets_rotsym(rotsym_quad):
    assert_list_equal(rotsym_quad.element_sets('1'), ['ALL', 'LOAD', 'YSYM', 'ROTSYM', 'SLOPEDSURFACE', 'LOADSURFACE'])


def test_get_mesh_index_beam_2d_sq_node_set_load(beam_2d_squ):
    pd.testing.assert_index_equal(beam_2d_squ.mesh_index('1', node_set='LOAD'), RD.beam_2d_squ_mesh_index_load)


def test_get_mesh_index_beam_2d_sq_node_set_fix(beam_2d_squ):
    pd.testing.assert_index_equal(beam_2d_squ.mesh_index('1', node_set='FIX'), RD.beam_2d_squ_mesh_index_fix)


def test_get_mesh_index_rotsym_quad_element_set_ysym(rotsym_quad):
    pd.testing.assert_index_equal(rotsym_quad.mesh_index('1', element_set='YSYM'), RD.rotsym_quad_mesh_index_ysym)


def test_get_mesh_index_fail_elset_nset(beam_2d_squ):
    with pytest.raises(ValueError, match=("Cannot make mesh index for element set and node set at same time\n"
                                          "Please specify at most one of element_set or node_set. Not both of them.")):
        beam_2d_squ.mesh_index('1', node_set='FIX', element_set='ALL')


def test_get_mesh_index_fail_nonexistant_node_set(beam_2d_squ):
    with pytest.raises(KeyError, match="Node set 'foo' not found in geometry '1'"):
        beam_2d_squ.mesh_index('1', node_set='foo')


def test_get_mesh_index_fail_nonexistant_element_set(beam_2d_squ):
    with pytest.raises(KeyError, match="Element set 'foo' not found in geometry '1'"):
        beam_2d_squ.mesh_index('1', element_set='foo')

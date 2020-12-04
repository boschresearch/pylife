import numpy as np
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


def test_get_nodes_2d(beam_2d_squ):
    pd.testing.assert_frame_equal(beam_2d_squ.nodes('1'), RD.beam_2d_squ_nodes)


def test_get_mesh_index(beam_2d_squ):
    pd.testing.assert_index_equal(beam_2d_squ.mesh_index('1'), RD.beam_2d_squ_mesh_index)


def test_get_mesh_coords(beam_2d_squ):
    pd.testing.assert_frame_equal(beam_2d_squ.mesh_coords('1'), RD.beam_2d_squ_mesh_coords)


def test_get_node_variable_displacement(beam_2d_squ):
    var_frame = beam_2d_squ.variable('1', 'STATE-2', 'DISPLACEMENT')
    pd.testing.assert_frame_equal(var_frame, RD.beam_2d_squ_node_displacement)


def test_get_element_variable_evol(beam_3d_hex):
    var_frame = beam_3d_hex.variable('1', 'STATE-2', 'EVOL')
    pd.testing.assert_frame_equal(var_frame, RD.beam_3d_hex_node_displacement)


def test_get_element_nodal_variable_stress(beam_2d_squ):
    var_frame = beam_2d_squ.variable('1', 'STATE-2', 'STRESS_CAUCHY')
    pd.testing.assert_frame_equal(var_frame, RD.beam_2d_squ_element_nodal_stress)


def test_unsupported_location():
    vm = vmap.VMAP('tests/vmap/testfiles/beam_at_integration_points.vmap')
    with pytest.raises(vmap.FeatureNotSupportedError,
                       match="Unsupported value location, sorry\nSupported: NODE, ELEMENT, ELEMENT NODAL"):
        vm.variable('1', 'STATE-2', 'STRESS_CAUCHY')

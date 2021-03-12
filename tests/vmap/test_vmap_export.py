import re

import numpy as np
import pandas as pd
import pytest

import pylife.vmap as vmap

import reference_data as RD


@pytest.fixture
def config():
    return vmap.VMAPExport('tests/vmap/testfiles/test.vmap')


def test_export(config):
    vmap_import = vmap.VMAPImport('demos/plate_with_hole.vmap')
    mesh = (vmap_import.make_mesh('1', 'STATE-2')
            .join_coordinates()
            .join_variable('STRESS_CAUCHY')
            .join_variable('DISPLACEMENT')
            .to_frame())
    int_type_1 = vmap.VMAPIntegrationType('GAUSS_TRIANGLE_3', 3, 2, 0.0,
                                          [0.166667, 0.166667, 0.666667, 0.166667, 0.166667, 0.666667],
                                          [0.333333, 0.333333, 0.333333])
    int_type_2 = vmap.VMAPIntegrationType('GAUSS_QUAD_9', 9, 2, 0.0,
                                          [-0.774597, -0.774597, 0, -0.774597, 0.774597, -0.774597, -0.774597, 0, 0, 0,
                                           0.774597, 0, -0.774597, 0.774597, 0, 0.774597, 0.774597, 0.774597],
                                          [0.308642, 0.493827, 0.308642, 0.493827, 0.790123, 0.493827, 0.308642,
                                           0.493827, 0.308642])
    config.create_dataset(int_type_1, int_type_2)
    coord_system_1 = vmap.VMAPCoordinateSystem(2, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    config.create_dataset(coord_system_1)
    element_type_1 = vmap.VMAPElementType('VMAP_ELEM_2D_TRIANGLE_6', 'Abaqus: CPS6M', 6, 2, 7, 5, -1, 3, 1)
    element_type_2 = vmap.VMAPElementType('VMAP_ELEM_2D_QUAD_8', 'Abaqus: CPS8', 8, 2, 9, 6, -1, 3, 1)
    config.create_dataset(element_type_1, element_type_2)
    config.create_geometry('1', mesh)

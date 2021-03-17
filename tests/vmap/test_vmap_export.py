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
            .join_variable('E')
            .to_frame())
    int_type_1 = vmap.VMAPIntegrationType('GAUSS_TRIANGLE_3', 3, 2, 0.0,
                                          [0.166667, 0.166667, 0.666667, 0.166667, 0.166667, 0.666667],
                                          [0.333333, 0.333333, 0.333333])
    int_type_2 = vmap.VMAPIntegrationType('GAUSS_QUAD_9', 9, 2, 0.0,
                                          [-0.774597, -0.774597, 0, -0.774597, 0.774597, -0.774597, -0.774597, 0, 0, 0,
                                           0.774597, 0, -0.774597, 0.774597, 0, 0.774597, 0.774597, 0.774597],
                                          [0.308642, 0.493827, 0.308642, 0.493827, 0.790123, 0.493827, 0.308642,
                                           0.493827, 0.308642])
    config.create_VMAP_dataset(True, int_type_1, int_type_2)
    coord_system_1 = vmap.VMAPCoordinateSystem(2, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    config.create_VMAP_dataset(True, coord_system_1)
    config.create_geometry('1', mesh)
    config.add_variable('STATE-2', '1', 'E', mesh)


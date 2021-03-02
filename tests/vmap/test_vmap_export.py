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
    config.create_geometry('1', mesh)

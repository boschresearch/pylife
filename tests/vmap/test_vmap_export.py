import re

import numpy as np
import pandas as pd
import pytest

import pylife.vmap as vmap

import reference_data as RD


@pytest.fixture
def config():
    return vmap.VMAPExport('d:/repos/pylife/dev/tests/vmap/testfiles/test.vmap')


def test_export(config):
    config.create_vmap_groups()

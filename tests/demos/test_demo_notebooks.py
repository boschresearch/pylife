# Copyright (c) 2019-2021 - for information on the respective copyright owner
# see the NOTICE file and/or the repository
# https://github.com/boschresearch/pylife
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import numpy as np
from testbook import testbook

pytestmark = pytest.mark.demos


@pytest.fixture(autouse=True)
def change_workingdir_dir(monkeypatch):
    monkeypatch.chdir('demos')


@testbook('demos/hotspot_plate.ipynb', execute=True)
def test_hotspot_plate(tb):
    tb.inject(
        """
        pd.testing.assert_index_equal(
            first_hotspot.index,
            pd.MultiIndex.from_tuples([(456, 5)], names=['element_id', 'node_id'])
        )
        """
    )
    tb.inject(
        """
        pd.testing.assert_index_equal(
            second_hotspot.index,
            pd.MultiIndex.from_tuples([(2852, 9)], names=['element_id', 'node_id'])
        )
        """
    )


@testbook('demos/local_stress_with_FE.ipynb', execute=True)
def test_local_stress_with_fe(tb):
    tb.inject("np.testing.assert_approx_equal(damage.max(), 0.002319614543675979)")



@testbook('demos/psd_optimizer.ipynb', execute=True)
def test_psd_optimizer(tb):
    tb.inject(
        """
        np.testing.assert_allclose(
            [rms_psd(psdSignal.psd_smoother(psd, fsel, fit)) for fit in np.linspace(0, 1, 4)],
            [19.30855280765641, 11.13723910736734, 9.997749625841077, np.nan]
        )
        """
    )


@testbook('demos/ramberg_osgood.ipynb', execute=True)
def test_ramberg_osgood(tb):
    tb.inject(
        """
        assert monotone_strain.max() == hyst_strain.max()
        assert hyst_strain.max() == - hyst_strain.min()
        """
    )
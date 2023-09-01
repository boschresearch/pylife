# Copyright (c) 2019-2023 - for information on the respective copyright owner
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
import pandas as pd

import pylife.strength.meanstress as MST


@pytest.fixture
def ms_sens():
    return pd.Series({'M': 0.5, 'M2': 0.5/3.})


def test_meanstress_collective_empty_fail():
    df = pd.DataFrame({'foo': [], 'bar': []})
    with pytest.raises(AttributeError, match="Load collective"):
        df.meanstress_transform


def test_meanstress_collective_empty_fkm_goodman(ms_sens):
    df = pd.DataFrame(columns=['from', 'to'])

    res = df.meanstress_transform.fkm_goodman(ms_sens, -1.).to_pandas()
    assert 'from' in res.columns
    assert 'to' in res.columns
    assert len(res.columns) == 2
    assert res.shape[0] == 0


def test_meanstress_collective_fkm_goodman_single_ms_sens(ms_sens):
    df = pd.DataFrame({
        'from': [-6., -4.,  -5./2., -1., -0.4, 0., 7./12.],
        'to': [-2., 0., 0.5, 1., 1.2, 4./3., 21./12.]
    }, index=[3, 4, 5, 6, 7, 8, 9])

    res = df.meanstress_transform.fkm_goodman(ms_sens, -1.)

    expected_amplitude = pd.Series(np.ones(7), name='amplitude', index=df.index)
    pd.testing.assert_series_equal(res.amplitude, expected_amplitude)

    expected_meanstress = pd.Series(np.zeros(7), name='meanstress', index=df.index)
    pd.testing.assert_series_equal(res.meanstress, expected_meanstress)


def test_meanstress_collective_fkm_goodman_multiple_ms_sens():
    df = pd.DataFrame({
        'from': [-6., -4.,  -5./2., -1., -0.4, 0., 7./12.],
        'to': [-2., 0., 0.5, 1., 1.2, 4./3., 21./12.]
    }, index=pd.Index([3, 4, 5, 6, 7, 8, 9], name='element_id'))

    ms_sens = pd.DataFrame({
        'M': [0.5, 0.4],
        'M2': [0.5/3., 0.4/3.]
    })

    expected_index = pd.MultiIndex.from_tuples([
        (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
        (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)
    ], names=[None, 'element_id'])

    res = df.meanstress_transform.fkm_goodman(ms_sens, -1.)

    expected_amplitude = pd.Series(
        [1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.1, 1.0, 0.96, 0.93, 0.91],
        index=expected_index,
        name='amplitude'
    )
    pd.testing.assert_series_equal(res.amplitude, expected_amplitude, rtol=1e-2)

    expected_meanstress = pd.Series(
        np.zeros(14),
        index=expected_index,
        name='meanstress'
    )
    pd.testing.assert_series_equal(res.meanstress, expected_meanstress, rtol=1e-2)

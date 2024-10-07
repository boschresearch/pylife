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

import sys, os, copy
import warnings
import pytest

import numpy as np
import pandas as pd
import numpy.testing as testing

import pylife.strength.meanstress as MST


@pytest.fixture
def ms_sens():
    return pd.Series({'M': 0.5, 'M2': 0.5/3.})


def test_meanstress_collective_empty_fail():
    df = pd.Series([], index=pd.MultiIndex.from_tuples([], names= ['foo', 'bar']), dtype=np.float64)

    with pytest.raises(AttributeError, match="Load collective"):
        df.meanstress_transform


def test_meanstress_collective_empty_fkm_goodman(ms_sens):
    from_intervals = pd.interval_range(0., 0., 0)
    to_intervals = pd.interval_range(0., 0., 0)
    index = pd.MultiIndex.from_product([from_intervals, to_intervals], names=['from', 'to'])

    ser = pd.Series([], index=index, dtype=np.float64)

    expected = pd.Series([],
                         index=pd.IntervalIndex(to_intervals, name='range'),
                         name='cycles', dtype=np.float64)
    res = ser.meanstress_transform.fkm_goodman(ms_sens, -1.).to_pandas()
    pd.testing.assert_series_equal(res, expected)


def test_meanstress_collective_fkm_goodman_single_ms_sens(ms_sens):
    fr = pd.IntervalIndex.from_breaks(np.linspace(-25./24., 1., 49), closed='left')
    to = pd.IntervalIndex.from_breaks(np.linspace(-1./24., 2., 49), closed='left')
    index = pd.MultiIndex.from_product([fr, to], names=['from', 'to'])

    rf = pd.Series(0.0, name='cycles', index=index)

    rf.loc[(14./24., 21./12.)] = 1
    rf.loc[(0., 4./3.)] = 3
    rf.loc[(-1., 1.)] = 5

    haigh = pd.Series({'M': 0.5, 'M2': 0.5/3.})
    R_goal = -1.

    expected = 2.0
    expected_interval = pd.Interval(expected - 1./96., expected + 1./96.)
    res = rf.meanstress_transform.fkm_goodman(haigh, R_goal).to_pandas()

    mask = res.index.get_level_values('range').overlaps(expected_interval)
    assert res.loc[mask].sum() == 9
    assert res.loc[~mask].sum() == 0


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

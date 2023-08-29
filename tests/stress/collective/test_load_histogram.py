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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import pytest
import pandas as pd
import numpy as np

import pylife.stress.collective


@pytest.fixture
def lc_matrix_range_mean():
    range_intervals = pd.interval_range(0., 12., 3)
    mean_intervals = pd.interval_range(-2., 4., 3)
    index = pd.MultiIndex.from_product([range_intervals, mean_intervals], names=['range', 'mean'])
    return pd.Series(1, index=index)


@pytest.fixture
def lc_matrix_from_to():
    from_intervals = pd.interval_range(-4., 8., 3)
    to_intervals = pd.interval_range(-2., 4., 3)
    index = pd.MultiIndex.from_product([from_intervals, to_intervals], names=['from', 'to'])
    return pd.Series(1, index=index)


def test_lc_matrix_fail_empty_fail():
    lcm = pd.Series(dtype=np.float64)
    with pytest.raises(AttributeError, match=r"Load collective matrix needs .* index levels"):
        lcm.load_collective


def test_lc_matrix_fail_no_interval_index():
    lcm = pd.Series([], index=pd.MultiIndex.from_tuples([], names=['from', 'to']), dtype=np.float64)
    with pytest.raises(AttributeError, match="Index of a load collective matrix must be pandas.IntervalIndex."):
        lcm.load_collective


def test_lc_matrix_fail_from_to_only_one_interval_index():
    emtpy_interval_index = pd.interval_range(0., 0., 0)
    index = pd.MultiIndex.from_product([emtpy_interval_index, []], names=['from', 'to'])
    lcm = pd.Series([], index=index, dtype=np.float64)
    with pytest.raises(AttributeError, match="Index of a load collective matrix must be pandas.IntervalIndex."):
        lcm.load_collective

    index = pd.MultiIndex.from_product([emtpy_interval_index, []], names=['to', 'from'])
    lcm = pd.Series([], index=index, dtype=np.float64)
    with pytest.raises(AttributeError, match="Index of a load collective matrix must be pandas.IntervalIndex."):
        lcm.load_collective


def test_lc_matrix_fail_range_interval_index():
    index = pd.Index([], name='range')
    lcm = pd.Series([], index=index, dtype=np.float64)
    with pytest.raises(AttributeError, match="Index of a load collective matrix must be pandas.IntervalIndex."):
        lcm.load_collective


def test_lc_matrix_range_no_mean_interval_index():
    emtpy_interval_index = pd.interval_range(0., 0., 0)
    index = pd.IntervalIndex(emtpy_interval_index, name='range')
    lcm = pd.Series([], index=index, dtype=np.float64)
    lcm.load_collective


def test_lc_matrix_fail_range_mean_only_one_interval_index():
    emtpy_interval_index = pd.interval_range(0., 0., 0)
    index = pd.MultiIndex.from_product([emtpy_interval_index, []], names=['range', 'mean'])
    lcm = pd.Series([], index=index, dtype=np.float64)
    with pytest.raises(AttributeError, match="Index of a load collective matrix must be pandas.IntervalIndex."):
        lcm.load_collective

    index = pd.MultiIndex.from_product([emtpy_interval_index, []], names=['mean', 'range'])
    lcm = pd.Series([], index=index, dtype=np.float64)
    with pytest.raises(AttributeError, match="Index of a load collective matrix must be pandas.IntervalIndex."):
        lcm.load_collective


def test_lc_matrix_valid_from_to(lc_matrix_from_to):
    lc_matrix_from_to.load_collective


def test_lc_matrix_valid_only_from(lc_matrix_from_to):
    lc_matrix_from_to.index = lc_matrix_from_to.index.droplevel('to')
    with pytest.raises(AttributeError, match=r"Load collective matrix needs .* index levels"):
        lc_matrix_from_to.load_collective


def test_rainflow_from_to_amplitude_left(lc_matrix_from_to):
    expected = pd.Series([1., 2., 3., 1., 0., 1., 3., 2., 1.],
                         name='amplitude',
                         index=lc_matrix_from_to.index)
    pd.testing.assert_series_equal(lc_matrix_from_to.load_collective.use_class_left().amplitude, expected)


def test_rainflow_from_to_amplitude_mid(lc_matrix_from_to):
    expected = pd.Series([0.5, 1.5, 2.5, 1.5, 0.5, 0.5, 3.5, 2.5, 1.5],
                         name='amplitude',
                         index=lc_matrix_from_to.index)
    pd.testing.assert_series_equal(lc_matrix_from_to.load_collective.amplitude, expected)


def test_rainflow_from_to_amplitude_right(lc_matrix_from_to):
    expected = pd.Series([0., 1., 2., 2., 1., 0., 4., 3., 2.],
                         name='amplitude',
                         index=lc_matrix_from_to.index)
    pd.testing.assert_series_equal(lc_matrix_from_to.load_collective.use_class_right().amplitude, expected)


@pytest.mark.parametrize('cycles_value', [1, 2])
def test_rainflow_from_to_amplitude_histogram(lc_matrix_from_to, cycles_value):
    expected_index = pd.IntervalIndex.from_arrays(
        [0., 0., 2., 0., 0., 0., 4., 2., 0.],
        [4., 6., 8., 6., 4., 4., 10., 8., 6.],
        name='amplitude'
    )
    expected = pd.Series(cycles_value, index=expected_index, name='cycles')
    matrix = lc_matrix_from_to * cycles_value

    result = matrix.load_collective.amplitude_histogram

    pd.testing.assert_series_equal(result, expected)


def test_rainflow_from_to_meanstress_left(lc_matrix_from_to):
    expected = pd.Series([-3., -2., -1., -1., 0., 1., 1., 2., 3.],
                         name='meanstress',
                         index=lc_matrix_from_to.index)
    pd.testing.assert_series_equal(lc_matrix_from_to.load_collective.use_class_left().meanstress, expected)


def test_rainflow_from_to_meanstress_mid(lc_matrix_from_to):
    expected = pd.Series([-1.5, -0.5, 0.5, 0.5, 1.5, 2.5, 2.5, 3.5, 4.5],
                         name='meanstress',
                         index=lc_matrix_from_to.index)
    pd.testing.assert_series_equal(lc_matrix_from_to.load_collective.meanstress, expected)


def test_rainflow_from_to_meanstress_right(lc_matrix_from_to):
    expected = pd.Series([0., 1., 2., 2., 3., 4., 4., 5., 6.],
                         name='meanstress',
                         index=lc_matrix_from_to.index)
    pd.testing.assert_series_equal(lc_matrix_from_to.load_collective.use_class_right().meanstress, expected)


def test_rainflow_from_to_upper_left(lc_matrix_from_to):
    expected = pd.Series([-2., 0., 2., 0., 0., 2., 4., 4., 4.],
                         name='upper',
                         index=lc_matrix_from_to.index)
    pd.testing.assert_series_equal(lc_matrix_from_to.load_collective.use_class_left().upper, expected)


def test_rainflow_from_to_upper_mid(lc_matrix_from_to):
    expected = pd.Series([-1., 1., 3., 2., 2., 3., 6., 6., 6.],
                         name='upper',
                         index=lc_matrix_from_to.index)
    pd.testing.assert_series_equal(lc_matrix_from_to.load_collective.upper, expected)


def test_rainflow_from_to_upper_right(lc_matrix_from_to):
    expected = pd.Series([0., 2., 4., 4., 4., 4., 8., 8., 8.],
                         name='upper',
                         index=lc_matrix_from_to.index)
    pd.testing.assert_series_equal(lc_matrix_from_to.load_collective.use_class_right().upper, expected)


def test_rainflow_from_to_lower_left(lc_matrix_from_to):
    expected = pd.Series([-4., -4., -4., -2., 0., 0., -2., 0., 2.],
                         name='lower',
                         index=lc_matrix_from_to.index)
    pd.testing.assert_series_equal(lc_matrix_from_to.load_collective.use_class_left().lower, expected)


def test_rainflow_from_to_lower_mid(lc_matrix_from_to):
    expected = pd.Series([-2., -2., -2., -1., 1., 2., -1., 1., 3.],
                         name='lower',
                         index=lc_matrix_from_to.index)
    pd.testing.assert_series_equal(lc_matrix_from_to.load_collective.lower, expected)


def test_rainflow_from_to_lower_right(lc_matrix_from_to):
    expected = pd.Series([0., 0., 0., 0., 2., 4., 0., 2., 4.],
                         name='lower',
                         index=lc_matrix_from_to.index)
    pd.testing.assert_series_equal(lc_matrix_from_to.load_collective.use_class_right().lower, expected)


def test_rainflow_R(lc_matrix_from_to):
    print(lc_matrix_from_to)
    expected = pd.Series([2., -np.inf, -2., -np.inf, 0., 0., -0.5, 0., 0.5],
                         name='R',
                         index=lc_matrix_from_to.index)
    pd.testing.assert_series_equal(lc_matrix_from_to.load_collective.use_class_left().R, expected)


def test_rainflow_cycles_from_to(lc_matrix_from_to):
    expected = pd.Series(1, index=lc_matrix_from_to.index, name='cycles')
    freq = lc_matrix_from_to.load_collective.cycles
    pd.testing.assert_series_equal(freq, expected)


def test_rainflow_from_to_scale_scalar(lc_matrix_from_to):
    from_intervals = pd.interval_range(-2., 4., 3)
    to_intervals = pd.interval_range(-1., 2., 3)
    expected_index = pd.MultiIndex.from_product([from_intervals, to_intervals], names=['from', 'to'])
    expected = pd.Series(1, index=expected_index, name='cycles')

    scaled = lc_matrix_from_to.load_collective.scale(0.5)
    assert isinstance(scaled, pylife.stress.collective.LoadHistogram)
    pd.testing.assert_series_equal(scaled.to_pandas(), expected)


def test_rainflow_from_to_scale_series(lc_matrix_from_to):
    from_intervals = pd.IntervalIndex.from_arrays(
        [
            -2., -8., -2., -8., -2, -8.,
            0., 0., 0., 0., 0., 0.,
            2., 8., 2., 8., 2., 8.
        ], [
            0., 0., 0., 0., 0., 0.,
            2., 8., 2., 8., 2., 8.,
            4., 16., 4., 16., 4., 16.
        ]
    )
    to_intervals = pd.IntervalIndex.from_arrays(
        [
            -1., -4., 0., 0., 1., 4.,
            -1., -4., 0., 0., 1., 4.,
            -1., -4., 0., 0., 1., 4.
        ], [
            0., 0., 1., 4, 2., 8.,
            0., 0., 1., 4, 2., 8.,
            0., 0., 1., 4, 2., 8.
        ]
    )
    foo_index = pd.Index(['x', 'y']*9, name='foo')
    expected_index = pd.MultiIndex.from_arrays([from_intervals, to_intervals, foo_index], names=['from', 'to', 'foo'])
    expected = pd.Series(1, index=expected_index, name='cycles')

    factors = pd.Series([0.5, 2.0], index=pd.Index(['x', 'y'], name='foo'), name='scale_factors')

    scaled = lc_matrix_from_to.load_collective.scale(factors)

    assert isinstance(scaled, pylife.stress.collective.LoadHistogram)
    pd.testing.assert_series_equal(scaled.to_pandas(), expected)


def test_rainflow_from_to_shift_scalar(lc_matrix_from_to):
    from_intervals = pd.interval_range(-0., 12., 3)
    to_intervals = pd.interval_range(2., 8., 3)
    expected_index = pd.MultiIndex.from_product([from_intervals, to_intervals], names=['from', 'to'])
    expected = pd.Series(1, index=expected_index, name='cycles')

    shifted = lc_matrix_from_to.load_collective.shift(4.)
    assert isinstance(shifted, pylife.stress.collective.LoadHistogram)
    pd.testing.assert_series_equal(shifted.to_pandas(), expected)


def test_rainflow_from_to_shift_series(lc_matrix_from_to):
    from_intervals = pd.IntervalIndex.from_arrays(
        [
            -2., 0., -2., -0., -2., 0.,
            2., 4., 2., 4., 2., 4.,
            6., 8., 6., 8., 6., 8.
        ], [
            2., 4., 2., 4., 2., 4.,
            6., 8., 6., 8., 6., 8.,
            10., 12., 10., 12., 10., 12.
        ]
    )
    to_intervals = pd.IntervalIndex.from_arrays(
        [
            0., 2., 2., 4., 4., 6.,
            0., 2., 2., 4., 4., 6.,
            0., 2., 2., 4., 4., 6.
        ], [
            2., 4., 4., 6., 6., 8.,
            2., 4., 4., 6., 6., 8.,
            2., 4., 4., 6., 6., 8.
        ]
    )
    foo_index = pd.Index(['x', 'y']*9, name='foo')
    expected_index = pd.MultiIndex.from_arrays([from_intervals, to_intervals, foo_index], names=['from', 'to', 'foo'])
    expected = pd.Series(1, index=expected_index, name='cycles')

    factors = pd.Series([2., 4.], index=pd.Index(['x', 'y'], name='foo'), name='shift_factors')

    shiftd = lc_matrix_from_to.load_collective.shift(factors)

    assert isinstance(shiftd, pylife.stress.collective.LoadHistogram)
    pd.testing.assert_series_equal(shiftd.to_pandas(), expected)


def test_lc_matrix_valid_range_mean(lc_matrix_range_mean):
    lc_matrix_range_mean.load_collective


def test_matrix_range_mean_amplitude_mid(lc_matrix_range_mean):
    expected = pd.Series([1., 1., 1., 3., 3., 3., 5., 5., 5.],
                         name='amplitude',
                         index=lc_matrix_range_mean.index)
    pd.testing.assert_series_equal(lc_matrix_range_mean.load_collective.amplitude, expected)


def test_matrix_range_mean_amplitude_left(lc_matrix_range_mean):
    expected = pd.Series([0., 0., 0., 2., 2., 2., 4., 4., 4.],
                         name='amplitude',
                         index=lc_matrix_range_mean.index)
    pd.testing.assert_series_equal(lc_matrix_range_mean.load_collective.use_class_left().amplitude, expected)


def test_matrix_range_mean_amplitude_right(lc_matrix_range_mean):
    expected = pd.Series([2., 2., 2., 4., 4., 4., 6., 6., 6.],
                         name='amplitude',
                         index=lc_matrix_range_mean.index)
    pd.testing.assert_series_equal(lc_matrix_range_mean.load_collective.use_class_right().amplitude, expected)


def test_matrix_rainge_mean_amplitude_histogram(lc_matrix_range_mean):
    expected_index = pd.IntervalIndex.from_arrays(
        [0., 0., 0., 2., 2., 2., 4., 4., 4.],
        [2., 2., 2., 4., 4., 4., 6., 6., 6.],
        name='amplitude'
    )
    result = lc_matrix_range_mean.load_collective.amplitude_histogram

    pd.testing.assert_index_equal(result.index, expected_index)


def test_matrix_range_mean_meanstress_left(lc_matrix_range_mean):
    expected = pd.Series([-2., 0., 2., -2., 0., 2., -2., 0., 2.],
                         name='meanstress',
                         index=lc_matrix_range_mean.index)
    pd.testing.assert_series_equal(lc_matrix_range_mean.load_collective.use_class_left().meanstress, expected)


def test_matrix_range_mean_meanstress_mid(lc_matrix_range_mean):
    expected = pd.Series([-1., 1., 3., -1., 1., 3., -1., 1., 3.],
                         name='meanstress',
                         index=lc_matrix_range_mean.index)
    pd.testing.assert_series_equal(lc_matrix_range_mean.load_collective.meanstress, expected)


def test_matrix_range_mean_meanstress_right(lc_matrix_range_mean):
    expected = pd.Series([0., 2., 4., 0., 2., 4., 0., 2., 4.],
                         name='meanstress',
                         index=lc_matrix_range_mean.index)
    pd.testing.assert_series_equal(lc_matrix_range_mean.load_collective.use_class_right().meanstress, expected)


def test_matrix_range_mean_meanstress_no_mean_defined():
    range_intervals = pd.interval_range(0., 12., 3)
    index = pd.IntervalIndex(range_intervals, name='range')
    rf = pd.Series(1., index=index)

    expected = pd.Series(np.zeros(3), name='meanstress', index=rf.index)
    pd.testing.assert_series_equal(rf.load_collective.use_class_right().meanstress, expected)


def test_matrix_range_mean_upper_left(lc_matrix_range_mean):
    expected = pd.Series([-2., 0., 2., 0., 2., 4., 2., 4., 6.],
                         name='upper',
                         index=lc_matrix_range_mean.index)
    pd.testing.assert_series_equal(lc_matrix_range_mean.load_collective.use_class_left().upper, expected)


def test_matrix_range_mean_upper_mid(lc_matrix_range_mean):
    expected = pd.Series([0., 2., 4., 2., 4., 6., 4., 6., 8.],
                         name='upper',
                         index=lc_matrix_range_mean.index)
    pd.testing.assert_series_equal(lc_matrix_range_mean.load_collective.upper, expected)


def test_matrix_range_mean_upper_right(lc_matrix_range_mean):
    expected = pd.Series([2., 4., 6., 4., 6., 8., 6., 8., 10.],
                         name='upper',
                         index=lc_matrix_range_mean.index)
    pd.testing.assert_series_equal(lc_matrix_range_mean.load_collective.use_class_right().upper, expected)


def test_matrix_range_mean_lower_left(lc_matrix_range_mean):
    expected = pd.Series([-2., 0., 2., -4., -2., 0., -6., -4., -2.],
                         name='lower',
                         index=lc_matrix_range_mean.index)
    pd.testing.assert_series_equal(lc_matrix_range_mean.load_collective.use_class_left().lower, expected)


def test_matrix_range_mean_lower_mid(lc_matrix_range_mean):
    expected = pd.Series([-2., 0., 2., -4., -2., 0., -6., -4., -2.],
                         name='lower',
                         index=lc_matrix_range_mean.index)
    pd.testing.assert_series_equal(lc_matrix_range_mean.load_collective.lower, expected)


def test_matrix_range_mean_lower_right(lc_matrix_range_mean):
    expected = pd.Series([-2., 0., 2., -4., -2., 0., -6., -4., -2.],
                         name='lower',
                         index=lc_matrix_range_mean.index)
    pd.testing.assert_series_equal(lc_matrix_range_mean.load_collective.use_class_right().lower, expected)


def test_matrix_range_mean_scale_scalar(lc_matrix_range_mean):
    range_intervals = pd.interval_range(0., 6., 3)
    mean_intervals = pd.interval_range(-1., 2., 3)
    expected_index = pd.MultiIndex.from_product([range_intervals, mean_intervals], names=['range', 'mean'])
    expected = pd.Series(1, index=expected_index, name='cycles')

    scaled = lc_matrix_range_mean.load_collective.scale(0.5)
    assert isinstance(scaled, pylife.stress.collective.LoadHistogram)
    pd.testing.assert_series_equal(scaled.to_pandas(), expected)


def test_matrix_range_mean_foo_scale_scalar(lc_matrix_range_mean):
    foo_index = pd.Index([1, 2, 3], name='foo')
    levels = lc_matrix_range_mean.index.levels
    total_index = pd.MultiIndex.from_product([levels[0], levels[1], foo_index])
    lc_matrix = pd.Series(1, index=total_index)

    range_intervals = pd.interval_range(0., 6., 3)
    mean_intervals = pd.interval_range(-1., 2., 3)
    expected_index = pd.MultiIndex.from_product(
        [range_intervals, mean_intervals, foo_index],
        names=['range', 'mean', 'foo']
    )
    expected = pd.Series(1, index=expected_index, name='cycles')

    scaled = lc_matrix.load_collective.scale(0.5)
    assert isinstance(scaled, pylife.stress.collective.LoadHistogram)
    pd.testing.assert_series_equal(scaled.to_pandas(), expected)


def test_matrix_range_mean_scale_series(lc_matrix_range_mean):
    range_intervals = pd.IntervalIndex.from_arrays(
        [
            0., 0., 0., 0., 0., 0.,
            2., 8., 2., 8., 2., 8.,
            4., 16., 4., 16., 4., 16.
        ], [
            2., 8., 2., 8., 2., 8.,
            4., 16., 4., 16., 4., 16.,
            6., 24., 6., 24., 6., 24.
        ]
    )
    mean_intervals = pd.IntervalIndex.from_arrays(
        [
            -1., -4., 0., 0., 1., 4.,
            -1., -4., 0., 0., 1., 4.,
            -1., -4., 0., 0., 1., 4.
        ], [
            0., 0., 1., 4, 2., 8.,
            0., 0., 1., 4, 2., 8.,
            0., 0., 1., 4, 2., 8.
        ]
    )
    foo_index = pd.Index(['x', 'y']*9, name='foo')
    expected_index = pd.MultiIndex.from_arrays([range_intervals, mean_intervals, foo_index], names=['range', 'mean', 'foo'])
    expected = pd.Series(1, index=expected_index, name='cycles')

    factors = pd.Series([0.5, 2.0], index=pd.Index(['x', 'y'], name='foo'), name='scale_factors')

    scaled = lc_matrix_range_mean.load_collective.scale(factors)

    assert isinstance(scaled, pylife.stress.collective.LoadHistogram)
    pd.testing.assert_series_equal(scaled.to_pandas(), expected)


def test_matrix_range_mean_shift_scalar(lc_matrix_range_mean):
    range_intervals = pd.interval_range(0., 12., 3)
    mean_intervals = pd.interval_range(2., 8., 3)
    expected_index = pd.MultiIndex.from_product([range_intervals, mean_intervals], names=['range', 'mean'])
    expected = pd.Series(1, index=expected_index, name='cycles')

    shifted = lc_matrix_range_mean.load_collective.shift(4.)
    assert isinstance(shifted, pylife.stress.collective.LoadHistogram)
    pd.testing.assert_series_equal(shifted.to_pandas(), expected)


def test_matrix_range_mean_shift_series(lc_matrix_range_mean):
    range_intervals = pd.IntervalIndex.from_arrays(
        [
            0., 0., 0., 0., 0., 0.,
            4., 4., 4., 4., 4., 4.,
            8., 8., 8., 8., 8., 8.
        ], [
            4., 4., 4., 4., 4., 4.,
            8., 8., 8., 8., 8., 8.,
            12., 12., 12., 12., 12., 12.
        ]
    )
    mean_intervals = pd.IntervalIndex.from_arrays(
        [
            0., 2., 2., 4., 4., 6.,
            0., 2., 2., 4., 4., 6.,
            0., 2., 2., 4., 4., 6.,
        ], [
            2., 4., 4., 6., 6., 8.,
            2., 4., 4., 6., 6., 8.,
            2., 4., 4., 6., 6., 8.
        ]
    )
    foo_index = pd.Index(['x', 'y']*9, name='foo')
    expected_index = pd.MultiIndex.from_arrays([range_intervals, mean_intervals, foo_index], names=['range', 'mean', 'foo'])
    expected = pd.Series(1, index=expected_index, name='cycles')

    factors = pd.Series([2., 4.], index=pd.Index(['x', 'y'], name='foo'), name='shift_factors')

    shiftd = lc_matrix_range_mean.load_collective.shift(factors)

    assert isinstance(shiftd, pylife.stress.collective.LoadHistogram)
    pd.testing.assert_series_equal(shiftd.to_pandas(), expected)


@pytest.mark.parametrize('range_interval', [
    (pd.interval_range(0., 12., 3)),
    (pd.interval_range(0., 6., 3))
])
def test_matrix_cumulative_range_only_range(range_interval):
    idx = pd.IntervalIndex(range_interval, name='range')
    result = pd.Series(1, index=idx).load_collective.cumulated_range()

    expected = pd.Series([1, 1, 1], name='cumulated_cycles', index=idx)

    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize('range_interval', [
    (pd.interval_range(0., 12., 3)),
    (pd.interval_range(0., 6., 3))
])
def test_matrix_cumulative_range_range_mean(range_interval):
    range_idx = pd.IntervalIndex(range_interval, name='range')
    mean_idx = pd.IntervalIndex(pd.interval_range(0., 1., 3), name='mean')
    idx = pd.MultiIndex.from_product([range_idx, mean_idx])
    print(pd.Series(1, index=idx))
    result = pd.Series(1, index=idx).load_collective.cumulated_range()

    expected = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3], name='cumulated_cycles', index=idx)

    pd.testing.assert_series_equal(result, expected)

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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import pytest
import pandas as pd
import numpy as np

import pylife.stress.rainflow


@pytest.fixture
def rainflow_matrix_range_mean():
    range_intervals = pd.interval_range(0., 12., 3)
    mean_intervals = pd.interval_range(-2., 4., 3)
    index = pd.MultiIndex.from_product([range_intervals, mean_intervals], names=['range', 'mean'])
    return pd.Series(1, index=index)


@pytest.fixture
def rainflow_matrix_from_to():
    from_intervals = pd.interval_range(-4., 8., 3)
    to_intervals = pd.interval_range(-2., 4., 3)
    index = pd.MultiIndex.from_product([from_intervals, to_intervals], names=['from', 'to'])
    return pd.Series(1, index=index)


def test_rainflow_matrix_signal_fail_empty_fail():
    rainflow_matrix = pd.Series(dtype=np.float64)
    with pytest.raises(AttributeError, match=r"Rainflow needs .* index levels"):
        rainflow_matrix.rainflow


def test_rainflow_matrix_signal_fail_no_interval_index():
    rfm = pd.Series([], index=pd.MultiIndex.from_tuples([], names=['from', 'to']), dtype=np.float64)
    with pytest.raises(AttributeError, match="Index of a rainflow matrix must be pandas.IntervalIndex."):
        rfm.rainflow


def test_rainflow_matrix_signal_fail_from_to_only_one_interval_index():
    emtpy_interval_index = pd.interval_range(0., 0., 0)
    index = pd.MultiIndex.from_product([emtpy_interval_index, []], names=['from', 'to'])
    rfm = pd.Series([], index=index, dtype=np.float64)
    with pytest.raises(AttributeError, match="Index of a rainflow matrix must be pandas.IntervalIndex."):
        rfm.rainflow

    index = pd.MultiIndex.from_product([emtpy_interval_index, []], names=['to', 'from'])
    rfm = pd.Series([], index=index, dtype=np.float64)
    with pytest.raises(AttributeError, match="Index of a rainflow matrix must be pandas.IntervalIndex."):
        rfm.rainflow


def test_rainflow_matrix_signal_fail_range_interval_index():
    index = pd.Index([], name='range')
    rfm = pd.Series([], index=index, dtype=np.float64)
    with pytest.raises(AttributeError, match="Index of a rainflow matrix must be pandas.IntervalIndex."):
        rfm.rainflow


def test_rainflow_matrix_signal_range_no_mean_interval_index():
    emtpy_interval_index = pd.interval_range(0., 0., 0)
    index = pd.IntervalIndex(emtpy_interval_index, name='range')
    rfm = pd.Series([], index=index, dtype=np.float64)
    rfm.rainflow


def test_rainflow_matrix_signal_fail_range_mean_only_one_interval_index():
    emtpy_interval_index = pd.interval_range(0., 0., 0)
    index = pd.MultiIndex.from_product([emtpy_interval_index, []], names=['range', 'mean'])
    rfm = pd.Series([], index=index, dtype=np.float64)
    with pytest.raises(AttributeError, match="Index of a rainflow matrix must be pandas.IntervalIndex."):
        rfm.rainflow

    index = pd.MultiIndex.from_product([emtpy_interval_index, []], names=['mean', 'range'])
    rfm = pd.Series([], index=index, dtype=np.float64)
    with pytest.raises(AttributeError, match="Index of a rainflow matrix must be pandas.IntervalIndex."):
        rfm.rainflow


def test_rainflow_matrix_signal_valid_from_to(rainflow_matrix_from_to):
    rainflow_matrix_from_to.rainflow


def test_rainflow_matrix_signal_valid_only_from(rainflow_matrix_from_to):
    rainflow_matrix_from_to.index = rainflow_matrix_from_to.index.droplevel('to')
    with pytest.raises(AttributeError, match=r"Rainflow needs .* index levels"):
        rainflow_matrix_from_to.rainflow


def test_rainflow_from_to_amplitude_left(rainflow_matrix_from_to):
    expected = pd.Series([1., 2., 3., 1., 0., 1., 3., 2., 1.],
                         name='amplitude',
                         index=rainflow_matrix_from_to.index)
    pd.testing.assert_series_equal(rainflow_matrix_from_to.rainflow.use_class_left().amplitude, expected)


def test_rainflow_from_to_amplitude_mid(rainflow_matrix_from_to):
    expected = pd.Series([0.5, 1.5, 2.5, 1.5, 0.5, 0.5, 3.5, 2.5, 1.5],
                         name='amplitude',
                         index=rainflow_matrix_from_to.index)
    pd.testing.assert_series_equal(rainflow_matrix_from_to.rainflow.amplitude, expected)


def test_rainflow_from_to_amplitude_right(rainflow_matrix_from_to):
    expected = pd.Series([0., 1., 2., 2., 1., 0., 4., 3., 2.],
                         name='amplitude',
                         index=rainflow_matrix_from_to.index)
    pd.testing.assert_series_equal(rainflow_matrix_from_to.rainflow.use_class_right().amplitude, expected)


def test_rainflow_from_to_meanstress_left(rainflow_matrix_from_to):
    expected = pd.Series([-3., -2., -1., -1., 0., 1., 1., 2., 3.],
                         name='meanstress',
                         index=rainflow_matrix_from_to.index)
    pd.testing.assert_series_equal(rainflow_matrix_from_to.rainflow.use_class_left().meanstress, expected)


def test_rainflow_from_to_meanstress_mid(rainflow_matrix_from_to):
    expected = pd.Series([-1.5, -0.5, 0.5, 0.5, 1.5, 2.5, 2.5, 3.5, 4.5],
                         name='meanstress',
                         index=rainflow_matrix_from_to.index)
    pd.testing.assert_series_equal(rainflow_matrix_from_to.rainflow.meanstress, expected)


def test_rainflow_from_to_meanstress_right(rainflow_matrix_from_to):
    expected = pd.Series([0., 1., 2., 2., 3., 4., 4., 5., 6.],
                         name='meanstress',
                         index=rainflow_matrix_from_to.index)
    pd.testing.assert_series_equal(rainflow_matrix_from_to.rainflow.use_class_right().meanstress, expected)


def test_rainflow_from_to_upper_left(rainflow_matrix_from_to):
    expected = pd.Series([-2., 0., 2., 0., 0., 2., 4., 4., 4.],
                         name='upper',
                         index=rainflow_matrix_from_to.index)
    pd.testing.assert_series_equal(rainflow_matrix_from_to.rainflow.use_class_left().upper, expected)


def test_rainflow_from_to_upper_mid(rainflow_matrix_from_to):
    expected = pd.Series([-1., 1., 3., 2., 2., 3., 6., 6., 6.],
                         name='upper',
                         index=rainflow_matrix_from_to.index)
    pd.testing.assert_series_equal(rainflow_matrix_from_to.rainflow.upper, expected)


def test_rainflow_from_to_upper_right(rainflow_matrix_from_to):
    expected = pd.Series([0., 2., 4., 4., 4., 4., 8., 8., 8.],
                         name='upper',
                         index=rainflow_matrix_from_to.index)
    pd.testing.assert_series_equal(rainflow_matrix_from_to.rainflow.use_class_right().upper, expected)


def test_rainflow_from_to_lower_left(rainflow_matrix_from_to):
    expected = pd.Series([-4., -4., -4., -2., 0., 0., -2., 0., 2.],
                         name='lower',
                         index=rainflow_matrix_from_to.index)
    pd.testing.assert_series_equal(rainflow_matrix_from_to.rainflow.use_class_left().lower, expected)


def test_rainflow_from_to_lower_mid(rainflow_matrix_from_to):
    expected = pd.Series([-2., -2., -2., -1., 1., 2., -1., 1., 3.],
                         name='lower',
                         index=rainflow_matrix_from_to.index)
    pd.testing.assert_series_equal(rainflow_matrix_from_to.rainflow.lower, expected)


def test_rainflow_from_to_lower_right(rainflow_matrix_from_to):
    expected = pd.Series([0., 0., 0., 0., 2., 4., 0., 2., 4.],
                         name='lower',
                         index=rainflow_matrix_from_to.index)
    pd.testing.assert_series_equal(rainflow_matrix_from_to.rainflow.use_class_right().lower, expected)


def test_rainflow_cycles_from_to(rainflow_matrix_from_to):
    expected = pd.Series(1, index=rainflow_matrix_from_to.index, name='cycles')
    freq = rainflow_matrix_from_to.rainflow.cycles
    pd.testing.assert_series_equal(freq, expected)


def test_rainflow_from_to_scale_scalar(rainflow_matrix_from_to):
    from_intervals = pd.interval_range(-2., 4., 3)
    to_intervals = pd.interval_range(-1., 2., 3)
    expected_index = pd.MultiIndex.from_product([from_intervals, to_intervals], names=['from', 'to'])
    expected = pd.Series(1, index=expected_index, name='cycles')

    scaled = rainflow_matrix_from_to.rainflow.scale(0.5)
    assert isinstance(scaled, pylife.stress.rainflow.RainflowMatrix)
    pd.testing.assert_series_equal(scaled.to_pandas(), expected)


def test_rainflow_from_to_scale_series(rainflow_matrix_from_to):
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

    scaled = rainflow_matrix_from_to.rainflow.scale(factors)

    assert isinstance(scaled, pylife.stress.rainflow.RainflowMatrix)
    pd.testing.assert_series_equal(scaled.to_pandas(), expected)


def test_rainflow_from_to_shift_scalar(rainflow_matrix_from_to):
    from_intervals = pd.interval_range(-0., 12., 3)
    to_intervals = pd.interval_range(2., 8., 3)
    expected_index = pd.MultiIndex.from_product([from_intervals, to_intervals], names=['from', 'to'])
    expected = pd.Series(1, index=expected_index, name='cycles')

    shifted = rainflow_matrix_from_to.rainflow.shift(4.)
    assert isinstance(shifted, pylife.stress.rainflow.RainflowMatrix)
    pd.testing.assert_series_equal(shifted.to_pandas(), expected)


def test_rainflow_from_to_shift_series(rainflow_matrix_from_to):
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

    shiftd = rainflow_matrix_from_to.rainflow.shift(factors)

    assert isinstance(shiftd, pylife.stress.rainflow.RainflowMatrix)
    pd.testing.assert_series_equal(shiftd.to_pandas(), expected)


def test_rainflow_matrix_signal_valid_range_mean(rainflow_matrix_range_mean):
    rainflow_matrix_range_mean.rainflow


def test_rainflow_range_mean_amplitude_mid(rainflow_matrix_range_mean):
    expected = pd.Series([1., 1., 1., 3., 3., 3., 5., 5., 5.],
                         name='amplitude',
                         index=rainflow_matrix_range_mean.index)
    pd.testing.assert_series_equal(rainflow_matrix_range_mean.rainflow.amplitude, expected)


def test_rainflow_range_mean_amplitude_left(rainflow_matrix_range_mean):
    expected = pd.Series([0., 0., 0., 2., 2., 2., 4., 4., 4.],
                         name='amplitude',
                         index=rainflow_matrix_range_mean.index)
    pd.testing.assert_series_equal(rainflow_matrix_range_mean.rainflow.use_class_left().amplitude, expected)


def test_rainflow_range_mean_amplitude_right(rainflow_matrix_range_mean):
    expected = pd.Series([2., 2., 2., 4., 4., 4., 6., 6., 6.],
                         name='amplitude',
                         index=rainflow_matrix_range_mean.index)
    pd.testing.assert_series_equal(rainflow_matrix_range_mean.rainflow.use_class_right().amplitude, expected)


def test_rainflow_range_mean_meanstress_left(rainflow_matrix_range_mean):
    expected = pd.Series([-2., 0., 2., -2., 0., 2., -2., 0., 2.],
                         name='meanstress',
                         index=rainflow_matrix_range_mean.index)
    pd.testing.assert_series_equal(rainflow_matrix_range_mean.rainflow.use_class_left().meanstress, expected)


def test_rainflow_range_mean_meanstress_mid(rainflow_matrix_range_mean):
    expected = pd.Series([-1., 1., 3., -1., 1., 3., -1., 1., 3.],
                         name='meanstress',
                         index=rainflow_matrix_range_mean.index)
    pd.testing.assert_series_equal(rainflow_matrix_range_mean.rainflow.meanstress, expected)


def test_rainflow_range_mean_meanstress_right(rainflow_matrix_range_mean):
    expected = pd.Series([0., 2., 4., 0., 2., 4., 0., 2., 4.],
                         name='meanstress',
                         index=rainflow_matrix_range_mean.index)
    pd.testing.assert_series_equal(rainflow_matrix_range_mean.rainflow.use_class_right().meanstress, expected)


def test_rainflow_range_mean_meanstress_no_mean_defined():
    range_intervals = pd.interval_range(0., 12., 3)
    index = pd.IntervalIndex(range_intervals, name='range')
    rf = pd.Series(1., index=index)

    expected = pd.Series(np.zeros(3), name='meanstress', index=rf.index)
    pd.testing.assert_series_equal(rf.rainflow.use_class_right().meanstress, expected)


def test_rainflow_range_mean_upper_left(rainflow_matrix_range_mean):
    expected = pd.Series([-2., 0., 2., 0., 2., 4., 2., 4., 6.],
                         name='upper',
                         index=rainflow_matrix_range_mean.index)
    pd.testing.assert_series_equal(rainflow_matrix_range_mean.rainflow.use_class_left().upper, expected)


def test_rainflow_range_mean_upper_mid(rainflow_matrix_range_mean):
    expected = pd.Series([0., 2., 4., 2., 4., 6., 4., 6., 8.],
                         name='upper',
                         index=rainflow_matrix_range_mean.index)
    pd.testing.assert_series_equal(rainflow_matrix_range_mean.rainflow.upper, expected)


def test_rainflow_range_mean_upper_right(rainflow_matrix_range_mean):
    expected = pd.Series([2., 4., 6., 4., 6., 8., 6., 8., 10.],
                         name='upper',
                         index=rainflow_matrix_range_mean.index)
    pd.testing.assert_series_equal(rainflow_matrix_range_mean.rainflow.use_class_right().upper, expected)


def test_rainflow_range_mean_lower_left(rainflow_matrix_range_mean):
    expected = pd.Series([-2., 0., 2., -4., -2., 0., -6., -4., -2.],
                         name='lower',
                         index=rainflow_matrix_range_mean.index)
    pd.testing.assert_series_equal(rainflow_matrix_range_mean.rainflow.use_class_left().lower, expected)


def test_rainflow_range_mean_lower_mid(rainflow_matrix_range_mean):
    expected = pd.Series([-2., 0., 2., -4., -2., 0., -6., -4., -2.],
                         name='lower',
                         index=rainflow_matrix_range_mean.index)
    pd.testing.assert_series_equal(rainflow_matrix_range_mean.rainflow.lower, expected)


def test_rainflow_range_mean_lower_right(rainflow_matrix_range_mean):
    expected = pd.Series([-2., 0., 2., -4., -2., 0., -6., -4., -2.],
                         name='lower',
                         index=rainflow_matrix_range_mean.index)
    pd.testing.assert_series_equal(rainflow_matrix_range_mean.rainflow.use_class_right().lower, expected)


def test_rainflow_range_mean_scale_scalar(rainflow_matrix_range_mean):
    range_intervals = pd.interval_range(0., 6., 3)
    mean_intervals = pd.interval_range(-1., 2., 3)
    expected_index = pd.MultiIndex.from_product([range_intervals, mean_intervals], names=['range', 'mean'])
    expected = pd.Series(1, index=expected_index, name='cycles')

    scaled = rainflow_matrix_range_mean.rainflow.scale(0.5)
    assert isinstance(scaled, pylife.stress.rainflow.RainflowMatrix)
    pd.testing.assert_series_equal(scaled.to_pandas(), expected)


def test_rainflow_range_mean_scale_series(rainflow_matrix_range_mean):
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

    scaled = rainflow_matrix_range_mean.rainflow.scale(factors)

    assert isinstance(scaled, pylife.stress.rainflow.RainflowMatrix)
    pd.testing.assert_series_equal(scaled.to_pandas(), expected)


def test_rainflow_range_mean_shift_scalar(rainflow_matrix_range_mean):
    range_intervals = pd.interval_range(0., 12., 3)
    mean_intervals = pd.interval_range(2., 8., 3)
    expected_index = pd.MultiIndex.from_product([range_intervals, mean_intervals], names=['range', 'mean'])
    expected = pd.Series(1, index=expected_index, name='cycles')

    shifted = rainflow_matrix_range_mean.rainflow.shift(4.)
    assert isinstance(shifted, pylife.stress.rainflow.RainflowMatrix)
    pd.testing.assert_series_equal(shifted.to_pandas(), expected)


def test_rainflow_range_mean_shift_series(rainflow_matrix_range_mean):
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

    shiftd = rainflow_matrix_range_mean.rainflow.shift(factors)

    assert isinstance(shiftd, pylife.stress.rainflow.RainflowMatrix)
    pd.testing.assert_series_equal(shiftd.to_pandas(), expected)

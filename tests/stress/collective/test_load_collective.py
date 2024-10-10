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


@pytest.mark.parametrize('df, expected', [
    (
        pd.DataFrame(columns=[1, 2], dtype=np.float64),
        pd.Series(dtype=np.float64)
    ),
    (
        pd.DataFrame([[-1., 1.], [2, -2]]),
        pd.Series([1., 2.])
    ),
    (
        pd.DataFrame([[-2., 2.], [3, -3]], index=[23, 42]),
        pd.Series([2., 3.], index=[23, 42])
    )
])
def test_load_collective_amplitude_from_to(df, expected):
    df.columns = ['from', 'to']
    expected.name = 'amplitude'
    pd.testing.assert_series_equal(df.load_collective.amplitude, expected)

    expected_cycles = pd.Series(1.0, name='cycles', index=df.index)
    pd.testing.assert_series_equal(df.load_collective.cycles, expected_cycles)


def test_load_collective_amplitude_from_to_with_cycles():
    df = pd.DataFrame([[-1., 1., 1e6], [2, -2, 2e6]])
    df.columns = ['from', 'to', 'cycles']

    expected_cycles = pd.Series([1e6, 2e6], name='cycles', index=df.index)
    pd.testing.assert_series_equal(df.load_collective.cycles, expected_cycles)


def test_load_collective_upper_from_to_with_cycles():
    df = pd.DataFrame([[-1., 1., 1e6], [2, -2, 2e6]])
    df.columns = ['from', 'to', 'cycles']

    expected_upper = pd.Series([1., 2.], name='upper', index=df.index)
    pd.testing.assert_series_equal(df.load_collective.upper, expected_upper)


def test_load_collective_lower_from_to_with_cycles():
    df = pd.DataFrame([[1000., 10000., 1e2], [20000., 2000., 2e2]])
    df.columns = ['from', 'to', 'cycles']

    expected_lower = pd.Series([1000., 2000.], name='lower', index=df.index)
    pd.testing.assert_series_equal(df.load_collective.lower, expected_lower)


@pytest.mark.parametrize('df, expected', [
    (
        pd.DataFrame(columns=[1, 2], dtype=np.float64),
        pd.Series(dtype=np.float64)
    ),
    (
        pd.DataFrame([[-1., 1.], [3, -1]]),
        pd.Series([0., 1.])
    ),
    (
        pd.DataFrame([[-2., 2.], [6, -2]], index=[23, 42]),
        pd.Series([0., 2.], index=[23, 42])
    )
])
def test_load_collective_mean_from_to(df, expected):
    df.columns = ['from', 'to']
    expected.name = 'meanstress'
    pd.testing.assert_series_equal(df.load_collective.meanstress, expected)

    expected_cycles = pd.Series(1.0, name='cycles', index=df.index)
    pd.testing.assert_series_equal(df.load_collective.cycles, expected_cycles)


@pytest.mark.parametrize('df, expected', [
    (
        pd.DataFrame(columns=[1, 2], dtype=np.float64),
        pd.Series(dtype=np.float64)
    ),
    (
        pd.DataFrame([
            [-1., 1.],
            [3, -1],
            [0., 0.],
            [-1.0, 0.0],
        ]),
        pd.Series([-1., -1./3., 0., -np.inf])
    ),
    (
        pd.DataFrame([[-2., 2.], [6, -2]], index=[23, 42]),
        pd.Series([-1., -1./3.], index=[23, 42])
    )
])
def test_load_collective_R_from_to(df, expected):
    df.columns = ['from', 'to']
    expected.name = 'R'
    pd.testing.assert_series_equal(df.load_collective.R, expected)

    expected_cycles = pd.Series(1.0, name='cycles', index=df.index)
    pd.testing.assert_series_equal(df.load_collective.cycles, expected_cycles)


@pytest.mark.parametrize('df, expected_upper, expected_lower', [
    (
        pd.DataFrame(columns=[1, 2], dtype=np.float64),
        pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)
    ),
    (
        pd.DataFrame([[-1., 1.], [3, -1]]),
        pd.Series([1., 3.]), pd.Series([-1., -1])
    ),
    (
        pd.DataFrame([[-2., 2.], [6, -2]], index=[23, 42]),
        pd.Series([2., 6.], index=[23, 42]),
        pd.Series([-2., -2.], index=[23, 42]),
    )
])
def test_load_collective_upper_lower_from_to(df, expected_upper, expected_lower):
    df.columns = ['from', 'to']
    expected_upper.name = 'upper'
    expected_lower.name = 'lower'
    pd.testing.assert_series_equal(df.load_collective.upper, expected_upper)
    pd.testing.assert_series_equal(df.load_collective.lower, expected_lower)


@pytest.mark.parametrize('df, expected', [
    (
        pd.DataFrame(columns=[1, 2], dtype=np.float64),
        pd.DataFrame(columns=[1, 2], dtype=np.float64),
    ),
    (
        pd.DataFrame([[-1., 1.], [2, -2]]),
        pd.DataFrame([[-2., 2.], [4, -4]]),
    ),
    (
        pd.DataFrame([[-2., 2.], [3, -3]], index=[23, 42]),
        pd.DataFrame([[-4., 4.], [6, -6]], index=[23, 42]),
    )
])
def test_load_collective_from_to_scale_scalar(df, expected):
    df.columns = ['from', 'to']
    expected.columns = ['from', 'to']
    pd.testing.assert_frame_equal(df.load_collective.scale(2.0).to_pandas(), expected)


def test_load_collective_from_to_scale_scalar_with_cycles():
    df = pd.DataFrame([[-1., 1., 1e6], [2, -2, 2e6]])
    df.columns = ['from', 'to', 'cycles']
    expected = pd.DataFrame([[-2., 2., 1e6], [4, -4, 2e6]])
    expected.columns = ['from', 'to', 'cycles']

    pd.testing.assert_frame_equal(df.load_collective.scale(2.0).to_pandas(), expected)


@pytest.mark.parametrize('df, expected', [
    (
        pd.DataFrame(columns=[1, 2], dtype=np.float64),
        pd.DataFrame(columns=[1, 2], dtype=np.float64),
    ),
    (
        pd.DataFrame([[-1., 1.], [2, -2]]),
        pd.DataFrame([[1., 3.], [4., 0.]]),
    ),
    (
        pd.DataFrame([[-2., 2.], [3., -3.]], index=[23, 42]),
        pd.DataFrame([[0., 4.], [5., -1.]], index=[23, 42]),
    )
])
def test_load_collective_from_to_shift_scalar(df, expected):
    df.columns = ['from', 'to']
    expected.columns = ['from', 'to']
    pd.testing.assert_frame_equal(df.load_collective.shift(2.0).to_pandas(), expected)


def test_load_collective_from_to_shift_scalar_with_cycles():
    df = pd.DataFrame([[-1., 1., 1e2], [2, -2, 2e2]])
    df.columns = ['from', 'to', 'cycles']
    expected = pd.DataFrame([[1., 3., 1e2], [4., 0., 2e2]])
    expected.columns = ['from', 'to', 'cycles']
    pd.testing.assert_frame_equal(df.load_collective.shift(2.0).to_pandas(), expected)


def test_load_collective_from_to_scale_series():
    df = pd.DataFrame([[-2., 2.], [3, -3]],
                      columns=['from', 'to'],
                      index=pd.Index([23, 42], name='foo_index'))

    scale_operand = pd.Series([2, 3, 4], index=pd.Index([6, 7, 8], name='scale_index'))

    expected_index = pd.MultiIndex.from_tuples([
        (23, 6), (23, 7), (23, 8),
        (42, 6), (42, 7), (42, 8),
    ], names=['foo_index', 'scale_index'])
    expected = pd.DataFrame([
        [-4., 4.],
        [-6., 6.],
        [-8., 8.],
        [6., -6.],
        [9., -9.],
        [12., -12.]
    ], columns=['from', 'to'], index=expected_index)

    pd.testing.assert_frame_equal(df.load_collective.scale(scale_operand).to_pandas(), expected)


def test_load_collective_from_to_shift_series():
    df = pd.DataFrame([[-2., 2.], [3, -3]],
                      columns=['from', 'to'],
                      index=pd.Index([23, 42], name='foo_index'))

    shift_operand = pd.Series([2, 3, 4], index=pd.Index([6, 7, 8], name='shift_index'))

    expected_index = pd.MultiIndex.from_tuples([
        (23, 6), (23, 7), (23, 8),
        (42, 6), (42, 7), (42, 8),
    ], names=['foo_index', 'shift_index'])
    expected = pd.DataFrame([
        [0., 4.],
        [1., 5.],
        [2., 6.],
        [5., -1.],
        [6., 0.],
        [7., 1.]
    ], columns=['from', 'to'], index=expected_index)

    pd.testing.assert_frame_equal(df.load_collective.shift(shift_operand).to_pandas(), expected)


@pytest.mark.parametrize('df, expected', [
    (
        pd.DataFrame(columns=[1, 2], dtype=np.float64),
        pd.Series(dtype=np.float64)
    ),
    (
        pd.DataFrame([[4., 1.], [2., 0.]]),
        pd.Series([2., 1.])
    ),
    (
        pd.DataFrame([[6., 2.], [4, -3]], index=[23, 42]),
        pd.Series([3., 2.], index=[23, 42])
    )
])
def test_load_collective_amplitude_range_mean(df, expected):
    df.columns = ['range', 'mean']
    expected.name = 'amplitude'
    pd.testing.assert_series_equal(df.load_collective.amplitude, expected)

    expected_cycles = pd.Series(1.0, name='cycles', index=df.index)
    pd.testing.assert_series_equal(df.load_collective.cycles, expected_cycles)


def test_load_collective_amplitude_range_mean_with_cycles():
    df = pd.DataFrame([[2., 1., 1e6], [0.1, 0.2, 2e6]])
    df.columns = ['range', 'mean', 'cycles']

    expected_cycles = pd.Series([1e6, 2e6], name='cycles', index=df.index)
    pd.testing.assert_series_equal(df.load_collective.cycles, expected_cycles)


@pytest.mark.parametrize('df, expected_upper, expected_lower', [
    (
        pd.DataFrame(columns=[1, 2], dtype=np.float64),
        pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)
    ),
    (
        pd.DataFrame([[4., 1.], [2., 0.]]),
        pd.Series([3., 1.]), pd.Series([-1., -1])
    ),
    (
        pd.DataFrame([[6., 2.], [4, -3]], index=[23, 42]),
        pd.Series([5., -1.], index=[23, 42]),
        pd.Series([-1., -5.], index=[23, 42]),
    )
])
def test_load_collective_upper_lower_range_mean(df, expected_upper, expected_lower):
    df.columns = ['range', 'mean']
    expected_upper.name = 'upper'
    expected_lower.name = 'lower'
    pd.testing.assert_series_equal(df.load_collective.upper, expected_upper)
    pd.testing.assert_series_equal(df.load_collective.lower, expected_lower)


def test_load_collective_upper_lower_range_mean_single_value():
    df = pd.DataFrame({'range': [2.0], 'mean': 0.0})
    expected = pd.DataFrame({'from': [-1.0], 'to': [1.0]})
    pd.testing.assert_frame_equal(df.load_collective.to_pandas(), expected)


@pytest.mark.parametrize('df, expected', [
    (
        pd.DataFrame(columns=[1, 2], dtype=np.float64),
        pd.Series(dtype=np.float64)
    ),
    (
        pd.DataFrame([[4., 1.], [2., 0.]]),
        pd.Series([1., 0.])
    ),
    (
        pd.DataFrame([[6., 2.], [4, -3]], index=[23, 42]),
        pd.Series([2., -3.], index=[23, 42])
    )
])
def test_load_collective_mean_range_mean(df, expected):
    df.columns = ['range', 'mean']
    expected.name = 'meanstress'
    pd.testing.assert_series_equal(df.load_collective.meanstress, expected)


@pytest.mark.parametrize('df, expected_amplitude, expected_mean', [
    (
        pd.DataFrame(columns=[1, 2], dtype=np.float64),
        pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)
    ),
    (
        pd.DataFrame([[4., 1.], [2., 0.]]),
        pd.Series([4., 2.]), pd.Series([2., 0.])
    ),
    (
        pd.DataFrame([[6., 2.], [4., -3.]], index=[23, 42]),
        pd.Series([6., 4.], index=[23, 42]),
        pd.Series([4., -6.], index=[23, 42]),
    )
])
def test_load_collective_mean_range_scale_scalar(df, expected_amplitude, expected_mean):
    df.columns = ['range', 'mean']
    expected_amplitude.name = 'amplitude'
    expected_mean.name = 'meanstress'
    scaled = df.load_collective.scale(2.0)
    pd.testing.assert_series_equal(scaled.amplitude, expected_amplitude)
    pd.testing.assert_series_equal(scaled.meanstress, expected_mean)


@pytest.mark.parametrize('df, expected_amplitude, expected_mean', [
    (
        pd.DataFrame(columns=[1, 2], dtype=np.float64),
        pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)
    ),
    (
        pd.DataFrame([[4., 1.], [2., 0.]]),
        pd.Series([2., 1.]), pd.Series([3., 2.])
    ),
    (
        pd.DataFrame([[6., 2.], [4., -3.]], index=[23, 42]),
        pd.Series([3., 2.], index=[23, 42]),
        pd.Series([4., -1.], index=[23, 42]),
    )
])
def test_load_collective_mean_range_shift_scalar(df, expected_amplitude, expected_mean):
    df.columns = ['range', 'mean']
    expected_amplitude.name = 'amplitude'
    expected_mean.name = 'meanstress'
    scaled = df.load_collective.shift(2.0)
    pd.testing.assert_series_equal(scaled.amplitude, expected_amplitude)
    pd.testing.assert_series_equal(scaled.meanstress, expected_mean)


@pytest.mark.parametrize('bins, expected_index_tuples, expected_data', [
    ([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)], [0, 2, 1]),
    ([1, 2, 3], [(1, 2), (2, 3)], [2, 1])
])
def test_load_collective_range_histogram_alter_bins(bins, expected_index_tuples, expected_data):
    df = pd.DataFrame({
        'range': [1.0, 2.0, 1.0],
        'mean': [0.0, 0.0, 0.0]
    }, columns=['range', 'mean'])

    expected = pd.Series(
        expected_data,
        name='cycles',
        index=pd.IntervalIndex.from_tuples(expected_index_tuples, name='range'),
    )

    result = df.load_collective.range_histogram(bins)

    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_load_collective_range_histogram_alter_ranges():
    df = pd.DataFrame(
        {'range': [1.0, 2.0, 1.0, 2.0, 1], 'mean': [0.0, 0.0, 0.0, 0.0, 0.0]},
        columns=['range', 'mean'],
    )

    expected_index = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3)], name='range')
    expected = pd.Series([0, 3, 2], name='cycles', index=expected_index)

    result = df.load_collective.range_histogram([0, 1, 2, 3])

    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_load_collective_range_histogram_interval_index():
    df = pd.DataFrame({
        'range': [1.0, 2.0, 1.0, 2.0, 1.0],
        'mean': [0.0, 0.0, 0.0, 0.0, 0.0]
    }, columns=['range', 'mean'])

    expected_index = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3)], name='range')
    expected = pd.Series([0, 3, 2], name='cycles', index=expected_index)

    result = df.load_collective.range_histogram(expected_index)

    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_load_collective_range_histogram_interval_arrays():
    df = pd.DataFrame({
        'range': [1.0, 2.0, 1.0, 2.0, 1.0],
        'mean': [0.0, 0.0, 0.0, 0.0, 0.0]
    }, columns=['range', 'mean'])

    expected_index = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3)], name='range')
    expected = pd.Series([0, 3, 2], name='cycles', index=expected_index)

    intervals = pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2), (2, 3)])
    result = df.load_collective.range_histogram(intervals)

    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_load_collective_range_histogram_unnested_grouped():
    element_idx = pd.Index([10, 20, 30], name='element_id')
    cycle_idx = pd.Index([0, 1, 2], name='cycle_number')
    idx = pd.MultiIndex.from_product((element_idx, cycle_idx))

    df = pd.DataFrame({
        'range': [0., 1., 2., 0., 1., 2., 0., 1., 2.],
        'mean': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }, columns=['range', 'mean'], index=idx)

    expected_intervals = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3)], name='range')
    expected_index = pd.MultiIndex.from_product([element_idx, expected_intervals])
    expected = pd.Series(1, name='cycles', index=expected_index)

    result = df.load_collective.range_histogram([0, 1, 2, 3], 'cycle_number')

    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_load_collective_range_histogram_nested_grouped():
    element_idx = pd.Index([10, 20], name='element_id')
    node_idx = pd.Index([100, 101], name='node_id')
    cycle_idx = pd.Index([0, 1], name='cycle_number')
    idx = pd.MultiIndex.from_product((element_idx, node_idx, cycle_idx))

    df = pd.DataFrame({
        'range': [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0],
        'mean': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }, columns=['range', 'mean'], index=idx)

    expected_intervals = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3)], name='range')
    expected_index = pd.MultiIndex.from_product([element_idx, node_idx, expected_intervals])
    expected = pd.Series([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0], name='cycles', index=expected_index)

    result = df.load_collective.range_histogram([0, 1, 2, 3], 'cycle_number')

    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_load_collective_strange_shift():
    upper_loads = pd.Series([1000., 2000., 1500])
    collective = pd.DataFrame({
        'to': upper_loads,
        'from': 0.0,
    })
    collective.index.name = 'load_block'

    mises = pd.Series([1.0, 2.0, 3.0], index=pd.Index([1, 2, 3], name='node_id'), name='mises')

    lower_stress = pd.Series([100., 200., 300.], index=mises.index, name='lower_stress')

    result = collective.load_collective.scale(mises).shift(lower_stress).to_pandas()

    expected_index = pd.MultiIndex.from_product([collective.index, mises.index])
    expected = pd.DataFrame({
        'to': [1100., 2200., 3300., 2100., 4200., 6300., 1600., 3200., 4800.],
        'from': [100., 200., 300.] * 3
    }, index=expected_index)

    pd.testing.assert_frame_equal(result, expected)


# GH-107
@pytest.mark.parametrize('bins, expected_index_tuples, expected_data', [
    ([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)], [0, 0, 0, 2, 0, 0, 0, 1, 0]),
    ([0, 2, 4], [(0, 2), (2, 4)], [2, 0, 1, 0])
])
def test_load_collective_histogram_alter_bins(bins, expected_index_tuples, expected_data):
    df = pd.DataFrame(
        {'range': [1.5, 2.5, 1.5], 'mean': [0.75, 1.25, 0.75]}, columns=['range', 'mean']
    )

    expected_intervals = pd.IntervalIndex.from_tuples(expected_index_tuples)
    expected = pd.Series(
        expected_data,
        name='cycles',
        index=pd.MultiIndex.from_product(
            [expected_intervals, expected_intervals], names=['range', 'mean']
        ),
        dtype=np.float64
    )

    result = df.load_collective.histogram(bins)

    pd.testing.assert_series_equal(result.to_pandas(), expected)


# GH-107
def test_load_collective_histogram_alter_ranges():
    df = pd.DataFrame({
        'range': [1., 2., 1., 2., 1],
        'mean': [0.5, 1.0, 0.5, 1.0, 0.5]
    }, columns=['range', 'mean'])

    expected_intervals = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3)])
    expected = pd.Series(
        [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        name='cycles',
        index=pd.MultiIndex.from_product(
            [expected_intervals, expected_intervals], names=['range', 'mean']
        ),
    )

    result = df.load_collective.histogram([0, 1, 2, 3])

    pd.testing.assert_series_equal(result.to_pandas(), expected)


# GH-107
def test_load_collective_histogram_interval_index():
    df = pd.DataFrame({
        'range': [1., 2., 1., 2., 1],
        'mean': [0.5, 1.0, 0.5, 1.0, 0.5]
    }, columns=['range', 'mean'])

    expected_intervals = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3)])
    expected = pd.Series(
        [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        name='cycles',
        index=pd.MultiIndex.from_product(
            [expected_intervals, expected_intervals], names=['range', 'mean']
        ),
    )

    result = df.load_collective.histogram(expected_intervals)

    pd.testing.assert_series_equal(result.to_pandas(), expected)


# GH-107
def test_load_collective_histogram_interval_array():
    df = pd.DataFrame({
        'range': [1., 2., 1., 2., 1],
        'mean': [0.5, 1.0, 0.5, 1.0, 0.5]
    }, columns=['range', 'mean'])

    expected_intervals = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3)])
    expected = pd.Series(
        [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        name='cycles',
        index=pd.MultiIndex.from_product(
            [expected_intervals, expected_intervals], names=['range', 'mean']
        ),
    )

    intervals = pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2), (2, 3)])
    result = df.load_collective.histogram(intervals)

    pd.testing.assert_series_equal(result.to_pandas(), expected)


# GH-107
def test_load_collective_histogram_unnested_grouped():
    element_idx = pd.Index([10, 20, 30], name='element_id')
    cycle_idx = pd.Index([0, 1, 2], name='cycle_number')
    idx = pd.MultiIndex.from_product((element_idx, cycle_idx))

    df = pd.DataFrame({
        'range': [1., 2., 1., 2., 1., 2., 1., 1., 1],
        'mean': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }, columns=['range', 'mean'], index=idx)

    expected_intervals = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3)])
    expected_ranges = expected_intervals.set_names(['range'])
    expected_means = expected_intervals.set_names(['mean'])

    expected_index = pd.MultiIndex.from_product(
        [element_idx, expected_ranges, expected_means]
    )
    expected = pd.Series(
        [0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
        name='cycles',
        index=expected_index,
        dtype=np.float64,
    )

    result = df.load_collective.histogram([0, 1, 2, 3], 'cycle_number')
    pd.testing.assert_series_equal(result.to_pandas(), expected)


# GH-107
def test_load_collective_histogram_nested_grouped():
    element_idx = pd.Index([10, 20], name='element_id')
    node_idx = pd.Index([100, 101], name='node_id')
    cycle_idx = pd.Index([0, 1], name='cycle_number')
    idx = pd.MultiIndex.from_product((element_idx, node_idx, cycle_idx))

    df = pd.DataFrame({
        'range': [1., 2., 1., 2., 1., 2., 1., 2.],
        'mean': [0, 0, 0, 0, 0, 0, 0, 0]
    }, columns=['range', 'mean'], index=idx)

    expected_intervals = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3)])
    expected_ranges = expected_intervals.set_names(['range'])
    expected_means = expected_intervals.set_names(['mean'])
    expected_index = pd.MultiIndex.from_product(
        [element_idx, node_idx, expected_ranges, expected_means]
    )
    expected = pd.Series(
        [0, 0, 0, 1, 0, 0, 1, 0, 0] * 4,
        name='cycles',
        index=expected_index,
        dtype=np.float64,
    )

    result = df.load_collective.histogram([0, 1, 2, 3], 'cycle_number')

    pd.testing.assert_series_equal(result.to_pandas(), expected)

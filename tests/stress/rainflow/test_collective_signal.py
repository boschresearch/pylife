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
def test_rainflow_collective_signal_amplitude_from_to(df, expected):
    df.columns = ['from', 'to']
    expected.name = 'amplitude'
    pd.testing.assert_series_equal(df.rainflow.amplitude, expected)

    expected_cycles = pd.Series(1.0, name='cycles', index=df.index)
    pd.testing.assert_series_equal(df.rainflow.cycles, expected_cycles)

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
def test_rainflow_collective_signal_mean_from_to(df, expected):
    df.columns = ['from', 'to']
    expected.name = 'meanstress'
    pd.testing.assert_series_equal(df.rainflow.meanstress, expected)

    expected_cycles = pd.Series(1.0, name='cycles', index=df.index)
    pd.testing.assert_series_equal(df.rainflow.cycles, expected_cycles)



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
def test_rainflow_collective_signal_upper_lower_from_to(df, expected_upper, expected_lower):
    df.columns = ['from', 'to']
    expected_upper.name = 'upper'
    expected_lower.name = 'lower'
    pd.testing.assert_series_equal(df.rainflow.upper, expected_upper)
    pd.testing.assert_series_equal(df.rainflow.lower, expected_lower)


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
def test_rainflow_collective_from_to_scale_scalar(df, expected):
    df.columns = ['from', 'to']
    expected.columns = ['from', 'to']
    pd.testing.assert_frame_equal(df.rainflow.scale(2.0).to_pandas(), expected)


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
def test_rainflow_collective_from_to_shift_scalar(df, expected):
    df.columns = ['from', 'to']
    expected.columns = ['from', 'to']
    pd.testing.assert_frame_equal(df.rainflow.shift(2.0).to_pandas(), expected)


def test_rainflow_collective_from_to_scale_series():
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

    pd.testing.assert_frame_equal(df.rainflow.scale(scale_operand).to_pandas(), expected)


def test_rainflow_collective_from_to_shift_series():
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

    pd.testing.assert_frame_equal(df.rainflow.shift(shift_operand).to_pandas(), expected)


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
def test_rainflow_collective_signal_amplitude_range_mean(df, expected):
    df.columns = ['range', 'mean']
    expected.name = 'amplitude'
    pd.testing.assert_series_equal(df.rainflow.amplitude, expected)

    expected_cycles = pd.Series(1.0, name='cycles', index=df.index)
    pd.testing.assert_series_equal(df.rainflow.cycles, expected_cycles)


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
def test_rainflow_collective_signal_upper_lower_range_mean(df, expected_upper, expected_lower):
    df.columns = ['range', 'mean']
    expected_upper.name = 'upper'
    expected_lower.name = 'lower'
    pd.testing.assert_series_equal(df.rainflow.upper, expected_upper)
    pd.testing.assert_series_equal(df.rainflow.lower, expected_lower)


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
def test_rainflow_collective_signal_mean_range_mean(df, expected):
    df.columns = ['range', 'mean']
    expected.name = 'meanstress'
    pd.testing.assert_series_equal(df.rainflow.meanstress, expected)


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
def test_rainflow_collective_signal_mean_range_scale_scalar(df, expected_amplitude, expected_mean):
    df.columns = ['range', 'mean']
    expected_amplitude.name = 'amplitude'
    expected_mean.name = 'meanstress'
    scaled = df.rainflow.scale(2.0)
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
def test_rainflow_collective_signal_mean_range_shift_scalar(df, expected_amplitude, expected_mean):
    df.columns = ['range', 'mean']
    expected_amplitude.name = 'amplitude'
    expected_mean.name = 'meanstress'
    scaled = df.rainflow.shift(2.0)
    pd.testing.assert_series_equal(scaled.amplitude, expected_amplitude)
    pd.testing.assert_series_equal(scaled.meanstress, expected_mean)

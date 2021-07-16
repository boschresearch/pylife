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


def matrix_index():
    range_intervals = pd.interval_range(-4., 8., 3)
    mean_intervals = pd.interval_range(-2., 4., 3)
    return pd.MultiIndex.from_product([range_intervals, mean_intervals])

@pytest.fixture
def rainflow_matrix_range_mean():
    index = matrix_index()
    index.names = ['range', 'mean']
    return pd.Series(1, index=index)


@pytest.fixture
def rainflow_matrix_from_to():
    index = matrix_index()
    index.names = ['from', 'to']
    return pd.Series(1, index=index)


def test_rainflow_signal_fail_empty_fail():
    rainflow_matrix = pd.Series(dtype=np.float64)
    with pytest.raises(AttributeError, match=r"RainflowAccessor needs .* index levels"):
        rainflow_matrix.rainflow


def test_rainflow_signal_valid_range_mean(rainflow_matrix_range_mean):
    rainflow_matrix_range_mean.rainflow


def test_rainflow_signal_valid_from_to(rainflow_matrix_from_to):
    rainflow_matrix_from_to.rainflow


def test_rainflow_signal_valid_only_from(rainflow_matrix_from_to):
    rainflow_matrix_from_to.index = rainflow_matrix_from_to.index.droplevel('to')
    with pytest.raises(AttributeError, match=r"RainflowAccessor needs .* index levels"):
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


def test_rainflow_range_mean_amplitude_right(rainflow_matrix_range_mean):
    expected = pd.Series([0., 0., 0., 2., 2., 2., 4., 4., 4.],
                         name='amplitude',
                         index=rainflow_matrix_range_mean.index)
    pd.testing.assert_series_equal(rainflow_matrix_range_mean.rainflow.use_class_right().amplitude, expected)


def test_rainflow_range_mean_amplitude_mid(rainflow_matrix_range_mean):
    expected = pd.Series([-1., -1., -1., 1., 1., 1., 3., 3., 3.],
                         name='amplitude',
                         index=rainflow_matrix_range_mean.index)
    pd.testing.assert_series_equal(rainflow_matrix_range_mean.rainflow.amplitude, expected)


def test_rainflow_range_mean_amplitude_left(rainflow_matrix_range_mean):
    expected = pd.Series([-2., -2., -2., 0., 0., 0., 2., 2., 2.],
                         name='amplitude',
                         index=rainflow_matrix_range_mean.index)
    pd.testing.assert_series_equal(rainflow_matrix_range_mean.rainflow.use_class_left().amplitude, expected)


def test_rainflow_frequency(rainflow_matrix_range_mean):
    expected = pd.Series(1, index=rainflow_matrix_range_mean.index, name='frequency')
    freq = rainflow_matrix_range_mean.rainflow.frequency
    pd.testing.assert_series_equal(freq, expected)

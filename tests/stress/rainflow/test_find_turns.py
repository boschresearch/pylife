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

from pylife.stress.rainflow.general import AbstractDetector
from pylife.stress.timesignal import TimeSignalGenerator

import pylife.stress.rainflow as RF


class DummyDetector(AbstractDetector):
    def process(self, samples):
        return self


def test_rainflow_partial_get_turns_general():
    tsgen = TimeSignalGenerator(10, {'number': 50,
                                     'amplitude_median': 1.0, 'amplitude_std_dev': 0.5,
                                     'frequency_median': 4, 'frequency_std_dev': 3,
                                     'offset_median': 0, 'offset_std_dev': 0.4},
                                None, None)

    signal_tot = tsgen.query(10000)
    _, turns_tot = DummyDetector(recorder=None)._new_turns(signal_tot)
    rfc_partial = DummyDetector(recorder=None)
    turns_partial = np.concatenate((
        rfc_partial._new_turns(signal_tot[:3424])[1],
        rfc_partial._new_turns(signal_tot[3424:])[1]))

    np.testing.assert_array_equal(turns_tot, turns_partial)


def test_rainflow_partial_signals_get_turns_splitturn():
    tsgen = TimeSignalGenerator(10, {'number': 50,
                                     'amplitude_median': 1.0, 'amplitude_std_dev': 0.5,
                                     'frequency_median': 4, 'frequency_std_dev': 3,
                                     'offset_median': 0, 'offset_std_dev': 0.4},
                                None, None)

    signal_tot = tsgen.query(10000)
    _, turns_tot = DummyDetector(recorder=None)._new_turns(signal_tot)
    turn_points, _ = RF.find_turns(signal_tot)
    turn_points = np.insert(turn_points, 0, 0)
    turn_num = turn_points.shape[0]
    split_points = [int(np.ceil(turn_num*x)) for x in [0.0, 0.137, 0.23, 0.42, 1.0]]
    rfc_partial = DummyDetector(recorder=None)
    turns_partial = np.empty(0)
    for i in range(len(split_points)-1):
        lower = turn_points[split_points[i]]
        upper = 10000 if split_points[i+1] == turn_points.shape[0] else turn_points[split_points[i+1]]
        turns_partial = np.concatenate((turns_partial, rfc_partial._new_turns(signal_tot[lower:upper])[1]))

    print(turns_tot[-5:])
    print(turns_partial[-5:])
    np.testing.assert_array_equal(turns_tot, turns_partial)


def test_rainflow_partial_get_turns_no_turns():
    samples = np.array([0., 1.])
    index, values = RF.find_turns(samples)
    assert len(index) == 0
    assert len(values) == 0


def test_rainflow_partial_get_turns_consecutive_duplicates():
    samples = np.array([1., 1., 0.5, 0.5, 1., 1., 1., -1., -1., 0.5, 1.])
    index, values = RF.find_turns(samples)
    np.testing.assert_array_equal(values, np.array([0.5, 1., -1.]))
    np.testing.assert_array_equal(index, np.array([2, 4, 7]))


def test_rainflow_duplicates_no_peak_up():
    samples = np.array([1., 2., 2., 3.])
    index, values = RF.find_turns(samples)
    assert len(index) == 0
    assert len(values) == 0


def test_rainflow_duplicates_no_peak_down():
    samples = np.array([3., 2., 2., 1.])
    index, values = RF.find_turns(samples)
    assert len(index) == 0
    assert len(values) == 0


def test_rainflow_find_turns_shifted_index():
    samples = np.array([32., 32., 32.1, 32.9, 33., 33., 33., 33., 33., 32.5, 32., 32., 32.7, 37.2, 40., 35.2, 33.])
    expected_index = [4, 10, 14]
    expected_values = [33., 32., 40.]
    index, values = RF.find_turns(samples)
    np.testing.assert_array_equal(index, expected_index)
    np.testing.assert_array_equal(values, expected_values)


def test_rainflow_find_turns_shifted_index_four_leading_dups():
    samples = np.array([32., 32., 32., 32., 32.1, 32.9, 33., 33., 33., 33., 33., 32.5, 32., 32., 32.7, 37.2, 40., 35.2, 33.])
    expected_index = [6, 12, 16]
    expected_values = [33., 32., 40.]
    index, values = RF.find_turns(samples)
    np.testing.assert_array_equal(index, expected_index)
    np.testing.assert_array_equal(values, expected_values)


def test_rainflow_find_turns_shifted_index_four_trailing_dups():
    samples = np.array([32., 32.1, 32.9, 33., 33., 33., 33., 33., 32.5, 32., 32., 32.7, 37.2, 40., 35.2, 33., 33., 33., 33., 33.])
    expected_index = [3, 9, 13]
    expected_values = [33., 32., 40.]
    index, values = RF.find_turns(samples)
    np.testing.assert_array_equal(index, expected_index)
    np.testing.assert_array_equal(values, expected_values)


def test_find_turns_leading_dups():
    samples = np.array([1., 1., 1., 2., 1.])
    expected_index = [3]
    expected_values = [2.]

    index, values = RF.find_turns(samples)
    np.testing.assert_array_equal(index, expected_index)
    np.testing.assert_array_equal(values, expected_values)


def test_find_turns_trailing_dups():
    samples = np.array([1., 2., 1., 1., 1.])
    expected_index = [1]
    expected_values = [2.]

    index, values = RF.find_turns(samples)
    np.testing.assert_array_equal(index, expected_index)
    np.testing.assert_array_equal(values, expected_values)


def test_find_turns_leading_and_trailing_dups():
    samples = np.array([0., 0., 1., 2., 2., 1., 0., 0., 1., 2., 2.])
    expected_index = [3, 6]
    expected_values = [2., 0.]

    index, values = RF.find_turns(samples)
    np.testing.assert_array_equal(index, expected_index)
    np.testing.assert_array_equal(values, expected_values)


def test_find_turns_leading_and_trailing_dups_no_turns():
    samples = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    expected_index = []
    expected_values = []

    index, values = RF.find_turns(samples)
    np.testing.assert_array_equal(index, expected_index)
    np.testing.assert_array_equal(values, expected_values)


def test_find_turns_flat_signal():
    samples = np.array([0.0, 0.0, 0.0, 0.0])
    expected_index = []
    expected_values = []

    index, values = RF.find_turns(samples)
    np.testing.assert_array_equal(index, expected_index)
    np.testing.assert_array_equal(values, expected_values)


@pytest.mark.parametrize('null', [None, np.nan, pd.NaT, float("nan"), pd.NA])
def test_find_turns_nan_next_to_turns_nan_after_turn(null):
    samples = np.array([0.0, 1.0, null, 0.0, null, 1.0])
    expected_index = [1, 3]
    expected_values = [1.0, 0.0]

    with pytest.warns(UserWarning, match="At least one NaN like value has been dropped from the input signal."):
        index, values = RF.find_turns(samples)
    np.testing.assert_array_equal(index, expected_index)
    np.testing.assert_array_equal(values, expected_values)


@pytest.mark.parametrize('null', [None, np.nan, pd.NaT, float("nan"), pd.NA])
def test_find_turns_nan_next_to_turns_nan_before_turns(null):
    samples = np.array([0.0, null, 1.0, null, 0.0, 1.0])
    expected_index = [2, 4]
    expected_values = [1.0, 0.0]

    with pytest.warns(UserWarning, match="At least one NaN like value has been dropped from the input signal."):
        index, values = RF.find_turns(samples)
    np.testing.assert_array_equal(index, expected_index)
    np.testing.assert_array_equal(values, expected_values)

# Copyright (c) 2019-2022 - for information on the respective copyright owner
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

__author__ = ["Johannes Mueller", "Benjamin Maier"]
__maintainer__ = __author__

import pytest
import pandas as pd
import numpy as np
import copy

from pylife.stress.rainflow.general import AbstractDetector
from pylife.stress.timesignal import TimeSignalGenerator

import pylife.stress.rainflow as RF


class DummyDetector(AbstractDetector):
    def process(self, samples):
        return self


@pytest.mark.parametrize("samples, expected_index, expected_values", [
    (
        # flush
        np.array([32., 32., 32.1, 32.9, 33., 33., 33., 33., 33., 32.5, 32., 32., 32.7, 37.2, 40., 35.2, 33.]),
        [4, 10, 14, 16],
        [33., 32., 40., 33.]
    ),(
        # falling_one
        np.array([1., 2., 1.]),
        [1, 2],
        [2., 1.]
    ),(
        # rising_one
        np.array([1., 2., 1., 1.5]),
        [1, 2, 3],
        [2., 1., 1.5]
    ),(
        # falling_one
        np.array([1., 2., 1.]),
        [1, 2],
        [2., 1.]
    ),(
        # falling_two
        np.array([1., 2., 1., 0.]),
        [1, 3],
        [2., 0.]
    ),(
        # rising_two
        np.array([1., 2., 1., 1.5, 4]),
        [1, 2, 4],
        [2., 1., 4]
    )
])
def test_rainflow_new_turns_flush(samples, expected_index, expected_values):

    index, values = DummyDetector(recorder=None)._new_turns(samples, flush=True)
    np.testing.assert_array_equal(index, expected_index)
    np.testing.assert_array_equal(values, expected_values)


@pytest.mark.parametrize("samples1, expected_index1, expected_values1, samples2, expected_index2, expected_values2", [
    (
        np.array([5., 2., 3.]),
        [1, 2],
        [2., 3.],
        np.array([4., 1., 0.]),
        [3],
        [4.],
    ),(
        np.array([2., 5., 2., 3.]),
        [1, 2, 3],
        [5., 2., 3.],
        np.array([0., 1., 0.]),
        [4, 5],
        [0., 1.]
    )
])
def test_rainflow_new_turns_flush_continue(samples1, expected_index1, expected_values1, samples2, expected_index2, expected_values2):
    detector = DummyDetector(recorder=None)

    index, values = detector._new_turns(samples1, flush=True)
    np.testing.assert_array_equal(index, expected_index1)
    np.testing.assert_array_equal(values, expected_values1)

    index, values = detector._new_turns(samples2, flush=False)
    np.testing.assert_array_equal(index, expected_index2)
    np.testing.assert_array_equal(values, expected_values2)


@pytest.mark.parametrize("samples1, expected_index1, expected_values1, samples2, expected_index2, expected_values2", [
    (
        np.array([2., 5., 2., 3.]),
        [1, 2, 3],
        [5., 2., 3.],
        np.array([0., 1., 0.]),
        [4, 5, 6],
        [0., 1., 0.],
    ),(
        np.array([2., 5., 2., 3.]),
        [1, 2, 3],
        [5., 2., 3.],
        np.array([0., 1., 0.5, 0.01, 0.]),
        [4, 5, 8],
        [0., 1., 0.],
    ),(
        np.array([2., 5., 2., 3.]),
        [1, 2, 3],
        [5., 2., 3.],
        np.array([4., 5., 6., 7.]),
        [7],
        [7.],
    ),
])
def test_rainflow_new_turns_double_flush_continue(samples1, expected_index1, expected_values1, samples2, expected_index2, expected_values2):
    detector = DummyDetector(recorder=None)

    index, values = detector._new_turns(samples1, flush=True)
    np.testing.assert_array_equal(index, expected_index1)
    np.testing.assert_array_equal(values, expected_values1)

    index, values = detector._new_turns(samples2, flush=True)
    np.testing.assert_array_equal(index, expected_index2)
    np.testing.assert_array_equal(values, expected_values2)


def test_rainflow_new_turns_flush_continue_four_times():
    detector = DummyDetector(recorder=None)

    samples = np.array([2., 5., 2., 3.])
    expected_index = [1, 2, 3]
    expected_values = [5., 2., 3.]
    index, values = detector._new_turns(samples, flush=True)
    np.testing.assert_array_equal(index, expected_index)
    np.testing.assert_array_equal(values, expected_values)

    samples = np.array([4., 5.])
    expected_index = []
    expected_values = []
    index, values = detector._new_turns(samples, flush=False)
    np.testing.assert_array_equal(index, expected_index)
    np.testing.assert_array_equal(values, expected_values)

    samples = np.array([4., 5., 6., 7.])
    expected_index = [5, 6, 9]
    expected_values = [5., 4., 7.]
    index, values = detector._new_turns(samples, flush=True)
    np.testing.assert_array_equal(index, expected_index)
    np.testing.assert_array_equal(values, expected_values)

    samples = np.array([4., 5., 6., 7.])
    expected_index = [10]
    expected_values = [4.]
    index, values = detector._new_turns(samples)
    np.testing.assert_array_equal(index, expected_index)
    np.testing.assert_array_equal(values, expected_values)

    expected_index = [13]
    expected_values = [7.]
    index, values = detector._new_turns(np.array([]), flush=True)


@pytest.mark.parametrize("samples, expected_index, expected_values", [
    (
        # flush
        np.array([32., 32., 32.1, 32.9, 33., 33., 33., 33., 33., 32.5, 32., 32., 32.7, 37.2, 40., 35.2, 33.]),
        [0, 4, 10, 14, 16],
        [32., 33., 32., 40., 33.]
    ),(
        # falling_one
        np.array([1., 2., 1.]),
        [0, 1, 2],
        [1., 2., 1.]
    ),(
        # falling_one
        np.array([-1., -2., -1.]),
        [0, 1, 2],
        [-1., -2., -1.]
    ),(
        # rising_one
        np.array([1., 2., 1., 1.5]),
        [0, 1, 2, 3],
        [1., 2., 1., 1.5]
    ),(
        # falling_one
        np.array([1., 2., 1.]),
        [0, 1, 2],
        [1., 2., 1.]
    ),(
        # falling_two
        np.array([1., 2., 1., 0.]),
        [0, 1, 3],
        [1., 2., 0.]
    ),(
        # rising_two
        np.array([1., 2., 1., 1.5, 4]),
        [0, 1, 2, 4],
        [1., 2., 1., 4]
    ),(
        np.array([1., 2., 3., 1., 1.5, 4]),
        [0, 2, 3, 5],
        [1., 3., 1., 4]
    ),
])
def test_rainflow_new_turns_preserve_start(samples, expected_index, expected_values):

    index, values = DummyDetector(recorder=None)._new_turns(samples, flush=True, preserve_start=True)
    np.testing.assert_array_equal(index, expected_index)
    np.testing.assert_array_equal(values, expected_values)


# with data frames
@pytest.mark.skip(reason="handled in process")
@pytest.mark.parametrize("samples, expected_index", [
    (
        # flush
        np.array([32., 32., 32.1, 32.9, 33., 33., 33., 33., 33., 32.5, 32., 32., 32.7, 37.2, 40., 35.2, 33.]),
        [4, 10, 14, 16],
    ),(
        # falling_one
        np.array([1., 2., 1.]),
        [1, 2],
    ),(
        # rising_one
        np.array([1., 2., 1., 1.5]),
        [1, 2, 3],
    ),(
        # falling_one
        np.array([1., 2., 1.]),
        [1, 2],
    ),(
        # falling_two
        np.array([1., 2., 1., 0.]),
        [1, 3],
    ),(
        # rising_two
        np.array([1., 2., 1., 1.5, 4]),
        [1, 2, 4],
    ),
])
def test_rainflow_new_turns_flush_df(samples, expected_index):

    index = pd.MultiIndex.from_product([range(len(samples)), (1,5,8)], names=["load_step", "node_id"])

    multiple_samples = np.concatenate([[samples], [samples*5], [samples*-2]]).T.reshape(-1,1)
    df_samples = pd.DataFrame(multiple_samples, index=index, columns=["L"])

    index, values = DummyDetector(recorder=None)._new_turns(df_samples, flush=True)
    np.testing.assert_array_equal(index, expected_index)

    # the expected values are those where the load_step is equal to the expected index
    expected_values = df_samples[df_samples.index.get_level_values("load_step").isin(expected_index)]
    expected_values = [df.reset_index(drop=True) for _, df in expected_values.groupby("load_step")]

    np.testing.assert_array_equal(values, expected_values)


@pytest.mark.skip(reason="handled in process")
@pytest.mark.parametrize("samples1, expected_index1, samples2, expected_index2", [
    (
        np.array([5., 2., 3.]),
        [1, 2],
        np.array([4., 1., 0.]),
        [3],
    ),(
        np.array([2., 5., 2., 3.]),
        [1, 2, 3],
        np.array([0., 1., 0.]),
        [4, 5],
    )
])
def test_rainflow_new_turns_flush_continue_df(samples1, expected_index1, samples2, expected_index2):

    detector = DummyDetector(recorder=None)

    # first call to _new_turns
    index = pd.MultiIndex.from_product([range(len(samples1)), (1,5,8)], names=["load_step", "node_id"])

    multiple_samples = np.concatenate([[samples1], [samples1*5], [samples1*-2]]).T.reshape(-1,1)
    df_samples = pd.DataFrame(multiple_samples, index=index, columns=["L"])

    index, values = detector._new_turns(df_samples, flush=True)
    np.testing.assert_array_equal(index, expected_index1)

    # the expected values are those where the load_step is equal to the expected index
    expected_values = df_samples[df_samples.index.get_level_values("load_step").isin(expected_index1)]
    expected_values = [df.reset_index(drop=True) for _, df in expected_values.groupby("load_step")]

    np.testing.assert_array_equal(values, expected_values)

    # second call to _new_turns
    index = pd.MultiIndex.from_product([range(len(samples2)), (1,5,8)], names=["load_step", "node_id"])

    multiple_samples = np.concatenate([[samples2], [samples2*5], [samples2*-2]]).T.reshape(-1,1)
    df_samples = pd.DataFrame(multiple_samples, index=index, columns=["L"])

    index, values = detector._new_turns(df_samples, flush=False)
    np.testing.assert_array_equal(index, expected_index2)

    # the expected values are those where the load_step is equal to the expected index
    expected_index2 = [index - len(samples1) for index in expected_index2]
    expected_values = df_samples[df_samples.index.get_level_values("load_step").isin(expected_index2)]
    expected_values = [df.reset_index(drop=True) for _, df in expected_values.groupby("load_step")]

    np.testing.assert_array_equal(values, expected_values)


@pytest.mark.skip(reason="handled in process")
@pytest.mark.parametrize("samples1, expected_index1, samples2, expected_index2", [
    (
        np.array([2., 5., 2., 3.]),
        [1, 2, 3],
        np.array([0., 1., 0.]),
        [4, 5, 6],
    ),
    (
        np.array([2., 5., 2., 3.]),
        [1, 2, 3],
        np.array([0., 1., 0.5, 0.01, 0.]),
        [4, 5, 8],
    ),
    (
        np.array([2., 5., 2., 3.]),
        [1, 2, 3],
        np.array([4., 5., 6., 7.]),
        [7],
    ),
])
def test_rainflow_new_turns_double_flush_continue_df(samples1, expected_index1, samples2, expected_index2):

    detector = DummyDetector(recorder=None)

    # first call to _new_turns
    index = pd.MultiIndex.from_product([range(len(samples1)), (1,5,8)], names=["load_step", "node_id"])

    multiple_samples = np.concatenate([[samples1], [samples1*5], [samples1*-2]]).T.reshape(-1,1)
    df_samples = pd.DataFrame(multiple_samples, index=index, columns=["L"])

    index, values = detector._new_turns(df_samples, flush=True)
    np.testing.assert_array_equal(index, expected_index1)

    # the expected values are those where the load_step is equal to the expected index
    expected_values = df_samples[df_samples.index.get_level_values("load_step").isin(expected_index1)]
    expected_values = [df.reset_index(drop=True) for _, df in expected_values.groupby("load_step")]

    np.testing.assert_array_equal(values, expected_values)

    # second call to _new_turns
    index = pd.MultiIndex.from_product([range(len(samples2)), (1,5,8)], names=["load_step", "node_id"])

    multiple_samples = np.concatenate([[samples2], [samples2*5], [samples2*-2]]).T.reshape(-1,1)
    df_samples = pd.DataFrame(multiple_samples, index=index, columns=["L"])

    detector2 = copy.deepcopy(detector)
    index, values = detector._new_turns(df_samples, flush=True)
    np.testing.assert_array_equal(index, expected_index2)

    # the expected values are those where the load_step is equal to the expected index
    adjusted_expected_index = [index - len(samples1) for index in expected_index2]
    expected_values = df_samples[df_samples.index.get_level_values("load_step").isin(adjusted_expected_index)]
    expected_values = [df.reset_index(drop=True) for _, df in expected_values.groupby("load_step")]

    np.testing.assert_array_equal(values, expected_values)

    # -----------------------
    # do the same again, with preserve_start=True, which must not have any effect
    index, values = detector2._new_turns(df_samples, flush=True, preserve_start=True)
    np.testing.assert_array_equal(index, expected_index2)

    # the expected values are those where the load_step is equal to the expected index
    expected_values = df_samples[df_samples.index.get_level_values("load_step").isin(adjusted_expected_index)]
    expected_values = [df.reset_index(drop=True) for _, df in expected_values.groupby("load_step")]

    np.testing.assert_array_equal(values, expected_values)


@pytest.mark.skip(reason="handled in process")
@pytest.mark.parametrize("samples, expected_index", [
    (
        # flush
        np.array([32., 32., 32.1, 32.9, 33., 33., 33., 33., 33., 32.5, 32., 32., 32.7, 37.2, 40., 35.2, 33.]),
        [0, 4, 10, 14, 16],
    ),(
        # falling_one
        np.array([1., 2., 1.]),
        [0, 1, 2],
    ),(
        # falling_one
        np.array([-1., -2., -1.]),
        [0, 1, 2],
    ),(
        # rising_one
        np.array([1., 2., 1., 1.5]),
        [0, 1, 2, 3],
    ),(
        # falling_one
        np.array([1., 2., 1.]),
        [0, 1, 2],
    ),(
        # falling_two
        np.array([1., 2., 1., 0.]),
        [0, 1, 3],
    ),(
        # rising_two
        np.array([1., 2., 1., 1.5, 4]),
        [0, 1, 2, 4],
    ),(
        np.array([1., 2., 3., 1., 1.5, 4]),
        [0, 2, 3, 5],
    ),
])
def test_rainflow_new_turns_preserve_start_df(samples, expected_index):
    index = pd.MultiIndex.from_product([range(len(samples)), (1,5,8)], names=["load_step", "node_id"])

    multiple_samples = np.concatenate([[samples], [samples*5], [samples*-2]]).T.reshape(-1,1)
    df_samples = pd.DataFrame(multiple_samples, index=index, columns=["L"])

    index, values = DummyDetector(recorder=None)._new_turns(df_samples, flush=True, preserve_start=True)
    np.testing.assert_array_equal(index, expected_index)

    # the expected values are those where the load_step is equal to the expected index
    expected_values = df_samples[df_samples.index.get_level_values("load_step").isin(expected_index)]
    expected_values = [df.reset_index(drop=True) for _, df in expected_values.groupby("load_step")]

    np.testing.assert_array_equal(values, expected_values)


@pytest.mark.skip(reason="handled in process")
def test_single_multiple_points():

    # generate a load sequence for three assessment points
    load_sequence_0 = pd.Series([100, -200, 100, -250, 200, 0, 200, -200])  # [N]
    load_sequence_1 = load_sequence_0 * 1.2
    load_sequence_2 = load_sequence_0 * 0.2     # last has infinite life

    index = pd.MultiIndex.from_product([range(len(load_sequence_0)), [0,1,2]], names=["load_step", "node_id"])

    load_sequence = pd.DataFrame(index=index, data={"load": 0})
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==0,"load"] = load_sequence_0.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==1,"load"] = load_sequence_1.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==2,"load"] = load_sequence_2.to_numpy()


    detector0 = DummyDetector(recorder=None)
    loads_indices1, load_turning_points1 = detector0._new_turns(load_sequence, flush=True, preserve_start=True)

    loads_indices2, load_turning_points2 = detector0._new_turns(load_sequence, flush=True, preserve_start=False)

    print(load_turning_points1)
    print(load_turning_points2)

    # only first row
    detector1 = DummyDetector(recorder=None)
    loads_indices3, load_turning_points3 = detector1._new_turns(load_sequence_0.to_numpy(), flush=True, preserve_start=True)

    loads_indices4, load_turning_points4 = detector1._new_turns(load_sequence_0.to_numpy(), flush=True, preserve_start=False)

    print(load_turning_points3)
    print(load_turning_points4)
    np.testing.assert_array_equal(loads_indices1, loads_indices3)

    np.testing.assert_array_equal(loads_indices2, loads_indices4)

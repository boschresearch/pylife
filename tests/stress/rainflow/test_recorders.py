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
import numpy as np
import pandas as pd
import pylife.stress.rainflow.recorders as RFR


def test_abstract_recorder_new_no_chunks():
    arr = RFR.AbstractRecorder()
    assert len(arr.chunks) == 0


def test_abstract_recorder_one_chunk():
    arr = RFR.AbstractRecorder()
    arr.report_chunk(23)

    np.testing.assert_array_equal(arr.chunks, [23])


def test_abstract_recorder_two_chunks():
    arr = RFR.AbstractRecorder()
    arr.report_chunk(42)
    arr.report_chunk(23)
    np.testing.assert_array_equal(arr.chunks, [42, 23])


def test_abstract_recorder_calc_chunk_local_index():
    arr = RFR.AbstractRecorder()
    arr.report_chunk(42)
    arr.report_chunk(23)

    assert arr.chunk_local_index(0) == (0, 0)
    assert arr.chunk_local_index(17) == (0, 17)
    assert arr.chunk_local_index(42) == (1, 0)
    assert arr.chunk_local_index(47) == (1, 5)


def test_abstract_recorder_calc_chunk_local_index_array():
    arr = RFR.AbstractRecorder()
    arr.report_chunk(42)
    arr.report_chunk(23)

    chunks, index = arr.chunk_local_index([0, 17, 42, 47])
    np.testing.assert_array_equal(chunks, [0, 0, 1, 1])
    np.testing.assert_array_equal(index, [0, 17, 0, 5])


def test_loopvalue_recorder_new_all_empty():
    lvr = RFR.LoopValueRecorder()
    assert len(lvr.values_from) == 0
    assert len(lvr.values_to) == 0


def test_full_rainflow_recorder_new_all_empty():
    fr = RFR.FullRecorder()
    assert len(fr.values_from) == 0
    assert len(fr.values_to) == 0
    assert len(fr.index_from) == 0
    assert len(fr.index_to) == 0


@pytest.mark.parametrize('value_from, value_to, index_from, index_to', [
    (23., 42., 11, 17),
    (46., 84., 22, 34)
])
def test_full_rainflow_recorder_record_value(value_from, value_to, index_from, index_to):
    fr = RFR.FullRecorder()
    fr.record_values(value_from, value_to)
    fr.record_index(index_from, index_to)
    np.testing.assert_array_equal(fr.values_from, [value_from])
    np.testing.assert_array_equal(fr.values_to, [value_to])
    np.testing.assert_array_equal(fr.index_from, [index_from])
    np.testing.assert_array_equal(fr.index_to, [index_to])


def test_loop_value_rainflow_recorder_record_two_values():
    vf1, vt1, vf2, vt2 = 23., 42., 46., 84.
    lvr = RFR.LoopValueRecorder()

    lvr.record_values(vf1, vt1)
    lvr.record_values(vf2, vt2)

    np.testing.assert_array_equal(lvr.values_from, [vf1, vf2])
    np.testing.assert_array_equal(lvr.values_to, [vt1, vt2])


def test_full_rainflow_recorder_record_two_values():
    vf1, vt1, if1, it1, vf2, vt2, if2, it2 = 23., 42., 11, 17, 46., 84., 22, 34
    fr = RFR.FullRecorder()

    fr.record_values(vf1, vt1)
    fr.record_index(if1, it1)

    fr.record_values(vf2, vt2)
    fr.record_index(if2, it2)

    np.testing.assert_array_equal(fr.values_from, [vf1, vf2])
    np.testing.assert_array_equal(fr.values_to, [vt1, vt2])
    np.testing.assert_array_equal(fr.index_from, [if1, if2])
    np.testing.assert_array_equal(fr.index_to, [it1, it2])


def test_full_rainflow_recorder_empty_collective_default():
    fr = RFR.FullRecorder()
    expected = pd.DataFrame({
        'from': [],
        'to': [],
        'index_from': [],
        'index_to': []
    })
    pd.testing.assert_frame_equal(fr.collective, expected)


def test_full_rainflow_recorder_two_non_zero_collective():
    vf1, vt1, if1, it1, vf2, vt2, if2, it2 = 23., 42., 11, 17, 46., 84., 22, 34
    fr = RFR.FullRecorder()
    fr.record_values(vf1, vt1)
    fr.record_index(if1, it1)
    fr.record_values(vf2, vt2)
    fr.record_index(if2, it2)

    expected = pd.DataFrame({
        'from': [vf1, vf2],
        'to': [vt1, vt2],
        'index_from': [if1, if2],
        'index_to': [it1, it2]
    })

    pd.testing.assert_frame_equal(fr.collective, expected)


def test_full_rainflow_recorder_empty_matrix_default():
    fr = RFR.FullRecorder()
    matrix, _, _ = fr.matrix()
    np.testing.assert_array_equal(matrix, np.zeros((10, 10)))


def test_full_rainflow_recorder_empty_matrix_5_bins():
    fr = RFR.FullRecorder()
    matrix, _, _ = fr.matrix(bins=5)
    np.testing.assert_array_equal(matrix, np.zeros((5, 5)))


@pytest.mark.parametrize('value_from, value_to, index_from, index_to', [
    (23., 42., 11, 17),
    (46., 84., 22, 34)
])
def test_full_rainflow_recorder_one_non_zero(value_from, value_to, index_from, index_to):
    fr = RFR.FullRecorder()
    fr.record_values(value_from, value_to)
    fr.record_index(index_from, index_to)

    expected_from = np.linspace(value_from - 0.5, value_from + 0.5, 11)
    expected_to = np.linspace(value_to - 0.5, value_to + 0.5, 11)
    expected_matrix = np.zeros((10, 10))
    expected_matrix[5, 5] = 1.

    matrix, vfrom, vto = fr.matrix()
    np.testing.assert_array_equal(expected_from, vfrom)
    np.testing.assert_array_equal(expected_to, vto)
    np.testing.assert_array_equal(expected_matrix, matrix)


def test_full_rainflow_recorder_two_non_zero():
    vf1, vt1, if1, it1, vf2, vt2, if2, it2 = 23., 42., 11, 17, 46., 84., 22, 34
    fr = RFR.FullRecorder()
    fr.record_values(vf1, vt1)
    fr.record_index(if1, it1)
    fr.record_values(vf2, vt2)
    fr.record_index(if2, it2)
    fr.record_values(vf2, vt2)
    fr.record_index(if2, it2)

    expected_from = np.linspace(vf1, vf2, 11)
    expected_to = np.linspace(vt1, vt2, 11)
    expected_matrix = np.zeros((10, 10))
    expected_matrix[0, 0] = 1.
    expected_matrix[-1, -1] = 2.

    matrix, vfrom, vto = fr.matrix()
    np.testing.assert_array_equal(expected_from, vfrom)
    np.testing.assert_array_equal(expected_to, vto)
    np.testing.assert_array_equal(expected_matrix, matrix)


def test_loopvalue_rainflow_recorder_empty_collective_default():
    lvr = RFR.LoopValueRecorder()
    expected = pd.DataFrame({'from': [], 'to': []})
    pd.testing.assert_frame_equal(lvr.collective, expected)


def test_loop_value_rainflow_recorder_record_two_values_collective():
    vf1, vt1, vf2, vt2 = 23., 42., 46., 84.
    lvr = RFR.LoopValueRecorder()

    lvr.record_values(vf1, vt1)
    lvr.record_values(vf2, vt2)

    expected = pd.DataFrame({
        'from': [vf1, vf2],
        'to': [vt1, vt2]
    })

    pd.testing.assert_frame_equal(lvr.collective, expected)

def test_lopvalue_rainflow_recorder_empty_matrix_series_default():
    fr = RFR.LoopValueRecorder()
    matrix = fr.matrix_series()
    assert matrix.index.names[0] == 'from'
    assert matrix.index.names[1] == 'to'
    np.testing.assert_array_equal(matrix.to_numpy(), np.zeros(100))


def test_loopvalue_rainflow_recorder_empty_matrix_series_5_bins():
    fr = RFR.LoopValueRecorder()
    matrix = fr.matrix_series(bins=5).to_numpy()
    np.testing.assert_array_equal(matrix, np.zeros(25))


@pytest.mark.parametrize('value_from, value_to', [
    (23., 42.),
    (46., 84.)
])
def test_loopvalue_rainflow_recorder_matrix_series_one_non_zero(value_from, value_to):
    fr = RFR.LoopValueRecorder()
    fr.record_values(value_from, value_to)

    expected_from = pd.IntervalIndex.from_breaks(np.linspace(value_from - 0.5, value_from + 0.5, 11))
    expected_to = pd.IntervalIndex.from_breaks(np.linspace(value_to - 0.5, value_to + 0.5, 11))
    expected_index = pd.MultiIndex.from_product([expected_from, expected_to], names=['from', 'to'])

    expected_matrix = np.zeros((10, 10))
    expected_matrix[5, 5] = 1.

    matrix = fr.matrix_series()
    pd.testing.assert_series_equal(matrix, pd.Series(expected_matrix.flatten(), index=expected_index))

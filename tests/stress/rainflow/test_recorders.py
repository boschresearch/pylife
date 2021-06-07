import pytest
import numpy as np
import pandas as pd
import pylife.stress.rainflow.recorders as RFR


def test_generic_rainflow_recorder_new_all_empty():
    grr = RFR.GenericRainflowRecorder()
    assert len(grr.values_from) == 0
    assert len(grr.values_to) == 0
    assert len(grr.index_from) == 0
    assert len(grr.index_to) == 0


@pytest.mark.parametrize('loop_from, loop_to, index_from, index_to', [
    (23., 42., 11, 17),
    (46., 84., 22, 34)
])
def test_generic_rainflow_recorder_record_loop(loop_from, loop_to, index_from, index_to):
    grr = RFR.GenericRainflowRecorder()
    grr.record(index_from, index_to, loop_from, loop_to)
    np.testing.assert_array_equal(grr.values_from, [loop_from])
    np.testing.assert_array_equal(grr.values_to, [loop_to])
    np.testing.assert_array_equal(grr.index_from, [index_from])
    np.testing.assert_array_equal(grr.index_to, [index_to])


def test_generic_rainflow_recorder_record_two_values():
    vf1, vt1, if1, it1, vf2, vt2, if2, it2 = 23., 42., 11, 17, 46., 84., 22, 34
    grr = RFR.GenericRainflowRecorder()
    grr.record(if1, it1, vf1, vt1)
    grr.record(if2, it2, vf2, vt2)

    np.testing.assert_array_equal(grr.values_from, [vf1, vf2])
    np.testing.assert_array_equal(grr.values_to, [vt1, vt2])
    np.testing.assert_array_equal(grr.index_from, [if1, if2])
    np.testing.assert_array_equal(grr.index_to, [it1, it2])


def test_generic_rainflow_recorder_empty_matrix_default():
    grr = RFR.GenericRainflowRecorder()
    matrix, _, _ = grr.matrix()
    np.testing.assert_array_equal(matrix, np.zeros((10, 10)))


def test_generic_rainflow_recorder_empty_matrix_5_bins():
    grr = RFR.GenericRainflowRecorder()
    matrix, _, _ = grr.matrix(bins=5)
    np.testing.assert_array_equal(matrix, np.zeros((5, 5)))


@pytest.mark.parametrize('loop_from, loop_to, index_from, index_to', [
    (23., 42., 11, 17),
    (46., 84., 22, 34)
])
def test_generic_rainflow_recorder_one_non_zero(loop_from, loop_to, index_from, index_to):
    grr = RFR.GenericRainflowRecorder()
    grr.record(index_from, index_to, loop_from, loop_to)

    expected_from = np.linspace(loop_from - 0.5, loop_from + 0.5, 11)
    expected_to = np.linspace(loop_to - 0.5, loop_to + 0.5, 11)
    expected_matrix = np.zeros((10, 10))
    expected_matrix[5, 5] = 1.

    matrix, vfrom, vto = grr.matrix()
    np.testing.assert_array_equal(expected_from, vfrom)
    np.testing.assert_array_equal(expected_to, vto)
    np.testing.assert_array_equal(expected_matrix, matrix)


def test_generic_rainflow_recorder_two_non_zero():
    vf1, vt1, if1, it1, vf2, vt2, if2, it2 = 23., 42., 11, 17, 46., 84., 22, 34
    grr = RFR.GenericRainflowRecorder()
    grr.record(if1, it1, vf1, vt1)
    grr.record(if2, it2, vf2, vt2)
    grr.record(if2, it2, vf2, vt2)

    expected_from = np.linspace(vf1, vf2, 11)
    expected_to = np.linspace(vt1, vt2, 11)
    expected_matrix = np.zeros((10, 10))
    expected_matrix[0, 0] = 1.
    expected_matrix[-1, -1] = 2.

    matrix, vfrom, vto = grr.matrix()
    np.testing.assert_array_equal(expected_from, vfrom)
    np.testing.assert_array_equal(expected_to, vto)
    np.testing.assert_array_equal(expected_matrix, matrix)


def test_generic_rainflow_recorder_empty_matrix_frame_default():
    grr = RFR.GenericRainflowRecorder()
    matrix = grr.matrix_frame()
    assert matrix.index.names[0] == 'from'
    assert matrix.index.names[1] == 'to'
    np.testing.assert_array_equal(matrix.to_numpy(), np.zeros((100, 1)))


def test_generic_rainflow_recorder_empty_matrix_frame_5_bins():
    grr = RFR.GenericRainflowRecorder()
    matrix = grr.matrix_frame(bins=5).to_numpy()
    np.testing.assert_array_equal(matrix, np.zeros((25, 1)))


@pytest.mark.parametrize('loop_from, loop_to, index_from, index_to', [
    (23., 42., 11, 17),
    (46., 84., 22, 34)
])
def test_generic_rainflow_recorder_matrix_frame_one_non_zero(loop_from, loop_to, index_from, index_to):
    grr = RFR.GenericRainflowRecorder()
    grr.record(index_from, index_to, loop_from, loop_to)

    expected_from = pd.IntervalIndex.from_breaks(np.linspace(loop_from - 0.5, loop_from + 0.5, 11))
    expected_to = pd.IntervalIndex.from_breaks(np.linspace(loop_to - 0.5, loop_to + 0.5, 11))
    expected_index = pd.MultiIndex.from_product([expected_from, expected_to], names=['from', 'to'])

    expected_matrix = np.zeros((10, 10))
    expected_matrix[5, 5] = 1.

    matrix = grr.matrix_frame()
    pd.testing.assert_frame_equal(matrix, pd.DataFrame(expected_matrix.flatten(), index=expected_index))

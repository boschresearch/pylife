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
    fr.record_values([value_from], [value_to])
    fr.record_index([index_from], [index_to])
    np.testing.assert_array_equal(fr.values_from, [value_from])
    np.testing.assert_array_equal(fr.values_to, [value_to])
    np.testing.assert_array_equal(fr.index_from, [index_from])
    np.testing.assert_array_equal(fr.index_to, [index_to])


def test_loop_value_rainflow_recorder_record_two_values():
    vf1, vt1, vf2, vt2 = 23., 42., 46., 84.
    lvr = RFR.LoopValueRecorder()

    lvr.record_values([vf1], [vt1])
    lvr.record_values([vf2], [vt2])

    np.testing.assert_array_equal(lvr.values_from, [vf1, vf2])
    np.testing.assert_array_equal(lvr.values_to, [vt1, vt2])


def test_full_rainflow_recorder_record_two_values():
    vf1, vt1, if1, it1, vf2, vt2, if2, it2 = 23., 42., 11, 17, 46., 84., 22, 34
    fr = RFR.FullRecorder()

    fr.record_values([vf1], [vt1])
    fr.record_index([if1], [it1])

    fr.record_values([vf2], [vt2])
    fr.record_index([if2], [it2])

    np.testing.assert_array_equal(fr.values_from, [vf1, vf2])
    np.testing.assert_array_equal(fr.values_to, [vt1, vt2])
    np.testing.assert_array_equal(fr.index_from, [if1, if2])
    np.testing.assert_array_equal(fr.index_to, [it1, it2])


def test_full_rainflow_recorder_empty_collective_default():
    fr = RFR.FullRecorder()
    expected = pd.DataFrame({
        'from': [],
        'to': [],
        'index_from': pd.Series([], dtype=np.uintp),
        'index_to': pd.Series([], dtype=np.uintp)
    })
    pd.testing.assert_frame_equal(fr.collective, expected)


def test_full_rainflow_recorder_two_non_zero_collective():
    vf1, vt1, if1, it1, vf2, vt2, if2, it2 = 23., 42., 11, 17, 46., 84., 22, 34
    fr = RFR.FullRecorder()
    fr.record_values([vf1], [vt1])
    fr.record_index([if1], [it1])
    fr.record_values([vf2], [vt2])
    fr.record_index([if2], [it2])

    expected = pd.DataFrame({
        'from': [vf1, vf2],
        'to': [vt1, vt2],
        'index_from': pd.Series([if1, if2], dtype=np.uintp),
        'index_to': pd.Series([it1, it2], dtype=np.uintp)
    })

    pd.testing.assert_frame_equal(fr.collective, expected)


def test_full_rainflow_recorder_empty_histogram_default():
    fr = RFR.FullRecorder()
    histogram, _, _ = fr.histogram_numpy()
    np.testing.assert_array_equal(histogram, np.zeros((10, 10)))


def test_full_rainflow_recorder_empty_histogram_5_bins():
    fr = RFR.FullRecorder()
    histogram, _, _ = fr.histogram_numpy(bins=5)
    np.testing.assert_array_equal(histogram, np.zeros((5, 5)))


@pytest.mark.parametrize('value_from, value_to, index_from, index_to', [
    (23., 42., 11, 17),
    (46., 84., 22, 34)
])
def test_full_rainflow_recorder_one_non_zero(value_from, value_to, index_from, index_to):
    fr = RFR.FullRecorder()
    fr.record_values([value_from], [value_to])
    fr.record_index([index_from], [index_to])

    expected_from = np.linspace(value_from - 0.5, value_from + 0.5, 11)
    expected_to = np.linspace(value_to - 0.5, value_to + 0.5, 11)
    expected_histogram = np.zeros((10, 10))
    expected_histogram[5, 5] = 1.

    histogram, vfrom, vto = fr.histogram_numpy()
    np.testing.assert_array_equal(expected_from, vfrom)
    np.testing.assert_array_equal(expected_to, vto)
    np.testing.assert_array_equal(expected_histogram, histogram)


def test_full_rainflow_recorder_two_non_zero():
    vf1, vt1, if1, it1, vf2, vt2, if2, it2 = 23., 42., 11, 17, 46., 84., 22, 34
    fr = RFR.FullRecorder()
    fr.record_values([vf1], [vt1])
    fr.record_index([if1], [it1])
    fr.record_values([vf2], [vt2])
    fr.record_index([if2], [it2])
    fr.record_values([vf2], [vt2])
    fr.record_index([if2], [it2])

    expected_from = np.linspace(vf1, vf2, 11)
    expected_to = np.linspace(vt1, vt2, 11)
    expected_histogram = np.zeros((10, 10))
    expected_histogram[0, 0] = 1.
    expected_histogram[-1, -1] = 2.

    histogram, vfrom, vto = fr.histogram_numpy()
    np.testing.assert_array_equal(expected_from, vfrom)
    np.testing.assert_array_equal(expected_to, vto)
    np.testing.assert_array_equal(expected_histogram, histogram)


def test_loopvalue_rainflow_recorder_empty_collective_default():
    lvr = RFR.LoopValueRecorder()
    expected = pd.DataFrame({'from': [], 'to': []})
    pd.testing.assert_frame_equal(lvr.collective, expected)


def test_loop_value_rainflow_recorder_record_two_values_collective():
    vf1, vt1, vf2, vt2 = 23., 42., 46., 84.
    lvr = RFR.LoopValueRecorder()

    lvr.record_values([vf1], [vt1])
    lvr.record_values([vf2], [vt2])

    expected = pd.DataFrame({
        'from': [vf1, vf2],
        'to': [vt1, vt2]
    })

    pd.testing.assert_frame_equal(lvr.collective, expected)


def test_lopvalue_rainflow_recorder_empty_histogram_default():
    fr = RFR.LoopValueRecorder()
    histogram = fr.histogram()
    assert histogram.index.names[0] == 'from'
    assert histogram.index.names[1] == 'to'
    np.testing.assert_array_equal(histogram.to_numpy(), np.zeros(100))


def test_loopvalue_rainflow_recorder_empty_histogram_5_bins():
    fr = RFR.LoopValueRecorder()
    histogram = fr.histogram(bins=5).to_numpy()
    np.testing.assert_array_equal(histogram, np.zeros(25))


@pytest.mark.parametrize('value_from, value_to', [
    (23., 42.),
    (46., 84.)
])
def test_loopvalue_rainflow_recorder_histogram_one_non_zero(value_from, value_to):
    fr = RFR.LoopValueRecorder()
    fr.record_values([value_from], [value_to])

    expected_from = pd.IntervalIndex.from_breaks(np.linspace(value_from - 0.5, value_from + 0.5, 11))
    expected_to = pd.IntervalIndex.from_breaks(np.linspace(value_to - 0.5, value_to + 0.5, 11))
    expected_index = pd.MultiIndex.from_product([expected_from, expected_to], names=['from', 'to'])

    expected_histogram = np.zeros((10, 10))
    expected_histogram[5, 5] = 1.

    histogram = fr.histogram()
    pd.testing.assert_series_equal(histogram, pd.Series(expected_histogram.flatten(), index=expected_index))


# fkm nonlinear recorder
def test_fkm_nonlinear_recorder_record_two_values():
    a1, a2, b1, b2, c1, c2, d1, d2, e1, e2, f1, f2 = 23., 42.,  46., 84.,  2.5, -2.2,  4.8, 2.3,  4.5, -0.2,  1.8, 0.3
    g1, g2 = 1., 2.
    h1, h2 = 4., 5.
    fr = RFR.FKMNonlinearRecorder()

    results_min_1 = pd.DataFrame(
        {"loads_min": [a1], "S_min": [c1], "epsilon_min": [e1], "epsilon_min_LF": [g1]},
    )
    results_max_1 = pd.DataFrame(
        {"loads_max": [b1], "S_max": [d1], "epsilon_max": [f1], "epsilon_max_LF": [h1]},
    )
    results_min_2 = pd.DataFrame(
        {"loads_min": [a2], "S_min": [c2], "epsilon_min": [e2], "epsilon_min_LF": [g2]},
    )
    results_max_2 = pd.DataFrame(
        {"loads_max": [b2], "S_max": [d2], "epsilon_max": [f2], "epsilon_max_LF": [h2]},
    )
    args_1 = [results_min_1, results_max_1] + [[False], [False], 1]
    args_2 = [results_min_2, results_max_2] + [[True], [False], 2]
    args_3 = [results_min_2, results_max_2] + [[True], [True], 2]

    fr.record_values_fkm_nonlinear(*args_1)
    fr.record_values_fkm_nonlinear(*args_2)
    fr.record_values_fkm_nonlinear(*args_3)

    np.testing.assert_array_equal(fr.loads_min, [a1, a2, a2])
    np.testing.assert_array_equal(fr.loads_max, [b1, b2, b2])
    np.testing.assert_array_equal(fr.S_min, [c1, c2, c2])
    np.testing.assert_array_equal(fr.S_max, [d1, d2, d2])
    np.testing.assert_array_equal(fr.R, [c1/d1, c2/d2, -1])
    np.testing.assert_array_equal(fr.epsilon_min, [e1, e2, e2])
    np.testing.assert_array_equal(fr.epsilon_max, [f1, f2, f2])
    np.testing.assert_array_equal(fr.S_a, [0.5*(d1-c1), 0.5*(d2-c2), 0.5*(d2-c2)])
    np.testing.assert_array_equal(fr.S_m, [0.5*(d1+c1), 0.5*(d2+c2), 0])
    np.testing.assert_array_equal(fr.epsilon_a, [0.5*(f1-e1), 0.5*(f2-e2), 0.5*(f2-e2)])
    np.testing.assert_array_equal(fr.epsilon_m, [0.5*(f1+e1), 0.5*(f2+e2), 0])
    np.testing.assert_array_equal(fr.collective["epsilon_min_LF"], [g1, g2, g2])
    np.testing.assert_array_equal(fr.collective["epsilon_max_LF"], [h1, h2, h2])
    np.testing.assert_array_equal(fr.is_closed_hysteresis, [False, True, True])
    np.testing.assert_array_equal(fr.collective["is_zero_mean_stress_and_strain"], [False, False, True])
    np.testing.assert_array_equal(fr.collective["run_index"], [1, 2, 2])


def test_fkm_nonlinear_recorder_empty_collective_default():
    fr = RFR.FKMNonlinearRecorder()

    # create appropriate empty MultiIndex
    index = pd.MultiIndex.from_product([[],[]], names=["hysteresis_index","assessment_point_index"])
    index = index.set_levels([index.levels[1].astype(np.int64)], level=[1])

    expected = pd.DataFrame(
        index=index,
        data={
            'loads_min': [],
            'loads_max': [],
            'S_min': [],
            'S_max': [],
            'R': [],
            'epsilon_min': [],
            'epsilon_max': [],
            "S_a": [],
            "S_m": [],
            "epsilon_a": [],
            "epsilon_m": [],
            "epsilon_min_LF": [],
            "epsilon_max_LF": [],
            "is_closed_hysteresis": [],
            "is_zero_mean_stress_and_strain": [],
            "run_index": np.array([], dtype=np.int64),
        }
    )

    pd.testing.assert_frame_equal(fr.collective, expected)


def test_fkm_nonlinear_recorder_two_non_zero_collective():
    a1, a2, b1, b2, c1, c2, d1, d2, e1, e2, f1, f2 = 23., 42.,  46., 84.,  2.5, -2.2,  4.8, 2.3,  0.5, -0.2,  1.8, 0.3
    g1, g2 = 1., 2.
    h1, h2 = 4., 5.
    fr = RFR.FKMNonlinearRecorder()

    results_min_1 = pd.DataFrame(
        {"loads_min": [a1], "S_min": [c1], "epsilon_min": [e1], "epsilon_min_LF": [g1]},
    )
    results_max_1 = pd.DataFrame(
        {"loads_max": [b1], "S_max": [d1], "epsilon_max": [f1], "epsilon_max_LF": [h1]},
    )
    results_min_2 = pd.DataFrame(
        {"loads_min": [a2], "S_min": [c2], "epsilon_min": [e2], "epsilon_min_LF": [g2]},
    )
    results_max_2 = pd.DataFrame(
        {"loads_max": [b2], "S_max": [d2], "epsilon_max": [f2], "epsilon_max_LF": [h2]},
    )
    args_1 = [results_min_1, results_max_1] + [[False], [False], 1]
    args_2 = [results_min_2, results_max_2] + [[True], [False], 2]
    args_3 = [results_min_2, results_max_2] + [[True], [True], 2]

    fr.record_values_fkm_nonlinear(*args_1)
    fr.record_values_fkm_nonlinear(*args_2)
    fr.record_values_fkm_nonlinear(*args_3)

    expected = pd.DataFrame(
        index=pd.MultiIndex.from_product([[0,1,2],[0]],names=["hysteresis_index","assessment_point_index"]),
        data={
            'loads_min': [a1, a2, a2],
            'loads_max': [b1, b2, b2],
            'S_min': [c1, c2, c2],
            'S_max': [d1, d2, d2],
            'R': [c1/d1, c2/d2, -1],
            'epsilon_min': [e1, e2, e2],
            'epsilon_max': [f1, f2, f2],
            "S_a": [0.5*(d1-c1), 0.5*(d2-c2), 0.5*(d2-c2)],
            "S_m": [0.5*(d1+c1), 0.5*(d2+c2), 0],
            "epsilon_a": [0.5*(f1-e1), 0.5*(f2-e2), 0.5*(f2-e2)],
            "epsilon_m": [0.5*(f1+e1), 0.5*(f2+e2), 0],
            "epsilon_min_LF": [g1, g2, g2],
            "epsilon_max_LF": [h1, h2, h2],
            "is_closed_hysteresis": [False, True, True],
            "is_zero_mean_stress_and_strain": [False, False, True],
            "run_index": [1, 2, 2],
        }
    )

    pd.testing.assert_frame_equal(fr.collective, expected)

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

import numpy as np
import pandas as pd

import pylife.stress.rainflow as RF
from pylife.stress.timesignal import TimeSignalGenerator


def make_empty_rainflow_matrix(f1, f2, t1, t2, num):
    f_idx = pd.IntervalIndex.from_breaks(np.linspace(f1, f2, num+1))
    t_idx = pd.IntervalIndex.from_breaks(np.linspace(t1, t2, num+1))
    m_idx = pd.MultiIndex.from_product([f_idx, t_idx], names=['from', 'to'])

    return pd.DataFrame(data=np.zeros(num*num), index=m_idx)


def test_rainflow_simple_sine():
    signal = np.array([0., 1., -1., 1., -1., 0])
    rfc = RF.RainflowCounterThreePoint()
    res = rfc.process(signal).get_rainflow_matrix_frame((np.linspace(-1.5, 1.5, 5), np.linspace(-0.25, 1.25, 5)))

    expected = make_empty_rainflow_matrix(-1.5, 1.5, -0.25, 1.25, 4)
    expected.loc[(-1, 1)] = 1

    pd.testing.assert_frame_equal(res, expected)


def test_rainflow_two_amplitudes():
    signal = np.array([0, 1., 0., 1., -1., 1., 0., 1., -1., 1., 0])
    rfc = RF.RainflowCounterThreePoint()
    res = rfc.process(signal).get_rainflow_matrix_frame((np.linspace(-1.25, 1.25, 5), np.linspace(-0.25, 1.25, 5)))

    expected = make_empty_rainflow_matrix(-1.25, 1.25, -0.25, 1.25, 4)
    expected.loc[(-1, 1)] = 1
    expected.loc[(1, 0)] = 2

    pd.testing.assert_frame_equal(res, expected)


def test_rainflow_hits():
    r'''
                                                                                                 1   2   3   4   5   6
    --------------------------------------------------------------------------------------------------------------------
    6                x                                                                         |   |   |   |   |   |   |
    ----------------/-\-------------------------------------------------------------------------------------------------
    5              /   \                   x                                   x               | 1 |   |   |   |   |   |
    --------------/-----\-----------------/-\---------------------------------/-\---------------------------------------
    4            /       \           x   /   \           x   x           x   /   \             | 1 |   |   | 3 |   |   |
    ------------/---------\---------/-\-/-----\---------/-\-/-\---------/-\-/-----\-/-----------------------------------
    3                      \       /   x       \       /   x   \       /   x       x           |   |   |   |   |   |   |
    ------------------------\-----/-------------\-----/---------\-----/-------------------------------------------------
    2                        \   /               \   /           \   /                         |   |   |   |   |   |   |
    --------------------------\-/-----------------\-/-------------\-/---------------------------------------------------
    1                          x                   x               x                           |   |   |   |   |   |   |
    --------------------------------------------------------------------------------------------------------------------
    '''
    signal = np.array([3, 6, 1, 4, 3, 5, 1, 4, 3, 4, 1, 4, 3, 5, 3, 4]) - 0.5
    rfc = RF.RainflowCounterThreePoint()
    res = rfc.process(signal).get_rainflow_matrix_frame((np.linspace(0, 6, 7), np.linspace(0, 6, 7)))

    expected = make_empty_rainflow_matrix(0, 6, 0, 6, 6)
    expected.loc[(1, 5)] = 1
    expected.loc[(4, 3)] = 3
    expected.loc[(1, 4)] = 1

    pd.testing.assert_frame_equal(res, expected)


def test_rainflow_haibach_example():
    r'''
    Example from fig 3.3-30 of E. Haibach "Betriebsfestigkeit"

                                                                                               1   2   3   4   5   6
    ------------------------------------------------------------------------------------------------------------------
    6                6                   6                                                   | 1 |   |   |   |   |   |
    ----------------/-\-----------------/-\---------------------------------------------------------------------------
    5      5       /   \                | |                                    5             |   |   |   |   |   |   |
    ------/-\-----/-----\--------------/---\----------------------------------/-\-------------------------------------
    4    /   \   /       \             |   |         4               4       /   \   4       | 1 |   | 1 |   |   |   |
    ----/-----\-/---------\-----------/-----\-------/-\-------------/-\-----/-----\-/-\-------------------------------
    3  /       3           \   3      |     |      /   \   3       /   \   /       3   \     |   | 2 |   |   | 1 |   |
    --/---------------------\-/-\----/-------\----/-----\-/-\-----/-----\-/-------------\-----------------------------
    2                        2   \   |       |   /       2   \   /       2               2   |   |   |   | 1 |   |   |
    ------------------------------\-/---------\-/-------------\-/-----------------------------------------------------
    1                              1           1               1                             |   |   |   |   |   |   |
    ------------------------------------------------------------------------------------------------------------------
    '''
    signal = np.array([2, 5, 3, 6, 2, 3, 1, 6, 1, 4, 2, 3, 1, 4, 2, 5, 3, 4, 2]) - 0.5
    rfc = RF.RainflowCounterThreePoint()
    res = rfc.process(signal).get_rainflow_matrix_frame((np.linspace(0, 6, 7), np.linspace(0, 6, 7)))

    expected = make_empty_rainflow_matrix(0, 6, 0, 6, 6)
    expected.loc[(1, 6)] = 1
    expected.loc[(1, 4)] = 1
    expected.loc[(2, 3)] = 2
    expected.loc[(3, 4)] = 1
    expected.loc[(4, 2)] = 1
    expected.loc[(5, 3)] = 1

    expected_residuals = np.array([2, 6, 1, 5, 2])

    pd.testing.assert_frame_equal(res, expected)
    np.testing.assert_array_equal(np.ceil(rfc.residuals()).astype(int), expected_residuals)


def test_rainflow_lecture_example():
    r'''
                                                                         1   2   3   4   5   6
    -------------------/\------------------------------------------------------------------------
    6                 /  \                           x                 |   | 1 |   |   |   |   |
    -----------------/----\-------------------------/-\------------------------------------------
    5               /      \               x       /   \               |   |   |   |   |   |   |
    ---------------/--------\-------------/-\-----/-----\----------------------------------------
    4             /          \   x       /   \   /       \             |   |   | 1 |   |   |   |
    -------------/------------\-/-\-----/-----\-/---------\--------------------------------------
    3           /              x   \   /       x           \           |   |   |   |   | 1 |   |
    -----------/--------------------\-/---------------------\------------------------------------
    2         /                      x                       \         |   |   |   |   |   |   |
    ---------/------------------------------------------------\-/--------------------------------
    1                                                          x       |   |   |   |   |   |   |
    ---------------------------------------------------------------------------------------------
    '''
    signal = np.array([1, 7, 4, 3, 4, 2, 5, 3, 6, 1, 2]) - 0.5
    rfc = RF.RainflowCounterThreePoint()
    res = rfc.process(signal).get_rainflow_matrix_frame((np.linspace(0., 6., 7), np.linspace(0., 6., 7)))

    expected = make_empty_rainflow_matrix(0, 6, 0, 6, 6)
    expected.loc[(2, 6)] = 1
    expected.loc[(3, 4)] = 1
    expected.loc[(5, 3)] = 1
    expected_residuals = np.array([1, 7, 1, 2])

    pd.testing.assert_frame_equal(res, expected)
    np.testing.assert_array_equal(np.ceil(rfc.residuals()).astype(int), expected_residuals)



def test_rainflow_lower_after_main():
    r'''
                                                 1   2   3   4   5   6
    ---------------------------------------------------------------------
    6        x                                 |   |   |   |   |   |   |
    --------/-\----------------------------------------------------------
    5      /   \   x           x               |   |   |   | 1 |   |   |
    ------/-----\-/-\---------/-\----------------------------------------
    4    /       x   \       /   \             |   |   |   |   |   |   |
    --\-/-------------\-----/-----\--------------------------------------
    3  x               \   /       \           |   |   |   |   |   |   |
    --------------------\-/----------------------------------------------
    2                    x                     |   |   |   |   |   |   |
    ---------------------------------------------------------------------
    1                                          |   |   |   |   |   |   |
    ---------------------------------------------------------------------
    '''
    signal = np.array([4, 3, 6, 4, 5, 2, 5, 3]) - 0.5
    rfc = RF.RainflowCounterThreePoint()
    res = rfc.process(signal).get_rainflow_matrix_frame((np.linspace(0., 6., 7), np.linspace(0., 6., 7)))

    expected = make_empty_rainflow_matrix(0, 6, 0, 6, 6)
    expected.loc[(4, 5)] = 1

    expected_residuals = np.array([4, 3, 6, 2, 5, 3])

    pd.testing.assert_frame_equal(res, expected)
    np.testing.assert_array_equal(np.ceil(rfc.residuals()).astype(int), expected_residuals)


def test_rainflow_lower_after_main_one_more_close():
    r'''
                                                 1   2   3   4   5   6
    --------------------------------|------------------------------------
    6        x                      |          |   |   |   |   |   |   |
    --------/-\---------------------|------------------------------------
    5      /   \   x           x    |          |   | 1 |   | 1 |   |   |
    ------/-----\-/-\---------/-\---|------------------------------------
    4    /       x   \       /   \  |          |   |   |   |   |   |   |
    --\-/-------------\-----/-----\-|------------------------------------
    3  x               \   /       \|          |   |   |   |   |   |   |
    --------------------\-/---------|------------------------------------
    2                    x          |\         |   |   |   |   |   |   |
    --------------------------------|-\-/--------------------------------
    1                               |  x       |   |   |   |   |   |   |
    --------------------------------|------------------------------------
    '''
    signal = np.array([4, 3, 6, 4, 5, 2, 5, 3]) - 0.5
    signal2 = np.array([1, 2]) - 0.5
    rfc = RF.RainflowCounterThreePoint()
    res = rfc.process(signal).process(signal2).get_rainflow_matrix_frame((np.linspace(0., 6., 7), np.linspace(0., 6., 7)))

    expected = make_empty_rainflow_matrix(0, 6, 0, 6, 6)
    expected.loc[(4, 5)] = 1
    expected.loc[(2, 5)] = 1

    expected_residuals = np.array([4, 3, 6, 1, 2])

    pd.testing.assert_frame_equal(res, expected)
    np.testing.assert_array_equal(np.ceil(rfc.residuals()).astype(int), expected_residuals)


def test_rainflow_dampening():
    r'''
                                                                                    1   2   3   4   5   6
    -------------------------------------------------------------------------------------------------------
    6           6                                                                 |   |   |   |   |   |   |
    -----------/-\-----------------------------------------------------------------------------------------
    5         /   \           5                                                   |   |   |   |   |   |   |
    ---------/-----\---------/-\---------------------------------------------------------------------------
    4       /       \       /   \   4                                             |   |   |   |   |   |   |
    -------/---------\-----/-----\-/-----------------------------------------------------------------------
    3     /           \   /       3                                               |   |   |   |   |   |   |
    -----/-------------\-/---------------------------------------------------------------------------------
    2   /               2                                                         |   |   |   |   |   |   |
    ---/---------------------------------------------------------------------------------------------------
    1 1                                                                           |   |   |   |   |   |   |
    -------------------------------------------------------------------------------------------------------
    '''
    signal = np.array([1, 6, 2, 5, 3, 4]) - 0.5
    rfc = RF.RainflowCounterThreePoint()
    res = rfc.process(signal).get_rainflow_matrix_frame((np.linspace(0., 6., 7), np.linspace(0., 6., 7)))

    expected = make_empty_rainflow_matrix(0, 6, 0, 6, 6)
    expected_residuals = np.array([1, 6, 2, 5, 3, 4])

    pd.testing.assert_frame_equal(res, expected)
    np.testing.assert_array_equal(np.ceil(rfc.residuals()).astype(int), expected_residuals)


def test_rainflow_dampening_closed():
    r'''
                                                                                    1   2   3   4   5   6
    -------------------------------------------------------------------------------------------------------
    6           6                                                                 |   |   |   |   |   |   |
    -----------/-\-----------------------------------------------------------------------------------------
    5         /   \           5                                                   |   |   |   |   |   |   |
    ---------/-----\---------/-\---------------------------------------------------------------------------
    4       /       \       /   \   4                                             |   |   | 1 |   |   |   |
    -------/---------\-----/-----\-/-\---------------------------------------------------------------------
    3     /           \   /       3   \                                           |   | 1 |   |   |   |   |
    -----/-------------\-/-------------\-------------------------------------------------------------------
    2   /               2               \                                         |   |   |   |   |   |   |
    ---/---------------------------------\-----------------------------------------------------------------
    1 1                                   1                                       |   |   |   |   |   |   |
    -------------------------------------------------------------------------------------------------------
    '''
    signal = np.array([1, 6, 2, 5, 3, 4, 1]) - 0.5
    rfc = RF.RainflowCounterThreePoint()
    res = rfc.process(signal).get_rainflow_matrix_frame((np.linspace(0., 6., 7), np.linspace(0., 6., 7)))

    expected = make_empty_rainflow_matrix(0, 6, 0, 6, 6)
    expected.loc[(3, 4)] = 1
    expected.loc[(2, 5)] = 1

    expected_residuals = np.array([1, 6, 1])

    pd.testing.assert_frame_equal(res, expected)
    np.testing.assert_array_equal(np.ceil(rfc.residuals()).astype(int), expected_residuals)


def test_rainflow_partial_signals_general_FKM():
    tsgen = TimeSignalGenerator(10, {'number': 50,
                                     'amplitude_median': 1.0, 'amplitude_std_dev': 0.5,
                                     'frequency_median': 4, 'frequency_std_dev': 3,
                                     'offset_median': 0, 'offset_std_dev': 0.4},
                                None, None)

    signal_tot = tsgen.query(10000)

    rfc_tot = RF.RainflowCounterFKM().process(signal_tot)
    rfc_partial = RF.RainflowCounterFKM().process(signal_tot[:3424]).process(signal_tot[3424:])

    np.testing.assert_array_almost_equal(rfc_tot.loops_from, rfc_partial.loops_from)
    np.testing.assert_array_almost_equal(rfc_tot.loops_to, rfc_partial.loops_to)


def test_rainflow_partial_signals_general_three_point():
    tsgen = TimeSignalGenerator(10, {'number': 50,
                                     'amplitude_median': 1.0, 'amplitude_std_dev': 0.5,
                                     'frequency_median': 4, 'frequency_std_dev': 3,
                                     'offset_median': 0, 'offset_std_dev': 0.4},
                                None, None)

    signal_tot = tsgen.query(10000)

    rfc_tot = RF.RainflowCounterThreePoint().process(signal_tot)
    rfc_partial = RF.RainflowCounterThreePoint().process(signal_tot[:3424]).process(signal_tot[3424:])

    np.testing.assert_array_almost_equal(rfc_tot.loops_from, rfc_partial.loops_from)
    np.testing.assert_array_almost_equal(rfc_tot.loops_to, rfc_partial.loops_to)


def test_rainflow_partial_signals_splitturn_FKM():
    tsgen = TimeSignalGenerator(10, {'number': 50,
                                     'amplitude_median': 1.0, 'amplitude_std_dev': 0.5,
                                     'frequency_median': 4, 'frequency_std_dev': 3,
                                     'offset_median': 0, 'offset_std_dev': 0.4},
                                None, None)

    signal_tot = tsgen.query(10000)
    rfc_tot = RF.RainflowCounterFKM().process(signal_tot)
    turn_points, _ = RF.find_turns(signal_tot)
    turn_points = np.insert(turn_points, 0, 0)
    turn_num = turn_points.shape[0]
    split_points = [int(np.ceil(turn_num*x)) for x in [0.0, 0.137, 0.23, 0.42, 1.0 ]]
    rfc_partial = RF.RainflowCounterFKM()
    for i in range(len(split_points)-1):
        lower = turn_points[split_points[i]]
        upper = 10000 if split_points[i+1] == turn_points.shape[0] else turn_points[split_points[i+1]]
        rfc_partial.process(signal_tot[lower:upper])

    np.testing.assert_array_almost_equal(rfc_tot.loops_from, rfc_partial.loops_from)
    np.testing.assert_array_almost_equal(rfc_tot.loops_to, rfc_partial.loops_to)


def test_rainflow_partial_signals_splitturn():
    tsgen = TimeSignalGenerator(10, {'number': 50,
                                     'amplitude_median': 1.0, 'amplitude_std_dev': 0.5,
                                     'frequency_median': 4, 'frequency_std_dev': 3,
                                     'offset_median': 0, 'offset_std_dev': 0.4},
                                None, None)

    signal_tot = tsgen.query(10000)

    rfc_tot = RF.RainflowCounterThreePoint().process(signal_tot)
    turn_points, _ = RF.find_turns(signal_tot)
    turn_points = np.insert(turn_points, 0, 0)
    turn_num = turn_points.shape[0]
    split_points = [int(np.ceil(turn_num*x)) for x in [0.0, 0.137, 0.23, 0.42, 1.0]]
    rfc_partial = RF.RainflowCounterThreePoint()
    for i in range(len(split_points)-1):
        lower = turn_points[split_points[i]]
        upper = 10000 if split_points[i+1] == turn_points.shape[0] else turn_points[split_points[i+1]]
        _, tot_turns = RF.find_turns(signal_tot[:upper])
        rfc_partial.process(signal_tot[lower:upper])

    np.testing.assert_array_almost_equal(rfc_tot.loops_from, rfc_partial.loops_from)
    np.testing.assert_array_almost_equal(rfc_tot.loops_to, rfc_partial.loops_to)


def test_rainflow_FKM_memory1_inner():
    signal = np.array([0., 100., 0., 80., 20., 60., 40., 100., 0., 80., 20., 60., 40., 45.])
    rfc = RF.RainflowCounterFKM().process(signal)

    np.testing.assert_array_equal(rfc.loops_from, np.array([60., 80., 100.]))
    np.testing.assert_array_equal(rfc.loops_to, np.array([40., 20., 0.]))
    np.testing.assert_array_equal(rfc.residuals(), np.array([100., 0., 80., 20., 60., 40.]))


def test_rainflow_FKM_memory1_2_3():
    signal = np.array([0.,
                       1., -1., 1., -2., -1., -2., 2., 0., 2., -2.,
                       1., -1., 1., -2., -1., -2., 2., 0., 2., -2.,
                       -1.8])
    rfc = RF.RainflowCounterFKM().process(signal)

    np.testing.assert_array_equal(rfc.loops_from, np.array([1., -2., 2., -2.,  1., -2., -2., 2., -2.]))
    np.testing.assert_array_equal(rfc.loops_to,   np.array([-1., -1., 0.,  2., -1.,  1., -1., 0.,  2.]))
    np.testing.assert_array_equal(rfc.residuals(), np.array([1., -2.]))

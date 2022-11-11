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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import unittest
import pytest
import numpy as np

import pylife.stress.rainflow as RF
import pylife.stress.rainflow.recorders as RFR


def process_signal(signal):
    fr = RFR.FullRecorder()
    dtor = RF.ThreePointDetector(recorder=fr).process(signal)

    return fr, dtor


def test_three_point_detector_new_no_residuals():
    fr = RFR.FullRecorder()
    dtor = RF.ThreePointDetector(recorder=fr)
    assert dtor.recorder is fr
    assert len(dtor.residuals) == 0


class TestThreePointsSimpleSine(unittest.TestCase):
    def setUp(self):
        signal = np.array([0., 1., -1., 1., -1., 0])
        self._fr, self._dtor = process_signal(signal)

    def test_values(self):
        np.testing.assert_array_equal(self._fr.values_from, np.array([-1.]))
        np.testing.assert_array_equal(self._fr.values_to, np.array([1.]))

    def test_indeces(self):
        np.testing.assert_array_equal(self._fr.index_from, np.array([2]))
        np.testing.assert_array_equal(self._fr.index_to, np.array([3]))

    def test_residuals(self):
        np.testing.assert_array_equal(self._dtor.residuals, np.array([0., 1., -1., 0.]))


class TestThreePointsSineIntermediateVals(unittest.TestCase):
    def setUp(self):
        signal = np.array([0., 1., 0., -1., 0., 1., 0., -1., 0.])
        self._fr, self._dtor = process_signal(signal)

    def test_values(self):
        np.testing.assert_array_equal(self._fr.values_from, np.array([-1.]))
        np.testing.assert_array_equal(self._fr.values_to, np.array([1.]))

    def test_indeces(self):
        np.testing.assert_array_equal(self._fr.index_from, np.array([3]))
        np.testing.assert_array_equal(self._fr.index_to, np.array([5]))

    def test_residuals(self):
        np.testing.assert_array_equal(self._dtor.residuals, np.array([0., 1., -1., 0.]))


class TestThreePointsTwoAmplitudes(unittest.TestCase):
    def setUp(self):
        signal = np.array([0., 1., 0., 1., -1., 1., 0., 1., -1., 1., 0])
        self._fr, self._dtor = process_signal(signal)

    def test_values(self):
        np.testing.assert_array_equal(self._fr.values_from, np.array([1., 1., -1.]))
        np.testing.assert_array_equal(self._fr.values_to, np.array([0., 0., 1.]))

    def test_indeces(self):
        np.testing.assert_array_equal(self._fr.index_from, np.array([1, 5, 4]))
        np.testing.assert_array_equal(self._fr.index_to, np.array([2, 6, 7]))

    def test_residuals(self):
        np.testing.assert_array_equal(self._dtor.residuals, np.array([0., 1., -1., 1., 0.]))


class TestThreePointsTwoAmplitudesSplit(unittest.TestCase):
    def setUp(self):
        signal = np.array([0., 1., 0., 1., -1., 1., 0., 1., -1., 1., 0])
        self._fr = RFR.FullRecorder()
        self._dtor = RF.ThreePointDetector(recorder=self._fr).process(signal[:4]).process(signal[4:])

    def test_values(self):
        np.testing.assert_array_equal(self._fr.values_from, np.array([1., 1., -1.]))
        np.testing.assert_array_equal(self._fr.values_to, np.array([0., 0., 1.]))

    def test_indeces(self):
        np.testing.assert_array_equal(self._fr.index_from, np.array([1, 5, 4]))
        np.testing.assert_array_equal(self._fr.index_to, np.array([2, 6, 7]))

    def test_chunks(self):
        np.testing.assert_array_equal(self._fr.chunks, [4, 7])

    def test_residuals(self):
        np.testing.assert_array_equal(self._dtor.residuals, np.array([0., 1., -1., 1., 0.]))


class TestThreePointHits(unittest.TestCase):
    r'''
                                                                                                 1   2   3   4   5   6
    --------------------------------------------------------------------------------------------------------------------
    6                R                                                                         |   |   |   |   |   |   |
    ----------------/-\-------------------------------------------------------------------------------------------------
    5              /   \                   2                                   R               | 1 |   |   |   |   |   |
    --------------/-----\-----------------/-\---------------------------------/-\---------------------------------------
    4            /       \           1   /   \           3---4           5---/   \   R         | 1 |   |   | 3 |   |   |
    ------------/---------\---------/-\-/-----\---------/-\-/-\---------/-\-/-----\-/-----------------------------------
    3          R           \       /---1       \       /   3   \       /   5       R           |   |   |   |   |   |   |
    ------------------------\-----/-------------\-----/---------\-----/-------------------------------------------------
    2                        \   /               \   /           \   /                         |   |   |   |   |   |   |
    --------------------------\-/-----------------\-/-------------\-/---------------------------------------------------
    1                          2-------------------4---------------R                           |   |   |   |   |   |   |
    --------------------------------------------------------------------------------------------------------------------
               0     1         2     3 4   5       6     7 8 9    10    11 12 13  14 15
    '''
    def setUp(self):
        signal = np.array([3., 6., 1., 4., 3., 5., 1., 4., 3., 4., 1., 4., 3., 5., 3., 4.])
        self._fr, self._dtor = process_signal(signal)

    def test_values(self):
        np.testing.assert_array_equal(self._fr.values_from, np.array([4., 1., 4., 1., 4.]))
        np.testing.assert_array_equal(self._fr.values_to, np.array([3., 5., 3., 4., 3.]))

    def test_indeces(self):
        np.testing.assert_array_equal(self._fr.index_from, np.array([3, 2, 7, 6, 11]))
        np.testing.assert_array_equal(self._fr.index_to, np.array([4, 5, 8, 9, 12]))

    def test_residuals(self):
        np.testing.assert_array_equal(self._dtor.residuals, np.array([3., 6., 1., 5., 3., 4.]))

    def test_residual_index(self):
        np.testing.assert_array_equal(self._dtor.residual_index, np.array([0, 1, 10, 13, 14, 15]))


class TestThreePointHaibach(unittest.TestCase):
    r'''
    Example from fig 3.3-30 of E. Haibach "Betriebsfestigkeit"

                                                                                               1   2   3   4   5   6
    ------------------------------------------------------------------------------------------------------------------
    6                R                   3                                                   | 1 |   |   |   |   |   |
    ----------------/-\-----------------/-\---------------------------------------------------------------------------
    5      1-------/   \                | |                                    R             |   |   |   |   |   |   |
    ------/-\-----/-----\--------------/---\----------------------------------/-\-------------------------------------
    4    /   \   /       \             |   |         5               6-------/   \---7       | 1 |   | 1 |   |   |   |
    ----/-----\-/---------\-----------/-----\-------/-\-------------/-\-----/-----\-/-\-------------------------------
    3  /       1           \   2      |     |      /   \---4       /   \   /       7   \     |   | 2 |   |   | 1 |   |
    --/---------------------\-/-\----/-------\----/-----\-/-\-----/-----\-/-------------\-----------------------------
    2 R                      2---\   |       |   /       4   \   /       6               R   |   |   |   | 1 |   |   |
    ------------------------------\-/---------\-/-------------\-/-----------------------------------------------------
    1                              3-----------5---------------R                             |   |   |   |   |   |   |
    --0----1---2-----3-------4-5---6-----7-----8-----9--10-11-12----13--14----15--16-17-18----------------------------
    '''
    def setUp(self):
        signal = np.array([2., 5., 3., 6., 2., 3., 1., 6., 1., 4., 2., 3., 1., 4., 2., 5., 3., 4., 2.])
        self._fr, self._dtor = process_signal(signal)

    def test_values(self):
        np.testing.assert_array_equal(self._fr.values_from, np.array([5., 2., 1., 2., 1., 4., 3.]))
        np.testing.assert_array_equal(self._fr.values_to, np.array([3., 3., 6., 3., 4., 2., 4.]))

    def test_indeces(self):
        np.testing.assert_array_equal(self._fr.index_from, np.array([1, 4, 6, 10, 8, 13, 16]))
        np.testing.assert_array_equal(self._fr.index_to, np.array([2, 5, 7, 11, 9, 14, 17]))

    def test_residuals(self):
        np.testing.assert_array_equal(self._dtor.residuals, np.array([2., 6., 1., 5., 2.]))

    def test_residual_index(self):
        np.testing.assert_array_equal(self._dtor.residual_index, np.array([0, 3, 12, 15, 18]))


@pytest.mark.skip(reason="Not yet implemented")
class TestThreePointHaibachLastSampleDuplicate(unittest.TestCase):
    def setUp(self):
        signal = np.array([2., 5., 3., 6., 2., 3., 1., 6., 1., 4., 2., 3., 1., 4., 2., 5., 3., 4., 2., 2., 2., 2.])
        self._fr, self._dtor = process_signal(signal)

    def test_residual_index(self):
        np.testing.assert_array_equal(self._dtor.residual_index, np.array([0, 3, 12, 15, 18]))


class TestThreePointLecture(unittest.TestCase):
    r'''
                       R                                                 1   2   3   4   5   6
    -------------------/\------------------------------------------------------------------------
    6                 /  \                           3                 |   | 1 |   |   |   |   |
    -----------------/----\-------------------------/-\------------------------------------------
    5               /      \               2-------/   \               |   |   |   |   |   |   |
    ---------------/--------\-------------/-\-----/-----\----------------------------------------
    4             /          x   1       /   \   /       \             |   |   | 1 |   |   |   |
    -------------/------------\-/-\-----/-----\-/---------\--------------------------------------
    3           /              1---\   /       2           \           |   |   |   |   | 1 |   |
    -----------/--------------------\-/---------------------\------------------------------------
    2         /                      3-----------------------\   R     |   |   |   |   |   |   |
    ---------/------------------------------------------------\-/--------------------------------
    1       R                                                  R       |   |   |   |   |   |   |
    ---------------------------------------------------------------------------------------------
            0          1     2 3 4   5     6   7     8         9 10
    '''
    def setUp(self):
        signal = np.array([1., 7., 4., 3., 4., 2., 5., 3., 6., 1., 2.])
        self._fr, self._dtor = process_signal(signal)

    def test_values(self):
        np.testing.assert_array_equal(self._fr.values_from, np.array([3., 5., 2.]))
        np.testing.assert_array_equal(self._fr.values_to, np.array([4., 3., 6.]))

    def test_indeces(self):
        np.testing.assert_array_equal(self._fr.index_from, np.array([3, 6, 5]))
        np.testing.assert_array_equal(self._fr.index_to, np.array([4, 7, 8]))

    def test_residuals(self):
        np.testing.assert_array_equal(self._dtor.residuals, np.array([1., 7., 1., 2.]))

    def test_residual_index(self):
        np.testing.assert_array_equal(self._dtor.residual_index, np.array([0, 1, 9, 10]))

class TestThreePointLowerAfterMain(unittest.TestCase):
    r'''
                                                 1   2   3   4   5   6
    ---------------------------------------------------------------------
    6          R                               |   |   |   |   |   |   |
    ----------/-\--------------------------------------------------------
    5        /   \   1           R             |   |   |   | 1 |   |   |
    --------/-----\-/-\---------/-\--------------------------------------
    4  R   /       1---\       /   \           |   |   |   |   |   |   |
    ----\-/-------------\-----/-----\------------------------------------
    3    R               \   /       R         |   |   |   |   |   |   |
    ----------------------\-/--------------------------------------------
    2                      R                   |   |   |   |   |   |   |
    ---------------------------------------------------------------------
    1                                          |   |   |   |   |   |   |
    ---------------------------------------------------------------------
       0 1    2    3 4     5    6    7
    '''
    def setUp(self):
        signal = np.array([4., 3., 6., 4., 5., 2., 5., 3.])
        self._fr, self._dtor = process_signal(signal)

    def test_values(self):
        np.testing.assert_array_equal(self._fr.values_from, np.array([4.]))
        np.testing.assert_array_equal(self._fr.values_to, np.array([5.]))

    def test_indeces(self):
        np.testing.assert_array_equal(self._fr.index_from, np.array([3]))
        np.testing.assert_array_equal(self._fr.index_to, np.array([4]))

    def test_residuals(self):
        np.testing.assert_array_equal(self._dtor.residuals, np.array([4., 3., 6., 2., 5., 3.]))

    def test_residual_index(self):
        np.testing.assert_array_equal(self._dtor.residual_index, np.array([0, 1, 2, 5, 6, 7]))


class TestThreePointLowerAfterMainClose(unittest.TestCase):
    r'''
                                      |          1   2   3   4   5   6
    ----------------------------------|----------------------------------
    6          R                      |        |   |   |   |   |   |   |
    ----------/-\---------------------|----------------------------------
    5        /   \   1           2    |        |   |   |   | 1 |   |   |
    --------/-----\-/-\---------/-\---|----------------------------------
    4  R   /       1---\       /   \  |        |   |   |   |   |   |   |
    ----\-/-------------\-----/-----\-|----------------------------------
    3    R               \   /       x|        |   |   |   |   |   |   |
    ----------------------\-/---------\----------------------------------
    2                      2----------|\   R   |   |   |   |   |   |   |
    ----------------------------------|-\-/------------------------------
    1                                 |  R     |   |   |   |   |   |   |
    ----------------------------------|----------------------------------
       0 1    2    3 4     5    6    7|  8 9
    '''
    def setUp(self):
        signal = np.array([4., 3., 6., 4., 5., 2., 5., 3., 1., 2.])
        self._fr, self._dtor = process_signal(signal)

    def test_values(self):
        np.testing.assert_array_equal(self._fr.values_from, np.array([4., 2.]))
        np.testing.assert_array_equal(self._fr.values_to, np.array([5., 5.]))

    def test_indeces(self):
        np.testing.assert_array_equal(self._fr.index_from, np.array([3, 5]))
        np.testing.assert_array_equal(self._fr.index_to, np.array([4, 6]))

    def test_residuals(self):
        np.testing.assert_array_equal(self._dtor.residuals, np.array([4., 3., 6., 1., 2.]))

    def test_residual_index(self):
        np.testing.assert_array_equal(self._dtor.residual_index, np.array([0, 1, 2, 8, 9]))


class TestThreePointDampening(unittest.TestCase):
    r'''
                                                                                    1   2   3   4   5   6
    -------------------------------------------------------------------------------------------------------
    6           R                                                                 |   |   |   |   |   |   |
    -----------/-\-----------------------------------------------------------------------------------------
    5         /   \           R                                                   |   |   |   |   |   |   |
    ---------/-----\---------/-\---------------------------------------------------------------------------
    4       /       \       /   \   R                                             |   |   |   |   |   |   |
    -------/---------\-----/-----\-/-----------------------------------------------------------------------
    3     /           \   /       R                                               |   |   |   |   |   |   |
    -----/-------------\-/---------------------------------------------------------------------------------
    2   /               R                                                         |   |   |   |   |   |   |
    ---/---------------------------------------------------------------------------------------------------
    1 R                                                                           |   |   |   |   |   |   |
    -------------------------------------------------------------------------------------------------------
      0        1        2     3   4 5
    '''
    def setUp(self):
        signal = np.array([1., 6., 2., 5., 3., 4.])
        self._fr, self._dtor = process_signal(signal)

    def test_values(self):
        np.testing.assert_array_equal(self._fr.values_from, np.array([]))
        np.testing.assert_array_equal(self._fr.values_to, np.array([]))

    def test_indeces(self):
        np.testing.assert_array_equal(self._fr.index_from, np.array([]))
        np.testing.assert_array_equal(self._fr.index_to, np.array([]))

    def test_residuals(self):
        np.testing.assert_array_equal(self._dtor.residuals, np.array([1., 6., 2., 5., 3., 4.]))

    def test_residual_index(self):
        np.testing.assert_array_equal(self._dtor.residual_index, np.array([0, 1, 2, 3, 4, 5]))


class TestThreePointDampeningClosed(unittest.TestCase):
    r'''
                                                                                    1   2   3   4   5   6
    -------------------------------------------------------------------------------------------------------
    6           R                                                                 |   |   |   |   |   |   |
    -----------/-\-----------------------------------------------------------------------------------------
    5         /   \           2                                                   |   |   |   |   |   |   |
    ---------/-----\---------/-\---------------------------------------------------------------------------
    4       /       \       /   \   1                                             |   |   | 1 |   |   |   |
    -------/---------\-----/-----\-/-\---------------------------------------------------------------------
    3     /           \   /       1---\                                           |   | 1 |   |   |   |   |
    -----/-------------\-/-------------\-------------------------------------------------------------------
    2   /               2---------------\                                         |   |   |   |   |   |   |
    ---/---------------------------------\-----------------------------------------------------------------
    1 R                                   R                                       |   |   |   |   |   |   |
    -------------------------------------------------------------------------------------------------------
      0        1        2     3   4 5     6
    '''
    def setUp(self):
        signal = np.array([1., 6., 2., 5., 3., 4., 1.])
        self._fr, self._dtor = process_signal(signal)

    def test_values(self):
        np.testing.assert_array_equal(self._fr.values_from, np.array([3., 2.]))
        np.testing.assert_array_equal(self._fr.values_to, np.array([4., 5.]))

    def test_indeces(self):
        np.testing.assert_array_equal(self._fr.index_from, np.array([4, 2]))
        np.testing.assert_array_equal(self._fr.index_to, np.array([5, 3]))

    def test_residuals(self):
        np.testing.assert_array_equal(self._dtor.residuals, np.array([1., 6., 1.]))

    def test_residual_index(self):
        np.testing.assert_array_equal(self._dtor.residual_index, np.array([0, 1, 6]))


test_signals = [
    np.array([0., 1., 0., 1., -1., 1., 0., 1., -1., 1., 0]),
    np.array([2., 5., 3., 6., 2., 3., 1., 6., 1., 4., 2., 3., 1., 4., 2., 5., 3., 4., 2.]),
    np.array([1., 7., 4., 3., 4., 2., 5., 3., 6., 1., 2.]),
    np.array([4., 3., 6., 4., 5., 2., 5., 3., 1., 2.]),
    np.array([1., 6., 2., 5., 3., 4., 1.])
]

@pytest.mark.parametrize('signal', test_signals)
def test_split_any_signal_anywhere_once(signal):
    reference_recorder, reference_detector = process_signal(signal)

    for split_point in range(1, len(signal)):
        fr = RFR.FullRecorder()
        dtor = RF.ThreePointDetector(recorder=fr)

        dtor.process(signal[:split_point]).process(signal[split_point:])

        np.testing.assert_array_equal(fr.values_from, reference_recorder.values_from)
        np.testing.assert_array_equal(fr.values_to, reference_recorder.values_to)
        np.testing.assert_array_equal(fr.index_from, reference_recorder.index_from)
        np.testing.assert_array_equal(fr.index_to, reference_recorder.index_to)
        np.testing.assert_array_equal(dtor.residuals, reference_detector.residuals)
        np.testing.assert_array_equal(dtor.residual_index, reference_detector.residual_index)


@pytest.mark.parametrize('signal', test_signals)
def test_split_any_signal_anywhere_twice(signal):
    reference_recorder, reference_detector = process_signal(signal)

    for split_point_1 in range(1, len(signal)):
        for split_point_2 in range(split_point_1 + 1, len(signal)):
            fr = RFR.FullRecorder()
            dtor = RF.ThreePointDetector(recorder=fr)

            (dtor
             .process(signal[:split_point_1])
             .process(signal[split_point_1:split_point_2])
             .process(signal[split_point_2:]))

            np.testing.assert_array_equal(fr.values_from, reference_recorder.values_from)
            np.testing.assert_array_equal(fr.values_to, reference_recorder.values_to)
            np.testing.assert_array_equal(fr.index_from, reference_recorder.index_from)
            np.testing.assert_array_equal(fr.index_to, reference_recorder.index_to)
            np.testing.assert_array_equal(dtor.residuals, reference_detector.residuals)
            np.testing.assert_array_equal(dtor.residual_index, reference_detector.residual_index)


@pytest.mark.parametrize('signal', test_signals)
def test_flipped_signals(signal):
    reference_recorder, reference_detector = process_signal(signal)
    flipped_recorder, flipped_detector = process_signal(-signal)

    np.testing.assert_array_equal(flipped_recorder.values_from, -np.asarray(reference_recorder.values_from))
    np.testing.assert_array_equal(flipped_recorder.values_to, -np.asarray(reference_recorder.values_to))
    np.testing.assert_array_equal(flipped_recorder.index_from, np.asarray(reference_recorder.index_from))
    np.testing.assert_array_equal(flipped_recorder.index_to, np.asarray(reference_recorder.index_to))
    np.testing.assert_array_equal(flipped_detector.residuals, -np.asarray(reference_detector.residuals))
    np.testing.assert_array_equal(flipped_detector.residual_index, np.asarray(reference_detector.residual_index))

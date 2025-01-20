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

__author__ = "Benjamin Maier"
__maintainer__ = __author__

import pytest
import unittest
import numpy as np
import pandas as pd
import copy

from pylife.stress.rainflow.fkm_nonlinear import FKMNonlinearDetector
import pylife.stress.rainflow.recorders as RFR
import pylife.materiallaws.notch_approximation_law as NAL
from pylife.materiallaws.notch_approximation_law_seegerbeste import SeegerBeste


@pytest.fixture(autouse=True)
def np_precision_2_print():
    old_prec = pd.get_option("display.precision")
    old_expand = pd.get_option("expand_frame_repr")
    with np.printoptions(precision=2):
        pd.set_option("display.precision", 2)
        pd.set_option("expand_frame_repr", False)
        yield
    pd.set_option("display.precision", old_prec)
    pd.set_option("expand_frame_repr", old_expand)


class TestIncomplete(unittest.TestCase):

    def setUp(self):

        signal = np.array([0, 500.])

        self._recorder = RFR.FKMNonlinearRecorder()
        E = 206e3    # [MPa] Young's modulus
        K = 2650     # 1184 [MPa]
        n = 0.187    # [-]
        K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

        # initialize notch approximation law
        extended_neuber = NAL.ExtendedNeuber(E, K, n, K_p)

        # first run
        detector = FKMNonlinearDetector(recorder=self._recorder, notch_approximation_law=extended_neuber)
        detector.process(signal)

        # second run
        self._detector = detector.process(signal)

    def test_values(self):
        np.testing.assert_array_equal(self._recorder.loads_min, np.array([]))
        np.testing.assert_array_equal(self._recorder.loads_max, np.array([]))
        np.testing.assert_array_equal(self._recorder.S_min, np.array([]))
        np.testing.assert_array_equal(self._recorder.S_max, np.array([]))
        np.testing.assert_array_equal(self._recorder.epsilon_min, np.array([]))
        np.testing.assert_array_equal(self._recorder.epsilon_max, np.array([]))

    def test_strain_values(self):
        np.testing.assert_array_equal(self._detector.strain_values_first_run, np.array([]))
        np.testing.assert_allclose(self._detector.strain_values_second_run, np.array([2.48e-3, 5.65e-5]), rtol=1e-1)
        np.testing.assert_allclose(self._detector.strain_values, np.array([2.48e-3, 5.65e-5]), rtol=1e-1)

    def test_epsilon_LF(self):
        collective = self._recorder.collective
        np.testing.assert_allclose(collective.epsilon_min_LF.to_numpy(), np.array([]))
        np.testing.assert_allclose(collective.epsilon_max_LF.to_numpy(), np.array([]))

    def test_interpolation(self):
        df = self._detector.interpolated_stress_strain_data(load_segment=0, n_points_per_branch=5)
        expected = pd.DataFrame(
            {
                "stress": [0.0, 122.02, 244.04, 366.06, 488.08],
                "strain": [0.0, 5.92e-4, 1.19e-3, 1.80e-3, 2.48e-3],
                "secondary_branch": [False, False, False, False, False],
                "hyst_index": -1,
                "load_segment": 0,
                "run_index": 2
            }
        )

        pd.testing.assert_frame_equal(df, expected, rtol=1e-1)


def test_no_binning_class():
    signal = np.array([100, 0, 80, 20, 60, 40])

    E = 206e3    # [MPa] Young's modulus
    K = 2650     # 1184 [MPa]
    n = 0.187    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    # initialize notch approximation law
    extended_neuber = NAL.ExtendedNeuber(E, K, n, K_p)

    detector = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(), notch_approximation_law=extended_neuber, binner=None
    )
    detector.process_hcm_first(signal).process_hcm_second(signal)

    recorder = detector.recorder
    np.testing.assert_array_equal(recorder.loads_min, np.array([40., 20., 0.]))
    np.testing.assert_array_equal(recorder.loads_max, np.array([60., 80., 100.]))
    np.testing.assert_allclose(recorder.S_min, np.array([39.997581, 19.997582, -0.002388]), rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(recorder.S_max, np.array([59.997581, 79.997574, 99.997488]), rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(recorder.epsilon_min, np.array([1.941866e-04, 9.709922e-05, 1.169416e-08]), rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(recorder.epsilon_max, np.array([0.000291, 0.000388, 0.000485]), rtol=1e-3, atol=1e-6)


class TestFKMMemory1Inner(unittest.TestCase):
    """Example given in FKM nonlinear 3.2.1, p.147 """


    def setUp(self):

        signal = np.array([100, 0, 80, 20, 60, 40])

        self._recorder = RFR.FKMNonlinearRecorder()
        E = 206e3    # [MPa] Young's modulus
        K = 2650     # 1184 [MPa]
        n = 0.187    # [-]
        K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

        # initialize notch approximation law
        extended_neuber = NAL.ExtendedNeuber(E, K, n, K_p)

        # first run
        detector = FKMNonlinearDetector(recorder=self._recorder, notch_approximation_law=extended_neuber)
        detector.process_hcm_first(signal)

        # second run
        self._detector = detector.process_hcm_second(signal)

    def test_values(self):
        np.testing.assert_array_equal(self._recorder.loads_min, np.array([40., 20., 0.]))
        np.testing.assert_array_equal(self._recorder.loads_max, np.array([60., 80., 100.]))
        np.testing.assert_allclose(self._recorder.S_min, np.array([39.997581, 19.997582, -0.002388]), rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(self._recorder.S_max, np.array([59.997581, 79.997574, 99.997488]), rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(self._recorder.epsilon_min, np.array([1.941866e-04, 9.709922e-05, 1.169416e-08]), rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(self._recorder.epsilon_max, np.array([0.000291, 0.000388, 0.000485]), rtol=1e-3, atol=1e-6)

    def test_strain_values(self):

        # regression test

        expected_first = np.array([4.85e-04, 1.17e-08, 3.88e-04, 9.70e-05, 2.91e-04])
        expected_second = np.array(
            [1.94e-04, 4.85e-04, 1.17e-08, 3.88e-04, 9.70e-05, 2.91e-04, 1.94e-04]
        )
        expected_total = np.concatenate((expected_first, expected_second))

        np.testing.assert_allclose(self._detector.strain_values, expected_total, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(self._detector.strain_values_first_run, expected_first, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(self._detector.strain_values_second_run, expected_second, rtol=1e-3, atol=1e-5)

    def test_epsilon_LF(self):
        collective = self._recorder.collective

        np.testing.assert_allclose(
            collective.epsilon_min_LF.to_numpy(), np.array([0.0, 0.0, 0.0]), rtol=1e-2
        )

        np.testing.assert_allclose(
            collective.epsilon_max_LF.to_numpy(),
            np.array([0.485, 0.485, 0.485]) * 1e-3,
            rtol=1e-2,
        )


    def test_interpolation(self):
        df = self._detector.interpolated_stress_strain_data(load_segment=5, n_points_per_branch=5)
        expected = pd.DataFrame(
            {
                "stress": [6.0e01, 5.5e01, 5.0e01, 4.5e01, 4.0e01],
                "strain": [2.9e-04, 2.7e-04, 2.4e-04, 2.2e-04, 1.9e-04],
                "secondary_branch": True,
                "hyst_index": 0,
                "load_segment": 5,
                "run_index": 2,
            }
        )
        pd.testing.assert_frame_equal(df, expected, rtol=1e-1)


class TestFKMMemory1_2_3(unittest.TestCase):
    """Example given in FKM nonlinear 3.2.2, p.150 """

    def setUp(self):
        signal = np.array([100, -100, 100, -200, -100, -200, 200, 0, 200, -200])

        self._recorder = RFR.FKMNonlinearRecorder()
        E = 206e3    # [MPa] Young's modulus
        K = 2650     # 1184 [MPa]
        n = 0.187    # [-]
        K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

        # initialize notch approximation law
        extended_neuber = NAL.ExtendedNeuber(E, K, n, K_p)

        # first run
        detector = FKMNonlinearDetector(recorder=self._recorder, notch_approximation_law=extended_neuber)
        detector.process_hcm_first(signal)

        # second run
        self._detector = detector.process_hcm_second(signal)

    def test_values(self):
        np.testing.assert_array_equal(self._recorder.loads_min, np.array([ -100.0, -100.0, -200.0, 0.0, -200.0, -100.0, -200.0, -200.0, 0.0, -200.0]))
        np.testing.assert_array_equal(self._recorder.loads_max, np.array([100.0, 100.0, -100.0, 200.0, 200.0, 100.0, 100.0, -100.0, 200.0, 200.0]))
        np.testing.assert_allclose(self._recorder.S_min, np.array([-99.997488, -99.997488, -199.898022, -0.096954, -199.898022, -99.936887, -199.898022, -199.898022, -0.096954, -199.898022]), rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(self._recorder.S_max, np.array([99.997488, 99.997488, -99.898146, 199.898022, 199.898022, 100.05809, 100.05809, -99.898146, 199.898022, 199.898022]), rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(self._recorder.epsilon_min, np.array([-0.000485, -0.000485, -0.000971, 0.0, -0.000971, -0.000486, -0.000971, -0.000971, 0.0, -0.000971]), rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(self._recorder.epsilon_max, np.array([0.000485, 0.000485, -0.000486, 0.000971, 0.000971, 0.000485, 0.000485, -0.000486, 0.000971, 0.000971]), rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(self._recorder.S_a, np.array([99.997488, 99.997488, 49.999938, 99.997488, 199.898022, 99.997488, 149.978056, 49.999938, 99.997488, 199.898022]), rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(self._recorder.S_m, np.array([0.0, 0.0, -149.898084, 99.900534, 0.0, 0.060602, -49.919966, -149.898084, 99.900534, 0.0]), rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(self._recorder.epsilon_a, np.array([ 0.000485, 0.000485, 0.000243, 0.000485, 0.000971, 0.000485, 0.000728, 0.000243, 0.000485, 0.000971]), rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(self._recorder.epsilon_m, np.array([ 0.0, 0.0, -0.000729, 0.000486, 0.0, -0.0, -0.000243, -0.000729, 0.000486, 0.0]), rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(self._recorder.is_closed_hysteresis, np.array([True, False, True, True, True, True, True, True, True, True]), rtol=1e-3, atol=1e-6)

    def test_strain_values(self):

        # regression test
        np.testing.assert_allclose(self._detector.strain_values, np.array([  4.85449192e-04, -4.85449192e-04,  4.85449192e-04, -9.71373378e-04,
            -4.85935881e-04, -9.71373378e-04,  9.71373378e-04,  4.74995178e-07,
            9.71373378e-04, -9.71373378e-04,  4.85152226e-04, -4.85746157e-04,
            4.85152226e-04, -9.71373378e-04, -4.85935881e-04, -9.71373378e-04,
            9.71373378e-04,  4.74995178e-07,  9.71373378e-04, -9.71373378e-04]), rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(self._detector.strain_values_first_run, np.array([4.85449192e-04, -4.85449192e-04,  4.85449192e-04, -9.71373378e-04,
            -4.85935881e-04, -9.71373378e-04,  9.71373378e-04,  4.74995178e-07,
            9.71373378e-04, -9.71373378e-04]), rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(self._detector.strain_values_second_run, np.array([4.85152226e-04, -4.85746157e-04,
            4.85152226e-04, -9.71373378e-04, -4.85935881e-04, -9.71373378e-04,
            9.71373378e-04,  4.74995178e-07,  9.71373378e-04, -9.713734e-04]), rtol=1e-3, atol=1e-5)


    def test_epsilon_LF(self):
        collective = self._recorder.collective

        np.testing.assert_allclose(
            collective.epsilon_min_LF.to_numpy(),
            np.array([-0.485, -0.485, -0.971, -0.971, -0.971, -0.971, -0.971, -0.971, -0.971, -0.971]) * 1e-3,
            rtol=1e-2
        )

        np.testing.assert_allclose(
            collective.epsilon_max_LF.to_numpy(),
            np.array([0.485, 0.485, 0.485, 0.971, 0.971, 0.971, 0.971, 0.971, 0.971, 0.971]) * 1e-3,
            rtol=1e-2
        )


class TestHCMExample1(unittest.TestCase):
    """Example 2.7.1 "Akademisches Beispiel", p.74 """

    def setUp(self):
        L = pd.Series([100, -200, 100, -250, 200, 0, 200, -200])
        c = 1.4
        gamma_L = (250+6.6)/250
        self.signal = c * gamma_L * L

        self._recorder = RFR.FKMNonlinearRecorder()
        E = 206e3    # [MPa] Young's modulus
        K = 1184     # [MPa]
        n = 0.187    # [-]
        K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

        # initialize notch approximation law
        extended_neuber = NAL.ExtendedNeuber(E, K, n, K_p)

        # first run
        detector = FKMNonlinearDetector(recorder=self._recorder, notch_approximation_law=extended_neuber)
        detector.process_hcm_first(self.signal)
        self._detector_1st = copy.deepcopy(detector)

        # second run
        self._detector = detector.process_hcm_second(self.signal)

    def test_values(self):
        np.testing.assert_allclose(self._recorder.loads_min, np.array([-143.696, -287.392000, 0.000000, -287.392000, -287.392000, -359.240000, 0.000000]))
        np.testing.assert_allclose(self._recorder.loads_max, np.array([143.696, 143.696000, 287.392000, 143.696000, 143.696000, 287.392000, 287.392000]))
        np.testing.assert_allclose(self._recorder.S_min, np.array([-142.462718, -258.875485, -23.628639, -256.454172, -256.454172, -299.686083, -23.628639]), rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(self._recorder.S_max, np.array([142.462718, 154.491647, 261.296798, 156.912960, 156.912960, 261.296798, 261.296798]), rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(self._recorder.epsilon_min, np.array([-0.000704, -0.001551, 0.000121, -0.001574, -0.001574, -0.002099, 0.000121]), rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(self._recorder.epsilon_max, np.array([0.000704, 0.000632, 0.001529, 0.000610, 0.000610, 0.001529, 0.001529]), rtol=1e-3, atol=1e-6)

        # data according to table 2.24
        np.testing.assert_allclose(self._recorder.S_a, np.array([142.47, 206.7, 142.47, 206.7, 206.7, 280.54, 142.47]), rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(self._recorder.S_m, np.array([0, -52.21, 118.87, -49.79, -49.79, -19.2, 118.87]), rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(self._recorder.epsilon_a, np.array([0.070e-2, 0.109e-2, 0.07e-2, 0.109e-2, 0.109e-2, 0.181e-2, 0.07e-2]), rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(self._recorder.epsilon_m, np.array([0, -0.046e-2, 0.082e-2, -0.048e-2, -0.048e-2, -0.029e-2, 0.082e-2]), rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(self._recorder.is_closed_hysteresis, np.array([False, True, True, True, True, True, True]), rtol=1e-3, atol=1e-5)

    def test_strain_values(self):

        # regression test
        np.testing.assert_allclose(self._detector.strain_values, np.array([0.000704, -0.001551,  0.000632, -0.002099,  0.001529,  0.000121, 0.001529, -0.001574,  0.00061, -0.001574,  0.00061, -0.002099, 0.001529,  0.000121,  0.001529, -0.001574]), rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(self._detector.strain_values_first_run, np.array([0.000704, -0.001551,  0.000632, -0.002099,  0.001529,  0.000121, 0.001529, -0.001574]), rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(self._detector.strain_values_second_run, np.array([0.00061, -0.001574,  0.00061, -0.002099, 0.001529,  0.000121,  0.001529, -0.001574]), rtol=1e-3, atol=1e-5)

    def test_epsilon_LF(self):
        collective = self._recorder.collective
        np.testing.assert_allclose(
            collective.epsilon_min_LF.to_numpy(),
            np.array([0.0, -1.55, -2.1, -2.1, -2.1, -2.1, -2.1]) * 1e-3,
            rtol=1e-2
        )

        np.testing.assert_allclose(
            collective.epsilon_max_LF.to_numpy(),
            np.array([0.70, 0.70, 1.53, 1.53, 1.53, 1.53, 1.53]) * 1e-3,
            rtol=1e-2
        )


class TestHCMExample2(unittest.TestCase):
    """Example 2.7.2, "Welle mit V-Kerbe", p.78 """

    def setUp(self):
        signal = 1266.25 * pd.Series([0.3, -0.3, 0.5, -0.5, 0.6, -0.6, 0.3, -0.3, 0.7, -0.7, 0.2, -0.2, 0.6, -0.6, 0.8, -0.8, 0.8, -0.8])

        self._recorder = RFR.FKMNonlinearRecorder()

        E = 206e3    # [MPa] Young's modulus
        K = 3.1148*(1251)**0.897 / (( np.min([0.338, 1033.*1251.**(-1.235)]) )**0.187)
        #K = 2650.5   # [MPa]
        n = 0.187    # [-]
        K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

        np.testing.assert_allclose(max(abs(signal)), 1013)

        # initialize notch approximation law
        extended_neuber = NAL.ExtendedNeuber(E, K, n, K_p)

        # first run
        detector = FKMNonlinearDetector(recorder=self._recorder, notch_approximation_law=extended_neuber)
        detector.process_hcm_first(signal)

        # second run
        self._detector = detector.process_hcm_second(signal)

    def test_values(self):
        np.testing.assert_allclose(self._recorder.loads_min, np.array([-379.875000, -633.125000, -379.875000, -759.750000, -253.250000, -759.750000, -886.375000, -1013.000000, -379.875000, -633.125000, -379.875000, -759.750000, -253.250000, -759.750000, -886.375000, -1013.000000, -1013.000000]))
        np.testing.assert_allclose(self._recorder.loads_max, np.array([379.875000, 633.125000, 379.875000, 759.750000, 253.250000, 759.750000, 886.375000, 1013.000000, 379.875000, 633.125000, 379.875000, 759.750000, 253.250000, 759.750000, 886.375000, 1013.000000, 1013.000000]))

        np.testing.assert_allclose(self._recorder.S_min, np.array([-372.009936, -594.481489, -341.838808, -688.090316, -167.306238, -679.185693, -761.498693, -829.680000, -289.717095, -567.897815, -331.002638, -677.254146, -165.138652, -677.018107, -759.331107, -829.680000, -829.680000]), rtol=5e-2, atol=1e-6)
        np.testing.assert_allclose(self._recorder.S_max, np.array([381.714834, 602.229958, 411.885962, 688.090316, 338.477585, 696.994940, 767.107794, 829.680000, 464.007675, 628.813632, 422.722132, 698.926487, 340.645171, 699.162526, 769.275380, 829.680000, 829.680000]), rtol=5e-2, atol=1e-6)
        np.testing.assert_allclose(self._recorder.epsilon_min, np.array([-0.001833, -0.003223, -0.002011, -0.004078, -0.001642, -0.004028, -0.004965, -0.006035, -0.002412, -0.003449, -0.002115, -0.004182, -0.001666, -0.004052, -0.004989, -0.006035, -0.006035]), rtol=5e-2, atol=1e-6)
        np.testing.assert_allclose(self._recorder.epsilon_max, np.array([0.001885, 0.003285, 0.001707, 0.004078, 0.000820, 0.004128, 0.005044, 0.006035, 0.001306, 0.003059, 0.001603, 0.003974, 0.000796, 0.004104, 0.005019, 0.006035, 0.006035]), rtol=8e-2, atol=1e-6)

        # data according to table 2.26
        np.testing.assert_allclose(self._recorder.S_a, np.array([381.7148907, 602.2304471, 381.7148907, 688.0911313, 252.8919183, 688.0911313, 767.1089576, 829.6814505, 381.7148907,  602.2304471, 381.7148907, 688.0911313, 252.8919183, 688.0911313, 767.1089576, 829.6814505, 829.6814505]), rtol=2e-2, atol=1e-6)
        # the values in the next line are only exact to rtol=2e-2
        np.testing.assert_allclose(self._recorder.epsilon_a, np.array([0.1885e-2, 0.3285e-2, 0.1885e-2, 0.4078e-2, 0.1231e-2, 0.4078e-2, 0.5044e-2, 0.6035e-2, 0.1885e-2, 0.3285e-2, 0.1885e-2, 0.4078e-2, 0.1231e-2, 0.4078e-2, 0.5044e-2, 0.6035e-2, 0.6035e-2]), rtol=1e-3, atol=4e-5)

        # values as given in table 2.26, note that the values are not equal (due to errors in the FKM example?)
        #np.testing.assert_allclose(self._recorder.S_m, np.array([0, 0, 30.17091928, 0, 79.9760653, 3.295529065, 0, 0, 82.29263249, 26.58372041, 41.00712551, 10.83620623, 82.14366097, 5.463124735, 2.16759567, 0, 0]), rtol=1e-3, atol=1e-6)
        #np.testing.assert_allclose(self._recorder.epsilon_m, np.array([0.0000e-2, 0.0000e-2, -0.0178e-2, 0.0000e-2, -0.0490e-2, -0.0028e-2, 0.0000e-2, 0.0000e-2, -0.0579e-2, -0.0226e-2, -0.0282e-2, -0.0104e-2, -0.0514e-2, -0.0052e-2, -0.0024e-2, 0.0000e-2, 0.0000]), rtol=1e-3, atol=1e-6)

        # regression tests
        np.testing.assert_allclose(self._recorder.S_m, np.array([ 4.852453, 3.874247, 35.023372, 0., 85.585193, 8.904656,2.804564, 0., 87.145085, 30.457968, 45.859578, 10.836206, 87.752788, 11.072252, 4.972159, 0., 0.]), rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(self._recorder.epsilon_m, np.array([ 2.564442e-05, 3.108080e-05, -1.519894e-04, 0.000000e+00, -4.111172e-04, 5.031508e-05, 3.921113e-05, 0.000000e+00,-5.532012e-04, -1.953252e-04, -2.562677e-04, -1.042783e-04, -4.35200012e-04, 2.62322621e-05, 1.51283129e-05, 0.0, 0.0]), rtol=1e-3, atol=1e-6)

    def test_strain_values(self):
        # regression test
        np.testing.assert_allclose(self._detector.strain_values, np.array([ 0.00188457, -0.00183328,  0.00328525, -0.00322309,  0.00407816, -0.00407816,
            0.00170694, -0.00201092,  0.00504351, -0.00496509,  0.00082001, -0.00164224,
            0.00412847, -0.00402784,  0.00603472, -0.00603472,  0.00603472, -0.00603472,
            0.00130573, -0.00241213,  0.00305884, -0.00344949,  0.00397388, -0.00418244,
            0.00160266, -0.0021152,   0.00501943, -0.00498917,  0.00079592, -0.00166632,
            0.00410439, -0.00405193,  0.00603472, -0.00603472,  0.00603472, -0.00603472]), rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(self._detector.strain_values_first_run, np.array([0.00188457, -0.00183328,  0.00328525, -0.00322309,  0.00407816, -0.00407816,
            0.00170694, -0.00201092,  0.00504351, -0.00496509,  0.00082001, -0.00164224,
            0.00412847, -0.00402784,  0.00603472, -0.00603472,  0.00603472, -0.00603472]), rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(self._detector.strain_values_second_run, np.array([
            0.00130573, -0.00241213,  0.00305884, -0.00344949,  0.00397388, -0.00418244,
            0.00160266, -0.0021152,   0.00501943, -0.00498917,  0.00079592, -0.00166632,
            0.00410439, -0.00405193,  0.00603472, -0.00603472,  0.00603472, -0.00603472]), rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize('vals, expected_loads_min, expected_loads_max', [
    (
        [200, 600, 1000, 60, 1500, 200, 80, 400, 1500, 700, 200],
        [60, 80, 200, 60, 80],
        [1000, 1500, 1000, 1500, 1500]
    ),
    (
        [0, 500], [], []
    ),
    (
        [100, -200, 100, -250, 200, 0, 200, -200],
        [-200,  0,  -200, -200, -250, 0],
        [100, 200, 100, 100, 200, 200]
    )
])
def test_edge_case_value_in_sample_tail_simple_signal(vals, expected_loads_min, expected_loads_max):
    signal = np.array(vals)

    E = 206e3    # [MPa] Young's modulus
    K = 3.1148*(1251)**0.897 / (( np.min([0.338, 1033.*1251.**(-1.235)]) )**0.187)
    #K = 2650.5   # [MPa]
    n = 0.187    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    extended_neuber = NAL.ExtendedNeuber(E, K, n, K_p)

    detector = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(), notch_approximation_law=extended_neuber
    )
    detector.process(signal).process(signal)

    loads_min = detector.recorder.loads_min
    loads_max = detector.recorder.loads_max

    np.testing.assert_allclose(loads_min, np.array(expected_loads_min))
    np.testing.assert_allclose(loads_max, np.array(expected_loads_max))

    detector.recorder.collective


@pytest.mark.parametrize('vals, expected_loads_min, expected_loads_max', [
    (
        [200, 600, 1000, 60, 1500, 200, 80, 400, 1500, 700, 200],
        [60, 120, 180, 80, 160, 240, 200, 400, 600, 60, 120, 180, 80, 160, 240],
        [1000, 2000, 3000, 1500, 3000, 4500, 1000, 2000, 3000, 1500, 3000, 4500, 1500, 3000, 4500]
    ),
    (
        [0, 500], [], []
    ),
    (
        [100, -200, 100, -250, 200, 0, 200, -200],
        [-200, -400, -600,  0, 0, 0,  -200, -400, -600, -200, -400, -600, -250, -500, -750, 0, 0, 0],
        [100,  200, 300, 200, 400, 600, 100, 200, 300, 100, 200, 300, 200, 400, 600, 200, 400, 600]
    )
])
def test_edge_case_value_in_sample_tail(vals, expected_loads_min, expected_loads_max):
    vals = np.array(vals)

    signal = pd.DataFrame({11: vals, 12: 2*vals, 13: 3*vals, 'load_step': range(len(vals))}).set_index('load_step').stack()
    signal.index.names = ['load_step', 'node_id']

    E = 206e3    # [MPa] Young's modulus
    K = 3.1148*(1251)**0.897 / (( np.min([0.338, 1033.*1251.**(-1.235)]) )**0.187)
    #K = 2650.5   # [MPa]
    n = 0.187    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    extended_neuber = NAL.ExtendedNeuber(E, K, n, K_p)

    detector = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(), notch_approximation_law=extended_neuber
    )
    detector.process(signal).process(signal)

    loads_min = detector.recorder.loads_min
    loads_max = detector.recorder.loads_max

    np.testing.assert_allclose(loads_min, np.array(expected_loads_min))
    np.testing.assert_allclose(loads_max, np.array(expected_loads_max))

    detector.recorder.collective


def test_flush_edge_case_load():
    mi_1 = pd.MultiIndex.from_product([range(9), range(3)], names=["load_step", "node_id"])

    signal_1 = pd.Series([
        0.0, 0.0, 0.0, 143.0, 171.0, 31.0, -287.0, -343.0, -63.0, 143.0, 171.0,
        31.0, -359.0, -429.0, -79.0, 287.0, 343.0, 63.0, 0.0, 0.0, 0.0, 287.0,
        343.0, 63.0, -287.0, -343.0, -63.0
    ], index=mi_1)

    mi_2 = pd.MultiIndex.from_product([range(9, 17), range(3)], names=["load_step", "node_id"])

    signal_2 = pd.Series([
        143.0, 171.0, 31.0, -287.0, -343.0, -63.0, 143.0, 171.0, 31.0, -359.0,
        -429.0, -79.0, 287.0, 343.0, 63.0, 0.0, 0.0, 0.0, 287.0, 343.0, 63.0,
        -287.0, -343.0, -63.0
    ], index=mi_2)

    E = 206e3    # [MPa] Young's modulus
    #K = 3.048*(1251)**0.07 / ((np.min([0.08, 1033.*1251.**(-1.05)]) )**0.07)
    K = 2650.5   # [MPa]
    n = 0.07    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    extended_neuber = NAL.ExtendedNeuber(E, K, n, K_p)

    detector = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(), notch_approximation_law=extended_neuber
    )

    detector.process(signal_1, flush=True).process(signal_2, flush=True)

    expected_load_min = pd.Series(
        [
            -143.0, -171.0, -31.0, -287.0, -343.0, -63.0, 0.0, 0.0, 0.0, -287.0, -343.0,
            -63.0, -287.0, -343.0, -63.0, -359.0, -429.0, -79.0, 0.0, 0.0, 0.0
        ],
        index=pd.MultiIndex.from_product(
            [[1, 2, 6, 8, 10, 4, 14], range(3)], names=["load_step", "node_id"]
        ),
        name="loads_min"
    )
    expected_load_max = pd.Series(
        [
            143.0, 171.0,  31.0, 143.0, 171.0,  31.0, 287.0, 343.0,  63.0, 143.0, 171.0,
            31.0, 143.0, 171.0,  31.0, 287.0, 343.0,  63.0, 287.0, 343.0, 63.0
        ],
        index=pd.MultiIndex.from_product(
            [[1, 3, 5, 9, 11, 7, 13], range(3)], names=["load_step", "node_id"]
        ),
        name="loads_max"
    )

    loads_min = detector.recorder.loads_min
    loads_max = detector.recorder.loads_max

    pd.testing.assert_series_equal(loads_min, expected_load_min)
    pd.testing.assert_series_equal(loads_max, expected_load_max)

    collective = detector.recorder.collective

    np.testing.assert_allclose(
        collective.epsilon_min_LF.to_numpy(),
        np.array([0., 0., 0., -1.4, -1.67, -0.31, -1.75, -2.08, -0.4, -1.75, -2.08, -0.4, -1.75, -2.08, -0.4, -1.75, -2.08, -0.4, -1.75, -2.08, -0.4 ]) * 1e-3,
        rtol=1e-1
    )

    np.testing.assert_allclose(
        collective.epsilon_max_LF.to_numpy(),
        np.array([0.71, 0.83, 0.17, 0.71, 0.83, 0.17, 1.4, 1.67, 0.31, 1.4, 1.67, 0.31, 1.4, 1.67, 0.31, 1.4, 1.67, 0.31, 1.4, 1.67, 0.31]) * 1e-3,
        rtol=1e-1
    )


def test_flush_edge_case_load_simple_signal():
    signal_1 = np.array([0.0, 143.0, -287.0, 143.0, -359.0, 287.0, 0.0, 287.0, -287.0])
    signal_2 = np.array([143.0, -287.0, 143.0, -359.0, 287.0, 0.0, 287.0, -287.0])

    E = 206e3    # [MPa] Young's modulus
    K = 3.048*(1251)**0.07 / (( np.min([0.08, 1033.*1251.**(-1.05)]) )**0.07)
    #K = 2650.5   # [MPa]
    n = 0.07    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    extended_neuber = NAL.ExtendedNeuber(E, K, n, K_p)

    detector = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(), notch_approximation_law=extended_neuber
    )

    detector.process(signal_1, flush=True).process(signal_2, flush=True)

    expected_load_min = np.array(
        [-143.0, -287.0, 0.0, -287.0, -287.0, -359.0, 0.0],
    )
    expected_load_max = np.array(
        [143.0, 143.0, 287.0, 143.0, 143.0, 287.0, 287.0],
    )

    loads_min = detector.recorder.loads_min
    loads_max = detector.recorder.loads_max

    np.testing.assert_allclose(loads_min, expected_load_min)
    np.testing.assert_allclose(loads_max, expected_load_max)


def test_flush_edge_case_S_simple_signal():
    signal_1 = np.array([0.0, 143.0, -287.0, 143.0, -359.0, 287.0, 0.0, 287.0, -287.0])
    signal_2 = np.array([143.0, -287.0, 143.0, -359.0, 287.0, 0.0, 287.0, -287.0])

    E = 206e3    # [MPa] Young's modulus
    K = 3.048*(1251)**0.07 / (( np.min([0.08, 1033.*1251.**(-1.05)]) )**0.07)
    #K = 2650.5   # [MPa]
    n = 0.07    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    extended_neuber = NAL.ExtendedNeuber(E, K, n, K_p)

    detector = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(), notch_approximation_law=extended_neuber
    )

    detector.process(signal_1, flush=True).process(signal_2, flush=True)

    expected_S_min = np.array([-48.0, -96.7, 1.42e-14, -96.7, -96.7, -121.0, 1.42e-14])
    expected_S_max = pd.Series([49.1, 49.1, 96.74, 49.1, 49.1, 96.75, 96.75])

    S_min = detector.recorder.S_min
    S_max = detector.recorder.S_max

    np.testing.assert_allclose(S_min, expected_S_min, rtol=1e-1, atol=0.0)
    np.testing.assert_allclose(S_max, expected_S_max, rtol=1e-1, atol=0.0)


def test_flush_edge_case_S():
    mi_1 = pd.MultiIndex.from_product([range(9), range(3)], names=["load_step", "node_id"])

    signal_1 = pd.Series([
        0.0, 0.0, 0.0, 143.0, 171.0, 31.0, -287.0, -343.0, -63.0, 143.0, 171.0,
        31.0, -359.0, -429.0, -79.0, 287.0, 343.0, 63.0, 0.0, 0.0, 0.0, 287.0,
        343.0, 63.0, -287.0, -343.0, -63.0
    ], index=mi_1)

    mi_2 = pd.MultiIndex.from_product([range(9, 17), range(3)], names=["load_step", "node_id"])

    signal_2 = pd.Series([
        143.0, 171.0, 31.0, -287.0, -343.0, -63.0, 143.0, 171.0, 31.0, -359.0,
        -429.0, -79.0, 287.0, 343.0, 63.0, 0.0, 0.0, 0.0, 287.0, 343.0, 63.0,
        -287.0, -343.0, -63.0
    ], index=mi_2)

    E = 206e3    # [MPa] Young's modulus
    K = 3.048*(1251)**0.07 / (( np.min([0.08, 1033.*1251.**(-1.05)]) )**0.07)
    #K = 2650.5   # [MPa]
    n = 0.07    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    extended_neuber = NAL.ExtendedNeuber(E, K, n, K_p)

    detector = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(), notch_approximation_law=extended_neuber
    )

    detector.process(signal_1, flush=True).process(signal_2, flush=True)

    expected_S_min = pd.Series(
        [
            -49.096964, -57.761134, -11.552227,  -96.749900, -115.522268,  -21.660425, -9.610801e-11,
            -1.143832e-10, -1.549e-07,  -96.749900, -115.522268,  -21.660425,
            -96.749900, -115.522268,  -21.660425, -121.298382, -144.402835,  -27.436539,
            -9.610801e-11, -1.143832e-10, -1.549e-07,
        ],
        index=pd.MultiIndex.from_product(
            [[1, 2, 6, 8, 10, 4, 14], range(3)], names=["load_step", "node_id"]
        ),
        name="S_min"
    )
    expected_S_max = pd.Series(
        [
            49.096964, 57.761134, 11.552227, 49.096964, 57.761134, 10.108198, 96.749900,
            115.522268, 21.660425, 49.096964, 57.761134, 10.108198, 49.096964, 57.761134,
            10.108198, 96.749900, 115.522268, 21.660425, 96.749900, 115.522268, 21.660425,
        ],
        index=pd.MultiIndex.from_product(
            [[1, 3, 5, 9, 11, 7, 13], range(3)], names=["load_step", "node_id"]
        ),
        name="S_max"
    )

    S_min = detector.recorder.S_min
    S_max = detector.recorder.S_max

    pd.testing.assert_series_equal(S_min, expected_S_min, rtol=1e-1)
    pd.testing.assert_series_equal(S_max, expected_S_max, rtol=1e-1)


@pytest.mark.parametrize('vals, num', [
    (
        1266.25 * pd.Series(
            [0.3, -0.3, 0.5, -0.5, 0.6, -0.6, 0.3, -0.3, 0.7, -0.7, 0.2, -0.2, 0.6, -0.6, 0.8, -0.8, 0.8, -0.8]
        ),
        1
    ),
    (
        np.array([100., -100., 100., -200., -100., -200., 200., 0., 200., -200.]),
        2
    ),
    (
        [100., 0., 80., 20., 60., 40.],
        3
    ),
    (
        [200., 600., 1000., 60., 1500., 200., 80., 400., 1500., 700., 200.],
        4
    ),
    (
        [0., 500.], 5
    ),
    (
        [100., -200., 100., -250., 200., 0., 200., -200.],
        6
    ),
    (
        pd.Series([100., -200., 100., -250., 200., 0., 200., -200.]) * (250+6.6)/250 * 1.4,
        7
    )
])
def test_edge_case_value_in_sample_tail_compare_simple(vals, num):
    vals = np.array(vals)
    signal = pd.DataFrame({11: vals, 12: 2*vals, 13: 3*vals, 'load_step': range(len(vals))}).set_index('load_step').stack()
    signal.index.names = ['load_step', 'node_id']

    signal_2 = signal.copy()
    index = signal.index.to_frame().reset_index(drop=True)
    index['load_step'] += len(vals)
    signal_2.index = index.set_index(signal.index.names).index

    E = 206e3    # [MPa] Young's modulus
    K = 3.1148*(1251)**0.897 / (( np.min([0.338, 1033.*1251.**(-1.235)]) )**0.187)
    #K = 2650.5   # [MPa]
    n = 0.187    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    extended_neuber = NAL.ExtendedNeuber(E, K, n, K_p)


    print("single")
    detector_simple = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(), notch_approximation_law=extended_neuber
    )
    detector_simple.process(vals).process(vals)

    print("multiple")
    detector_multiindex = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(), notch_approximation_law=extended_neuber
    )
    detector_multiindex.process(signal).process(signal)

    simple_collective = detector_simple.recorder.collective
    simple_collective.index = simple_collective.index.droplevel('assessment_point_index')
    multi_collective = detector_multiindex.recorder.collective

    pd.testing.assert_frame_equal(
        simple_collective,
        multi_collective.groupby('hysteresis_index').first(),
    )

    with open(f'tests/stress/rainflow/reference-fkm-nonlinear/reference_process-process-{num}.json') as f:
        reference = f.read()

    assert multi_collective.to_json(indent=4) == reference



@pytest.mark.parametrize('vals, num', [
    (
        1266.25 * pd.Series(
            [0.3, -0.3, 0.5, -0.5, 0.6, -0.6, 0.3, -0.3, 0.7, -0.7, 0.2, -0.2, 0.6, -0.6, 0.8, -0.8, 0.8, -0.8]
        ),
        1
    ),
    (
        np.array([100., -100., 100., -200., -100., -200., 200., 0., 200., -200.]),
        2
    ),
    (
        [100., 0., 80., 20., 60., 40.],
        3
    ),
    (
        [200., 600., 1000., 60., 1500., 200., 80., 400., 1500., 700., 200.],
        4
    ),
    (
        [0., 500.], 5
    ),
    (
        [100., -200., 100., -250., 200., 0., 200., -200.],
        6
    ),
    (
        pd.Series([100., -200., 100., -250., 200., 0., 200., -200.]) * (250+6.6)/250 * 1.4,
        7
    )
])
def test_hcm_first_second(vals, num):
    vals = np.array(vals)
    signal = pd.DataFrame({11: vals, 12: 2*vals, 13: 3*vals, 'load_step': range(len(vals))}).set_index('load_step').stack()
    signal.index.names = ['load_step', 'node_id']

    E = 206e3    # [MPa] Young's modulus
    K = 3.1148*(1251)**0.897 / (( np.min([0.338, 1033.*1251.**(-1.235)]) )**0.187)
    #K = 2650.5   # [MPa]
    n = 0.187    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    extended_neuber = NAL.ExtendedNeuber(E, K, n, K_p)

    detector_simple = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(), notch_approximation_law=extended_neuber
    )
    detector_simple.process_hcm_first(vals).process_hcm_second(vals)

    detector_multiindex = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(), notch_approximation_law=extended_neuber
    )
    detector_multiindex.process_hcm_first(signal).process_hcm_second(signal)

    simple_collective = detector_simple.recorder.collective
    simple_collective.index = simple_collective.index.droplevel('assessment_point_index')
    multi_collective = detector_multiindex.recorder.collective

    pd.testing.assert_frame_equal(
        simple_collective,
        multi_collective.groupby('hysteresis_index').first(),
    )

    with open(f'tests/stress/rainflow/reference-fkm-nonlinear/reference_first_second-{num}.json') as f:
        reference = f.read()

    assert multi_collective.to_json(indent=4) == reference


def test_element_id():
    vals = np.array([100., -200., 100., -250., 200., 0., 200., -200.]) * (250+6.6)/250 * 1.4
    signal = pd.DataFrame({11: vals, 12: 2*vals, 13: 3*vals, 'load_step': range(len(vals))}).set_index('load_step').stack()
    signal.index.names = ['load_step', 'element_id']

    E = 206e3    # [MPa] Young's modulus
    K = 3.1148*(1251)**0.897 / (( np.min([0.338, 1033.*1251.**(-1.235)]) )**0.187)
    #K = 2650.5   # [MPa]
    n = 0.187    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    extended_neuber = NAL.ExtendedNeuber(E, K, n, K_p)

    detector = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(), notch_approximation_law=extended_neuber
    )
    detector.process_hcm_first(signal).process_hcm_second(signal)

    with open(f'tests/stress/rainflow/reference-fkm-nonlinear/reference_first_second-7.json') as f:
        reference = f.read()

    assert detector.recorder.collective.to_json(indent=4) == reference
    assert detector.recorder.loads_max.index.names == ["load_step", "element_id"]


@pytest.fixture
def detector_seeger_beste():
    E = 206e3    # [MPa] Young's modulus
    K = 1184.0   # [MPa]
    n = 0.187    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    seeger_beste = SeegerBeste(E, K, n, K_p)
    return FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(), notch_approximation_law=seeger_beste
    )


@pytest.fixture
def detector_interpolate(detector_seeger_beste):
    vals = pd.Series([160, -200, 250, -250, 230, 0, 260]) * 800.0/260.0
    return detector_seeger_beste.process(vals, flush=False).process(vals, flush=True)


@pytest.mark.parametrize("load_segment, n_points_per_branch, expected", [
    (
        0, 10,
        {
            "stress": [0.0, -4.2e+1, -8.4e+1, -1.3e+2, -1.7e+2, -2.1e+2, -2.5e+2, -2.9e+2, -3.4e+2, -3.8e+2],
            "strain": [0.0, -2.0e-4, -4.1e-4, -6.2e-4, -8.4e-4, -1.1e-3, -1.5e-3, -2.0e-3, -2.8e-3, -4.0e-3],
            "secondary_branch": False,
            "hyst_index": -1,
            "load_segment": 0,
            "run_index": 1,
        }
    ),
    (
        0, 5,
        {
            "stress": [0.0e+00, -9.4e+01, -1.9e+02, -2.8e+02, -3.8e+02],
            "strain": [0.0e+00, -4.6e-04, -9.7e-04, -1.8e-03, -4.0e-03],
            "secondary_branch": False,
            "hyst_index": -1,
            "load_segment": 0,
            "run_index": 1
        }
    ),
    (
        1, 5,
        {
            "stress": [-3.77e+02, -1.89e+02, 0.00e+00, 1.89e+02, 3.77e+02],
            "strain": [-4.03e-03, -3.11e-03, -2.09e-03, -3.40e-04, 4.03e-03],
            "secondary_branch": True,
            "hyst_index": 0,
            "load_segment": 1,
            "run_index": 1,
        }
    ),
    (
        2, 5,
        {
            "stress": [3.77e+02, 3.89e+02, 4.01e+02, 4.12e+02, 4.24e+02],
            "strain": [4.03e-03, 4.48e-03, 4.98e-03, 5.55e-03, 6.18e-03],
            "secondary_branch": False,
            "hyst_index": -1,
            "load_segment": 2,
            "run_index": 1,
        }
    ),
])
def test_interpolation_like_in_demo_load_segment(detector_interpolate, load_segment, n_points_per_branch, expected):
    df = detector_interpolate.interpolated_stress_strain_data(load_segment=load_segment, n_points_per_branch=n_points_per_branch)
    expected = pd.DataFrame(expected)
    pd.testing.assert_frame_equal(df, expected, rtol=1e-1)


@pytest.mark.parametrize("hyst_index, expected", [
    (
        0,
        {
            "stress": [-3.77e+02, -1.89e+02, 0.00e+00, 1.89e+02, 3.77e+02],
            "strain": [-4.03e-03, -3.11e-03, -2.09e-03, -3.40e-04, 4.03e-03],
            "secondary_branch": [True, True, True, True, True],
            "hyst_index": 0,
            "load_segment": [1, 1, 1, 1, 1],
            "run_index": 1,
        }
    ),
    (
        1,
        {
            "stress": [4.06e+02, 2.64e+02, 1.23e+02, -1.78e+01, -1.59e+02, -1.59e+02, -1.78e+01, 1.23e+02, 2.64e+02, 4.06e+02],
            "strain": [5.19e-03, 4.50e-03, 3.80e-03, 2.93e-03, 1.51e-03, 1.51e-03, 2.20e-03, 2.91e-03, 3.77e-03, 5.19e-03],
            "secondary_branch": [True, True, True, True, True, True, True, True, True, True],
            "hyst_index": 1,
            "load_segment": [5, 5, 5, 5, 5, 6, 6, 6, 6, 6],
            "run_index": 1,
        }
    ),
    (
        5,
        {
            "stress": [4.31e+02, 2.17e+02, 3.44e+00, -2.10e+02, -4.24e+02, -4.24e+02, -2.10e+02, 3.44e+00, 2.17e+02, 4.31e+02],
            "strain": [6.59e-03, 5.54e-03, 4.30e-03, 1.62e-03, -6.18e-03, -6.18e-03, -5.13e-03, -3.89e-03, -1.21e-03, 6.59e-03],
            "secondary_branch": [True, True, True, True, True, True, True, True, True, True],
            "hyst_index": 5,
            "load_segment": [12, 12, 12, 12, 12, 16, 16, 16, 16, 16],
            "run_index": 2,
        }
    )
])
def test_interpolation_like_in_demo_hyst_index(detector_interpolate, hyst_index, expected):
    df = detector_interpolate.interpolated_stress_strain_data(hysteresis_index=hyst_index, n_points_per_branch=5)
    expected = pd.DataFrame(expected)
    pd.testing.assert_frame_equal(df, expected, rtol=1e-1)


def test_interpolation_everything(detector_interpolate):
    df = detector_interpolate.interpolated_stress_strain_data(n_points_per_branch=3)
    expected = pd.DataFrame(
        {
            "stress": [
                0.0e+00, -1.9e+02, -3.8e+02, -3.8e+02, 0.0e+00, 3.8e+02, 3.8e+02, 4.0e+02, 4.2e+02,
                4.2e+02, 1.1e+00, -4.2e+02, -4.2e+02, -8.1e+00, 4.1e+02, 4.1e+02, 1.2e+02, -1.6e+02,
                -1.6e+02, 1.2e+02, 4.1e+02, -4.2e+02, 1.1e+00, 4.2e+02, 4.2e+02, 4.3e+02, 4.3e+02,
                4.3e+02, 2.6e+01, -3.8e+02, -3.8e+02, 2.3e+01, 4.2e+02, 4.2e+02, 2.3e+01, -3.8e+02,
                4.3e+02, 3.4e+00, -4.2e+02, -4.2e+02, -1.0e+01, 4.0e+02, 4.0e+02, 1.2e+02, -1.6e+02,
                -1.6e+02, 1.2e+02, 4.0e+02, -4.2e+02, 3.4e+00, 4.3e+02, 4.3e+02, 4.3e+02, 4.3e+02
            ],
            "strain": [
                0.0e+00, -9.7e-04, -4.0e-03, -4.0e-03, -2.1e-03, 4.0e-03, 4.0e-03, 5.0e-03, 6.2e-03,
                6.2e-03, 3.9e-03, -6.0e-03, -6.0e-03, -3.9e-03, 5.2e-03, 5.2e-03, 3.8e-03, 1.5e-03,
                1.5e-03, 2.9e-03, 5.2e-03, -6.0e-03, -3.8e-03, 6.2e-03, 6.2e-03, 6.4e-03, 6.6e-03,
                6.6e-03, 4.5e-03, -3.7e-03, -3.7e-03, -1.6e-03, 6.3e-03, 6.3e-03, 4.2e-03, -3.7e-03,
                6.6e-03, 4.3e-03, -6.2e-03, -6.2e-03, -4.0e-03, 5.1e-03, 5.1e-03, 3.7e-03, 1.4e-03,
                1.4e-03, 2.8e-03, 5.1e-03, -6.2e-03, -3.9e-03, 6.6e-03, 6.6e-03, 6.6e-03, 6.6e-03
            ],
            "secondary_branch": [
                False, False, False, True, True, True, False, False, False, True, True, True,
                True, True, True, True, True, True, True, True, True, True, True, True,
                False, False, False, True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True, True, True, True,
                True, True, True, False, False, False
            ],
            "hyst_index": [
                -1, -1, -1, 0, 0, 0, -1, -1, -1, 2, 2, 2, -1, -1, -1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                -1, -1, -1, -1, -1, -1, 3, 3, 3, 3, 3, 3, 5, 5, 5, -1, -1, -1, 4, 4, 4, 4, 4, 4,
                5, 5, 5, -1, -1, -1
            ],
            "load_segment": [
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7,
                8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15,
                16, 16, 16, 17, 17, 17],
            "run_index": [
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
            ],
        }
    )
    pd.testing.assert_frame_equal(df, expected, rtol=1e-1)


def test_interpolation_everything_first_run(detector_seeger_beste):
    vals = pd.Series([160, -200, 250, -250, 230, 0, 260]) * 800.0/260.0
    detector_seeger_beste.process(vals, flush=False)
    df = detector_seeger_beste.interpolated_stress_strain_data(n_points_per_branch=3)

    expected = pd.DataFrame(
        {
            "stress": [
                0.00e+00, -1.89e+02, -3.77e+02, -3.77e+02, 0.00e+00, 3.77e+02, 3.77e+02, 4.01e+02, 4.24e+02,
                4.24e+02, 1.15e+00, -4.22e+02, -4.22e+02, -8.06e+00, 4.06e+02, 4.06e+02, 1.23e+02, -1.59e+02
            ],
            "strain": [
                0.00e+00, -9.69e-04, -4.03e-03, -4.03e-03, -2.09e-03, 4.03e-03, 4.03e-03, 4.98e-03, 6.18e-03,
                6.18e-03, 3.92e-03, -6.05e-03, -6.05e-03, -3.86e-03, 5.19e-03, 5.19e-03, 3.79e-03, 1.51e-03
            ],
            "secondary_branch": [
                False, False, False, True, True, True, False, False, False,
                True, True, True, True, True, True, True, True, True
            ],
            "hyst_index": [
                -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
            ],
            "load_segment": [
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5
            ],
            "run_index": [
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            ],
        }
    )
    pd.testing.assert_frame_equal(df, expected, rtol=1e-1)


def test_history_guideline_at_once():
    # Fig 2.3 FKM NL Guideline

    signal = pd.Series([0., 200., -50., 250., -300., 150., -120., 350., 349.])
    signal.index.name = "load_step"

    recorder = RFR.FKMNonlinearRecorder()
    E = 206e3    # [MPa] Young's modulus
    K = 2650     # 1184 [MPa]
    n = 0.187    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    # initialize notch approximation law
    extended_neuber = NAL.ExtendedNeuber(E, K, n, K_p)

    # first run
    detector = FKMNonlinearDetector(recorder=recorder, notch_approximation_law=extended_neuber)

    detector.process(signal)

    df = detector.history()

    expected = pd.DataFrame(
        {
            "load": [200., -50., 200., 250., -250., -300., 150., -120., 150., 300., 350.],
            "stress": [2.0e+02, -4.9e+01, 2.0e+02, 2.5e+02, -2.5e+02, -3.0e+02, 1.5e+02, -1.2e+02, 1.5e+02, 3.0e+02, 3.5e+02],
            "strain": [9.9e-04, -2.4e-04, 9.9e-04, 1.2e-03, -1.2e-03, -1.5e-03, 7.3e-04, -6.0e-04, 7.3e-04, 1.5e-03, 1.7e-03],
            "secondary_branch": [False, True, True, False, True, False, True, True, True, True, False],
            "load_step": [1, 2, -1, 3, -1, 4, 5, 6, -1, -1, 7],
            "turning_point": [0, 1, -1, 2, -1, 3, 4, 5, -1, -1, 6],
            "load_segment": np.arange(11),
            "run_index": 1,
            "hyst_from": [0, -1, -1,  1, -1,  3,  2, -1, -1, -1, -1],
            "hyst_to": [-1,  0, -1, -1,  1, -1, -1,  2, -1,  3, -1],
            "hyst_close": [-1, -1,  0, -1, -1, -1, -1, -1,  2, -1, -1],
        }
    ).set_index(["load_segment", "load_step", "run_index", "turning_point", "hyst_from", "hyst_to", "hyst_close"])

    pd.testing.assert_frame_equal(df, expected, rtol=1e-1)


@pytest.mark.parametrize("split_point", [5])
def test_history_guideline_at_split(split_point):
    # Fig 2.3 FKM NL Guideline

    signal = pd.Series([0., 200., -50., 250., -300., 150., -120., 350., 349.])
    signal.index.name = "load_step"

    recorder = RFR.FKMNonlinearRecorder()
    E = 206e3    # [MPa] Young's modulus
    K = 2650     # 1184 [MPa]
    n = 0.187    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    # initialize notch approximation law
    extended_neuber = NAL.ExtendedNeuber(E, K, n, K_p)

    # first run
    detector = FKMNonlinearDetector(recorder=recorder, notch_approximation_law=extended_neuber)

    detector.process(signal[:split_point]).process(signal[split_point:])

    df = detector.history()

    expected = pd.DataFrame(
        {
            "load": [200., -50., 200., 250., -250., -300., 150., -120., 150., 300., 350.],
            "stress": [2.0e+02, -4.9e+01, 2.0e+02, 2.5e+02, -2.5e+02, -3.0e+02, 1.5e+02, -1.2e+02, 1.5e+02, 3.0e+02, 3.5e+02],
            "strain": [9.9e-04, -2.4e-04, 9.9e-04, 1.2e-03, -1.2e-03, -1.5e-03, 7.3e-04, -6.0e-04, 7.3e-04, 1.5e-03, 1.7e-03],
            "secondary_branch": [False, True, True, False, True, False, True, True, True, True, False],
            "load_step": [1, 2, -1, 3, -1, 4, 5, 6, -1, -1, 7],
            "turning_point": [0, 1, -1, 2, -1, 3, 4, 5, -1, -1, 6],
            "load_segment": np.arange(11),
            "run_index": [1] * split_point + [2] * (11-split_point),
            "hyst_from": [0, -1, -1,  1, -1,  3,  2, -1, -1, -1, -1],
            "hyst_to": [-1,  0, -1, -1,  1, -1, -1,  2, -1,  3, -1],
            "hyst_close": [-1, -1,  0, -1, -1, -1, -1, -1,  2, -1, -1],
        }
    ).set_index(["load_segment", "load_step", "run_index", "turning_point", "hyst_from", "hyst_to", "hyst_close"])

    pd.testing.assert_frame_equal(df, expected, rtol=1e-1)


@pytest.fixture
def unitdetector():
    recorder = RFR.FKMNonlinearRecorder()
    E = 1.0    # [MPa] Young's modulus
    K = 1e6     # 1184 [MPa]
    n = 1.0    # [-]
    K_p = 1.0    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    # initialize notch approximation law
    extended_neuber = NAL.ExtendedNeuber(E, K, n, K_p)

    return FKMNonlinearDetector(recorder=recorder, notch_approximation_law=extended_neuber)


def test_epsilon_max_LF_unclosed_hysteresis(unitdetector):
    signal = pd.Series([0.0, -100.0, 100.0, 0.0, 200.0, 100.0])
    signal.index.name = "load_step"

    unitdetector.process(signal)

    assert np.all([np.diff(unitdetector.recorder.epsilon_min_LF) <= 0.0])
    assert np.all([np.diff(unitdetector.recorder.epsilon_max_LF) >= 0.0])


def test_epsilon_min_LF_flip(unitdetector):
    signal = pd.Series([10.0, -1.0, 260.0, -250.0, 60.0, -280.0, -100.0])
    signal.index.name = "load_step"

    unitdetector.process(signal)

    assert np.all([np.diff(unitdetector.recorder.epsilon_min_LF) <= 0.0])


def test_epsilon_max_LF_flip(unitdetector):
    signal = -1.0 * pd.Series([10.0, -1.0, 260.0, -250.0, 60.0, -280.0, -100.0])
    signal.index.name = "load_step"

    unitdetector.process(signal)

    assert np.all([np.diff(unitdetector.recorder.epsilon_max_LF) >= 0.0])

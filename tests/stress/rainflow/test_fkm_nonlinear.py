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

import pylife.stress.rainflow as RF
from pylife.stress.rainflow.fkm_nonlinear import FKMNonlinearDetector
import pylife.stress.rainflow.recorders as RFR
import pylife.materiallaws.notch_approximation_law


class TestFKMMemory1Inner(unittest.TestCase):
    """Example given in FKM nonlinear 3.2.1, p.147 """


    def setUp(self):

        signal = np.array([100,0,80,20,60,40])

        self._recorder = RFR.FKMNonlinearRecorder()
        E = 206e3    # [MPa] Young's modulus
        K = 2650     # 1184 [MPa]
        n = 0.187    # [-]
        K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

        # initialize notch approximation law
        extended_neuber = pylife.materiallaws.notch_approximation_law.ExtendedNeuber(E, K, n, K_p)

        # wrap the notch approximation law by a binning class, which precomputes the values
        maximum_absolute_load = max(abs(signal))
        extended_neuber_binned = pylife.materiallaws.notch_approximation_law.Binned(
            extended_neuber, maximum_absolute_load, 100)

        # first run
        detector = FKMNonlinearDetector(recorder=self._recorder, notch_approximation_law=extended_neuber_binned)
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
        np.testing.assert_allclose(self._detector.strain_values, np.array([4.854492e-04, 1.169416e-08, 3.883614e-04, 9.709922e-05, 2.912740e-04, 1.941866e-04, 4.854492e-04, 1.169416e-08, 3.883614e-04, 9.709922e-05, 2.912740e-04, 1.941866e-04]), rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(self._detector.strain_values_first_run, np.array([4.854492e-04, 1.169416e-08, 3.883614e-04, 9.709922e-05, 2.912740e-04]), rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(self._detector.strain_values_second_run, np.array([1.941866e-04, 4.854492e-04, 1.169416e-08, 3.883614e-04, 9.709922e-05, 2.912740e-04, 1.941866e-04]), rtol=1e-3, atol=1e-5)


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
        extended_neuber = pylife.materiallaws.notch_approximation_law.ExtendedNeuber(E, K, n, K_p)

        # wrap the notch approximation law by a binning class, which precomputes the values
        maximum_absolute_load = max(abs(signal))
        extended_neuber_binned = pylife.materiallaws.notch_approximation_law.Binned(
            extended_neuber, maximum_absolute_load, 100)

        # first run
        detector = FKMNonlinearDetector(recorder=self._recorder, notch_approximation_law=extended_neuber_binned)
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
        extended_neuber = pylife.materiallaws.notch_approximation_law.ExtendedNeuber(E, K, n, K_p)

        # wrap the notch approximation law by a binning class, which precomputes the values
        maximum_absolute_load = max(abs(self.signal))
        extended_neuber_binned = pylife.materiallaws.notch_approximation_law.Binned(
            extended_neuber, maximum_absolute_load, 100)

        # first run
        detector = FKMNonlinearDetector(recorder=self._recorder, notch_approximation_law=extended_neuber_binned)
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

    def test_plotting(self):

        plotting_data = self._detector.interpolated_stress_strain_data(n_points_per_branch=3, only_hystereses=False)

        strain_values_primary = plotting_data["strain_values_primary"]
        stress_values_primary = plotting_data["stress_values_primary"]
        hysteresis_index_primary = plotting_data["hysteresis_index_primary"]
        strain_values_secondary = plotting_data["strain_values_secondary"]
        stress_values_secondary = plotting_data["stress_values_secondary"]
        hysteresis_index_secondary = plotting_data["hysteresis_index_secondary"]

        # plot resulting stress-strain curve
        sampling_parameter = 50    # choose larger for smoother plot
        plotting_data_fine = self._detector_1st.interpolated_stress_strain_data(n_points_per_branch=sampling_parameter)

        strain_values_primary_fine = plotting_data_fine["strain_values_primary"]
        stress_values_primary_fine = plotting_data_fine["stress_values_primary"]
        hysteresis_index_primary_fine = plotting_data_fine["hysteresis_index_primary"]
        strain_values_secondary_fine = plotting_data_fine["strain_values_secondary"]
        stress_values_secondary_fine = plotting_data_fine["stress_values_secondary"]
        hysteresis_index_secondary_fine = plotting_data_fine["hysteresis_index_secondary"]

        # the following plots the test case for visual debugging
        if False:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12,6))
            # load-time diagram
            import matplotlib
            matplotlib.rcParams.update({'font.size': 14})
            axes[0].plot(self.signal, "o-", lw=2)
            axes[0].grid()
            axes[0].set_xlabel("t [s]")
            axes[0].set_ylabel("L [N]")
            axes[0].set_title("Scaled load sequence")

            # stress-strain diagram
            axes[1].plot(strain_values_primary_fine, stress_values_primary_fine, "y-", lw=1)
            axes[1].plot(strain_values_secondary_fine, stress_values_secondary_fine, "y-.", lw=1)
            axes[1].grid()
            axes[1].set_xlabel(r"$\epsilon$")
            axes[1].set_ylabel(r"$\sigma$ [MPa]")
            axes[1].set_title("Material response")

            plt.savefig("test_fkm_nonlinear.png")


        strain_values_primary_reference = np.array([
             0.,          0.00034608,  0.00070365,         np.nan, -0.00070365, -0.00070365,
            -0.00104958, -0.00155125,         np.nan, -0.00155125, -0.00155125, -0.00179771,
            -0.00209921,         np.nan, -0.00178372, -0.00209921, -0.00209921, -0.00209921,])
        stress_values_primary_reference = np.array([
            0.,           71.23135917,  142.46271834,           np.nan, -142.46271834,
            -142.46271834, -200.66910165, -258.87548495,           np.nan, -258.87548495,
            -258.87548495, -279.28078419, -299.68608343,           np.nan, -299.68608343,
            -299.68608343, -299.68608343, -299.68608343,
        ])
        hysteresis_index_primary_reference = np.array([0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 6, 6, 6, 6, 6])
        strain_values_secondary_reference = np.array([
            np.nan,  7.03647041e-04,  7.03647041e-04,  1.14870755e-05,
            -7.03647041e-04,             np.nan, -1.55125063e-03, -1.55125063e-03,
            -5.43592015e-04,  6.32115565e-04,  6.32115565e-04, -3.75543053e-04,
            -1.55125063e-03,             np.nan, -2.09920894e-03, -2.09920894e-03,
            -7.15382913e-04,  1.52864389e-03,  1.52864389e-03,  8.36483922e-04,
            1.21349805e-04,  1.21349805e-04,  8.13509770e-04,  1.52864389e-03,
                        np.nan,  1.52864389e-03,  1.52864389e-03,  1.52864389e-03,
            1.52864389e-03,  2.57497805e-04, -1.57385738e-03, -1.57385738e-03,
            -5.66198760e-04,  6.09508820e-04,  6.09508820e-04, -3.98149798e-04,
            -1.57385738e-03,             np.nan, -1.57385738e-03, -1.57385738e-03,
            -1.57385738e-03, -1.57385738e-03, -5.66198760e-04,  6.09508820e-04,
            6.09508820e-04, -3.98149798e-04, -1.57385738e-03,             np.nan,
            -1.57385738e-03, -1.67878923e-03, -1.78372203e-03,             np.nan,
            -2.09920894e-03, -2.09920894e-03, -7.15382913e-04,  1.52864389e-03,
            1.52864389e-03,  8.36483922e-04,  1.21349805e-04,  1.21349805e-04,
            8.13509770e-04,  1.52864389e-03,             np.nan,  1.52864389e-03,
            1.52864389e-03,  1.52864389e-03,  1.52864389e-03,  2.57497805e-04,
            -1.57385738e-03])
        stress_values_secondary_reference = np.array([
            np.nan,  142.46271834,  142.46271834,    0.,         -142.46271834,
                    np.nan, -258.87548495, -258.87548495,  -52.19191877,  154.4916474,
            154.4916474,   -52.19191877, -258.87548495,           np.nan, -299.68608343,
            -299.68608343,  -19.19464281,  261.29679782,  261.29679782,  118.83407947,
            -23.62863887,  -23.62863887,  118.83407947,  261.29679782,           np.nan,
            261.29679782,  261.29679782,  261.29679782,  261.29679782,    2.42131287,
            -256.45417208, -256.45417208,  -49.77060591,  156.91296027,  156.91296027,
            -49.77060591, -256.45417208,           np.nan, -256.45417208, -256.45417208,
            -256.45417208, -256.45417208,  -49.77060591,  156.91296027,  156.91296027,
            -49.77060591, -256.45417208,           np.nan, -256.45417208, -278.07012776,
            -299.68608343,           np.nan, -299.68608343, -299.68608343,  -19.19464281,
            261.29679782,  261.29679782,  118.83407947,  -23.62863887,  -23.62863887,
            118.83407947,  261.29679782,           np.nan,  261.29679782,  261.29679782,
            261.29679782,  261.29679782,    2.42131287, -256.45417208])
        hysteresis_index_secondary_reference = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7])

        np.testing.assert_allclose(strain_values_primary_reference, strain_values_primary, rtol=1e-3)
        np.testing.assert_allclose(stress_values_primary_reference, stress_values_primary, rtol=1e-3)
        np.testing.assert_allclose(hysteresis_index_primary_reference, hysteresis_index_primary)

        np.testing.assert_allclose(strain_values_secondary_reference, strain_values_secondary, rtol=1e-3)
        np.testing.assert_allclose(stress_values_secondary_reference, stress_values_secondary, rtol=1e-3)
        np.testing.assert_allclose(hysteresis_index_secondary_reference, hysteresis_index_secondary)


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
        extended_neuber = pylife.materiallaws.notch_approximation_law.ExtendedNeuber(E, K, n, K_p)

        # wrap the notch approximation law by a binning class, which precomputes the values
        maximum_absolute_load = max(abs(signal))

        extended_neuber_binned = pylife.materiallaws.notch_approximation_law.Binned(
            extended_neuber, maximum_absolute_load, 100)

        # first run
        detector = FKMNonlinearDetector(recorder=self._recorder, notch_approximation_law=extended_neuber_binned)
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

    extended_neuber = pylife.materiallaws.notch_approximation_law.ExtendedNeuber(E, K, n, K_p)

    maximum_absolute_load = max(abs(signal))

    extended_neuber_binned = pylife.materiallaws.notch_approximation_law.Binned(
        extended_neuber, maximum_absolute_load, 100
    )

    detector = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(),
        notch_approximation_law=extended_neuber_binned
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
    K = 3.048*(1251)**0.07 / (( np.min([0.08, 1033.*1251.**(-1.05)]) )**0.07)
    #K = 2650.5   # [MPa]
    n = 0.07    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    extended_neuber = pylife.materiallaws.notch_approximation_law.ExtendedNeuber(E, K, n, K_p)

    maximum_absolute_load = max(abs(pd.concat([signal_1, signal_2])))

    extended_neuber_binned = pylife.materiallaws.notch_approximation_law.Binned(
        extended_neuber, maximum_absolute_load, 100
    )

    detector = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(),
        notch_approximation_law=extended_neuber_binned
    )

    detector.process(signal_1, flush=True).process(signal_2, flush=True)

    expected_load_min = pd.Series(
        [
            -143.0, -171.0, -31.0, -287.0, -343.0, -63.0, 0.0, 0.0, 0.0, -287.0, -343.0,
            -63.0, -287.0, -343.0, -63.0, -359.0, -429.0, -79.0, 0.0, 0.0, 0.0
        ],
        index=pd.MultiIndex.from_product(
            [[1, 2, 6, 8, 10, 4, 14], range(3)], names=["load_step", "node_id"]
        )
    )
    expected_load_max = pd.Series(
        [
            143.0, 171.0,  31.0, 143.0, 171.0,  31.0, 287.0, 343.0,  63.0, 143.0, 171.0,
            31.0, 143.0, 171.0,  31.0, 287.0, 343.0,  63.0, 287.0, 343.0, 63.0
        ],
        index=pd.MultiIndex.from_product(
            [[1, 3, 5, 9, 11, 7, 13], range(3)], names=["load_step", "node_id"]
        )
    )

    loads_min = detector.recorder.loads_min
    loads_max = detector.recorder.loads_max

    pd.testing.assert_series_equal(loads_min, expected_load_min)
    pd.testing.assert_series_equal(loads_max, expected_load_max)


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

    extended_neuber = pylife.materiallaws.notch_approximation_law.ExtendedNeuber(E, K, n, K_p)

    maximum_absolute_load = max(abs(pd.concat([signal_1, signal_2])))

    extended_neuber_binned = pylife.materiallaws.notch_approximation_law.Binned(
        extended_neuber, maximum_absolute_load, 100
    )

    detector = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(),
        notch_approximation_law=extended_neuber_binned
    )

    detector.process(signal_1, flush=True).process(signal_2, flush=True)

    expected_S_min = pd.Series(
        [
            -49.096964, -57.761134, -11.552227,  -96.749900, -115.522268,  -21.660425, -9.610801e-11,
            -1.143832e-10, -1.241333e-07,  -96.749900, -115.522268,  -21.660425,
            -96.749900, -115.522268,  -21.660425, -121.298382, -144.402835,  -27.436539,
            -9.610801e-11, -1.143832e-10, -1.241333e-07,
        ],
        index=pd.MultiIndex.from_product(
            [[1, 2, 6, 8, 10, 4, 14], range(3)], names=["load_step", "node_id"]
        )
    )
    expected_S_max = pd.Series(
        [
            49.096964, 57.761134, 11.552227, 49.096964, 57.761134, 10.108198, 96.749900,
            115.522268, 21.660425, 49.096964, 57.761134, 10.108198, 49.096964, 57.761134,
            10.108198, 96.749900, 115.522268, 21.660425, 96.749900, 115.522268, 21.660425,
        ],
        index=pd.MultiIndex.from_product(
            [[1, 3, 5, 9, 11, 7, 13], range(3)], names=["load_step", "node_id"]
        )
    )

    S_min = detector.recorder.S_min
    S_max = detector.recorder.S_max

    pd.testing.assert_series_equal(S_min, expected_S_min, check_index=False)
    pd.testing.assert_series_equal(S_max, expected_S_max, check_index=False)

    pd.testing.assert_series_equal(S_min, expected_S_min)
    pd.testing.assert_series_equal(S_max, expected_S_max)


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

    extended_neuber = pylife.materiallaws.notch_approximation_law.ExtendedNeuber(E, K, n, K_p)

    maximum_absolute_load_simple = max(abs(vals))
    maximum_absolute_load_multiple = signal.abs().groupby('node_id').max()


    print("single")
    extended_neuber_binned_simple = pylife.materiallaws.notch_approximation_law.Binned(
        extended_neuber, maximum_absolute_load_simple, 100
    )
    detector_simple = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(),
        notch_approximation_law=extended_neuber_binned_simple
    )
    detector_simple.process(vals).process(vals)

    print("multiple")
    extended_neuber_binned_multiple = pylife.materiallaws.notch_approximation_law.Binned(
        extended_neuber, maximum_absolute_load_multiple, 100
    )
    detector_multiindex = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(),
        notch_approximation_law=extended_neuber_binned_multiple
    )
    detector_multiindex.process(signal).process(signal)

    simple_collective = detector_simple.recorder.collective
    simple_collective.index = simple_collective.index.droplevel('assessment_point_index')
    simple_collective.pop('debug_output')
    multi_collective = detector_multiindex.recorder.collective

    if 'debug_output' in multi_collective:
        multi_collective.pop('debug_output')
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

    extended_neuber = pylife.materiallaws.notch_approximation_law.ExtendedNeuber(E, K, n, K_p)

    maximum_absolute_load_simple = max(abs(vals))
    maximum_absolute_load_multiple = signal.abs().groupby('node_id').max()

    extended_neuber_binned_simple = pylife.materiallaws.notch_approximation_law.Binned(
        extended_neuber, maximum_absolute_load_simple, 100
    )
    extended_neuber_binned_multiple = pylife.materiallaws.notch_approximation_law.Binned(
        extended_neuber, maximum_absolute_load_multiple, 100
    )
    detector_simple = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(),
        notch_approximation_law=extended_neuber_binned_simple
    )
    detector_simple.process_hcm_first(vals).process_hcm_second(vals)

    detector_multiindex = FKMNonlinearDetector(
        recorder=RFR.FKMNonlinearRecorder(),
        notch_approximation_law=extended_neuber_binned_multiple
    )
    detector_multiindex.process_hcm_first(signal).process_hcm_second(signal)

    simple_collective = detector_simple.recorder.collective
    simple_collective.index = simple_collective.index.droplevel('assessment_point_index')
    simple_collective.pop('debug_output')
    multi_collective = detector_multiindex.recorder.collective

    if 'debug_output' in multi_collective:
        multi_collective.pop('debug_output')

    pd.testing.assert_frame_equal(
        simple_collective,
        multi_collective.groupby('hysteresis_index').first(),
    )

    with open(f'tests/stress/rainflow/reference-fkm-nonlinear/reference_first_second-{num}.json') as f:
        reference = f.read()

    assert multi_collective.to_json(indent=4) == reference

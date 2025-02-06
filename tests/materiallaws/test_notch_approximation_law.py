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
import numpy as np
from pylife.materiallaws.notch_approximation_law import ExtendedNeuber, NotchApproxBinner
from .data import *


@pytest.mark.parametrize('load, strain, stress', [
    (3.592, 0.0017e-2, 3.592),
    (7.185, 0.0035e-2, 7.185),
    (176.057, 8.71e-4, 172.639),
    (359.3, 0.0021, 299.78)
])
def test_extended_neuber_primary_example_1(load, strain, stress):
    """ example under 2.7.1, p.74 of FKM nonlinear "Akademisches Beispiel" """
    notch_approximation_law = ExtendedNeuber(E=206e3, K=1184, n=0.187, K_p=3.5)

    assert np.isclose(notch_approximation_law.stress(load), stress, rtol=1e-3)
    assert np.isclose(notch_approximation_law.strain(stress), strain, rtol=1e-1)
    assert np.isclose(notch_approximation_law.load(stress), load, rtol=1e-3)


@pytest.mark.parametrize('delta_load, delta_strain, delta_stress', [
    (100.587, 0.0488e-2, 100.578),
    (201.174, 0.0978e-2, 200.794),
    (402.349, 0.2019e-2, 389.430),
    (700.518, 0.4051e-2, 590.289)
]) # values from matrix AST in FKM guideline nonlinear on page 162, chapter 3.4.1
def test_extended_neuber_secondary_example_1(delta_load, delta_strain, delta_stress):
    """ example under 2.7.1, p.74 of FKM nonlinear "Akademisches Beispiel" """
    notch_approximation_law = ExtendedNeuber(E=206e3, K=1184, n=0.187, K_p=3.5)

    assert np.isclose(notch_approximation_law.stress_secondary_branch(delta_load), delta_stress, rtol=1e-3)
    assert np.isclose(notch_approximation_law.strain_secondary_branch(delta_stress), delta_strain, rtol=1e-3)
    assert np.isclose(notch_approximation_law.load_secondary_branch(delta_stress), delta_load, rtol=1e-3)


@pytest.mark.parametrize('load, strain, stress', [
    (10.13, 0.0049e-2, 10.130),
    (20.26, 0.0098e-2, 20.260),
    (1013.00, 0.6035e-2, 829.681),
])
def test_extended_neuber_primary_example_2(load, strain, stress):
    """ example under 2.7.2, p.78 of FKM nonlinear, "Welle mit V-Kerbe" """
    notch_approximation_law = ExtendedNeuber(E=206e3, K=2650.5, n=0.187, K_p=3.5)

    assert np.isclose(notch_approximation_law.stress(load), stress, rtol=1e-1)
    assert np.isclose(notch_approximation_law.strain(stress), strain, rtol=1e-1)
    assert np.isclose(notch_approximation_law.load(stress), load, rtol=1e-1)


@pytest.mark.parametrize('delta_load, delta_strain, delta_stress', [
    (101.30, 0.0492e-2, 101.300),
    (506.50, 0.2462e-2, 505.784),
    (1002.87, 0.4990e-2, 978.725),
    (1499.24, 0.8017e-2, 1362.884),
    (2005.74, 1.1897e-2, 1649.573)
]) # values from matrix AST in FKM guideline nonlinear on page 171, chapter 3.4.2
def test_extended_neuber_secondary_example_2(delta_load, delta_strain, delta_stress):
    """ example under 2.7.1, p.74 of FKM nonlinear "Akademisches Beispiel" """
    notch_approximation_law = ExtendedNeuber(E=206e3, K=2650.5, n=0.187, K_p=3.5)

    assert np.isclose(notch_approximation_law.stress_secondary_branch(delta_load), delta_stress, rtol=1e-3)
    assert np.isclose(notch_approximation_law.strain_secondary_branch(delta_stress), delta_strain, rtol=1e-3)
    assert np.isclose(notch_approximation_law.load_secondary_branch(delta_stress), delta_load, rtol=1e-3)


def test_extended_neuber_example_no_binning_vectorized():
    E = 206e3    # [MPa] Young's modulus
    K = 1184     # [MPa]
    n = 0.187    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    notch_approximation_law = ExtendedNeuber(E, K, n, K_p)

    load = np.array([150.0, 175.0, 200.0])
    stress = notch_approximation_law.stress(load)
    stress_secondary_branch = notch_approximation_law.stress_secondary_branch(load)

    np.testing.assert_allclose(stress, np.array([148.463622, 171.674936, 193.702502]), rtol=1e-3)
    np.testing.assert_allclose(stress_secondary_branch, np.array([149.92007905, 174.81829204, 199.63066067]), rtol=1e-3)


@pytest.mark.parametrize('stress, load', [
    (22, 42),
    (40, 40),
    (120, 80),
    (220, 180),
    (320, 180)
])
def test_derivatives(stress, load):
    """ Test the analytically derived derivatives of stress and strain formulas """

    E = 206e3    # [MPa] Young's modulus
    K = 1184     # [MPa]
    n = 0.187    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)


    # initialize notch approximation law and damage parameter
    notch_approximation_law = ExtendedNeuber(E, K, n, K_p)

    load = np.array(load, dtype=float)
    stress = np.array(stress, dtype=float)
    h = 0.1

    # test derivative of stress
    numeric_derivative = (notch_approximation_law._stress_implicit(stress+h, load) - notch_approximation_law._stress_implicit(stress-h, load)) / (2*h)
    derivative = notch_approximation_law._d_stress_implicit(stress, load)

    assert np.isclose(numeric_derivative, derivative)

    # test derivative of secondary_stress
    numeric_derivative = (notch_approximation_law._stress_secondary_implicit(stress+h, load) - notch_approximation_law._stress_secondary_implicit(stress-h, load)) / (2*h)
    derivative = notch_approximation_law._d_stress_secondary_implicit(stress, load)

    assert np.isclose(numeric_derivative, derivative)


@pytest.mark.parametrize('E, K, n, L', [
    (260e3, 1184, 0.187, np.array([100, -200, 100, -250, 200, 100, 200, -200])),
    (100e3, 1500, 0.4, np.array([-100, 100, -200])),
    (200e3, 1000, 0.2, np.array([100, 10])),
])
def test_load(E, K, n, L):
    c = 1.4
    gamma_L = (250+6.6)/250
    L = c * gamma_L * L

    # initialize notch approximation law and damage parameter
    notch_approximation_law = ExtendedNeuber(E, K, n, K_p=3.5)

    # The "load" method is the inverse operation of "stress",
    # i.e., ``L = load(stress(L))`` and ``S = stress(load(stress))``.
    stress = notch_approximation_law.stress(L)
    load = notch_approximation_law.load(stress)
    stress2 = notch_approximation_law.stress(load)

    np.testing.assert_allclose(L, load, rtol=1e-3)
    np.testing.assert_allclose(stress, stress2, rtol=1e-3)


@pytest.mark.parametrize("L, expected", [
    (150.0, [148.5, 7.36e-4]),
    (175.0, [171.7, 8.67e-4]),
    (200.0, [193.7, 1.00e-3])
])
def test_primary_scalar(L, expected):
    notch_approximation_law = ExtendedNeuber(E=206e3, K=1184., n=0.187, K_p=3.5)
    result = notch_approximation_law.primary(L)

    assert result.shape == (2, )
    np.testing.assert_allclose(result, expected, rtol=1e-2)


def test_primary_vectorized():
    notch_approximation_law = ExtendedNeuber(E=206e3, K=1184., n=0.187, K_p=3.5)

    result = notch_approximation_law.primary([150.0, 175.0, 200.0])
    expected = [[148.4, 7.36e-4], [171.7, 8.67e-4], [193.7, 1.00e-3]]

    assert result.shape == (3, 2)
    np.testing.assert_allclose(result, expected, rtol=1e-2)


@pytest.mark.parametrize("L, expected", [
    (100.0, [100.0, 4.88e-4]),
    (400.0, [386.0, 1.99e-3]),
    (600.0, [533.0, 3.28e-3])
])
def test_secondary_scalar(L, expected):
    notch_approximation_law = ExtendedNeuber(E=206e3, K=1184., n=0.187, K_p=3.5)
    result = notch_approximation_law.secondary(L)

    assert result.shape == (2, )
    np.testing.assert_allclose(result, expected, rtol=1e-2)


def test_secondary_vectorized():
    notch_approximation_law = ExtendedNeuber(E=206e3, K=1184., n=0.187, K_p=3.5)

    result = notch_approximation_law.secondary([100.0, 400.0, 600.0])
    expected = [[100.0, 4.88e-4], [386.0, 1.99e-3], [533.0, 3.28e-3]]

    assert result.shape == (3, 2)
    np.testing.assert_allclose(result, expected, rtol=1e-2)


def test_binner_uninitialized():
    unbinned = ExtendedNeuber(E=206e3, K=1184., n=0.187, K_p=3.5)
    binned = NotchApproxBinner(unbinned)

    with pytest.raises(RuntimeError, match="NotchApproxBinner not initialized."):
        binned.primary(100.0)

    with pytest.raises(RuntimeError, match="NotchApproxBinner not initialized."):
        binned.secondary(100.0)


@pytest.mark.parametrize("L, expected", [
    (120.0, [148.0, 7.36e-4]),
    (160.0, [193.7, 1.00e-3]),
    (200.0, [193.7, 1.00e-3]),
])
def test_binner_initialized_five_points_primary_scalar(L, expected):
    unbinned = ExtendedNeuber(E=206e3, K=1184., n=0.187, K_p=3.5)
    binned = NotchApproxBinner(unbinned, number_of_bins=4).initialize(max_load=200.0)

    result = binned.primary(L)

    np.testing.assert_allclose(result, np.array([expected]), rtol=1e-2)


@pytest.mark.parametrize("L, expected", [
    (-120.0, [-148.0, -7.36e-4]),
    (-160.0, [-193.7, -1.00e-3]),
    (-200.0, [-193.7, -1.00e-3]),
])
def test_binner_initialized_five_points_primary_scalar_symmetry(L, expected):
    unbinned = ExtendedNeuber(E=206e3, K=1184., n=0.187, K_p=3.5)
    binned = NotchApproxBinner(unbinned, number_of_bins=4).initialize(max_load=200.0)

    result = binned.primary(L)

    np.testing.assert_allclose(result, np.array([expected]), rtol=1e-2)


@pytest.mark.parametrize("L, expected", [
    (120.0, [124.0, 6.10e-4]),
    (160.0, [171.7, 8.67e-4]),
    (200.0, [193.7, 1.00e-3])
])
def test_binner_initialized_nine_points_primary_scalar(L, expected):
    unbinned = ExtendedNeuber(E=206e3, K=1184., n=0.187, K_p=3.5)
    binned = NotchApproxBinner(unbinned, number_of_bins=8).initialize(max_load=200.0)

    result = binned.primary(L)

    np.testing.assert_allclose(result, np.array([expected]), rtol=1e-2)


def test_binner_initialized_nine_points_primary_out_of_scale():
    unbinned = ExtendedNeuber(E=206e3, K=1184., n=0.187, K_p=3.5)
    binned = NotchApproxBinner(unbinned, number_of_bins=8).initialize(max_load=200.0)

    with pytest.raises(
        ValueError,
        match="Requested load `400.0`, higher than initialized maximum load `200.0`",
    ):
        binned.primary(400.0)


@pytest.mark.parametrize("L, expected", [
    (120.0, [[148.0, 7.36e-4], [182.9, 9.34e-4], [214.3, 1.15e-3]]),
    (160.0, [[193.7, 1.00e-3], [233.3, 1.30e-3], [266.7, 1.64e-3]]),
    (200.0, [[193.7, 1.00e-3], [233.3, 1.30e-3], [266.7, 1.64e-3]])
])
def test_binner_initialized_five_points_primary_vectorized(L, expected):
    unbinned = ExtendedNeuber(E=206e3, K=1184., n=0.187, K_p=3.5)
    binned = NotchApproxBinner(unbinned, number_of_bins=4).initialize(
        max_load=[200.0, 250.0, 300.0]
    )

    result = binned.primary([L, 1.25*L, 1.5*L])

    assert result.shape == (3, 2)

    np.testing.assert_allclose(result, expected, rtol=1e-2)


@pytest.mark.parametrize("L, expected", [
    (20.0, [297.0, 1.48e-3]),
    (340.0, [533.0, 3.28e-3]),
    (600.0, [533.0, 3.28e-3])
])
def test_binner_initialized_one_bin_secondary_scalar(L, expected):
    unbinned = ExtendedNeuber(E=206e3, K=1184., n=0.187, K_p=3.5)
    binned = NotchApproxBinner(unbinned, number_of_bins=1).initialize(max_load=300.0)

    result = binned.secondary(L)

    np.testing.assert_allclose(result, np.array([expected]), rtol=1e-2)


@pytest.mark.parametrize("L, expected", [
    (-20.0, [-297.0, -1.48e-3]),
    (-340.0, [-533.0, -3.28e-3]),
    (-600.0, [-533.0, -3.28e-3])
])
def test_binner_initialized_one_bin_secondary_scalar_symmetry(L, expected):
    unbinned = ExtendedNeuber(E=206e3, K=1184., n=0.187, K_p=3.5)
    binned = NotchApproxBinner(unbinned, number_of_bins=1).initialize(max_load=300.0)

    result = binned.secondary(L)

    np.testing.assert_allclose(result, np.array([expected]), rtol=1e-2)


@pytest.mark.parametrize("L, expected", [
    (20.0, [100.0, 4.88e-4]),
    (400.0, [386.0, 1.99e-3]),
    (600.0, [533.0, 3.28e-3])
])
def test_binner_initialized_three_points_secondary_scalar(L, expected):
    unbinned = ExtendedNeuber(E=206e3, K=1184., n=0.187, K_p=3.5)
    binned = NotchApproxBinner(unbinned, number_of_bins=3).initialize(max_load=300.0)

    result = binned.secondary(L)

    np.testing.assert_allclose(result, np.array([expected]), rtol=1e-2)


def test_binner_initialized_three_points_secondary_out_of_scale():
    unbinned = ExtendedNeuber(E=206e3, K=1184., n=0.187, K_p=3.5)
    binned = NotchApproxBinner(unbinned, number_of_bins=3).initialize(max_load=300.0)

    with pytest.raises(
        ValueError,
        match="Requested load `700.0`, higher than initialized maximum delta load `600.0`",
    ):
        binned.secondary(700.0)


@pytest.mark.parametrize("L, expected", [
    (20.0, [[100.0, 4.88e-4], [133.3, 6.47e-4], [200.0, 9.74e-4]]),
    (400.0, [[386.0, 1.99e-3], [490.0, 2.81e-3], [638.2, 4.90e-3]]),
    (600.0, [[533.0, 3.28e-3], [638.2, 4.90e-03], [784.8, 9.26e-3]])
])
def test_binner_initialized_three_points_secondary_vectorized(L, expected):
    unbinned = ExtendedNeuber(E=206e3, K=1184., n=0.187, K_p=3.5)
    binned = NotchApproxBinner(unbinned, number_of_bins=3).initialize(max_load=[300.0, 400.0, 600.0])

    result = binned.secondary(([L, 1.25*L, 1.5*L]))

    assert result.shape == (3, 2)

    np.testing.assert_allclose(result, expected, rtol=1e-2)


def test_binner_zero_first_load():
    unbinned = ExtendedNeuber(E=206e3, K=1184., n=0.187, K_p=3.5)
    binned = NotchApproxBinner(unbinned, number_of_bins=3).initialize(max_load=[0.0, 300.0])

    result = binned.secondary(([0.0, 400.0]))

    expected = [[0.0, 0.0], [386.0, 1.99e-3]]
    np.testing.assert_allclose(result, expected, rtol=1e-2)


def test_binner_total_zero_load():
    unbinned = ExtendedNeuber(E=206e3, K=1184., n=0.187, K_p=3.5)
    with pytest.raises(
        ValueError,
        match="NotchApproxBinner must have at least one non zero point in max_load",
    ):
        NotchApproxBinner(unbinned, number_of_bins=3).initialize(max_load=[0.0, 0.0])

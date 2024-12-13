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
import pandas as pd

import pylife.strength.damage_parameter
import pylife.materiallaws.notch_approximation_law
from pylife.materiallaws.notch_approximation_law_seegerbeste import SeegerBeste
from .data import *

@pytest.mark.parametrize('load, strain, stress', [
    (3.592, 0.0017e-2, 3.592),
    (7.185, 0.0035e-2, 7.185),
    (176.057, 8.71e-4, 172.639),
    (359.3, 0.0019, 283.86)
])
def test_seeger_beste_primary_example_1(load, strain, stress):
    """ example under 2.7.1, p.74 of FKM nonlinear "Akademisches Beispiel" """
    notch_approximation_law = SeegerBeste(E=206e3, K=1184, n=0.187, K_p=3.5)

    assert np.isclose(notch_approximation_law.stress(load), stress, rtol=1e-1)
    assert np.isclose(notch_approximation_law.strain(stress), strain, rtol=1e-1)
    assert np.isclose(notch_approximation_law.load(stress), load, rtol=1e-1)


@pytest.mark.parametrize('delta_load, delta_strain, delta_stress', [
    (100.59, 0.000488, 100.568),
    (301.76, 0.001466, 295.844),
    (499.34, 0.002456, 449.183),
    (700.52, 0.003605, 559.407),
]) # values from matrix AST in FKM guideline nonlinear on page 180, chapter 3.5.1
def test_seeger_beste_secondary_example_1(delta_load, delta_strain, delta_stress):
    """ example under 2.7.1, p.74 of FKM nonlinear "Akademisches Beispiel" """
    notch_approximation_law = SeegerBeste(E=206e3, K=1184, n=0.187, K_p=3.5)

    assert np.isclose(notch_approximation_law.stress_secondary_branch(delta_load), delta_stress, rtol=1e-3)
    assert np.isclose(notch_approximation_law.strain_secondary_branch(delta_stress), delta_strain, rtol=1e-3)
    assert np.isclose(notch_approximation_law.load_secondary_branch(delta_stress), delta_load, rtol=1e-3)


@pytest.mark.parametrize('load, strain, stress', [
    (10.13, 0.0049e-2, 10.130),
    (20.26, 0.0098e-2, 20.260),
    (1013.00, 0.53e-2, 784.89),
])
def test_seeger_beste_example_2(load, strain, stress):
    """ example under 2.7.2, p.78 of FKM nonlinear, "Welle mit V-Kerbe" """
    notch_approximation_law = SeegerBeste(E=206e3, K=2650.5, n=0.187, K_p=3.5)

    assert np.isclose(notch_approximation_law.stress(load), stress, rtol=1e-1)
    assert np.isclose(notch_approximation_law.strain(stress), strain, rtol=1e-1)
    assert np.isclose(notch_approximation_law.load(stress), load, rtol=1e-1)


@pytest.mark.parametrize('delta_load, delta_strain, delta_stress', [
    (101.3, 0.000492, 101.3),
    (496.37, 0.00241, 495.095),
    (1002.87, 0.004879, 960.634),
    (1499.24, 0.007441, 1304.533),
    (2005.74, 0.010471, 1560.847),
]) # values from matrix AST in FKM guideline nonlinear on page 189, chapter 3.5.2
def test_seeger_beste_secondary_example_2(delta_load, delta_strain, delta_stress):
    """ example under 2.7.1, p.74 of FKM nonlinear "Akademisches Beispiel" """
    notch_approximation_law = SeegerBeste(E=206e3, K=2650.5, n=0.187, K_p=3.5)

    assert np.isclose(notch_approximation_law.stress_secondary_branch(delta_load), delta_stress, rtol=1e-3)
    assert np.isclose(notch_approximation_law.strain_secondary_branch(delta_stress), delta_strain, rtol=1e-3)
    assert np.isclose(notch_approximation_law.load_secondary_branch(delta_stress), delta_load, rtol=1e-3)


def test_seeger_beste_example_no_binning():
    E = 206e3    # [MPa] Young's modulus
    K = 1184     # [MPa]
    n = 0.187    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    notch_approximation_law = SeegerBeste(E, K, n, K_p)

    stress = notch_approximation_law.stress(150.0)
    stress_secondary_branch = notch_approximation_law.stress_secondary_branch(150.0)

    assert np.isclose(stress, 147.1, rtol=1e-3)
    assert np.isclose(stress_secondary_branch, 149.8, rtol=1e-3)


@pytest.mark.skip(reason="Derivatives not implemented at the moment, left in the code because it might be useful in the future for performance optimization with gradient-based root finding algorithms.")
@pytest.mark.parametrize('stress, load', [
    (4, 9),
    (10, 15),
    (22, 42),
])
def test_derivatives(stress, load):
    """ Test the analytically derived derivatives of stress and strain formulas """

    E = 206e3    # [MPa] Young's modulus
    K = 1184     # [MPa]
    n = 0.187    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)


    # initialize notch approximation law and damage parameter
    notch_approximation_law = SeegerBeste(E, K, n, K_p)

    load = float(load)
    stress = float(stress)
    h = 1e-5

    # test derivative of stress
    numeric_derivative = (notch_approximation_law._stress_implicit(stress+h, load) - notch_approximation_law._stress_implicit(stress-h, load)) / (2*h)
    derivative = notch_approximation_law._d_stress_implicit(stress, load)

    assert np.isclose(numeric_derivative, derivative.values[0])

    # test derivative of secondary_stress
    numeric_derivative = (notch_approximation_law._stress_secondary_implicit(stress+h, load) - notch_approximation_law._stress_secondary_implicit(stress-h, load)) / (2*h)
    derivative = notch_approximation_law._d_stress_secondary_implicit(stress, load)

    assert np.isclose(numeric_derivative, derivative.values[0])


@pytest.mark.parametrize('E, K, n, L', [
    (260e3, 1184, 0.187, pd.Series([100, -200, 100, -250, 200, 100, 200, -200])),
    (100e3, 1500, 0.4, pd.Series([-100, 100, -200])),
    (200e3, 1000, 0.2, pd.Series([100, 10])),
])
def test_seeger_beste_load(E, K, n, L):
    c = 1.4
    gamma_L = (250+6.6)/250
    L = c * gamma_L * L

    # initialize notch approximation law and damage parameter
    notch_approximation_law = SeegerBeste(E, K, n, K_p=3.5)

    # The "load" method is the inverse operation of "stress",
    # i.e., ``L = load(stress(L))`` and ``S = stress(load(stress))``.
    stress = notch_approximation_law.stress(L)
    load = notch_approximation_law.load(stress)
    stress2 = notch_approximation_law.stress(load)

    np.testing.assert_allclose(L, load, rtol=1e-3)
    np.testing.assert_allclose(stress, stress2, rtol=1e-3)



@pytest.mark.parametrize('E, K, n, L', [
    (260e3, 1184, 0.187, pd.Series([100, -200, 100, -250, 200, 100, 200, -200])),
    (100e3, 1500, 0.4, pd.Series([-100, 100, -200])),
    (200e3, 1000, 0.2, pd.Series([100, 10])),
])
def test_seeger_beste_load_secondary_branch(E, K, n, L):
    c = 1.4
    gamma_L = (250+6.6)/250
    L = c * gamma_L * L

    # initialize notch approximation law and damage parameter
    notch_approximation_law = SeegerBeste(E, K, n, K_p=3.5)

    # The "load" method is the inverse operation of "stress",
    # i.e., ``L = load(stress(L))`` and ``S = stress(load(stress))``.
    stress = notch_approximation_law.stress_secondary_branch(L)
    load = notch_approximation_law.load_secondary_branch(stress)
    stress2 = notch_approximation_law.stress_secondary_branch(load)

    np.testing.assert_allclose(L, load, rtol=1e-3)
    np.testing.assert_allclose(stress, stress2, rtol=1e-3)



@pytest.mark.parametrize("L, expected", [
    (7.18, [7.18, 3.50e-5]),
    (179.6, [172.9, 8.73e-4]),
    (359.2, [283.7, 1.86e-3])
]) # Values from FKM NL Guideline p. 135
def test_primary_scalar(L, expected):
    notch_approximation_law = SeegerBeste(E=206e3, K=1184., n=0.187, K_p=3.5)
    result = notch_approximation_law.primary(L)

    assert result.shape == (2, )
    np.testing.assert_allclose(result, expected, rtol=1e-2)


def test_primary_vectorized():
    notch_approximation_law = SeegerBeste(E=206e3, K=1184., n=0.187, K_p=3.5)

    result = notch_approximation_law.primary([7.18, 179.6, 359.2])
    expected = [[7.18, 3.50e-5], [172.9, 8.73e-4], [283.7, 1.86e-3]]

    assert result.shape == (3, 2)
    np.testing.assert_allclose(result, expected, rtol=1e-2)


@pytest.mark.parametrize("L, expected", [
    (359.2, [345.9, 1.75e-3]),
    (362.8, [348.9, 1.77e-3]),
    (718.5, [567.7, 3.72e-3])
])
def test_secondary_scalar(L, expected):
    notch_approximation_law = SeegerBeste(E=206e3, K=1184., n=0.187, K_p=3.5)
    result = notch_approximation_law.secondary(L)

    assert result.shape == (2, )
    np.testing.assert_allclose(result, expected, rtol=1e-2)


def test_secondary_vectorized():
    notch_approximation_law = SeegerBeste(E=206e3, K=1184., n=0.187, K_p=3.5)

    result = notch_approximation_law.secondary([359.2, 362.8, 718.5])
    expected = [[345.9, 1.75e-3], [348.9, 1.77e-3], [567.7, 3.72e-3]]

    assert result.shape == (3, 2)
    np.testing.assert_allclose(result, expected, rtol=1e-2)

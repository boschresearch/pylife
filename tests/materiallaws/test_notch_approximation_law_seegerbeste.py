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


def test_seeger_beste_example_1():
    """ example under 2.7.1, p.74 of FKM nonlinear "Akademisches Beispiel" """

    E = 206e3    # [MPa] Young's modulus
    K = 1184     # [MPa]
    n = 0.187    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    L = pd.Series([100, -200, 100, -250, 200, 0, 200, -200])
    c = 1.4
    gamma_L = (250+6.6)/250
    L = c * gamma_L * L

    # initialize notch approximation law and damage parameter
    #notch_approximation_law = ExtendedNeuber(E, K, n, K_p)
    notch_approximation_law = SeegerBeste(E, K, n, K_p)
#    damage_parameter_type = pylife.strength.damage_parameter.P_RAM()

    maximum_absolute_load = max(abs(L))
#    print(maximum_absolute_load)

    # the FKM example seems to round here, real value is 359.24
    assert np.isclose(maximum_absolute_load, 359.3, rtol=1e-3)

    binned_notch_approximation_law = pylife.materiallaws.notch_approximation_law.Binned(notch_approximation_law, maximum_absolute_load)

    # some rows of PFAD are given in the FKM nonlinear example on p.76
    pd.testing.assert_series_equal(binned_notch_approximation_law._lut_primary_branch.iloc[0], \
        pd.Series([3.59, 0.0017e-2, 3.59]), check_names=False, check_index=False, rtol=1e-3, atol=1e-5)

    pd.testing.assert_series_equal(binned_notch_approximation_law._lut_primary_branch.iloc[1], \
        pd.Series([7.18, 0.0035e-2, 7.18]), check_names=False, check_index=False, rtol=1e-3, atol=1e-5)

    # this row seems off in the FKM nonlinear example, error is as high as 5%
    pd.testing.assert_series_equal(binned_notch_approximation_law._lut_primary_branch.iloc[49], \
        #pd.Series([179.62, 8.73e-4, 172.93]), check_names=False, check_index=False, rtol=10e-2, atol=1e-5)
        pd.Series([179.62, 8.73e-4, 172.93]), check_names=False, check_index=False, rtol=1e-3, atol=1e-5)

    pd.testing.assert_series_equal(binned_notch_approximation_law._lut_primary_branch.iloc[99], \
        pd.Series([359.24, 0.001859, 283.86]), check_names=False, check_index=False, rtol=1e-3, atol=1e-5)

    # matrix AST on page 162, chapter 3.4.1
    pd.testing.assert_frame_equal(
        binned_notch_approximation_law._lut_secondary_branch, expected_matrix_AST_162_seeger_beste, rtol=1e-3, atol=1e-5)

    assert np.isclose(binned_notch_approximation_law._lut_primary_branch.load.max(), maximum_absolute_load)


def test_seeger_beste_example_2():
    """ example under 2.7.2, p.78 of FKM nonlinear, "Welle mit V-Kerbe" """

    E = 206e3    # [MPa] Young's modulus
    K = 2650.5   # [MPa]
    n = 0.187    # [-]
    K_p = 3.5    # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)

    L = 1266.25 * pd.Series([0.3, -0.3, 0.5, -0.5, 0.6, -0.6, 0.3, -0.3, 0.7, -0.7, 0.2, -0.2, 0.6, -0.6, 0.8, -0.8, 0.8, -0.8])

    # initialize notch approximation law and damage parameter
    notch_approximation_law = SeegerBeste(E, K, n, K_p)

    maximum_absolute_load = max(abs(L))

    # the FKM example seems to round here, real value is 359.24
    assert np.isclose(maximum_absolute_load, 1013, rtol=1e-3)

    binned_notch_approximation_law = pylife.materiallaws.notch_approximation_law.Binned(
        notch_approximation_law, maximum_absolute_load)

    # some rows of PFAD are given in the FKM nonlinear example on p.79
    pd.testing.assert_series_equal(binned_notch_approximation_law._lut_primary_branch.iloc[0], \
        pd.Series([10.13, 0.0049e-2, 10.130]), check_names=False, check_index=False, rtol=1e-3, atol=1e-5)

    pd.testing.assert_series_equal(binned_notch_approximation_law._lut_primary_branch.iloc[1], \
        pd.Series([20.26, 0.01e-2, 20.260]), check_names=False, check_index=False, rtol=1e-3, atol=1e-5)

    # this row seems off in the FKM nonlinear example, error is as high as 5%
    pd.testing.assert_series_equal(binned_notch_approximation_law._lut_primary_branch.iloc[99], \
        pd.Series([1013.00, 0.53e-2, 784.890]), check_names=False, check_index=False, rtol=1e-3, atol=1e-5)


    # matrix AST on page 171, chapter 3.4.2
    pd.testing.assert_frame_equal(
        binned_notch_approximation_law._lut_secondary_branch, expected_matrix_AST_171_seeger_beste, rtol=2e-3, atol=1e-5)


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


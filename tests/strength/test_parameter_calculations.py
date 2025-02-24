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

__author__ = "Benjamin Maier"
__maintainer__ = __author__

import pylife.strength.fkm_nonlinear.parameter_calculations
import pytest
import pandas as pd
import numpy as np


@pytest.mark.parametrize(
    "P_A, expected_beta", [
    (1e-10, 6.3613409),
    (1e-5, 4.26489),
    (5e-5, 3.890591),
    (1e-2, 2.326347),
    (0.5, 0),
    (0.75, -0.674489),
    (0.9, -1.281551),
    (1-1e-10, -6.36134),
])
def test_compute_beta_success(P_A, expected_beta):
    beta = pylife.strength.fkm_nonlinear.parameter_calculations.compute_beta(P_A)
    assert np.isclose(beta, expected_beta)


@pytest.mark.parametrize("P_A", [-1.0, 0.0, 1.01, 2.0])
def test_compute_beta_fails(P_A):

    expected_error_message = f"Could not compute the value of beta for P_A={P_A}, " \
        "the optimizer did not find a solution."
    with pytest.raises(RuntimeError, match=expected_error_message):
        pylife.strength.fkm_nonlinear.parameter_calculations.compute_beta(P_A)


@pytest.mark.parametrize(
    "MatGroupFKM, R_m, K_prime, n_prime", [
    ("Steel", 1000., 2058.867, 0.187),
    ("Steel", 1500., 3252.742, 0.187),
    ("Steel", 10., 30.09696, 0.187),
    ("SteelCast", 800., 1565.1487161427392, 0.176),
    ("Al_wrought", 200., 434.411, 0.128)
])
def test_calculate_cyclic_assessment_parameters_Rm(MatGroupFKM, R_m, K_prime, n_prime):

    assessment_parameters = pd.Series({
        "MatGroupFKM": MatGroupFKM,
        "R_m": R_m
    })

    result = pylife.strength.fkm_nonlinear.parameter_calculations.\
        calculate_cyclic_assessment_parameters(assessment_parameters)

    assert np.isclose(result["K_prime"], K_prime, atol=1e-15, rtol=1e-6)
    assert result["n_prime"] == n_prime


@pytest.mark.parametrize(
    "MatGroupFKM, R_m, P_A, P_RAM_Z_WS, P_RAM_D_WS, d_1, d_2", [
    ("Steel", 1000., 1e-5, 819.00, 335.02, -0.302, -0.197),
    ("Steel", 1500., 1e-5, 1039.09, 486.49, -0.302, -0.197),
    ("Steel", 10., 1e-5, 54.86, 4.84, -0.302, -0.197),
    ("SteelCast", 800., 1e-5, 418.63, 143.64, -0.289, -0.189),
    ("Al_wrought", 200., 1e-5, 175.37, 36.6, -0.238, -0.167),
    ("Steel", 1000., 0.5, 819.00/0.71, 335.02/0.71, -0.302, -0.197),
    ("Steel", 1500., 0.5, 1039.09/0.71, 486.49/0.71, -0.302, -0.197),
    ("Steel", 10., 0.5, 54.86/0.71, 4.84/0.71, -0.302, -0.197),
    ("SteelCast", 800., 0.5, 418.63/0.51, 143.64/0.51, -0.289, -0.189),
    ("Al_wrought", 200., 0.5, 175.37/0.61, 36.6/0.61, -0.238, -0.167)
])
def test_calculate_material_woehler_parameters_P_RAM_Rm(MatGroupFKM, R_m, P_A, P_RAM_Z_WS, P_RAM_D_WS, d_1, d_2):
    assessment_parameters = pd.Series({
        "MatGroupFKM": MatGroupFKM,
        "R_m": R_m,
        "P_A": P_A,
    })

    result = pylife.strength.fkm_nonlinear.parameter_calculations.\
        calculate_material_woehler_parameters_P_RAM(assessment_parameters)

    assert np.isclose(result["P_RAM_Z_WS"], P_RAM_Z_WS, atol=1e-15, rtol=1e-3)
    assert np.isclose(result["P_RAM_D_WS"], P_RAM_D_WS, atol=1e-15, rtol=1e-3)
    assert result["d_1"] == d_1
    assert result["d_2"] == d_2


@pytest.mark.parametrize(
    "MatGroupFKM, R_m, P_A, P_RAJ_Z_WS, P_RAJ_D_WS, d_RAJ", [
    ("Steel", 300., 1e-5, 433.67, 0.0897, -0.63),
    ("Steel", 500., 1e-5, 661.32, 0.198, -0.63),
    ("Steel", 3., 1e-5, 9.6642, 7.129e-05, -0.63),
    ("SteelCast", 800., 1e-5, 417.83, 0.1111, -0.66),
    ("Al_wrought", 200., 1e-5, 145.17, 0.00922, -0.61),
    ("Steel", 300., 0.5, 433.67/0.39, 0.0897/0.39, -0.63),
    ("Steel", 500., 0.5, 661.32/0.39, 0.198/0.39, -0.63),
    ("Steel", 3., 0.5, 9.6642/0.39, 7.129e-05/0.39, -0.63),
    ("SteelCast", 800., 0.5, 417.83/0.40, 0.1111/0.40, -0.66),
    ("Al_wrought", 200., 0.5, 145.17/0.36, 0.00922/0.36, -0.61),
])
def test_calculate_material_woehler_parameters_P_RAJ_Rm(MatGroupFKM, R_m, P_A, P_RAJ_Z_WS, P_RAJ_D_WS, d_RAJ):
    assessment_parameters = pd.Series({
        "MatGroupFKM": MatGroupFKM,
        "R_m": R_m,
        "P_A": P_A,
    })

    result = pylife.strength.fkm_nonlinear.parameter_calculations.\
        calculate_material_woehler_parameters_P_RAJ(assessment_parameters)

    assert np.isclose(result["P_RAJ_Z_WS"], P_RAJ_Z_WS, atol=1e-15, rtol=1e-3)
    assert np.isclose(result["P_RAJ_D_WS"], P_RAJ_D_WS, atol=1e-15, rtol=1e-3)
    assert result["d_RAJ"] == d_RAJ


@pytest.mark.parametrize(
    "MatGroupFKM, A_ref, A_sigma, R_m, G, n_st, n_bm, n_P", [
    ("Steel", 500, 50, 300, 1.2, 1.0797751623, 1, 1.0797751623),
    ("Steel", 500, 150, 100, 1.6, 1.0409486145706073, 1.1199957472253255, 1.1658580213991747),
    ("Steel", 500, 700, 120, 1.2, 0.9888469207193641, 1.1261126442652365, 1.1135530206648199),
    ("SteelCast", 500, 50, 300, 1.2, 1.1659144011798317, 1, 1.1659144011798317),
    ("SteelCast", 500, 150, 100, 1.6, 1.0835740181764668, 1.078888669504796, 1.1690557307803737),
    ("SteelCast", 500, 700, 120, 1.2, 0.9778182326161683, 1.1377030648858324, 1.1124668001486624),
    ("Al_wrought", 500, 50, 300, 1.2, 1.1220184543019633, 1, 1.1220184543019633),
    ("Al_wrought", 500, 150, 100, 1.6, 1.0620474909369635, 1, 1.0620474909369635),
    ("Al_wrought", 500, 700, 120, 1.2, 0.9833171148443156, 1, 0.9833171148443156),
    ("Al_wrought", 500, 700, 120, 1.7, 0.9833171148443156, 1.0349367378977414, 1.0176710071559947),
])
def test_calculate_nonlocal_parameters(MatGroupFKM, A_ref, A_sigma, R_m, G, n_st, n_bm, n_P):
    assessment_parameters = pd.Series({
        "MatGroupFKM": MatGroupFKM,
        "A_ref": A_ref,
        "A_sigma": A_sigma,
        "R_m": R_m,
        "G": G,
    })

    result = pylife.strength.fkm_nonlinear.parameter_calculations.\
        calculate_nonlocal_parameters(assessment_parameters)

    assert np.isclose(result["n_st"], n_st, atol=1e-15)
    assert np.isclose(result["n_bm"], n_bm, atol=1e-15)
    assert np.isclose(result["n_P"], n_P, atol=1e-15)

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

import pytest
import numpy as np
import pandas as pd
import numpy.testing as testing
import pylife
import pylife.strength.fkm_nonlinear
from pylife.strength.fkm_nonlinear.constants import FKMNLConstants
import pylife.strength.fkm_nonlinear.parameter_calculations
import pylife.strength.fkm_nonlinear.parameter_calculations as parameter_calculations

def test_material_constants():

    # create table of material constants, typed in independently of the other constants
    data = ([10, 10, 10.03, 10, 101.7],
            [0.826, 0.826, 0.695, 0.826, 0.26],
            [3.33E-5, 3.33E-5, 5.15E-6, 3.33E-5, 5.18E-7],
            [1.55, 1.55, 1.63, 1.55, 2.04],
            [-0.63, -0.63, -0.66, -0.63, -0.61],
            [0.39, 0.39, 0.4, 0.39, 0.36],
            [30, 30, 15, 30, 20],
            [680, 680, 680, 680, 270],
            [0.27, 0.27, 0.25, 0.27, 0.27],
            [0.43, 0.43, 0.42, 0.43, 0.43],
            [400, 400, 400, 400, 133],
            [0.187, 0.187, 0.176, 0.187, 0.128],
            [3.1148, 3.1148, 1.732, 3.1148, 9.12],
            [1033, 1033, 0.847, 1033, 895.9],
            [0.897, 0.897, 0.982, 0.897, 0.742],
            [-1.235, -1.235, -0.181, -1.235, -1.183],
            [0.338, 0.338, 1E20, 0.338, 1E20],
            [0.35, 0.35, 0.35, 0.35, 1],
            [-0.1, -0.1, 0.05, -0.1, -0.04],
            [206000, 206000, 206000, 206000, 70000])

    index = ['aPZ',
              'bPZ',
              'aPD',
              'bPD',
              'd',
              'f25',
              'k_st',
              'RmRef',
              'a_RP',
              'b_RP',
              'RmNmin',
              'n_apo',
              'a_sig',
              'a_eps',
              'b_sig',
              'b_eps',
              'eps_gr',
              'a_M',
              'b_M',
              'E']

    columns = ['CaseHard_Steel', 'Stainless_Steel', 'SteelCast', 'Steel', 'Al_wrought']

    df_reference = pd.DataFrame(data=data, index=index, columns=columns)
    df_to_test = FKMNLConstants().to_pandas()

    # check if previously defined constants match the constants defined in FKMNLConstants()
    pd.testing.assert_series_equal(df_to_test.loc["E",:], df_reference.loc["E",["Steel", "SteelCast", "Al_wrought"]])
    pd.testing.assert_series_equal(df_to_test.loc["a_sigma",:], df_reference.loc["a_sig",["Steel", "SteelCast", "Al_wrought"]], check_names=False)
    pd.testing.assert_series_equal(df_to_test.loc["a_epsilon",:], df_reference.loc["a_eps",["Steel", "SteelCast", "Al_wrought"]], check_names=False)
    pd.testing.assert_series_equal(df_to_test.loc["b_sigma",:], df_reference.loc["b_sig",["Steel", "SteelCast", "Al_wrought"]], check_names=False)
    pd.testing.assert_series_equal(df_to_test.loc["b_epsilon",:], df_reference.loc["b_eps",["Steel", "SteelCast", "Al_wrought"]], check_names=False)
    pd.testing.assert_series_equal(df_to_test.loc["epsilon_grenz",["Steel"]], df_reference.loc["eps_gr",["Steel"]], check_names=False)

    # aPZ and bPZ is different for P_RAM and P_RAJ! Compare for P_RAJ.
    pd.testing.assert_series_equal(df_to_test.loc["a_PZ_RAJ",:], df_reference.loc["aPZ",["Steel", "SteelCast", "Al_wrought"]], check_names=False)
    pd.testing.assert_series_equal(df_to_test.loc["b_PZ_RAJ",:], df_reference.loc["bPZ",["Steel", "SteelCast", "Al_wrought"]], check_names=False)
    pd.testing.assert_series_equal(df_to_test.loc["a_PD_RAJ",:], df_reference.loc["aPD",["Steel", "SteelCast", "Al_wrought"]], check_names=False)
    pd.testing.assert_series_equal(df_to_test.loc["b_PD_RAJ",:], df_reference.loc["bPD",["Steel", "SteelCast", "Al_wrought"]], check_names=False)

    # d or d_1 is different for P_RAM and P_RAJ!, Here, consider d for P_RAJ
    pd.testing.assert_series_equal(df_to_test.loc["d_RAJ",:], df_reference.loc["d",["Steel", "SteelCast", "Al_wrought"]], check_names=False)

    pd.testing.assert_series_equal(df_to_test.loc["k_st",:], df_reference.loc["k_st",["Steel", "SteelCast", "Al_wrought"]], check_names=False)
    pd.testing.assert_series_equal(df_to_test.loc["a_RP",:], df_reference.loc["a_RP",["Steel", "SteelCast", "Al_wrought"]], check_names=False)
    pd.testing.assert_series_equal(df_to_test.loc["b_RP",:], df_reference.loc["b_RP",["Steel", "SteelCast", "Al_wrought"]], check_names=False)
    pd.testing.assert_series_equal(df_to_test.loc["R_m_N_min",:], df_reference.loc["RmNmin",["Steel", "SteelCast", "Al_wrought"]], check_names=False)
    pd.testing.assert_series_equal(df_to_test.loc["a_M",:], df_reference.loc["a_M",["Steel", "SteelCast", "Al_wrought"]], check_names=False)
    pd.testing.assert_series_equal(df_to_test.loc["b_M",:], df_reference.loc["b_M",["Steel", "SteelCast", "Al_wrought"]], check_names=False)


def test_material_constants_existing_material():
    assessment_parameters = {
        "MatGroupFKM": "Al_wrought",
    }
    constants = FKMNLConstants().for_material_group(assessment_parameters=assessment_parameters)
    assert constants["E"] == 70e3


def test_material_constants_unknown_material_fails():
    assessment_parameters = {
        "MatGroupFKM": "new_material",
    }
    with pytest.raises(KeyError, match="new_material"):
        FKMNLConstants().for_material_group(assessment_parameters=assessment_parameters)


def test_material_constants_new_material():

    assert "new_material" not in FKMNLConstants()

    FKMNLConstants().add_custom_material("new_material", {"E": 99e3, "n_prime": 0.123})

    assessment_parameters = {"MatGroupFKM": "new_material"}

    # retrieve constants
    constants = FKMNLConstants().for_material_group(assessment_parameters=assessment_parameters)

    assert constants["E"] == 99e3
    assert constants["n_prime"] == 0.123

    # constants are now also updated directly in the module
    assert "new_material" in FKMNLConstants()
    assert FKMNLConstants()["new_material"]["E"] == 99e3


def test_material_constants_new_material_conflict():
    with pytest.raises(ValueError, match="Material `Al_wrought` already exists."):
        FKMNLConstants().add_custom_material("Al_wrought", {"E": 99e3, "n_prime": 0.123})


def test_computation_functions_1():
    """Example 2.10.2, "Welle mit V-Kerbe", p.138 """

    assessment_parameters = pd.Series({
        "MatGroupFKM":        "Steel",
        "FinishingFKM":        "none",
        "R_m":                 1251,
        "R_z":                  200,
        "P_A":             0.000072,
        "P_L":             0.000072,
        "c":                    1.0,
        "A_sigma":             4.71,
        "A_ref":                500,
        "G":               0.133333,
        "s_L":                   10,
        "K_p":                  3.5,
        "x_Einsatz":             50,
        "r":                    0.5,
        "LSD_s":                  1,
    })

    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_cyclic_assessment_parameters(assessment_parameters)

    # calculate the parameters for the material woehler curve
    # (for both P_RAM and P_RAJ, the variable names do not interfere)
    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_material_woehler_parameters_P_RAM(assessment_parameters)
    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_material_woehler_parameters_P_RAJ(assessment_parameters)
    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_nonlocal_parameters(assessment_parameters)

    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_roughness_parameter(assessment_parameters)
    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_failure_probability_factor_P_RAM(assessment_parameters)
    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_failure_probability_factor_P_RAJ(assessment_parameters)
    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_component_woehler_parameters_P_RAM(assessment_parameters)
    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_component_woehler_parameters_P_RAJ(assessment_parameters)

    assessment_parameters_reference = pd.Series({
        "MatGroupFKM": "Steel",
        "FinishingFKM": "none",
        "R_m": 1251,
        "R_z": 200,
        "P_A": 0.000072,
        "P_L": 0.000072,
        "c": 1.0,
        "A_sigma": 4.71,
        "A_ref": 500,
        "G": 0.133333,
        "s_L": 10,
        "K_p": 3.5,
        "x_Einsatz": 50,
        "r": 0.5,
        "LSD_s": 1,
        "n_prime": 0.187,
        "E": 206000.0,
        "K_prime": 2650.509115,
        "notes": "P_A not 0.5 (but 7.2e-05): scale P_RAM woehler curve by f_2.5% = 0.71.\n" \
                "P_A not 0.5 (but 7.2e-05): scale P_RAJ woehler curve by f_2.5% = 0.39.\n",
        "P_RAM_Z_WS": 934.067766,
        "P_RAM_D_WS": 411.669722,
        "d_1": -0.302,
        "d_2": -0.197,
        "P_RAJ_Z_WS": 1410.5846217,
        "P_RAJ_D_WS": 0.820838,
        "d_RAJ": -0.63,
        "n_st": 1.168239,
        "n_bm_": 0.495764968,
        "n_bm": 1.0,
        "n_P": 1.168239,
        "K_RP": 0.745648,
        "beta": 3.801,
        "gamma_M_RAM": 1.211,
        "gamma_M_RAJ": 1.45,
        "f_RAM": 1.39,
        "P_RAM_Z": 671,
        "P_RAM_D": 296,
        "f_RAJ": 1.910894,
        "P_RAJ_Z": 738.18061,
        "P_RAJ_D_0": 0.429557,
        "P_RAJ_D": 0.429557
    })

    print(assessment_parameters["notes"])

    pd.testing.assert_series_equal(assessment_parameters_reference, assessment_parameters, rtol=5e-3)


def test_computation_functions_2():
    """Example 2.10.2, "Welle mit V-Kerbe", p.138 """

    assessment_parameters = pd.Series({
        "MatGroupFKM":        "Steel",
        "FinishingFKM":        "none",
        "R_m":                  600,
        "R_z":                  250,
        "P_A":             0.000072,
        "P_L":                  2.5,
        "c":                    1.4,
        "A_sigma":            339.4,
        "A_ref":                500,
        "G":               0.133333,
        "s_L":                   10,
        "K_p":                  3.5,
        "x_Einsatz":           3000,
        "r":                     15,
        "LSD_s":                  1,
    })

    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_cyclic_assessment_parameters(assessment_parameters)

    # calculate the parameters for the material woehler curve
    # (for both P_RAM and P_RAJ, the variable names do not interfere)
    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_material_woehler_parameters_P_RAM(assessment_parameters)
    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_material_woehler_parameters_P_RAJ(assessment_parameters)
    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_nonlocal_parameters(assessment_parameters)

    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_roughness_parameter(assessment_parameters)
    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_failure_probability_factor_P_RAM(assessment_parameters)
    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_component_woehler_parameters_P_RAM(assessment_parameters)
    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_failure_probability_factor_P_RAJ(assessment_parameters)
    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_component_woehler_parameters_P_RAJ(assessment_parameters)

    assessment_parameters_reference = pd.Series({
        "MatGroupFKM": "Steel",
        "FinishingFKM": "none",
        "R_m": 600,
        "R_z": 250,
        "P_A": 0.000072,
        "P_L": 2.5,
        "c": 1.4,
        "A_sigma": 339.4,
        "A_ref": 500,
        "G": 0.133333,
        "s_L": 10,
        "K_p": 3.5,
        "x_Einsatz": 3000,
        "r": 15,
        "LSD_s": 1,
        "n_prime": 0.187,
        "E": 206000.0,
        "K_prime": 1184.470952,
        "notes": "P_A not 0.5 (but 7.2e-05): scale P_RAM woehler curve by f_2.5% = 0.71.\n" \
                "P_A not 0.5 (but 7.2e-05): scale P_RAJ woehler curve by f_2.5% = 0.39.\n",
        "P_RAM_Z_WS": 606.82453,
        "P_RAM_D_WS": 209.397432,
        "d_1": -0.302,
        "d_2": -0.197,
        "P_RAJ_Z_WS": 768.8073,
        "P_RAJ_D_WS": 0.262811,
        "d_RAJ": -0.63,
        "n_st": 1.012998,
        "n_bm_": 0.719782499,
        "n_bm": 1.0,
        "n_P": 1.012998,
        "K_RP": 0.8531,
        "beta": 3.80119,
        "gamma_M_RAM": 1.211,
        "f_RAM": 1.40174,
        "P_RAM_Z": 432.9077,
        "P_RAM_D": 149.3838,
        "gamma_M_RAJ": 1.45,
        "f_RAJ": 1.9414,
        "P_RAJ_Z": 395.99,
        "P_RAJ_D_0": 0.135361,
        "P_RAJ_D": 0.135361,
    })

    print(assessment_parameters["notes"])

    pd.testing.assert_series_equal(assessment_parameters_reference, assessment_parameters, rtol=1e-3)


@pytest.mark.parametrize("P_A,expected_gamma_M_RAM,expected_gamma_M_RAJ", [
    (0.5, 1, 1),
    (2.3e-1, 1.1, 1.2),
    (1e-3, 1.1, 1.2),
    (7.2e-5, 1.2, 1.45),
    (1e-5, 1.3, 1.7),
    (1e-6, 1.393, 1.901),
    (1e-7, 1.488, 2.162)
])
def test_calculate_component_woehler_parameters_P_RAM(P_A, expected_gamma_M_RAM, expected_gamma_M_RAJ):

    assessment_parameters = pd.Series({
        "MatGroupFKM": "Steel",
        "K_p": 3,
        "n_P": 1.1,
        "K_RP": 0.8,
        "P_A": P_A,
        "P_RAM_Z_WS": 1000,
        "P_RAM_D_WS": 500,
        "P_RAJ_Z_WS": 1000,
        "P_RAJ_D_WS": 500,
    })

    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_failure_probability_factor_P_RAM(assessment_parameters)
    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_component_woehler_parameters_P_RAM(assessment_parameters)
    assert np.isclose(assessment_parameters["gamma_M_RAM"], expected_gamma_M_RAM, rtol=0.05)
    assert np.isclose(assessment_parameters["f_RAM"], assessment_parameters["gamma_M_RAM"] / (1.1*0.8), rtol=0.05)

    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_failure_probability_factor_P_RAJ(assessment_parameters)
    assessment_parameters = pylife.strength.fkm_nonlinear.parameter_calculations.calculate_component_woehler_parameters_P_RAJ(assessment_parameters)
    assert np.isclose(assessment_parameters["gamma_M_RAJ"], expected_gamma_M_RAJ, rtol=0.05)
    assert np.isclose(assessment_parameters["f_RAJ"], assessment_parameters["gamma_M_RAJ"] / (1.1*1.1*0.8*0.8), rtol=0.05)


@pytest.mark.parametrize(
    "K_p", [1.0-1e-5, 0.0, 0.9, -1.0]
)
def test_Kp_range(K_p):
    assessment_parameters = pd.Series({
        "K_p": K_p,
        "G": 1,
    })

    load_sequence = pd.Series([1,2])

    with pytest.raises(AssertionError, match="K_p should be at least 1"):

        result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard\
            .perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
            calculate_P_RAM=True, calculate_P_RAJ=True)


@pytest.mark.parametrize(
    "G", ["1", "2", pd.DataFrame([3.0, 3.0]), pd.Series([4], index=pd.MultiIndex.from_arrays([[0]]))]
)
def test_G_wrong_format(G):
    assessment_parameters = pd.Series({
        "K_p": 1.1,
        "G": G,
    })

    load_sequence = pd.Series([1,2])

    with pytest.raises(AssertionError, match="stress gradient G is in a wrong format"):

        result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard\
            .perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
            calculate_P_RAM=True, calculate_P_RAJ=True)


def test_assessment_single_point_1():

    # this is example 2.7.1, 2.10.1 "akademisches Beispiel" in the FKM nonlinear document
    assessment_parameters = pd.Series({
        'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
        'FinishingFKM': 'none',  # type of surface finisihing
        'R_m': 600,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
        #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
        'R_z': 250,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

        'P_A': 7.2e-5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
        # beta: 0.5,             # damage index, specify this as an alternative to P_A

        'P_L': 2.5,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlichkeit der Lastfolge)
        'c':   1.4,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
        'A_sigma': 339.4,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
        'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
        'G': 2/15,               # [mm^-1] (de: bezogener Spannungsgradient)
        's_L': 10,               # [MPa] standard deviation of Gaussian distribution
        'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
        'x_Einsatz': 3000,       # [-] (de: Einsatzdurchlaufzahl)
        'r': 15,                 # [mm] radius (?)
    })

    # load sequence
    load_sequence = pd.Series([100, -200, 100, -250, 200, 0, 200, -200])  # [N]

    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=True, calculate_P_RAJ=True)

    assert result["P_RAM_is_life_infinite"] == False
    assert np.isclose(result["P_RAM_lifetime_n_cycles"], 13935)   # originially 146818, but with rounded gamma_M
    assert np.isclose(result["P_RAM_lifetime_n_times_load_sequence"], 3483, rtol=1e-2)

    assert result["P_RAJ_is_life_infinite"] == False
    assert np.isclose(result["P_RAJ_lifetime_n_cycles"], 54868, rtol=1e-2)      # 31000 in FKM document (probably rounded), calculation with given (wrong) formulas of document: 30948
    assert np.isclose(result["P_RAJ_lifetime_n_times_load_sequence"], 7737*54868/30948, rtol=1e-2)      # calculation with given (wrong) formulas of document: 7737


def test_assessment_single_point_2():

    # this is the example 2.7.2, 2.10.2 "Welle mit V-Kerbe" in the FKM nonlinear document
    assessment_parameters = pd.Series({
        'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
        'FinishingFKM': 'none',  # type of surface finisihing
        'R_m': 1251,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
        #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
        'R_z': 200,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

        'P_A': 7.2e-5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
        # beta: 0.5,             # damage index, specify this as an alternative to P_A

        'P_L': 50,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlilchkeit der Lastfolge)
        'c':   1.0,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
        'A_sigma': 4.71,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
        'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
        'G': 2/0.5,               # [mm^-1] (de: bezogener Spannungsgradient)
        'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
        'x_Einsatz': 50,       # [-] (de: Einsatzdurchlaufzahl)
        'r': 0.5,                 # [mm] radius (?)
    })

    # load sequence
    load_sequence = 1266.25 * pd.Series([0.3, -0.3, 0.5, -0.5, 0.6, -0.6, 0.3, -0.3, 0.7, -0.7, 0.2, -0.2, 0.6, -0.6, 0.8, -0.8, 0.8, -0.8])  # [N]

    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=True, calculate_P_RAJ=True)


    assert result["P_RAM_is_life_infinite"] == False
    assert np.isclose(result["P_RAM_lifetime_n_cycles"], 624.6530563, rtol=1e-2)    # originally 642.96
    assert np.isclose(result["P_RAM_lifetime_n_times_load_sequence"], 69, rtol=1e-2)

    assert result["P_RAJ_is_life_infinite"] == False
    assert np.isclose(result["P_RAJ_lifetime_n_cycles"], 1500, rtol=2e-2)      # 722 in FKM document, calculation with given (wrong) formulas of document: 722
    assert np.isclose(result["P_RAJ_lifetime_n_times_load_sequence"], 80/722*1500, rtol=2e-2)       # 80 in FKM document


def test_assessment_single_point_3_infinite_life():

    # this is example 2.7.1, 2.10.1 "akademisches Beispiel" in the FKM nonlinear document
    # but with a larger R_m and lower load to make infinite life
    assessment_parameters = pd.Series({
        'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
        'FinishingFKM': 'none',  # type of surface finisihing
        'R_m': 1000,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
        #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
        'R_z': 50,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

        'P_A': 7.2e-5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
        # beta: 0.5,             # damage index, specify this as an alternative to P_A

        'P_L': 2.5,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlilchkeit der Lastfolge)
        'c':   1.4,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
        'A_sigma': 339.4,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
        'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
        'G': 2/15,               # [mm^-1] (de: bezogener Spannungsgradient)
        's_L': 10,               # [MPa] standard deviation of Gaussian distribution
        'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
        'x_Einsatz': 3000,       # [-] (de: Einsatzdurchlaufzahl)
        'r': 15,                 # [mm] radius (?)
    })

    # load sequence
    load_sequence = 0.5*pd.Series([100, -200, 100, -250, 200, 0, 200, -200])  # [N]

    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=True, calculate_P_RAJ=True)


    assert result["P_RAM_is_life_infinite"] == True
    assert result["P_RAJ_is_life_infinite"] == True


def test_assessment_multiple_points_p_ram():

    # this is example 2.7.1, 2.10.1 "akademisches Beispiel" in the FKM nonlinear document,
    # with an additional second points
    assessment_parameters = pd.Series({
        'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
        'FinishingFKM': 'none',  # type of surface finisihing
        'R_m': 600,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
        #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
        'R_z': 250,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

        'P_A': 7.2e-5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
        # beta: 0.5,             # damage index, specify this as an alternative to P_A

        'P_L': 2.5,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlilchkeit der Lastfolge)
        'c':   1.4,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
        'A_sigma': 339.4,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
        'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
        'G': 2/15,               # [mm^-1] (de: bezogener Spannungsgradient)
        's_L': 10,               # [MPa] standard deviation of Gaussian distribution
        'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
        'x_Einsatz': 3000,       # [-] (de: Einsatzdurchlaufzahl)
        'r': 15,                 # [mm] radius (?)
        "max_load_independently_for_nodes": True,     # boolean, optional, default is False, whether the load scaling as part of the failure probability is done for all nodes at once or per node.
    })

    # generate a load sequence for three assessment points
    load_sequence_0 = pd.Series([100, -200, 100, -250, 200, 0, 200, -200])  # [N]
    load_sequence_1 = load_sequence_0 * 1.2
    load_sequence_2 = load_sequence_0 * 0.2     # last has infinite life

    index = pd.MultiIndex.from_product([range(len(load_sequence_0)), [0,1,2]], names=["load_step", "node_id"])

    load_sequence = pd.Series(index=index)
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==0] = load_sequence_0.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==1] = load_sequence_1.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==2] = load_sequence_2.to_numpy()

    # perform assessment of all points at once
    result_multiple = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                                      calculate_P_RAM=True, calculate_P_RAJ=False)


    # perform assessment again for all points, one by one
    result = [{} for _ in range(3)]
    result[0] = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence_0,
                                                                              calculate_P_RAM=True, calculate_P_RAJ=False)

    result[1] = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence_1,
                                                                              calculate_P_RAM=True, calculate_P_RAJ=False)

    result[2] = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence_2,
                                                                              calculate_P_RAM=True, calculate_P_RAJ=False)

    # assert that the results are equal
    for i in range(3):
        assert result[i]["P_RAM_is_life_infinite"] == result_multiple["P_RAM_is_life_infinite"][i]
        assert np.isclose(result[i]["P_RAM_lifetime_n_cycles"], result_multiple["P_RAM_lifetime_n_cycles"][i])
        assert np.isclose(result[i]["P_RAM_lifetime_n_times_load_sequence"], result_multiple["P_RAM_lifetime_n_times_load_sequence"][i])


def test_assessment_multiple_points_p_raj():

    # this is example 2.7.1, 2.10.1 "akademisches Beispiel" in the FKM nonlinear document,
    # with an additional second points
    assessment_parameters = pd.Series({
        'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
        'FinishingFKM': 'none',  # type of surface finisihing
        'R_m': 500,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
        #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
        'R_z': 250,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

        'P_A': 7.2e-5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
        # beta: 0.5,             # damage index, specify this as an alternative to P_A

        'P_L': 2.5,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlilchkeit der Lastfolge)
        'c':   1.4,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
        'A_sigma': 339.4,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
        'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
        'G': 2/15,               # [mm^-1] (de: bezogener Spannungsgradient)
        's_L': 10,               # [MPa] standard deviation of Gaussian distribution
        'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
        'x_Einsatz': 3000,       # [-] (de: Einsatzdurchlaufzahl)
        'r': 15,                 # [mm] radius (?)
        'n_bins': 1000,          # number of bins or classes for P_RAJ computation (default: 200, a larger value gives more accurate results but longer runtimes)
        "max_load_independently_for_nodes": True,     # boolean, optional, default is False, whether the load scaling as part of the failure probability is done for all nodes at once or per node.
    })

    # generate a load sequence for three assessment points
    load_sequence_0 = pd.Series([100, -200, 100, -250, 200, 0, 200, -200])  # [N]
    load_sequence_1 = load_sequence_0 * 1.0
    load_sequence_2 = load_sequence_0 * 1.0         # use the same load sequence, because P_RAJ_klass_max will be different otherwise

    index = pd.MultiIndex.from_product([range(len(load_sequence_0)), [0,1,2]], names=["load_step", "node_id"])

    load_sequence = pd.Series(index=index)
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==0] = load_sequence_0.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==1] = load_sequence_1.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==2] = load_sequence_2.to_numpy()

    # perform assessment of all points at once
    result_multiple = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                                      calculate_P_RAM=False, calculate_P_RAJ=True)

    # perform assessment again for all points, one by one
    result = [{} for _ in range(3)]
    result[0] = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence_0,
                                                                              calculate_P_RAM=False, calculate_P_RAJ=True)

    result[1] = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence_1,
                                                                              calculate_P_RAM=False, calculate_P_RAJ=True)

    result[2] = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence_2,
                                                                              calculate_P_RAM=False, calculate_P_RAJ=True)

    # assert that the results are equal
    for i in range(3):
        assert result[i]["P_RAJ_is_life_infinite"] == result_multiple["P_RAJ_is_life_infinite"][i]
        assert np.isclose(result[i]["P_RAJ_lifetime_n_cycles"], result_multiple["P_RAJ_lifetime_n_cycles"][i])
        assert np.isclose(result[i]["P_RAJ_lifetime_n_times_load_sequence"], result_multiple["P_RAJ_lifetime_n_times_load_sequence"][i])
        assert np.isclose(result[i]["P_RAJ_miner_lifetime_n_times_load_sequence"], result_multiple["P_RAJ_miner_lifetime_n_times_load_sequence"][i])


def test_assessment_multiple_points_p_ram_with_spatially_varying_gradient():

    # this is example 2.7.1, 2.10.1 "akademisches Beispiel" in the FKM nonlinear document,
    # with an additional second points
    assessment_parameters = pd.Series({
        'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
        'FinishingFKM': 'none',  # type of surface finisihing
        'R_m': 600,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
        #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
        'R_z': 250,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

        'P_A': 7.2e-5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
        # beta: 0.5,             # damage index, specify this as an alternative to P_A

        'P_L': 2.5,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlilchkeit der Lastfolge)
        'c':   1.4,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
        'A_sigma': 339.4,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
        'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
        'G': 2/15,               # [mm^-1] (de: bezogener Spannungsgradient)
        's_L': 10,               # [MPa] standard deviation of Gaussian distribution
        'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
        'x_Einsatz': 3000,       # [-] (de: Einsatzdurchlaufzahl)
        'r': 15,                 # [mm] radius (?)
        "max_load_independently_for_nodes": True,     # boolean, optional, default is False, whether the load scaling as part of the failure probability is done for all nodes at once or per node.
    })

    # generate a load sequence for three assessment points
    load_sequence_0 = pd.Series([100, -200, 100, -250, 200, 0, 200, -200])  # [N]
    load_sequence_1 = load_sequence_0 * 1.2
    load_sequence_2 = load_sequence_0 * 0.2     # last has infinite life

    index = pd.MultiIndex.from_product([range(len(load_sequence_0)), [0,1,2]], names=["load_step", "node_id"])

    G = pd.Series(index=pd.Index([5,8,9], name="node_id_anynamehere"), data=[0.2, 0.8, 0.1])
    assessment_parameters.G = G

    load_sequence = pd.Series(index=index)
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==0] = load_sequence_0.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==1] = load_sequence_1.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==2] = load_sequence_2.to_numpy()

    # perform assessment of all points at once
    result_multiple = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                                      calculate_P_RAM=True, calculate_P_RAJ=False)


    # perform assessment again for all points, one by one
    result = [{} for _ in range(3)]
    assessment_parameters.G = 0.2
    result[0] = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence_0,
                                                                              calculate_P_RAM=True, calculate_P_RAJ=False)

    assessment_parameters.G = 0.8
    result[1] = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence_1,
                                                                              calculate_P_RAM=True, calculate_P_RAJ=False)

    assessment_parameters.G = 0.1
    result[2] = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence_2,
                                                                              calculate_P_RAM=True, calculate_P_RAJ=False)

    # assert that the results are equal
    for i in range(3):
        assert result[i]["P_RAM_is_life_infinite"] == result_multiple["P_RAM_is_life_infinite"][i]
        assert np.isclose(result[i]["P_RAM_lifetime_n_cycles"], result_multiple["P_RAM_lifetime_n_cycles"][i])
        assert np.isclose(result[i]["P_RAM_lifetime_n_times_load_sequence"], result_multiple["P_RAM_lifetime_n_times_load_sequence"][i])


def test_assessment_multiple_points_p_raj_with_spatially_varying_gradient():

    # this is example 2.7.1, 2.10.1 "akademisches Beispiel" in the FKM nonlinear document,
    # with an additional second points
    assessment_parameters = pd.Series({
        'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
        'FinishingFKM': 'none',  # type of surface finisihing
        'R_m': 500,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
        #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
        'R_z': 250,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

        'P_A': 7.2e-5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
        # beta: 0.5,             # damage index, specify this as an alternative to P_A

        'P_L': 2.5,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlilchkeit der Lastfolge)
        'c':   1.4,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
        'A_sigma': 339.4,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
        'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
        'G': 2/15,               # [mm^-1] (de: bezogener Spannungsgradient)
        's_L': 10,               # [MPa] standard deviation of Gaussian distribution
        'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
        'x_Einsatz': 3000,       # [-] (de: Einsatzdurchlaufzahl)
        'r': 15,                 # [mm] radius (?)
        'n_bins': 1000,          # number of bins or classes for P_RAJ computation (default: 200, a larger value gives more accurate results but longer runtimes)
        "max_load_independently_for_nodes": True,     # boolean, optional, default is False, whether the load scaling as part of the failure probability is done for all nodes at once or per node.
    })

    # generate a load sequence for three assessment points
    load_sequence_0 = pd.Series([100, -200, 100, -250, 200, 0, 200, -200])  # [N]
    load_sequence_1 = load_sequence_0 * 1.0
    load_sequence_2 = load_sequence_0 * 1.0         # use the same load sequence, because P_RAJ_klass_max will be different otherwise

    index = pd.MultiIndex.from_product([range(len(load_sequence_0)), [0,1,2]], names=["load_step", "node_id"])

    G = pd.Series(index=pd.Index([5,8,9], name="node_id_anynamehere"), data=[0.2, 0.8, 0.1])
    assessment_parameters.G = G

    load_sequence = pd.Series(index=index)
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==0] = load_sequence_0.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==1] = load_sequence_1.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==2] = load_sequence_2.to_numpy()

    # perform assessment of all points at once
    result_multiple = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                                      calculate_P_RAM=False, calculate_P_RAJ=True)

    # perform assessment again for all points, one by one
    result = [{} for _ in range(3)]
    assessment_parameters.G = 0.2
    result[0] = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence_0,
                                                                              calculate_P_RAM=False, calculate_P_RAJ=True)

    assessment_parameters.G = 0.8
    result[1] = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence_1,
                                                                              calculate_P_RAM=False, calculate_P_RAJ=True)

    assessment_parameters.G = 0.1
    result[2] = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence_2,
                                                                              calculate_P_RAM=False, calculate_P_RAJ=True)

    # assert that the results are equal
    for i in range(3):
        assert result[i]["P_RAJ_is_life_infinite"] == result_multiple["P_RAJ_is_life_infinite"][i]
        assert np.isclose(result[i]["P_RAJ_lifetime_n_cycles"], result_multiple["P_RAJ_lifetime_n_cycles"][i])
        assert np.isclose(result[i]["P_RAJ_lifetime_n_times_load_sequence"], result_multiple["P_RAJ_lifetime_n_times_load_sequence"][i])
        assert np.isclose(result[i]["P_RAJ_miner_lifetime_n_times_load_sequence"], result_multiple["P_RAJ_miner_lifetime_n_times_load_sequence"][i])


@pytest.mark.parametrize(
    'load_sequence', [
    (pd.Series([200, 600, 1000, 200, 60, 1200])),
    (pd.Series([200, 600, 1000, 200, 60, 500, 1500, 700, 1200, -20])),
    (pd.Series([200, 600, 1000, 60, 1500])),
    (pd.Series([200, 600, 1000, 60, 1500, 200, 80, 400, 1500, 700, 200])),
    (pd.Series([200, 600, 1000, 200, 60, 500, 100, 700, 1260, 1500, 800, 900, 500, 900, 700, 1200])),
    (pd.Series([100, -200, 100, -250, 200, 0, 200, -200])),
    (pd.Series([100, -100, 100, -200, -100, -200, 200, 0, 200, -200])),
    (pd.Series([0.3, -0.3, 0.5, -0.5, 0.6, -0.6, 0.3, -0.3, 0.7, -0.7, 0.2, -0.2, 0.6, -0.6, 0.8, -0.8, 0.8, -0.8])),
])
def test_comparison_P_RAM_woehler(load_sequence):

    assessment_parameters = pd.Series({
        'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
        'FinishingFKM': 'none',  # type of surface finisihing
        'R_m': 600,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
        #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
        'R_z': 250,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

        'P_A': 7.2e-5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
        # beta: 0.5,             # damage index, specify this as an alternative to P_A

        'P_L': 2.5,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlichkeit der Lastfolge)
        'c':   1.4,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
        'A_sigma': 339.4,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
        'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
        'G': 2/15,               # [mm^-1] (de: bezogener Spannungsgradient)
        's_L': 10,               # [MPa] standard deviation of Gaussian distribution
        'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
        'x_Einsatz': 3000,       # [-] (de: Einsatzdurchlaufzahl)
        'r': 15,                 # [mm] radius (?)
    })

    # load sequence
    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=True, calculate_P_RAJ=False)

    # compare assessment result with manual Woehler curve assessment (actually the same as in the algorithm)
    component_woehler_curve_P_RAM = result["P_RAM_woehler_curve"]
    collective = result["P_RAM_collective"]

    # first HCM run
    D1 = 0
    for index, row in collective.loc[collective["run_index"] == 1].iterrows():
        P_RAM = row["P_RAM"]

        if P_RAM > component_woehler_curve_P_RAM.P_RAM_D:
            N = component_woehler_curve_P_RAM.calc_N(P_RAM)
        else:
            N = 1e3 * (P_RAM / component_woehler_curve_P_RAM.P_RAM_Z) ** (1/component_woehler_curve_P_RAM.d_2)

        if row["is_closed_hysteresis"]:
            D1 += 1/N
        else:
            D1 += 0.5/N

    # second HCM run
    D2 = 0
    for index, row in collective.loc[collective["run_index"] == 2].iterrows():
        P_RAM = row["P_RAM"]

        if P_RAM > component_woehler_curve_P_RAM.P_RAM_D:
            N = component_woehler_curve_P_RAM.calc_N(P_RAM)
        else:
            N = 1e3 * (P_RAM / component_woehler_curve_P_RAM.P_RAM_Z) ** (1/component_woehler_curve_P_RAM.d_2)

        if row["is_closed_hysteresis"]:
            D2 += 1/N
        else:
            D2 += 0.5/N


    n2 = len(collective.loc[collective["run_index"] == 2])
    N = (1 + (1-D1)/D2) * n2
    print(f"D1: {D1}, D2: {D2}, n2: {n2}, N: {N}",  result["P_RAM_lifetime_n_cycles"])

    assert np.isclose(N, result["P_RAM_lifetime_n_cycles"])


@pytest.mark.parametrize(
    'load_sequence', [
    (pd.Series([200, 600, 1000, 200, 60, 1200])),
    (pd.Series([200, 600, 1000, 200, 60, 500, 1500, 700, 1200, -20])),
    (pd.Series([200, 600, 1000, 60, 1500])),
    (pd.Series([200, 600, 1000, 60, 1500, 200, 80, 400, 1500, 700, 200])),
    (pd.Series([200, 600, 1000, 200, 60, 500, 100, 700, 1260, 1500, 800, 900, 500, 900, 700, 1200])),
    (pd.Series([100, -200, 100, -250, 200, 0, 200, -200])),
    (pd.Series([100, -100, 100, -200, -100, -200, 200, 0, 200, -200])),
])
def test_comparison_P_RAJ_woehler(load_sequence):

    assessment_parameters = pd.Series({
        'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
        'FinishingFKM': 'none',  # type of surface finisihing
        'R_m': 600,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
        #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
        'R_z': 250,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

        'P_A': 7.2e-5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
        # beta: 0.5,             # damage index, specify this as an alternative to P_A

        'P_L': 2.5,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlichkeit der Lastfolge)
        'c':   1.4,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
        'A_sigma': 339.4,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
        'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
        'G': 2/15,               # [mm^-1] (de: bezogener Spannungsgradient)
        's_L': 10,               # [MPa] standard deviation of Gaussian distribution
        'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
        'x_Einsatz': 3000,       # [-] (de: Einsatzdurchlaufzahl)
        'r': 15,                 # [mm] radius (?)
    })

    # load sequence
    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=False, calculate_P_RAJ=True)

    # compare assessment result with manual Woehler curve assessment (actually the one from the P_RAM algorithm)
    component_woehler_curve_P_RAJ = result["P_RAJ_woehler_curve"]
    collective = result["P_RAJ_collective"]

    # first HCM run
    D1 = 0
    for index, row in collective.loc[collective["run_index"] == 1].iterrows():
        P_RAJ = row["P_RAJ"]
        print(f"P_RAJ: {P_RAJ}, P_RAJ_D: {component_woehler_curve_P_RAJ.P_RAJ_D.values[0]}")

        if P_RAJ > component_woehler_curve_P_RAJ.P_RAJ_D.values[0]:
            N = component_woehler_curve_P_RAJ.calc_N(P_RAJ)

            if row["is_closed_hysteresis"]:
                D1 += 1/N
            else:
                D1 += 0.5/N

    # second HCM run
    D2 = 0
    for index, row in collective.loc[collective["run_index"] == 2].iterrows():
        P_RAJ = row["P_RAJ"]

        if P_RAJ > component_woehler_curve_P_RAJ.P_RAJ_D.values[0]:
            N = component_woehler_curve_P_RAJ.calc_N(P_RAJ)

            if row["is_closed_hysteresis"]:
                D2 += 1/N
            else:
                D2 += 0.5/N


    n2 = len(collective.loc[collective["run_index"] == 2])
    N = (1 + (1-D1)/D2) * n2
    print(f"D1: {D1}, D2: {D2}, n2: {n2}, N: {N}",  result["P_RAJ_lifetime_n_cycles"])
    print(N, result["P_RAJ_miner_lifetime_n_cycles"])

    assert np.isclose(N, result["P_RAJ_lifetime_n_cycles"], rtol=0.17)

    assert np.isclose(N, result["P_RAJ_miner_lifetime_n_cycles"], rtol=0.05)


def test_trailing_zero_has_no_effect():

    # this is example 2.7.1, 2.10.1 "akademisches Beispiel" in the FKM nonlinear document
    assessment_parameters = pd.Series({
        'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
        'FinishingFKM': 'none',  # type of surface finisihing
        'R_m': 600,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
        #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
        'R_z': 250,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

        'P_A': 7.2e-5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
        # beta: 0.5,             # damage index, specify this as an alternative to P_A

        'P_L': 2.5,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlichkeit der Lastfolge)
        'c':   1.4,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
        'A_sigma': 339.4,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
        'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
        'G': 2/15,               # [mm^-1] (de: bezogener Spannungsgradient)
        's_L': 10,               # [MPa] standard deviation of Gaussian distribution
        'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
        'x_Einsatz': 3000,       # [-] (de: Einsatzdurchlaufzahl)
        'r': 15,                 # [mm] radius (?)
    })

    # assessment without trailing zero
    # -----------------------------

    # load sequence
    load_sequence = pd.Series([100, -200, 100, -250, 200, 0, 200, -200])  # [N]

    result1 = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=True, calculate_P_RAJ=True)

    # assessment with trailing zero
    # -----------------------------

    # load sequence
    load_sequence = pd.Series([100, -200, 100, -250, 200, 0, 200, -200, 0])  # [N]

    result2 = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=True, calculate_P_RAJ=True)

    assert result1["P_RAM_is_life_infinite"] == result2["P_RAM_is_life_infinite"]
    assert np.isclose(result1["P_RAM_lifetime_n_cycles"], result2["P_RAM_lifetime_n_cycles"])
    assert np.isclose(result1["P_RAM_lifetime_n_times_load_sequence"], result2["P_RAM_lifetime_n_times_load_sequence"])

    assert result1["P_RAJ_is_life_infinite"] == result2["P_RAJ_is_life_infinite"]
    assert np.isclose(result1["P_RAJ_lifetime_n_cycles"], result2["P_RAJ_lifetime_n_cycles"])
    assert np.isclose(result1["P_RAJ_lifetime_n_times_load_sequence"], result2["P_RAJ_lifetime_n_times_load_sequence"])


def test_middle_zero_has_no_effect():

    # this is example 2.7.1, 2.10.1 "akademisches Beispiel" in the FKM nonlinear document
    assessment_parameters = pd.Series({
        'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
        'FinishingFKM': 'none',  # type of surface finisihing
        'R_m': 600,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
        #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
        'R_z': 250,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

        'P_A': 7.2e-5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
        # beta: 0.5,             # damage index, specify this as an alternative to P_A

        'P_L': 2.5,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlichkeit der Lastfolge)
        'c':   1.4,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
        'A_sigma': 339.4,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
        'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
        'G': 2/15,               # [mm^-1] (de: bezogener Spannungsgradient)
        's_L': 10,               # [MPa] standard deviation of Gaussian distribution
        'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
        'x_Einsatz': 3000,       # [-] (de: Einsatzdurchlaufzahl)
        'r': 15,                 # [mm] radius (?)
    })

    # assessment without trailing zero
    # ---------------------------------

    # load sequence
    load_sequence = pd.Series([100, -200, 100, -250, 200, 0, 200, -200])  # [N]

    result1 = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=True, calculate_P_RAJ=True)

    # assessment with zeros in the middle
    # ------------------------------------

    # load sequence
    load_sequence = pd.Series([100, 0, -200, 0, 0, 0, 100, -250, 200, 0, 200, 0, -200])  # [N]

    result2 = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=True, calculate_P_RAJ=True)

    assert result1["P_RAM_is_life_infinite"] == result2["P_RAM_is_life_infinite"]
    assert np.isclose(result1["P_RAM_lifetime_n_cycles"], result2["P_RAM_lifetime_n_cycles"])
    assert np.isclose(result1["P_RAM_lifetime_n_times_load_sequence"], result2["P_RAM_lifetime_n_times_load_sequence"])

    assert result1["P_RAJ_is_life_infinite"] == result2["P_RAJ_is_life_infinite"]
    assert np.isclose(result1["P_RAJ_lifetime_n_cycles"], result2["P_RAJ_lifetime_n_cycles"])
    assert np.isclose(result1["P_RAJ_lifetime_n_times_load_sequence"], result2["P_RAJ_lifetime_n_times_load_sequence"])


def test_direction_of_constant_amplitude():
    # this is example 2.7.1, 2.10.1 "akademisches Beispiel" in the FKM nonlinear document
    assessment_parameters = pd.Series({
        'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
        'FinishingFKM': 'none',  # type of surface finisihing
        'R_m': 600,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
        #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
        'R_z': 250,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

        'P_A': 7.2e-5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
        # beta: 0.5,             # damage index, specify this as an alternative to P_A

        'P_L': 2.5,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlichkeit der Lastfolge)
        'c':   1.4,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
        'A_sigma': 339.4,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
        'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
        'G': 2/15,               # [mm^-1] (de: bezogener Spannungsgradient)
        's_L': 10,               # [MPa] standard deviation of Gaussian distribution
        'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
        'x_Einsatz': 3000,       # [-] (de: Einsatzdurchlaufzahl)
        'r': 15,                 # [mm] radius (?)
    })

    # assessment without trailing zero
    # -----------------------------

    # load sequence
    load_sequence = pd.Series([100, -100])  # [N]

    result1 = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=True, calculate_P_RAJ=True)

    # assessment with trailing zero
    # -----------------------------

    # load sequence
    load_sequence = pd.Series([-100, 100])  # [N]

    result2 = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=True, calculate_P_RAJ=True)

    assert result1["P_RAM_is_life_infinite"] == result2["P_RAM_is_life_infinite"]
    assert np.isclose(result1["P_RAM_lifetime_n_cycles"], result2["P_RAM_lifetime_n_cycles"], rtol=0.05)
    assert np.isclose(result1["P_RAM_lifetime_n_times_load_sequence"], result2["P_RAM_lifetime_n_times_load_sequence"], rtol=0.05)

    assert result1["P_RAJ_is_life_infinite"] == result2["P_RAJ_is_life_infinite"]
    assert np.isclose(result1["P_RAJ_lifetime_n_cycles"], result2["P_RAJ_lifetime_n_cycles"], rtol=0.05)
    assert np.isclose(result1["P_RAJ_lifetime_n_times_load_sequence"], result2["P_RAJ_lifetime_n_times_load_sequence"], rtol=0.05)

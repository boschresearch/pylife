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
import pylife.vmap
import pylife.stress.equistress
import pylife.strength.fkm_load_distribution
import pylife.strength.fkm_nonlinear
import pylife.strength.damage_parameter
import pylife.strength.woehler_fkm_nonlinear
import pylife.materiallaws
import pylife.stress.rainflow
import pylife.stress.rainflow.recorders
import pylife.stress.rainflow.fkm_nonlinear
import pylife.materiallaws.notch_approximation_law
import pylife.materiallaws.notch_approximation_law_seegerbeste


def test_woehler_curve_P_RAM_collective_has_no_index():

    # Parameters for 2.7.1 "akademisches Beispiel" in the FKM nonlinear document
    assessment_parameters = pd.Series({
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
        "P_RAM_Z_WS": 606.82453,
        "P_RAM_D_WS": 209.397432,
        "d_1": -0.302,
        "d_2": -0.197,
        "P_RAJ_Z_WS": 274.482,
        "P_RAJ_D_WS": 0.262811,
        "d_RAJ": -0.56,
        "n_st": 1.012998,
        "n_bm": 1.0,
        "n_P": 1.012998,
        "K_RP": 0.8531,
        "gamma_M_RAM": 1.2,
        "f_RAM": 1.388585,
        "P_RAM_Z": 437.009206,
        "gamma_M_RAJ": 1.45,
        "f_RAJ": 1.941559,
        "P_RAJ_Z": 141.371959,
        "P_RAJ_D_0": 0.135361,
        "P_RAJ_D": 0.135361,
    })

    # create woehler curve objects
    assessment_parameters["P_RAM_D"] = assessment_parameters.P_RAM_D_WS / assessment_parameters.f_RAM
    component_woehler_curve_parameters = assessment_parameters[["P_RAM_Z", "P_RAM_D", "d_1", "d_2"]]
    component_woehler_curve_P_RAM = component_woehler_curve_parameters.woehler_P_RAM

    # create rainflow for P_RAM
    # initialize notch approximation law
    E, K_prime, n_prime, K_p = assessment_parameters[["E", "K_prime", "n_prime", "K_p"]]
    extended_neuber = pylife.materiallaws.notch_approximation_law.ExtendedNeuber(E, K_prime, n_prime, K_p)

    load_sequence_list = pd.Series([143.696, -287.392, 143.696, -359.240, 287.392, 0.000, 287.392, -287.392])

    # create recorder object
    recorder = pylife.stress.rainflow.recorders.FKMNonlinearRecorder()

    # create detector object
    detector = pylife.stress.rainflow.fkm_nonlinear.FKMNonlinearDetector(
        recorder=recorder, notch_approximation_law=extended_neuber
    )

    # perform HCM algorithm, first run
    detector.process_hcm_first(load_sequence_list)

    # perform HCM algorithm, second run
    detector.process_hcm_second(load_sequence_list)

    collective = recorder.collective.reset_index(drop=True)

    # define damage parameter
    damage_parameter = pylife.strength.damage_parameter.P_RAM(collective, assessment_parameters)
    #display(damage_parameter.collective)

    # compute the effect of the damage parameter with the woehler curve
    damage_calculator = pylife.strength.fkm_nonlinear.damage_calculator\
        .DamageCalculatorPRAM(damage_parameter.collective, component_woehler_curve_P_RAM)

    # Infinite life assessment
    is_life_infinite = damage_calculator.is_life_infinite
    assert is_life_infinite == False

    assert np.isclose(component_woehler_curve_P_RAM.fatigue_strength_limit, 150, rtol=1e-2)
    assert np.isclose(damage_calculator.P_RAM_max, 323, rtol=1e-2)

    # finite life assessment
    lifetime_n_cycles = damage_calculator.lifetime_n_cycles
    lifetime_n_times_load_sequence = damage_calculator.lifetime_n_times_load_sequence

    assert np.isclose(np.round(lifetime_n_cycles), 14618)
    assert np.isclose(np.round(lifetime_n_times_load_sequence), 3655)

    assert np.isclose(component_woehler_curve_P_RAM.d_1, -0.302)
    assert np.isclose(component_woehler_curve_P_RAM.d_2, -0.197)
    assert np.isclose(component_woehler_curve_P_RAM.P_RAM_Z, 437.0, rtol=1e-3)
    assert np.isclose(component_woehler_curve_P_RAM.P_RAM_D, 150.8, rtol=1e-3)
    assert np.isclose(damage_calculator.P_RAM_max, 322.9, rtol=1e-3)
    assert np.isclose(component_woehler_curve_P_RAM.fatigue_strength_limit, 150.8, rtol=1e-3)
    assert np.isclose(component_woehler_curve_P_RAM.fatigue_life_limit, 221637.4, rtol=1e-3)


def test_woehler_curve_P_RAM_collective_has_MultiIndex():

    # Parameters for 2.7.1 "akademisches Beispiel" in the FKM nonlinear document
    assessment_parameters = pd.Series({
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
        "P_RAM_Z_WS": 606.82453,
        "P_RAM_D_WS": 209.397432,
        "d_1": -0.302,
        "d_2": -0.197,
        "P_RAJ_Z_WS": 274.482,
        "P_RAJ_D_WS": 0.262811,
        "d_RAJ": -0.56,
        "n_st": 1.012998,
        "n_bm": 1.0,
        "n_P": 1.012998,
        "K_RP": 0.8531,
        "gamma_M_RAM": 1.2,
        "f_RAM": 1.388585,
        "P_RAM_Z": 437.009206,
        "gamma_M_RAJ": 1.45,
        "f_RAJ": 1.941559,
        "P_RAJ_Z": 141.371959,
        "P_RAJ_D_0": 0.135361,
        "P_RAJ_D": 0.135361,
    })

    # create woehler curve objects
    assessment_parameters["P_RAM_D"] = assessment_parameters.P_RAM_D_WS / assessment_parameters.f_RAM
    component_woehler_curve_parameters = assessment_parameters[["P_RAM_Z", "P_RAM_D", "d_1", "d_2"]]
    component_woehler_curve_P_RAM = component_woehler_curve_parameters.woehler_P_RAM

    # create rainflow for P_RAM
    # initialize notch approximation law
    E, K_prime, n_prime, K_p = assessment_parameters[["E", "K_prime", "n_prime", "K_p"]]
    extended_neuber = pylife.materiallaws.notch_approximation_law.ExtendedNeuber(E, K_prime, n_prime, K_p)

    load_sequence_list = pd.Series([143.696, -287.392, 143.696, -359.240, 287.392, 0.000, 287.392, -287.392])

    # create recorder object
    recorder = pylife.stress.rainflow.recorders.FKMNonlinearRecorder()

    # create detector object
    detector = pylife.stress.rainflow.fkm_nonlinear.FKMNonlinearDetector(
        recorder=recorder, notch_approximation_law=extended_neuber
    )

    # perform HCM algorithm, first run
    detector.process_hcm_first(load_sequence_list)

    # perform HCM algorithm, second run
    detector.process_hcm_second(load_sequence_list)

    # define damage parameter
    damage_parameter = pylife.strength.damage_parameter.P_RAM(recorder.collective, assessment_parameters)
    #display(damage_parameter.collective)

    # compute the effect of the damage parameter with the woehler curve
    damage_calculator = pylife.strength.fkm_nonlinear.damage_calculator\
        .DamageCalculatorPRAM(damage_parameter.collective, component_woehler_curve_P_RAM)

    # Infinite life assessment
    is_life_infinite = damage_calculator.is_life_infinite
    assert is_life_infinite == False

    assert np.isclose(component_woehler_curve_P_RAM.fatigue_strength_limit, 150, rtol=1e-2)
    assert np.isclose(damage_calculator.P_RAM_max, 323, rtol=1e-2)

    # finite life assessment
    lifetime_n_cycles = damage_calculator.lifetime_n_cycles
    lifetime_n_times_load_sequence = damage_calculator.lifetime_n_times_load_sequence

    assert np.isclose(np.round(lifetime_n_cycles), 14618)
    assert np.isclose(np.round(lifetime_n_times_load_sequence), 3655)


def test_woehler_curve_P_RAJ_has_no_index():

    # Parameters for 2.10.1 "akademisches Beispiel" in the FKM nonlinear document
    assessment_parameters = pd.Series({
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
        "P_RAM_Z_WS": 606.82453,
        "P_RAM_D_WS": 209.397432,
        "d_1": -0.302,
        "d_2": -0.197,
        "P_RAJ_Z_WS": 274.482,
        "P_RAJ_D_WS": 0.262811,
        "d_RAJ": -0.56,
        "n_st": 1.012998,
        "n_bm": 1.0,
        "n_P": 1.012998,
        "K_RP": 0.8531,
        "gamma_M_RAM": 1.2,
        "f_RAM": 1.388585,
        "P_RAM_Z": 437.009206,
        "gamma_M_RAJ": 1.45,
        "f_RAJ": 1.941559,
        "P_RAJ_Z": 141.371959,
        "P_RAJ_D_0": 0.135361,
        "P_RAJ_D": 0.135361,
    })

    # create woehler curve objects
    component_woehler_curve_parameters = assessment_parameters[["P_RAJ_Z", "P_RAJ_D_0", "d_RAJ"]]
    component_woehler_curve_P_RAJ = component_woehler_curve_parameters.woehler_P_RAJ

    # create rainflow for P_RAJ
    # initialize notch approximation law
    E, K_prime, n_prime, K_p = assessment_parameters[["E", "K_prime", "n_prime", "K_p"]]
    seeger_beste = pylife.materiallaws.notch_approximation_law_seegerbeste.SeegerBeste(E, K_prime, n_prime, K_p)

    load_sequence_list = pd.Series([143.696, -287.392, 143.696, -359.240, 287.392, 0.000, 287.392, -287.392])
    # create recorder object
    recorder = pylife.stress.rainflow.recorders.FKMNonlinearRecorder()

    # create detector object
    detector = pylife.stress.rainflow.fkm_nonlinear.FKMNonlinearDetector(
        recorder=recorder, notch_approximation_law=seeger_beste
    )

    # perform HCM algorithm, first run
    detector.process_hcm_first(load_sequence_list)

    # perform HCM algorithm, second run
    detector.process_hcm_second(load_sequence_list)

    # remove index from collective
    collective = recorder.collective.reset_index(drop=True)

    # define damage parameter
    damage_parameter = pylife.strength.damage_parameter.P_RAJ(collective, assessment_parameters,\
                                                              component_woehler_curve_P_RAJ)

    # compute the effect of the damage parameter with the woehler curve
    damage_calculator = pylife.strength.fkm_nonlinear.damage_calculator\
        .DamageCalculatorPRAJ(damage_parameter.collective, assessment_parameters, component_woehler_curve_P_RAJ)

    # Infinite life assessment
    is_life_infinite = damage_calculator.is_life_infinite
    assert is_life_infinite == False

    assert np.isclose(component_woehler_curve_P_RAJ.fatigue_strength_limit, 0.135, rtol=1e-2)
    assert np.isclose(damage_calculator.P_RAJ_max, 0.749, rtol=1e-2)

    # finite life assessment
    lifetime_n_cycles = damage_calculator.lifetime_n_cycles
    lifetime_n_times_load_sequence = damage_calculator.lifetime_n_times_load_sequence

    assert np.isclose(np.round(lifetime_n_cycles), 30947)
    assert np.isclose(np.round(lifetime_n_times_load_sequence), 7737)

    assert np.isclose(component_woehler_curve_P_RAJ.d, -0.56, rtol=1e-3)
    assert np.isclose(component_woehler_curve_P_RAJ.P_RAJ_D.squeeze(),  0.135358, rtol=1e-3)
    assert np.isclose(component_woehler_curve_P_RAJ.P_RAJ_Z, 141.371959, rtol=1e-3)
    assert np.isclose(damage_calculator.P_RAJ_max, 0.7490233616724475, rtol=1e-3)
    assert np.isclose(component_woehler_curve_P_RAJ.fatigue_strength_limit, 0.135361, rtol=1e-3)
    assert np.isclose(component_woehler_curve_P_RAJ.fatigue_life_limit, 245945.45, rtol=1e-3)
    assert np.isclose(component_woehler_curve_P_RAJ.fatigue_strength_limit_final.squeeze(),  0.135358, rtol=1e-3)
    assert np.isclose(component_woehler_curve_P_RAJ.fatigue_life_limit_final.squeeze(), 245945.45, rtol=1e-3)


def test_woehler_curve_P_RAJ_has_MultiIndex():

    # Parameters for 2.10.1 "akademisches Beispiel" in the FKM nonlinear document
    assessment_parameters = pd.Series({
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
        "P_RAM_Z_WS": 606.82453,
        "P_RAM_D_WS": 209.397432,
        "d_1": -0.302,
        "d_2": -0.197,
        "P_RAJ_Z_WS": 274.482,
        "P_RAJ_D_WS": 0.262811,
        "d_RAJ": -0.56,
        "n_st": 1.012998,
        "n_bm": 1.0,
        "n_P": 1.012998,
        "K_RP": 0.8531,
        "gamma_M_RAM": 1.2,
        "f_RAM": 1.388585,
        "P_RAM_Z": 437.009206,
        "gamma_M_RAJ": 1.45,
        "f_RAJ": 1.941559,
        "P_RAJ_Z": 141.371959,
        "P_RAJ_D_0": 0.135361,
        "P_RAJ_D": 0.135361,
    })

    # create woehler curve objects
    component_woehler_curve_parameters = assessment_parameters[["P_RAJ_Z", "P_RAJ_D_0", "d_RAJ"]]
    component_woehler_curve_P_RAJ = component_woehler_curve_parameters.woehler_P_RAJ

    # create rainflow for P_RAJ
    # initialize notch approximation law
    E, K_prime, n_prime, K_p = assessment_parameters[["E", "K_prime", "n_prime", "K_p"]]
    seeger_beste = pylife.materiallaws.notch_approximation_law_seegerbeste.SeegerBeste(E, K_prime, n_prime, K_p)

    load_sequence_list = pd.Series([143.696, -287.392, 143.696, -359.240, 287.392, 0.000, 287.392, -287.392])

    # create recorder object
    recorder = pylife.stress.rainflow.recorders.FKMNonlinearRecorder()

    # create detector object
    detector = pylife.stress.rainflow.fkm_nonlinear.FKMNonlinearDetector(
        recorder=recorder, notch_approximation_law=seeger_beste
    )

    # perform HCM algorithm, first run
    detector.process_hcm_first(load_sequence_list)

    # perform HCM algorithm, second run
    detector.process_hcm_second(load_sequence_list)

    # define damage parameter
    damage_parameter = pylife.strength.damage_parameter.P_RAJ(recorder.collective, assessment_parameters,\
                                                              component_woehler_curve_P_RAJ)

    # compute the effect of the damage parameter with the woehler curve
    damage_calculator = pylife.strength.fkm_nonlinear.damage_calculator\
        .DamageCalculatorPRAJ(damage_parameter.collective, assessment_parameters, component_woehler_curve_P_RAJ)

    # Infinite life assessment
    is_life_infinite = damage_calculator.is_life_infinite
    assert is_life_infinite == False

    assert np.isclose(component_woehler_curve_P_RAJ.fatigue_strength_limit, 0.135, rtol=1e-2)
    assert np.isclose(damage_calculator.P_RAJ_max, 0.749, rtol=1e-2)

    # finite life assessment
    lifetime_n_cycles = damage_calculator.lifetime_n_cycles
    lifetime_n_times_load_sequence = damage_calculator.lifetime_n_times_load_sequence

    assert np.isclose(np.round(lifetime_n_cycles), 30947)
    assert np.isclose(np.round(lifetime_n_times_load_sequence), 7737)

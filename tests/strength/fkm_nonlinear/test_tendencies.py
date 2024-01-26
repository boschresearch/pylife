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
import copy
import pandas as pd
import numpy.testing as testing
import unittest.mock as mock

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
import pylife.strength.fkm_nonlinear.assessment_nonlinear_standard
import pylife.strength.fkm_nonlinear.parameter_calculations
import pylife.strength.fkm_nonlinear.parameter_calculations as parameter_calculations


def test_tendency_roughness():

    assessment_parameters = pd.Series({
        'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
        'R_m': 600,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
        #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
        'R_z': 10,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

        'P_A': 7.2e-5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
        # beta: 0.5,             # damage index, specify this as an alternative to P_A

        'P_L': 2.5,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlichkeit der Lastfolge)
        'c':   1.4,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
        'A_sigma': 339.4,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
        'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
        'G': 2/15,               # [mm^-1] (de: bezogener Spannungsgradient)
        's_L': 10,               # [MPa] standard deviation of Gaussian distribution
        'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
    })

    load_sequence = pd.Series([440.,-440.])  # [N]

    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard\
        .perform_fkm_nonlinear_assessment(
            assessment_parameters.copy(), load_sequence.copy(), calculate_P_RAM=True, calculate_P_RAJ=True)

    previous_N_P_RAM = result["P_RAM_lifetime_n_cycles"]
    previous_N_P_RAJ = result["P_RAJ_lifetime_n_cycles"]

    for i in range(1,5):
        # increase roughness, should decrease lifetime
        assessment_parameters["R_z"] = assessment_parameters.R_z + i*100

        # perform assessment with function
        result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard\
            .perform_fkm_nonlinear_assessment(
                assessment_parameters.copy(), load_sequence.copy(), calculate_P_RAM=True, calculate_P_RAJ=True)

        N_P_RAM = result["P_RAM_lifetime_n_cycles"]
        N_P_RAJ = result["P_RAJ_lifetime_n_cycles"]

        assert N_P_RAM < previous_N_P_RAM
        assert N_P_RAJ < previous_N_P_RAJ

        previous_N_P_RAM = N_P_RAM
        previous_N_P_RAJ = N_P_RAJ


def test_tendency_ultimate_tensile_strength():

    assessment_parameters = pd.Series({
        'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
        'R_m': 800,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
        #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
        'R_z': 10,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

        'P_A': 7.2e-5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
        # beta: 0.5,             # damage index, specify this as an alternative to P_A

        'P_L': 2.5,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlichkeit der Lastfolge)
        'c':   1.4,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
        'A_sigma': 339.4,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
        'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
        'G': 2/15,               # [mm^-1] (de: bezogener Spannungsgradient)
        's_L': 10,               # [MPa] standard deviation of Gaussian distribution
        'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
    })

    load_sequence = pd.Series([440.,-440.])  # [N]

    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard\
        .perform_fkm_nonlinear_assessment(
            assessment_parameters.copy(), load_sequence.copy(), calculate_P_RAM=True, calculate_P_RAJ=True)

    previous_N_P_RAM = result["P_RAM_lifetime_n_cycles"]
    previous_N_P_RAJ = result["P_RAJ_lifetime_n_cycles"]

    print(f"reference: {previous_N_P_RAM}, {previous_N_P_RAJ}")

    for i in range(1,5):
        # decrease R_m, should decrease lifetime
        assessment_parameters["R_m"] = 800-i*100

        # perform assessment with function
        result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard\
            .perform_fkm_nonlinear_assessment(
                assessment_parameters.copy(), load_sequence.copy(), calculate_P_RAM=True, calculate_P_RAJ=True)

        N_P_RAM = result["P_RAM_lifetime_n_cycles"]
        N_P_RAJ = result["P_RAJ_lifetime_n_cycles"]

        print(f"{i}: R_m: {assessment_parameters.R_m}, N: {N_P_RAM}, {N_P_RAJ}")

        assert N_P_RAM < previous_N_P_RAM
        assert N_P_RAJ < previous_N_P_RAJ

        previous_N_P_RAM = N_P_RAM
        previous_N_P_RAJ = N_P_RAJ


def test_tendency_G():

    assessment_parameters = pd.Series({
        'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
        'R_m': 300,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
        #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
        'R_z': 10,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

        'P_A': 7.2e-5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
        # beta: 0.5,             # damage index, specify this as an alternative to P_A

        'P_L': 2.5,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlichkeit der Lastfolge)
        'c':   1.4,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
        'A_sigma': 339.4,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
        'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
        'G': 0.15,               # [mm^-1] (de: bezogener Spannungsgradient)
        's_L': 10,               # [MPa] standard deviation of Gaussian distribution
        'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
    })

    load_sequence = pd.Series([440.,-440.])  # [N]

    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard\
        .perform_fkm_nonlinear_assessment(
            assessment_parameters.copy(), load_sequence.copy(), calculate_P_RAM=True, calculate_P_RAJ=True)

    previous_N_P_RAM = result["P_RAM_lifetime_n_cycles"]
    previous_N_P_RAJ = result["P_RAJ_lifetime_n_cycles"]

    print(f"reference: {previous_N_P_RAM}, {previous_N_P_RAJ}")

    for i in range(1,5):
        # increase G, should increase lifetime
        assessment_parameters["G"] = 0.15 + i*0.8

        # perform assessment with function
        result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard\
            .perform_fkm_nonlinear_assessment(
                assessment_parameters.copy(), load_sequence.copy(), calculate_P_RAM=True, calculate_P_RAJ=True)

        N_P_RAM = result["P_RAM_lifetime_n_cycles"]
        N_P_RAJ = result["P_RAJ_lifetime_n_cycles"]

        print(f"{i}: G: {assessment_parameters.G:.3f}, _n_bm: {result['assessment_parameters'].n_bm_}, N: {N_P_RAM}, {N_P_RAJ}")

        assert N_P_RAM >= previous_N_P_RAM
        assert N_P_RAJ >= previous_N_P_RAJ

        previous_N_P_RAM = N_P_RAM
        previous_N_P_RAJ = N_P_RAJ


def test_tendency_A_sigma():

    assessment_parameters = pd.Series({
        'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
        'R_m': 600,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
        #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
        'R_z': 10,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

        'P_A': 7.2e-5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
        # beta: 0.5,             # damage index, specify this as an alternative to P_A

        'P_L': 2.5,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlichkeit der Lastfolge)
        'c':   1.4,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
        'A_sigma': 500,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
        'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
        'G': 0.15,               # [mm^-1] (de: bezogener Spannungsgradient)
        's_L': 10,               # [MPa] standard deviation of Gaussian distribution
        'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
    })

    load_sequence = pd.Series([440.,-440.])  # [N]

    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard\
        .perform_fkm_nonlinear_assessment(
            assessment_parameters.copy(), load_sequence.copy(), calculate_P_RAM=True, calculate_P_RAJ=True)

    previous_N_P_RAM = result["P_RAM_lifetime_n_cycles"]
    previous_N_P_RAJ = result["P_RAJ_lifetime_n_cycles"]

    print(f"reference: {previous_N_P_RAM}, {previous_N_P_RAJ}")

    for i in range(1,5):
        # decrease A_sigma, should decrease lifetime
        assessment_parameters["A_sigma"] = 500-i*123

        # perform assessment with function
        result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard\
            .perform_fkm_nonlinear_assessment(
                assessment_parameters.copy(), load_sequence.copy(), calculate_P_RAM=True, calculate_P_RAJ=True)

        N_P_RAM = result["P_RAM_lifetime_n_cycles"]
        N_P_RAJ = result["P_RAJ_lifetime_n_cycles"]

        print(f"{i}: A_sigma: {assessment_parameters.A_sigma:.3f}, N: {N_P_RAM}, {N_P_RAJ}")

        assert N_P_RAM > previous_N_P_RAM
        assert N_P_RAJ > previous_N_P_RAJ

        previous_N_P_RAM = N_P_RAM
        previous_N_P_RAJ = N_P_RAJ

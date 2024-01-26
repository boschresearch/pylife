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

import pylife
import pylife.strength.fkm_nonlinear.assessment_nonlinear_standard

@pytest.mark.parametrize('P_A, P_L, expected_N_P_RAM, expected_N_P_RAJ', [
    (0.5, 2.5, 4610.050004939043, 8361.629594861868),
    (2.3e-1, 2.5, 576.8268529416085, 1185.6596018665978),
    (1e-3, 2.5, 426.7595636392213, 759.5296689480264),
    (7.2e-5, 2.5, 293.3547748823362, 483.4596907502259),
    (1e-5, 2.5, 212.78302272819474, 350.3771237505745),
    (0.5, 50, 2607.3833616750567, 4812.919793600275),
    (2.3e-1, 50, 400.87343303960074, 693.3767766143047),
    (1e-3, 50,  302.37420376152386, 459.4215309897427),
    (7.2e-5, 50,  208.93141665517766, 285.1618663384134),
    (1e-5, 50, 152.06628705115037, 208.80564563950222)
])
def test_probabilities(P_A, P_L, expected_N_P_RAM, expected_N_P_RAJ):

    assessment_parameters = pd.Series({
            'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
            'FinishingFKM': 'none',  # type of surface finisihing
            'R_m': 600,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
            #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
            'R_z': 250,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

            'P_A': P_A,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
            # beta: 0.5,             # damage index, specify this as an alternative to P_A

            'P_L': P_L,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlilchkeit der Lastfolge)
            'c':   3,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
            'A_sigma': 339.4,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
            'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
            'G': 2/15,               # [mm^-1] (de: bezogener Spannungsgradient)
            's_L': 10,               # [MPa] standard deviation of Gaussian distribution
            'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
            'r': 15,                 # [mm] radius (?)
    })

    load_sequence = pd.Series([200, -200, -180])  # [N]

    result1 = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=True, calculate_P_RAJ=True)

    N_P_RAM = result1["P_RAM_lifetime_n_cycles"]
    N_P_RAJ = result1["P_RAJ_lifetime_n_cycles"]

    print(f"{P_A}, {P_L},  {N_P_RAM} = {expected_N_P_RAM} ({100*(1-N_P_RAM/expected_N_P_RAM)} %), {N_P_RAJ} = {expected_N_P_RAJ}")

    # Note, the high tolerance results from the rounding in the FKM where the gamma_M factors are given
    assert np.isclose(N_P_RAM, expected_N_P_RAM, rtol=0.04)
    assert np.isclose(N_P_RAJ, expected_N_P_RAJ, rtol=0.087)


def test_probability_P_A():

    # calculate with P_A = 50% with functions
    assessment_parameters = pd.Series({
            'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
            'R_m': 600,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
            #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
            'R_z': 250,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

            'P_A': 0.5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
            # beta: 0.5,             # damage index, specify this as an alternative to P_A

            'P_L': 50,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlilchkeit der Lastfolge)
            'c':   3,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
            'A_sigma': 339.4,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
            'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
            'G': 2/15,               # [mm^-1] (de: bezogener Spannungsgradient)
            'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
            'r': 15,                 # [mm] radius (?)
            "n_bins": 1000,           # increase precision by increasing number of bins
    })

    load_sequence = pd.Series([200, -200, -180])  # [N]

    result1 = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=True, calculate_P_RAJ=True)

    P_RAM_N_max_bearable = result1['P_RAM_N_max_bearable']
    assert P_RAM_N_max_bearable(1e-6) == result1['P_RAM_lifetime_N_1ppm']
    assert P_RAM_N_max_bearable(0.1) == result1['P_RAM_lifetime_N_10']
    assert P_RAM_N_max_bearable(0.5) == result1['P_RAM_lifetime_N_50']
    assert P_RAM_N_max_bearable(0.9) == result1['P_RAM_lifetime_N_90']

    P_RAJ_N_max_bearable = result1['P_RAJ_N_max_bearable']
    assert P_RAJ_N_max_bearable(1e-6) == result1['P_RAJ_lifetime_N_1ppm']
    assert P_RAJ_N_max_bearable(0.1) == result1['P_RAJ_lifetime_N_10']
    assert P_RAJ_N_max_bearable(0.5) == result1['P_RAJ_lifetime_N_50']
    assert P_RAJ_N_max_bearable(0.9) == result1['P_RAJ_lifetime_N_90']

    # for P_A = 10%
    for P_A in np.logspace(-6, -1, 5):
        assessment_parameters["P_A"] = P_A

        result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                                calculate_P_RAM=True, calculate_P_RAJ=True)
        print(f'P_RAM, P_A: {P_A}, {result["P_RAM_lifetime_n_cycles"]}, expected: {P_RAM_N_max_bearable(P_A, True)} (error: {100*(1-(result["P_RAM_lifetime_n_cycles"]/P_RAM_N_max_bearable(P_A, True)))}')
        assert np.isclose(result["P_RAM_lifetime_n_cycles"], P_RAM_N_max_bearable(P_A, True))

        print(f'P_RAJ, P_A: {P_A}, {result["P_RAJ_lifetime_n_cycles"]}, expected: {P_RAJ_N_max_bearable(P_A, True)} (error: {100*(1-(result["P_RAJ_lifetime_n_cycles"]/P_RAJ_N_max_bearable(P_A, True)))}')
        assert np.isclose(result["P_RAJ_lifetime_n_cycles"], P_RAJ_N_max_bearable(P_A, True), rtol=5e-2)

    # P_RAM, P_A: 1e-06, 201.1337445238546, expected: 84.936467390315 (error: -136.80493279708634
    # P_RAJ, P_A: 1e-06, 382.6148580109982, expected: 384.07225404822026 (error: 0.3794588184542702
    # P_RAM, P_A: 1.778279410038923e-05, 271.69060660599837, expected: 134.9431579209367 (error: -101.3370746556725
    # P_RAJ, P_A: 1.778279410038923e-05, 509.0653430714803, expected: 508.42132610429104 (error: -0.1266699357644896
    # P_RAM, P_A: 0.00031622776601683794, 385.1249356500931, expected: 230.76240039404595 (error: -66.89241184545676
    # P_RAJ, P_A: 0.00031622776601683794, 699.8053724539884, expected: 703.7139153215512 (error: 0.5554164529739114
    # P_RAM, P_A: 0.005623413251903491, 439.3341660535216, expected: 282.52440822868493 (error: -55.50308336471568
    # P_RAJ, P_A: 0.005623413251903491, 803.4837691850523, expected: 798.4717410139905 (error: -0.6277026366264371


def test_failure_probability():

    # calculate with P_A = 50% with functions
    assessment_parameters = pd.Series({
            'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
            'R_m': 600,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
            #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
            'R_z': 250,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

            'P_A': 0.5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
            # beta: 0.5,             # damage index, specify this as an alternative to P_A

            'P_L': 50,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlilchkeit der Lastfolge)
            'c':   3,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
            'A_sigma': 339.4,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
            'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
            'G': 2/15,               # [mm^-1] (de: bezogener Spannungsgradient)
            's_L': 10,               # [MPa] standard deviation of Gaussian distribution
            'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
            'r': 15,                 # [mm] radius (?)
    })

    load_sequence = pd.Series([200, -200, -180])  # [N]

    result1 = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=True, calculate_P_RAJ=True)


    P_RAM_N_max_bearable = result1['P_RAM_N_max_bearable']
    P_RAJ_N_max_bearable = result1['P_RAJ_N_max_bearable']
    P_RAM_failure_probability = result1['P_RAM_failure_probability']
    P_RAJ_failure_probability = result1['P_RAJ_failure_probability']


    for P_A in list(np.logspace(-7, -1, 5)) + [0.5, 0.7, 0.9, 0.99]:

        assert np.isclose(P_RAM_failure_probability(P_RAM_N_max_bearable(P_A)), P_A)
        assert np.isclose(P_RAJ_failure_probability(P_RAJ_N_max_bearable(P_A)), P_A)


@pytest.mark.parametrize('P_A', [
    0.49,
    2.3e-1,
    1e-3,
    7.2e-5,
    1e-5,
    5e-5,
    1e-6,
    1e-7
])
def test_probability_functions(P_A):

    # calculate with P_A = 50% with functions
    assessment_parameters = pd.Series({
            'MatGroupFKM': 'Steel',  # [Steel, SteelCast, Al_wrought] material group
            'R_m': 600,              # [MPa] ultimate tensile strength (de: Zugfestigkeit)
            #'K_RP': 1,               # [-] surface roughness factor, set to 1 for polished surfaces or determine from the diagrams below
            'R_z': 250,              # [um] average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly

            'P_A': 0.5,           # [-] (one of [0.5, 2.3e-1, 1e-3, 7.2e-5, 1e-5], failure probability (de: auszulegende Ausfallwahrscheinlichkeit)
            # beta: 0.5,             # damage index, specify this as an alternative to P_A

            'P_L': 50,              # [%] (one of 2.5%, 50%) (de: Auftretenswahrscheinlilchkeit der Lastfolge)
            'c':   3,              # [MPa/N] (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt, c = sigma_I / L_REF)
            'A_sigma': 339.4,        # [mm^2] (de: Hochbeanspruchte Oberfläche des Bauteils)
            'A_ref': 500,            # [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
            'G': 2/15,               # [mm^-1] (de: bezogener Spannungsgradient)
            'K_p': 3.5,              # [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
            'r': 15,                 # [mm] radius (?)
            "n_bins": 500
    })

    load_sequence = pd.Series([200, -200, -180])  # [N]

    result1 = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=True, calculate_P_RAJ=True)

    P_RAM_N_max_bearable = result1['P_RAM_N_max_bearable']
    P_RAJ_N_max_bearable = result1['P_RAJ_N_max_bearable']

    # calculate with given probability
    assessment_parameters["P_A"] = P_A

    result2 = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=True, calculate_P_RAJ=True)

    assert np.isclose(result2["P_RAM_lifetime_n_cycles"], P_RAM_N_max_bearable(P_A, True))
    assert np.isclose(result2["P_RAJ_lifetime_n_cycles"], P_RAJ_N_max_bearable(P_A, True), rtol=5e-2)
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
import pylife.strength.fkm_nonlinear.parameter_calculations
import pylife.strength.fkm_nonlinear.parameter_calculations as parameter_calculations


@pytest.fixture(autouse=True)
def switch_off_pandas_cow_behavior():
    pd.options.mode.copy_on_write = False
    yield
    pd.options.mode.copy_on_write = True


@pytest.mark.parametrize(
    'load_sequence, n_hystereses_hcm_1, n_hystereses_hcm_2', [
    (pd.Series([0, 500]), 0, 1),
    (pd.Series([0, 500, 0]), 0, 1),
    (pd.Series([500, 0]), 0, 1),
    (pd.Series([0, 500, -500]), 0, 1),
    (pd.Series([0, 500, -500, 0]), 0.0, 1),  # flush=False in first HCM
    (pd.Series([500, -500, 0]), 0, 1),     # flush=False in first HCM
    (pd.Series([500, -500, -200, 0]), 0, 1),     # flush=False in first HCM
    (pd.Series([200, 500, -500, 0]), 0.0, 1),     # flush=False in first HCM
    (pd.Series([200, 500, -500, -200, 0]), 0.0, 1),     # flush=False in first HCM
    (pd.Series([500, -500]), 0, 1),
    (pd.Series([200, 600, 1000, 200, 60, 1200]), 1, 2),
    (pd.Series([200, 600, 1000, 200, 60, 500, 1500, 700, 1200, -20]), 2, 3),
    (pd.Series([200, 600, 1000, 60, 1500, 200, 80, 400, 1500, 700, 200]), 2, 3),
    (pd.Series([200, 600, 1000, 200, 60, 500, 100, 700, 1260, 1500, 800, 900, 500, 900, 700, 1200]), 4, 6),
    (pd.Series([100, -200, 100, -250, 200, 0, 200, -200]), 2.5, 4),   # FKM nonlinear 2.7.1
    (pd.Series([100, -100, 100, -200, -100, -200, 200, 0, 200, -200]), 4.5, 5),
    (pd.Series([3, -3, 5, -5, 6, -6, 3, -3, 7, -7, 2, -2, 6, -6, 8, -8, 8, -8]), 8, 9), # FKM nonlinear 2.7.2
])
def test_number_of_closed_hystereses(load_sequence, n_hystereses_hcm_1, n_hystereses_hcm_2):

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

    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=True, calculate_P_RAJ=False)

    collective = result["P_RAM_collective"]
    result_n_hystereses_hcm_1 = collective[(collective.run_index==1) & ~collective.is_closed_hysteresis].count().values[0]*0.5 \
        + collective[(collective.run_index==1) & collective.is_closed_hysteresis].count().values[0]

    result_n_hystereses_hcm_2 = collective[(collective.run_index==2) & ~collective.is_closed_hysteresis].count().values[0]*0.5 \
        + collective[(collective.run_index==2) & collective.is_closed_hysteresis].count().values[0]

    print(collective[["loads_min","loads_max","run_index","is_closed_hysteresis"]].reset_index(drop=True))
    print(collective)
    print(f"load: {load_sequence}, n_hyst_1: {result_n_hystereses_hcm_1}, "
        f"n_hyst_2: {result_n_hystereses_hcm_2} (expected: {n_hystereses_hcm_1}, {n_hystereses_hcm_2}")

    assert result_n_hystereses_hcm_1 == n_hystereses_hcm_1
    assert result_n_hystereses_hcm_2 == n_hystereses_hcm_2


@pytest.mark.parametrize(
    "load_sequences", [
    (
        pd.Series([0, 500]),
        pd.Series([0, 500, 0]),
        pd.Series([500, 0]),
        pd.Series([500, 500, 0]),
        pd.Series([500, 500, 0, 0]),
        pd.Series([0, 0, 500, 500, 0, 0]),
        pd.Series([0, 0, 500]),
        pd.Series([0, 0, 500, 500])
    ),(
        -1*pd.Series([0, 500]),
        -1*pd.Series([0, 500, 0]),
        -1*pd.Series([500, 0]),
        -1*pd.Series([500, 500, 0]),
        -1*pd.Series([500, 500, 0, 0]),
        -1*pd.Series([0, 0, 500, 500, 0, 0]),
        -1*pd.Series([0, 0, 500]),
        -1*pd.Series([0, 0, 500, 500])
    ),(
        pd.Series([100, 500]),
        pd.Series([100, 500, 100]),
        pd.Series([500, 100]),
        pd.Series([500, 500, 100]),
        pd.Series([500, 500, 100, 100]),
        pd.Series([100, 100, 500, 500, 100, 100]),
        pd.Series([100, 100, 500]),
        pd.Series([100, 100, 500, 500])
    ),(
        -1*pd.Series([100, 500]),
        -1*pd.Series([100, 500, 100]),
        -1*pd.Series([500, 100]),
        -1*pd.Series([500, 500, 100]),
        -1*pd.Series([500, 500, 100, 100]),
        -1*pd.Series([100, 100, 500, 500, 100, 100]),
        -1*pd.Series([100, 100, 500]),
        -1*pd.Series([100, 100, 500, 500])
    ),(
        pd.Series([0, 500, -500]),
        pd.Series([0, 500, -500, 0]),
        pd.Series([500, -500, 0]),
        pd.Series([500, -500, -200, 0]),
        pd.Series([200, 500, -500, 0]),
        pd.Series([200, 500, -500, -200, 0]),
        pd.Series([0, 200, 500, -500, -200, 0]),
        pd.Series([0, 200, 500, -500, -200]),
        pd.Series([0, 200, 500, -500]),
        pd.Series([200, 500, -500]),
        pd.Series([500, -500, -200]),
        pd.Series([500, -500]),
    ),(
        -1*pd.Series([0, 500, -500]),
        -1*pd.Series([0, 500, -500, 0]),
        -1*pd.Series([500, -500, 0]),
        -1*pd.Series([500, -500, -200, 0]),
        -1*pd.Series([200, 500, -500, 0]),
        -1*pd.Series([200, 500, -500, -200, 0]),
        -1*pd.Series([0, 200, 500, -500, -200, 0]),
        -1*pd.Series([0, 200, 500, -500, -200]),
        -1*pd.Series([0, 200, 500, -500]),
        -1*pd.Series([200, 500, -500]),
        -1*pd.Series([500, -500, -200]),
        -1*pd.Series([500, -500]),
    )
])
def test_load_sequences_with_same_lifetimes(load_sequences):

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


    load_sequence = load_sequences[0]
    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                                calculate_P_RAM=True, calculate_P_RAJ=False)

    N_reference = result["P_RAM_lifetime_n_cycles"]


    for load_sequence in load_sequences:

        result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                                calculate_P_RAM=True, calculate_P_RAJ=False)

        N = result["P_RAM_lifetime_n_cycles"]
        assert N == N_reference


@pytest.mark.parametrize(
    'load_sequence', [
    pd.Series([0, 500, 0]),
    pd.Series([0, 500]),
    pd.Series([500, 0]),
    pd.Series([0, 500, -500]),
    pd.Series([0, 500, -500, 0]),
    pd.Series([500, -500, 0]),
    pd.Series([500, -500, -200, 0]),
    pd.Series([200, 500, -500, 0]),
    pd.Series([200, 500, -500, -200, 0]),
    pd.Series([0, 200, 500, -500, -200, 0]),
    pd.Series([0, 200, 500, -500, -200]),
    pd.Series([0, 200, 500, -500]),
    pd.Series([200, 500, -500]),
    pd.Series([500, -500, -200]),
    pd.Series([500, -500]),
    pd.Series([200, 600, 1000, 200, 60, 1200]),
    pd.Series([200, 600, 1000, 200, 60, 500, 1500, 700, 1200, -20]),
    pd.Series([200, 600, 1000, 60, 1500]),
    pd.Series([200, 600, 1000, 60, 1500, 200, 80, 400, 1500, 700, 200]),
    pd.Series([200, 600, 1000, 200, 60, 500, 100, 700, 1260, 1500, 800, 900, 500, 900, 700, 1200]),
    pd.Series([100, -200, 100, -250, 200, 0, 200, -200]),
    pd.Series([100, -100, 100, -200, -100, -200, 200, 0, 200, -200]),
])
def test_repeated_load_sequence_P_RAM(load_sequence):

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

    for prefactor in [1, -1]:

        load_sequence = load_sequence*prefactor

        # simulation with single collective
        print(f"load_sequence: {load_sequence}")
        result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                                calculate_P_RAM=True, calculate_P_RAJ=False)

        N1 = result["P_RAM_lifetime_n_cycles"]
        print(f"N1: {N1}\n\n")
        print(f"infinite: {result['P_RAM_is_life_infinite']}")


        # same simulation but with repeated collective
        # load sequence
        load_sequence_2 = pd.concat([load_sequence, load_sequence], ignore_index=True)

        result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence_2,
                                                                                calculate_P_RAM=True, calculate_P_RAJ=False)
        N2 = result["P_RAM_lifetime_n_cycles"]
        print(f"N2: {N2}")
        print(f"infinite: {result['P_RAM_is_life_infinite']}")

        assert np.isclose(N1, N2)

        # same simulation but with repeated collective
        # load sequence
        load_sequence_3 = pd.concat([load_sequence, load_sequence, load_sequence, load_sequence, load_sequence, load_sequence], ignore_index=True)

        result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence_3,
                                                                                calculate_P_RAM=True, calculate_P_RAJ=False)
        N3 = result["P_RAM_lifetime_n_cycles"]
        print(f"N3: {N3}")
        print(f"infinite: {result['P_RAM_is_life_infinite']}")


        assert np.isclose(N1, N3)


@pytest.mark.parametrize(
    'original_load_sequence', [
    pd.Series([0, 500, 0]),
    pd.Series([0, 500]),
    pd.Series([500, 0]),
    pd.Series([0, 500, -500]),
    pd.Series([0, 500, -500, 0]),
    pd.Series([500, -500, 0]),
    pd.Series([500, -500, -200, 0]),
    pd.Series([200, 500, -500, 0]),
    pd.Series([200, 500, -500, -200, 0]),
    pd.Series([0, 200, 500, -500, -200, 0]),
    pd.Series([0, 200, 500, -500, -200]),
    pd.Series([0, 200, 500, -500]),
    pd.Series([200, 500, -500]),
    pd.Series([500, -500, -200]),
    pd.Series([500, -500]),
    pd.Series([200, 600, 1000, 200, 60, 1200]),
    pd.Series([200, 600, 1000, 200, 60, 500, 1500, 700, 1200, -20]),
    pd.Series([200, 600, 1000, 60, 1500]),
    pd.Series([200, 600, 1000, 60, 1500, 200, 80, 400, 1500, 700, 200]),
    pd.Series([200, 600, 1000, 200, 60, 500, 100, 700, 1260, 1500, 800, 900, 500, 900, 700, 1200]),
    pd.Series([100, -200, 100, -250, 200, 0, 200, -200]),
    pd.Series([100, -100, 100, -200, -100, -200, 200, 0, 200, -200]),
])
def test_repeated_load_sequence_multiple_points_P_RAM(original_load_sequence):

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

    # generate a load sequence for three assessment points
    load_sequence_0 = original_load_sequence
    load_sequence_1 = load_sequence_0 * 0.7
    load_sequence_2 = load_sequence_0 * 0.2

    index = pd.MultiIndex.from_product([range(len(load_sequence_0)), [0,1,2]], names=["load_step", "node_id"])

    load_sequence = pd.Series(index=index)
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==0] = load_sequence_0.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==1] = load_sequence_1.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==2] = load_sequence_2.to_numpy()

    # perform assessment of all points at once
    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                                      calculate_P_RAM=True, calculate_P_RAJ=False)

    N1 = result["P_RAM_lifetime_n_cycles"]
    print(f"N1: {N1}")
    print(f"infinite: {result['P_RAM_is_life_infinite']}")

    # same simulation but with repeated collective
    # load sequence
    load_sequence_r2 = pd.concat([original_load_sequence, original_load_sequence], ignore_index=True)

    # generate a load sequence for three assessment points
    load_sequence_0 = load_sequence_r2
    load_sequence_1 = load_sequence_0 * 0.7
    load_sequence_2 = load_sequence_0 * 0.2

    index = pd.MultiIndex.from_product([range(len(load_sequence_0)), [0,1,2]], names=["load_step", "node_id"])

    load_sequence = pd.Series(index=index)
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==0] = load_sequence_0.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==1] = load_sequence_1.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==2] = load_sequence_2.to_numpy()

    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=True, calculate_P_RAJ=False)
    N2 = result["P_RAM_lifetime_n_cycles"]
    print(f"N2: {N2}")
    print(f"infinite: {result['P_RAM_is_life_infinite']}")

    # same simulation but with repeated collective
    # load sequence
    load_sequence_r3 = pd.concat([original_load_sequence, original_load_sequence, original_load_sequence, original_load_sequence, original_load_sequence, original_load_sequence], ignore_index=True)

    # generate a load sequence for three assessment points
    load_sequence_0 = load_sequence_r3
    load_sequence_1 = load_sequence_0 * 0.7
    load_sequence_2 = load_sequence_0 * 0.2

    index = pd.MultiIndex.from_product([range(len(load_sequence_0)), [0,1,2]], names=["load_step", "node_id"])

    load_sequence = pd.Series(index=index)
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==0] = load_sequence_0.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==1] = load_sequence_1.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==2] = load_sequence_2.to_numpy()

    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=True, calculate_P_RAJ=False)
    N3 = result["P_RAM_lifetime_n_cycles"]
    print(f"N3: {N3}")
    print(f"infinite: {result['P_RAM_is_life_infinite']}")

    assert np.isclose(N1, N2).all()
    assert np.isclose(N1, N3).all()


@pytest.mark.parametrize(
    'load_sequence', [
    (pd.Series([0, 500, 0])),
    (pd.Series([0, 500])),
    (pd.Series([500, 0])),
    (pd.Series([200, 600, 1000, 200, 60, 1200])),
    (pd.Series([200, 600, 1000, 200, 60, 500, 1500, 700, 1200, -20])),
    (pd.Series([200, 600, 1000, 60, 1500])),
    (pd.Series([200, 600, 1000, 60, 1500, 200, 80, 400, 1500, 700, 200])),
    (pd.Series([200, 600, 1000, 200, 60, 500, 100, 700, 1260, 1500, 800, 900, 500, 900, 700, 1200])),
    (pd.Series([100, -200, 100, -250, 200, 0, 200, -200])),
    (pd.Series([100, -100, 100, -200, -100, -200, 200, 0, 200, -200])),
])
def test_repeated_load_sequence_P_RAJ(load_sequence):

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

    # simulation with single collective

    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=False, calculate_P_RAJ=True)
    N1 = result["P_RAJ_lifetime_n_cycles"]

    # same simulation but with repeated collective
    # load sequence
    load_sequence_2 = pd.concat([load_sequence, load_sequence], ignore_index=True)

    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence_2,
                                                                            calculate_P_RAM=False, calculate_P_RAJ=True)
    N2 = result["P_RAJ_lifetime_n_cycles"]

    # same simulation but with repeated collective
    # load sequence
    load_sequence_3 = pd.concat([load_sequence, load_sequence, load_sequence, load_sequence, load_sequence, load_sequence], ignore_index=True)

    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence_3,
                                                                            calculate_P_RAM=False, calculate_P_RAJ=True)
    N3 = result["P_RAJ_lifetime_n_cycles"]

    print(f"{N1}, {N2}, {N3}, {(N1-N2)/N1*100:+.2f}%, {(N1-N3)/N1*100:+.2f}%")

    assert np.isclose(N1, N2, rtol=0.25)
    assert np.isclose(N1, N3, rtol=0.25)


@pytest.mark.parametrize(
    'original_load_sequence', [
    (pd.Series([0, 500, 0])),
    (pd.Series([0, 500])),
    (pd.Series([500, 0])),
    (pd.Series([200, 600, 1000, 200, 60, 1200])),
    (pd.Series([200, 600, 1000, 200, 60, 500, 1500, 700, 1200, -20])),
    (pd.Series([200, 600, 1000, 60, 1500])),
    (pd.Series([200, 600, 1000, 60, 1500, 200, 80, 400, 1500, 700, 200])),
    (pd.Series([200, 600, 1000, 200, 60, 500, 100, 700, 1260, 1500, 800, 900, 500, 900, 700, 1200])),
    (pd.Series([100, -200, 100, -250, 200, 0, 200, -200])),
    (pd.Series([100, -100, 100, -200, -100, -200, 200, 0, 200, -200])),
])
def test_repeated_load_sequence_multiple_points_P_RAJ(original_load_sequence):

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

    # generate a load sequence for three assessment points
    load_sequence_0 = original_load_sequence
    load_sequence_1 = load_sequence_0 * 1.2
    load_sequence_2 = load_sequence_0 * 0.2

    index = pd.MultiIndex.from_product([range(len(load_sequence_0)), [0,1,2]], names=["load_step", "node_id"])

    load_sequence = pd.Series(index=index)
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==0] = load_sequence_0.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==1] = load_sequence_1.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==2] = load_sequence_2.to_numpy()

    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=False, calculate_P_RAJ=True)
    N1 = result["P_RAJ_lifetime_n_cycles"]
    print(f"N1: {N1}")

    # same simulation but with repeated collective
    # load sequence
    load_sequence_r2 = pd.concat([original_load_sequence, original_load_sequence], ignore_index=True)

    # generate a load sequence for three assessment points
    load_sequence_0 = load_sequence_r2
    load_sequence_1 = load_sequence_0 * 1.2
    load_sequence_2 = load_sequence_0 * 0.2

    index = pd.MultiIndex.from_product([range(len(load_sequence_0)), [0,1,2]], names=["load_step", "node_id"])

    load_sequence = pd.Series(index=index)
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==0] = load_sequence_0.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==1] = load_sequence_1.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==2] = load_sequence_2.to_numpy()

    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=False, calculate_P_RAJ=True)
    N2 = result["P_RAJ_lifetime_n_cycles"]
    print(f"N2: {N2}")

    # same simulation but with repeated collective
    # load sequence
    load_sequence_r3 = pd.concat([original_load_sequence, original_load_sequence, original_load_sequence, original_load_sequence, original_load_sequence, original_load_sequence], ignore_index=True)

    # generate a load sequence for three assessment points
    load_sequence_0 = load_sequence_r3
    load_sequence_1 = load_sequence_0 * 1.2
    load_sequence_2 = load_sequence_0 * 0.2

    index = pd.MultiIndex.from_product([range(len(load_sequence_0)), [0,1,2]], names=["load_step", "node_id"])

    load_sequence = pd.Series(index=index)
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==0] = load_sequence_0.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==1] = load_sequence_1.to_numpy()
    load_sequence.loc[load_sequence.index.get_level_values("node_id")==2] = load_sequence_2.to_numpy()

    result = pylife.strength.fkm_nonlinear.assessment_nonlinear_standard.perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence,
                                                                            calculate_P_RAM=False, calculate_P_RAJ=True)
    N3 = result["P_RAJ_lifetime_n_cycles"]

    assert np.isclose(N1, N2, rtol=0.3).all()
    assert np.isclose(N1, N3, rtol=0.3).all()

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
import pylife.strength.woehler_fkm_nonlinear


@pytest.mark.parametrize(
    "P_RAM_Z, P_RAM_D, d_1, d_2", [
    (400, 150, -0.3, -0.2),
    (400, 150, -0.1, -0.2),
    (500, 355, -0.4, -0.9)
])
def test_woehler_curve_P_RAM(P_RAM_Z, P_RAM_D, d_1, d_2):
    assessment_parameters = pd.Series({
        "P_RAM_Z": P_RAM_Z,
        "P_RAM_D": P_RAM_D,
        "d_1": d_1,
        "d_2": d_2
    })

    component_woehler_curve_P_RAM = assessment_parameters.woehler_P_RAM

    # assert that getters work as expected
    assert P_RAM_Z == component_woehler_curve_P_RAM.P_RAM_Z
    assert P_RAM_D == component_woehler_curve_P_RAM.P_RAM_D
    assert d_1 == component_woehler_curve_P_RAM.d_1
    assert d_2 == component_woehler_curve_P_RAM.d_2

    assert component_woehler_curve_P_RAM.fatigue_strength_limit == P_RAM_D

    # evaluate curve at some points (as function of N)
    assert component_woehler_curve_P_RAM.calc_P_RAM(1e3) == P_RAM_Z
    assert component_woehler_curve_P_RAM.calc_P_RAM(1e10) == P_RAM_D
    assert component_woehler_curve_P_RAM.calc_P_RAM(component_woehler_curve_P_RAM.fatigue_life_limit) == P_RAM_D
    assert component_woehler_curve_P_RAM.calc_P_RAM(component_woehler_curve_P_RAM.fatigue_life_limit*0.5) > P_RAM_D

    # evaluate curve at some points (as function of P_RAM)
    assert np.isinf(component_woehler_curve_P_RAM.calc_N(0))
    assert np.isinf(component_woehler_curve_P_RAM.calc_N(component_woehler_curve_P_RAM.fatigue_strength_limit))
    assert np.isclose(component_woehler_curve_P_RAM.calc_N(component_woehler_curve_P_RAM.fatigue_strength_limit+1e-12), component_woehler_curve_P_RAM.fatigue_life_limit)

    assert np.isclose(component_woehler_curve_P_RAM.calc_N(P_RAM_Z), 1e3)

    # check inverse
    for N in np.logspace(0, np.log10(component_woehler_curve_P_RAM.fatigue_life_limit)-1e-10):
        P = component_woehler_curve_P_RAM.calc_P_RAM(N)

        assert np.isclose(component_woehler_curve_P_RAM.calc_N(P), N)

    for P_RAM in np.logspace(np.log10(P_RAM_D)+1e-10, np.log10(component_woehler_curve_P_RAM.calc_P_RAM(1))):
        N = component_woehler_curve_P_RAM.calc_N(P_RAM)

        assert np.isclose(component_woehler_curve_P_RAM.calc_P_RAM(N), P_RAM)

    # test that slopes are correct (for function of P_RAM)
    # left branch
    for P_RAM in [5*P_RAM_Z, 2*P_RAM_Z, 1.1*P_RAM_Z]:
        log_P = np.log10(P_RAM)

        h = 1e-3
        log_P_1 = log_P + h
        log_P_2 = log_P - h

        log_N_1 = np.log10(component_woehler_curve_P_RAM.calc_N(10**log_P_1))
        log_N_2 = np.log10(component_woehler_curve_P_RAM.calc_N(10**log_P_2))

        d_numeric = (log_N_1 - log_N_2) / (2*h)
        assert np.isclose(d_numeric, 1/d_1)

    # right branch
    for P_RAM in [0.9*P_RAM_Z, 1.1*P_RAM_D]:
        log_P = np.log10(P_RAM)

        h = 1e-3
        log_P_1 = log_P + h
        log_P_2 = log_P - h

        log_N_1 = np.log10(component_woehler_curve_P_RAM.calc_N(10**log_P_1))
        log_N_2 = np.log10(component_woehler_curve_P_RAM.calc_N(10**log_P_2))

        d_numeric = (log_N_1 - log_N_2) / (2*h)
        assert np.isclose(d_numeric, 1/d_2)

    # test that slopes are correct (for function of N)
    # left branch
    for N in [1e1, 1e2, 5e2]:
        log_N = np.log10(N)

        h = 1e-3
        log_N_1 = log_N - h
        log_N_2 = log_N + h

        log_P_1 = np.log10(component_woehler_curve_P_RAM.calc_P_RAM(10**log_N_1))
        log_P_2 = np.log10(component_woehler_curve_P_RAM.calc_P_RAM(10**log_N_2))

        d_numeric = (log_P_2 - log_P_1) / (2*h)
        assert np.isclose(d_numeric, d_1)

    # right branch
    for N in np.logspace(np.log10(1e3)+0.1, np.log10(component_woehler_curve_P_RAM.fatigue_life_limit)-0.1):
        log_N = np.log10(N)

        h = 1e-3
        log_N_1 = log_N - h
        log_N_2 = log_N + h

        log_P_1 = np.log10(component_woehler_curve_P_RAM.calc_P_RAM(10**log_N_1))
        log_P_2 = np.log10(component_woehler_curve_P_RAM.calc_P_RAM(10**log_N_2))

        d_numeric = (log_P_2 - log_P_1) / (2*h)
        assert np.isclose(d_numeric, d_2)


@pytest.mark.parametrize(
    "P_RAM_Z, P_RAM_D, d_1, d_2", [
    (100, 200, -0.3, -0.2),
    (300, 150, -0.1, 0.2),
    (500, 355, 0.4, -0.9)
])
def test_woehler_curve_P_RAM_exceptions(P_RAM_Z, P_RAM_D, d_1, d_2):
    assessment_parameters = pd.Series({
        "P_RAM_Z": P_RAM_Z,
        "P_RAM_D": P_RAM_D,
        "d_1": d_1,
        "d_2": d_2
    })

    with pytest.raises(ValueError, match="has to be"):
        component_woehler_curve_P_RAM = assessment_parameters.woehler_P_RAM


def test_woehler_curve_P_RAM_change_params():
    ap = pd.Series({
        "P_RAM_Z": 400.0,
        "P_RAM_D": 150.0,
        "d_1": -0.3,
        "d_2": -0.2
    })

    _ = ap.woehler_P_RAM

    ap["d_2"] = -0.9

    assert ap.woehler_P_RAM.to_pandas()["d_2"] == -0.9


# ---------- P_RAJ --------------
@pytest.mark.parametrize(
    "P_RAJ_Z, P_RAJ_D_0, d_RAJ", [
    (150, 0.1, -0.5),
    (150, 10, -0.5),
    (150, 100, -0.1),
    (500, 20, -0.7),
    (600, 20, -0.2),
])
def test_woehler_curve_P_RAJ(P_RAJ_Z, P_RAJ_D_0, d_RAJ):
    assessment_parameters = pd.Series({
        "P_RAJ_Z": P_RAJ_Z,
        "P_RAJ_D_0": P_RAJ_D_0,
        "d_RAJ": d_RAJ
    })

    component_woehler_curve_P_RAJ = assessment_parameters.woehler_P_RAJ

    # assert that getters work as expected
    assert P_RAJ_Z == component_woehler_curve_P_RAJ.P_RAJ_Z
    assert P_RAJ_D_0 == component_woehler_curve_P_RAJ.P_RAJ_D
    assert d_RAJ == component_woehler_curve_P_RAJ.d

    assert component_woehler_curve_P_RAJ.fatigue_strength_limit == P_RAJ_D_0
    assert component_woehler_curve_P_RAJ.fatigue_strength_limit_final == P_RAJ_D_0

    # test updating of P_RAJ_D
    component_woehler_curve_P_RAJ.update_P_RAJ_D(2*P_RAJ_D_0)

    assert 2*P_RAJ_D_0 == component_woehler_curve_P_RAJ.P_RAJ_D
    assert component_woehler_curve_P_RAJ.fatigue_strength_limit_final == 2*P_RAJ_D_0
    assert component_woehler_curve_P_RAJ.fatigue_strength_limit == P_RAJ_D_0

    assert P_RAJ_Z == component_woehler_curve_P_RAJ.P_RAJ_Z
    assert d_RAJ == component_woehler_curve_P_RAJ.d

    # set back to original value
    component_woehler_curve_P_RAJ.update_P_RAJ_D(P_RAJ_D_0)


    # evaluate curve at some points (as function of N)
    assert component_woehler_curve_P_RAJ.calc_P_RAJ(1e0) == P_RAJ_Z
    assert component_woehler_curve_P_RAJ.calc_P_RAJ(1e10) == P_RAJ_D_0
    assert component_woehler_curve_P_RAJ.calc_P_RAJ(component_woehler_curve_P_RAJ.fatigue_life_limit) == P_RAJ_D_0
    assert component_woehler_curve_P_RAJ.calc_P_RAJ(component_woehler_curve_P_RAJ.fatigue_life_limit_final) == P_RAJ_D_0
    assert component_woehler_curve_P_RAJ.calc_P_RAJ(component_woehler_curve_P_RAJ.fatigue_life_limit*0.5) > P_RAJ_D_0

    # evaluate curve at some points (as function of P_RAJ)
    assert np.isinf(component_woehler_curve_P_RAJ.calc_N(0))
    assert np.isinf(component_woehler_curve_P_RAJ.calc_N(component_woehler_curve_P_RAJ.fatigue_strength_limit))

    assert np.isclose(component_woehler_curve_P_RAJ.calc_N(component_woehler_curve_P_RAJ.fatigue_strength_limit+1e-12), component_woehler_curve_P_RAJ.fatigue_life_limit)

    assert np.isclose(component_woehler_curve_P_RAJ.calc_N(P_RAJ_Z), 1e0)

    # check inverse
    for N in np.logspace(0, np.log10(component_woehler_curve_P_RAJ.fatigue_life_limit)-1e-10):
        P = component_woehler_curve_P_RAJ.calc_P_RAJ(N)

        assert np.isclose(component_woehler_curve_P_RAJ.calc_N(P), N)

    for P_RAJ in np.logspace(np.log10(P_RAJ_D_0)+1e-10, np.log10(component_woehler_curve_P_RAJ.calc_P_RAJ(1))):
        N = component_woehler_curve_P_RAJ.calc_N(P_RAJ)

        assert np.isclose(component_woehler_curve_P_RAJ.calc_P_RAJ(N), P_RAJ)

    # test that slopes are correct (for function of P_RAJ)
    for P_RAJ in [0.9*P_RAJ_Z, 1.1*P_RAJ_D_0]:
        log_P = np.log10(P_RAJ)

        h = 1e-3
        log_P_1 = log_P + h
        log_P_2 = log_P - h

        log_N_1 = np.log10(component_woehler_curve_P_RAJ.calc_N(10**log_P_1))
        log_N_2 = np.log10(component_woehler_curve_P_RAJ.calc_N(10**log_P_2))

        d_numeric = (log_N_1 - log_N_2) / (2*h)
        assert np.isclose(d_numeric, 1/d_RAJ)

    # test that slopes are correct (for function of N)
    for N in np.logspace(np.log10(1e0)+0.1, np.log10(component_woehler_curve_P_RAJ.fatigue_life_limit)-0.1):
        log_N = np.log10(N)

        h = 1e-3
        log_N_1 = log_N - h
        log_N_2 = log_N + h

        log_P_1 = np.log10(component_woehler_curve_P_RAJ.calc_P_RAJ(10**log_N_1))
        log_P_2 = np.log10(component_woehler_curve_P_RAJ.calc_P_RAJ(10**log_N_2))

        d_numeric = (log_P_2 - log_P_1) / (2*h)
        assert np.isclose(d_numeric, d_RAJ)


@pytest.mark.parametrize(
    "P_RAJ_Z, P_RAJ_D_0, d_RAJ", [
    (100, 200, -0.3),
    (300, 150, 0.2)
])
def test_woehler_curve_P_RAJ_exceptions(P_RAJ_Z, P_RAJ_D_0, d_RAJ):
    assessment_parameters = pd.Series({
        "P_RAJ_Z": P_RAJ_Z,
        "P_RAJ_D_0": P_RAJ_D_0,
        "d_RAJ": d_RAJ
    })

    with pytest.raises(ValueError, match="has to be"):
        component_woehler_curve_P_RAJ = assessment_parameters.woehler_P_RAJ


def test_woehler_curve_P_RAJ_change_params():
    ap = pd.Series({
        "P_RAJ_Z": 150.0,
        "P_RAJ_D_0": 0.1,
        "d_RAJ": -0.5,
    })

    _ = ap.woehler_P_RAJ

    ap["d_RAJ"] = -0.7

    assert ap.woehler_P_RAJ.to_pandas()["d_RAJ"] == -0.7

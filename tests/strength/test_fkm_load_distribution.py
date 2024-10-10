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

import pylife.strength.fkm_load_distribution


@pytest.mark.parametrize('P_A, resulting_gamma_L, result', [
    (1e-5,   1.11956,  [111.956, -223.912,  111.956, -279.89, 223.912, 0., 223.912, -223.912]),
    (7.2e-5, 1.1064,   [110.64, -221.28,  110.64, -276.6, 221.28, 0., 221.28, -221.28]),
    (1e-3,   1.08652,  [108.652, -217.304, 108.652, -271.63, 217.304, 0., 217.304, -217.304]),
    (2.3e-1, 1.020692, [102.0692, -204.1384,  102.0692, -255.173, 204.1384, 0., 204.1384, -204.1384]),
])
def test_load_distribution_normal_1(P_A, resulting_gamma_L, result):

    # test with a plain series
    load_sequence = pd.Series([100, -200, 100, -250, 200, 0, 200, -200])  # [N]

    assessment_parameters = pd.Series({"P_A": P_A, "P_L": 50, "s_L": 10})

    # FKMLoadDistributionNormal, uses assessment_parameters.s_L, assessment_parameters.P_L, assessment_parameters.P_A
    scaled_load_sequence = (
        load_sequence.fkm_safety_normal_from_stddev.scaled_load_sequence(
            assessment_parameters
        )
    )
    gamma_L = load_sequence.fkm_safety_normal_from_stddev.gamma_L(assessment_parameters)

    assert np.isclose(gamma_L, resulting_gamma_L)
    np.testing.assert_allclose(scaled_load_sequence, result)

    # ------

    # test with a DataFrame with one column and multiple nodes
    load_sequence0 = pd.DataFrame(data={
            "col0": (np.array([[1],[0.1]]) * np.array([load_sequence.values])).T.flatten()
        },
        index=pd.MultiIndex.from_product([range(len(load_sequence.values)), [0,1]], names=["load_step", "node_id"])
    )
    scaled_load_sequence = load_sequence0.fkm_safety_normal_from_stddev.scaled_load_sequence(assessment_parameters)
    gamma_L = load_sequence0.fkm_safety_normal_from_stddev.gamma_L(assessment_parameters)

    assert np.isclose(gamma_L, resulting_gamma_L)
    np.testing.assert_allclose(scaled_load_sequence.col0[scaled_load_sequence.index.get_level_values("node_id")==0], np.array(result))
    np.testing.assert_allclose(scaled_load_sequence.col0[scaled_load_sequence.index.get_level_values("node_id")==1], np.array(result)*0.1)

    # ------

    # test with a DataFrame with multiple columns and multiple nodes
    load_sequence1 = pd.DataFrame(data={
            "col0": (np.array([[1],[0.1]]) * np.array([load_sequence.values])).T.flatten(),
            "col1": range(2*len(load_sequence))
        },
        index=pd.MultiIndex.from_product([range(len(load_sequence.values)), [0,1]], names=["load_step", "node_id"])
    )
    scaled_load_sequence = load_sequence1.fkm_safety_normal_from_stddev.scaled_load_sequence(assessment_parameters)
    gamma_L = load_sequence1.fkm_safety_normal_from_stddev.gamma_L(assessment_parameters)

    assert np.isclose(gamma_L, resulting_gamma_L)
    np.testing.assert_allclose(scaled_load_sequence.col0[scaled_load_sequence.index.get_level_values("node_id")==0], np.array(result))
    np.testing.assert_allclose(scaled_load_sequence.col0[scaled_load_sequence.index.get_level_values("node_id")==1], np.array(result)*0.1)
    np.testing.assert_allclose(scaled_load_sequence.col1, range(2*len(load_sequence.values)))

    # ------

    # test with a DataFrame with multiple columns and one node
    load_sequence2 = pd.DataFrame(data={
            "col0": np.array(load_sequence.values),
            "col1": range(len(load_sequence))
        },
        index=pd.MultiIndex.from_product([range(len(load_sequence.values)), [0]], names=["load_step", "node_id"]))
    scaled_load_sequence = load_sequence2.fkm_safety_normal_from_stddev.scaled_load_sequence(assessment_parameters)
    gamma_L = load_sequence2.fkm_safety_normal_from_stddev.gamma_L(assessment_parameters)

    assert np.isclose(gamma_L, resulting_gamma_L)
    np.testing.assert_allclose(scaled_load_sequence.col0, np.array(result))
    np.testing.assert_allclose(scaled_load_sequence.col1, range(len(load_sequence.values)))


@pytest.mark.parametrize('P_A, resulting_gamma_L, result', [
    (1e-5,   1.03956, [103.956, -207.912, 103.956, -259.89, 207.912, 0., 207.912, -207.912]),
    (7.2e-5, 1.0264, [102.64, -205.28,  102.64, -256.6, 205.28, 0., 205.28, -205.28]),
    (1e-3,  1.00652, [100.652, -201.304,  100.652, -251.63, 201.304, 0., 201.304, -201.304]),
    (2.3e-1, 0.940692, [94.0692, -188.1384, 94.0692, -235.173, 188.1384, 0., 188.1384, -188.1384]),
])
def test_load_distribution_normal_2(P_A, resulting_gamma_L, result):

    # test with a plain series
    load_sequence = pd.Series([100, -200, 100, -250, 200, 0, 200, -200])  # [N]

    assessment_parameters = pd.Series({
        "P_A":               P_A,
        "P_L":                  2.5,
        "s_L":                   10,
    })

    # FKMLoadDistributionNormal, uses assessment_parameters.s_L, assessment_parameters.P_L, assessment_parameters.P_A
    scaled_load_sequence = load_sequence.fkm_safety_normal_from_stddev.scaled_load_sequence(assessment_parameters)
    gamma_L = load_sequence.fkm_safety_normal_from_stddev.gamma_L(assessment_parameters)

    assert np.isclose(gamma_L, resulting_gamma_L)
    np.testing.assert_allclose(scaled_load_sequence, result)

    # ------

    # test with a DataFrame with one column and multiple nodes
    load_sequence0 = pd.DataFrame(data={
            "col0": (np.array([[1],[0.1]]) * np.array([load_sequence.values])).T.flatten()
        },
        index=pd.MultiIndex.from_product([range(len(load_sequence.values)), [0,1]], names=["load_step", "node_id"])
    )
    scaled_load_sequence = load_sequence0.fkm_safety_normal_from_stddev.scaled_load_sequence(assessment_parameters)
    gamma_L = load_sequence0.fkm_safety_normal_from_stddev.gamma_L(assessment_parameters)

    assert np.isclose(gamma_L, resulting_gamma_L)
    np.testing.assert_allclose(scaled_load_sequence.col0[scaled_load_sequence.index.get_level_values("node_id")==0], np.array(result))
    np.testing.assert_allclose(scaled_load_sequence.col0[scaled_load_sequence.index.get_level_values("node_id")==1], np.array(result)*0.1)

    # ------

    # test with a DataFrame with multiple columns and multiple nodes
    load_sequence1 = pd.DataFrame(data={
            "col0": (np.array([[1],[0.1]]) * np.array([load_sequence.values])).T.flatten(),
            "col1": range(2*len(load_sequence))
        },
        index=pd.MultiIndex.from_product([range(len(load_sequence.values)), [0,1]], names=["load_step", "node_id"])
    )
    scaled_load_sequence = load_sequence1.fkm_safety_normal_from_stddev.scaled_load_sequence(assessment_parameters)
    gamma_L = load_sequence1.fkm_safety_normal_from_stddev.gamma_L(assessment_parameters)

    assert np.isclose(gamma_L, resulting_gamma_L)
    np.testing.assert_allclose(scaled_load_sequence.col0[scaled_load_sequence.index.get_level_values("node_id")==0], np.array(result))
    np.testing.assert_allclose(scaled_load_sequence.col0[scaled_load_sequence.index.get_level_values("node_id")==1], np.array(result)*0.1)
    np.testing.assert_allclose(scaled_load_sequence.col1, range(2*len(load_sequence.values)))


@pytest.mark.parametrize('P_L, resulting_gamma_L, result', [
    (2.5,  1.015313, [101.531312, -203.062625,  101.531312, -253.828281, 203.062625, 0., 203.062625, -203.062625]),
    (50, 1.063163, [106.316336, -212.632672,  106.316336, -265.790839,  212.632672,  0., 212.632672, -212.632672]),
])
def test_load_distribution_lognormal(P_L, resulting_gamma_L, result):

    # test with a plain series
    load_sequence = pd.Series([100, -200, 100, -250, 200, 0, 200, -200])  # [N]

    # FKMLoadDistributionLognormal, uses assessment_parameters.LSD_s, assessment_parameters.P_L, assessment_parameters.P_A
    assessment_parameters = pd.Series({"P_A": 7.2e-5, "P_L": P_L, "LSD_s": 1e-2})

    scaled_load_sequence = load_sequence.fkm_safety_lognormal_from_stddev.scaled_load_sequence(assessment_parameters)
    gamma_L = load_sequence.fkm_safety_lognormal_from_stddev.gamma_L(assessment_parameters)

    assert np.isclose(gamma_L, resulting_gamma_L)
    np.testing.assert_allclose(scaled_load_sequence, result)

    # ------

    # test with a DataFrame with one column and multiple nodes
    load_sequence0 = pd.DataFrame(data={
            "col0": (np.array([[1],[0.1]]) * np.array([load_sequence.values])).T.flatten()
        },
        index=pd.MultiIndex.from_product([range(len(load_sequence.values)), [0,1]], names=["load_step", "node_id"])
    )
    scaled_load_sequence = load_sequence0.fkm_safety_lognormal_from_stddev.scaled_load_sequence(assessment_parameters)
    gamma_L = load_sequence0.fkm_safety_lognormal_from_stddev.gamma_L(assessment_parameters)

    assert np.isclose(gamma_L, resulting_gamma_L)
    np.testing.assert_allclose(scaled_load_sequence.col0[scaled_load_sequence.index.get_level_values("node_id")==0], np.array(result))
    np.testing.assert_allclose(scaled_load_sequence.col0[scaled_load_sequence.index.get_level_values("node_id")==1], np.array(result)*0.1)

    # ------

    # test with a DataFrame with multiple columns and multiple nodes
    load_sequence1 = pd.DataFrame(data={
            "col0": (np.array([[1],[0.1]]) * np.array([load_sequence.values])).T.flatten(),
            "col1": range(2*len(load_sequence))
        },
        index=pd.MultiIndex.from_product([range(len(load_sequence.values)), [0,1]], names=["load_step", "node_id"])
    )
    scaled_load_sequence = load_sequence1.fkm_safety_lognormal_from_stddev.scaled_load_sequence(assessment_parameters)
    gamma_L = load_sequence1.fkm_safety_lognormal_from_stddev.gamma_L(assessment_parameters)

    assert np.isclose(gamma_L, resulting_gamma_L)
    np.testing.assert_allclose(scaled_load_sequence.col0[scaled_load_sequence.index.get_level_values("node_id")==0], np.array(result))
    np.testing.assert_allclose(scaled_load_sequence.col0[scaled_load_sequence.index.get_level_values("node_id")==1], np.array(result)*0.1)
    np.testing.assert_allclose(scaled_load_sequence.col1, range(2*len(load_sequence.values)))


def test_load_distribution_blanket():

    # test with a plain series
    load_sequence = pd.Series([100, -200, 100, -250, 200, 0, 200, -200])  # [N]

    # FKMLoadDistributionBlanket, uses input_parameters.P_L
    assessment_parameters = pd.Series({
        "P_L":                  2.5,
    })

    scaled_load_sequence = load_sequence.fkm_safety_blanket.scaled_load_sequence(assessment_parameters)
    gamma_L = load_sequence.fkm_safety_blanket.gamma_L(assessment_parameters)

    assert np.isclose(gamma_L, 1.1)
    np.testing.assert_allclose(scaled_load_sequence, [ 110., -220.,  110., -275.,  220.,    0.,  220., -220.])


def test_load_distribution_blanket_dataframe_single_column():

    # create a load_sequence with two nodes (node_id's 0 and 1) and eight values for each node
    # (the values of the second node are scaled by a factor of 10)
    # load_step node_id col0
    # 0         0        100
    # 0         1       1000
    # 1         0       -200
    # 1         1      -2000
    # 2         0        100
    # 2         1       1000
    # ...

    values = [100, -200, 100, -250, 200, 0, 200, -200]
    load_sequence = pd.DataFrame(data={
            "col0": (np.array([[1],[10]]) * np.array([values])).T.flatten()
        },
        index=pd.MultiIndex.from_product([range(len(values)), [0,1]], names=["load_step", "node_id"])
    )

    # FKMLoadDistributionBlanket, uses input_parameters.P_L
    assessment_parameters = pd.Series({
        "P_L":                  2.5,
    })

    scaled_load_sequence = load_sequence.fkm_safety_blanket.scaled_load_sequence(assessment_parameters)
    gamma_L = load_sequence.fkm_safety_blanket.gamma_L(assessment_parameters)

    assert np.isclose(gamma_L, 1.1)
    np.testing.assert_allclose(scaled_load_sequence.col0[scaled_load_sequence.index.get_level_values("node_id")==0], [110., -220.,  110., -275.,  220.,    0.,  220., -220.])
    np.testing.assert_allclose(scaled_load_sequence.col0[scaled_load_sequence.index.get_level_values("node_id")==1], [1100., -2200.,  1100., -2750.,  2200.,    0.,  2200., -2200.])


def test_load_distribution_blanket_dataframe_multiple_columns():

    # create a load_sequence with two nodes (node_id's 0 and 1) and eight values for each node
    # (the values of the second node are scaled by a factor of 10)
    # load_step node_id col0
    # 0         0        100
    # 0         1       1000
    # 1         0       -200
    # 1         1      -2000
    # 2         0        100
    # 2         1       1000
    # ...

    values = [100, -200, 100, -250, 200, 0, 200, -200]
    load_sequence = pd.DataFrame(data={
            "col0": (np.array([[1],[10]]) * np.array([values])).T.flatten(),
            "col1": range(16)
        },
        index=pd.MultiIndex.from_product([range(len(values)), [0,1]], names=["load_step", "node_id"])
    )

    # FKMLoadDistributionBlanket, uses input_parameters.P_L
    assessment_parameters = pd.Series({
        "P_L":                  2.5,
    })

    scaled_load_sequence = load_sequence.fkm_safety_blanket.scaled_load_sequence(assessment_parameters)
    gamma_L = load_sequence.fkm_safety_blanket.gamma_L(assessment_parameters)

    assert np.isclose(gamma_L, 1.1)
    np.testing.assert_allclose(scaled_load_sequence.col0[scaled_load_sequence.index.get_level_values("node_id")==0], [110., -220.,  110., -275.,  220.,    0.,  220., -220.])
    np.testing.assert_allclose(scaled_load_sequence.col0[scaled_load_sequence.index.get_level_values("node_id")==1], [1100., -2200.,  1100., -2750.,  2200.,    0.,  2200., -2200.])
    np.testing.assert_allclose(scaled_load_sequence.col1, range(16))

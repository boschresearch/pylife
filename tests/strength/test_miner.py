# Copyright (c) 2019-2020 - for information on the respective copyright owner
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

import numpy as np
import pandas as pd
import pdb
import pytest
import os
import sys

from . import data
from .data import collective

import pylife.strength.miner as miner
import pylife.strength.meanstress
from pylife.strength import miner
from pylife.stress.rainflow import *
from pylife.stress.timesignal import TimeSignalGenerator


# parameters for the example given in Haibach2006
sn_curve_parameters_haibach = {
    "ND_50": 1E6,
    "k_1": 4,
    "SD_50": 100
}
# parameters for the example given in Haibach2006
sn_curve_parameters_elementar = {
    "ND_50": 2E6,
    "k_1": 7,
    "SD_50": 125
}


@pytest.fixture(scope="module")
def miner_base():
    return miner.MinerBase(ND_50=10**6, k_1=5, SD_50=200)


@pytest.fixture(scope="module")
def miner_elementar():
    return miner.MinerElementar(ND_50=10**6, k_1=6, SD_50=200)


@pytest.fixture(scope="function")
def miner_haibach():
    m = miner.MinerHaibach(ND_50=10**6, k_1=6, SD_50=200)
    m.setup(collective)
    return m


def generate_transformed_pylife_collective():
    tsgen = TimeSignalGenerator(10, {'number': 50,
                                     'amplitude_median': 1.0, 'amplitude_std_dev': 0.5,
                                     'frequency_median': 4, 'frequency_std_dev': 3,
                                     'offset_median': 0, 'offset_std_dev': 0.4},
                                None, None)

    y = tsgen.query(2000000)

    rfc = RainflowCounterThreePoint().process(y)
    rfm = rfc.get_rainflow_matrix_frame(128)

    transformed = rfm.meanstress_hist.FKM_goodman(pd.Series({'M': 0.3, 'M2': 0.1}), R_goal=-1)

    return transformed


@pytest.fixture(scope="module")
def transformed_pylife_collective():
    return generate_transformed_pylife_collective()


def test_get_accumulated_from_relative_collective():
    coll_acc = miner.get_accumulated_from_relative_collective(
        data.coll_haibach_mod_rel
    )
    assert np.allclose(data.coll_haibach_mod_acc,
                       coll_acc
                       )


class TestMinerBase:
    coll = collective

    def test_init(self, miner_base):
        """Dummy test case for demonstration"""
        np.testing.assert_almost_equal(miner_base.ND_50, 10**6)

        with pytest.raises(TypeError):
            miner.MinerBase(ND_50="str", k_1="str", SD_50="str")

        with pytest.raises(ValueError):
            miner.MinerBase(ND_50=15, k_1=10, SD_50=200)

    def test__parse_collective_splitting_of_array(self, miner_base):
        miner_base._parse_collective(self.coll)

#        assert np.allclose(miner_base.S_collective, self.coll[:, 0])
#        assert np.allclose(miner_base.N_collective_accumulated, self.coll[:, 1])
        assert np.allclose(miner_base.S_collective, data.S_collective_normalized)
        assert np.allclose(miner_base.N_collective_accumulated, data.N_collective_accumulated)

    def test__parse_collective_relative_cycles(self, miner_base):
        miner_base._parse_collective(self.coll)

        assert np.allclose(miner_base.N_collective_accumulated, data.N_collective_accumulated)

    def test__parse_collective_relative(self, miner_base):
        miner_base._parse_collective(self.coll)

        assert np.allclose(miner_base.collective_relative,
                           np.stack((data.S_collective_normalized, data.N_collective_relative), axis=1))

    def test_calc_A_not_derived_class_fail(self, miner_base):
        with pytest.raises(NotImplementedError):
            miner_base.calc_A(self.coll)

    def test_pylife_collective_transformation(self, miner_base,
                                              transformed_pylife_collective):
        coll = transformed_pylife_collective
        parsed = miner_base._transform_pylife_collective(coll)
        # load classes have to be in ascending order
        assert np.all(np.diff(parsed[:, 0]) > 0 )
        # the frequencies are turned into accumulated frequencies in descending order
        assert np.all(np.diff(parsed[:, 1]) <= 0 )

    def test_parse_pylife_collective(self, miner_base,
                                     transformed_pylife_collective):
        coll = transformed_pylife_collective
        miner_base.setup(coll)


class TestMinerElementar:
    coll = collective

    def test_setup(self, miner_elementar):
        miner_elementar.setup(self.coll)

        H0 = self.coll[0, 1]
        np.testing.assert_almost_equal(H0, 190733.0)
        np.testing.assert_almost_equal(H0, miner_elementar.H0)
        # calculation of Zeitfestigkeitsfaktor
        # ND_50 = 10**6
        # k = 6
        z = 1.3180413239445 # = (ND_50 / H0)**(1. / k)
        np.testing.assert_almost_equal(miner_elementar.zeitfestigkeitsfaktor_collective, z)

    def test_A(self, miner_elementar):
        miner_elementar.setup(self.coll)

        # test requires k = 6
        A = miner_elementar.calc_A(self.coll)
        np.testing.assert_almost_equal(A, 862.99608075)

    def test_d_m(self, miner_elementar):
        if miner_elementar.A is None:
            miner_elementar.calc_A(self.coll)

        d_m = miner_elementar.effective_damage_sum(miner_elementar.A)
        d_m_test = 0.36900120961677296
        np.testing.assert_almost_equal(d_m, d_m_test)

    def test_d_m_limits(self, miner_elementar):
        # lower limit for d_m = 1
        assert miner_elementar.effective_damage_sum(A=16) == 1.0
        # upper limit for d_m = 0.3
        assert miner_elementar.effective_damage_sum(A=1975.308641975309) == 0.3

    def test_N_predict(self, miner_elementar):
        load_level = 400
        A = miner_elementar.calc_A(self.coll)
        N_woehler_load_level = (
            miner_elementar.ND_50 * (load_level / miner_elementar.SD_50)**(-miner_elementar.k_1)
        )
        N_predict = N_woehler_load_level * A
        np.testing.assert_almost_equal(miner_elementar.N_predict(load_level, A), N_predict)
        np.testing.assert_almost_equal(miner_elementar.N_predict(load_level), N_predict)

    def test_damage_accumulation_validation(self):
        """The test uses data from the book Haibach2006

        See miner.py for reference.
        The examples can be found on page 271.
        """
        coll = data.coll_elementar_acc
        load_level = coll[:, 0].max()
        expected_N = 2167330
        sn_curve_parameters = sn_curve_parameters_elementar
        miner_elementar = miner.MinerElementar(**sn_curve_parameters)
        miner_elementar.setup(coll)
        # some rounding is assumed in the example from the book
        # so in respect to millions of cycles a small neglectable tolerance is accepted
        np.testing.assert_approx_equal(expected_N, miner_elementar.N_predict(load_level), significant=6)


class TestMinerHaibach:
    def test_calc_A_no_collective_specified(self):
        m = miner.MinerHaibach(ND_50=10**6, k_1=5, SD_50=200)
        with pytest.raises(RuntimeError):
            m.calc_A(load_level=240)

    def test_calc_A_load_level_smaller_ND_50(self, miner_haibach):
        A = miner_haibach.calc_A(load_level=(miner_haibach.SD_50/ 2.))
        assert A == np.inf

    def test_calc_A_split_damage_regions(self, miner_haibach):
        load_level = 400
        miner_haibach.calc_A(load_level=load_level)
        assert np.array_equal(
            miner_haibach.evaluated_load_levels[load_level]["i_full_damage"],
            data.i_full_damage
        )
        assert np.array_equal(
            miner_haibach.evaluated_load_levels[load_level]["i_reduced_damage"],
            data.i_reduced_damage
        )

    def test_calc_A(self, miner_haibach):
        load_level = 400
        A = miner_haibach.calc_A(load_level)
        np.testing.assert_almost_equal(A, 1000.342377197)

    def test_N_predict(self, miner_haibach):
        load_level = 400
        N_woehler_load_level = (
            miner_haibach.ND_50 * (load_level / miner_haibach.SD_50)**(-miner_haibach.k_1)
        )
        A = miner_haibach.calc_A(load_level)
        N_predict = N_woehler_load_level * A
        np.testing.assert_almost_equal(miner_haibach.N_predict(load_level, A), N_predict)
        np.testing.assert_almost_equal(miner_haibach.N_predict(load_level), N_predict)

    @pytest.mark.parametrize("load_level, predicted_N", [
        (100, 935519000),
        (150, 61605400),
        (200, 11994100),
        (300, 1761820),
        (400, 505981),
        (600, 99382),
        (800, 31103)
    ])
    def test_damage_accumulation_validation(self, load_level, predicted_N):
        """The test uses data from the book Haibach2006

        See miner.py for reference.
        The examples can be found on page 292.
        """
        coll = data.coll_haibach_mod_acc
        sn_curve_parameters = sn_curve_parameters_haibach
        miner_haibach = miner.MinerHaibach(**sn_curve_parameters)
        miner_haibach.setup(coll)
        # some rounding is assumed in the example from the book
        # so in respect to millions of cycles a small neglectable tolerance is accepted
        np.testing.assert_approx_equal(predicted_N, miner_haibach.N_predict(load_level), significant=5)

    def test_N_predict_inf_rule(self):
        coll = data.coll_haibach_mod_acc
        sn_curve_parameters = sn_curve_parameters_haibach
        miner_haibach = miner.MinerHaibach(**sn_curve_parameters)
        miner_haibach.setup(coll)
        load_level = 50
        expected_N_100MPa = 935519000
        N_pred_ignored = miner_haibach.N_predict(load_level, ignore_inf_rule=True)
        N_pred_not_ignored = miner_haibach.N_predict(load_level, ignore_inf_rule=False)
        assert N_pred_not_ignored == np.inf
        assert N_pred_ignored > expected_N_100MPa

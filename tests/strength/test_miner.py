# Copyright (c) 2019-2021 - for information on the respective copyright owner
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

import pylife.strength.miner as miner
from pylife.strength import miner
from pylife.stress.rainflow import *


from . import data

def make_collective_from_raw_data(raw_collective):
    smallest_length = np.abs(np.diff(raw_collective[:, 0])).min()
    amplitude_left = raw_collective[:, 0] - smallest_length/2.
    amplitude_right = raw_collective[:, 0] + smallest_length/2

    decumulated = np.append(np.abs(np.diff(raw_collective[:, 1])), raw_collective[-1, 1])
    coll = pd.Series(decumulated,
                     index=pd.IntervalIndex.from_arrays(2.*amplitude_left, 2.*amplitude_right, name='range'),
                     name='cycles')
    return coll

@pytest.fixture
def collective(request):
    request.cls.coll = make_collective_from_raw_data(data.collective)



# parameters for the example given in Haibach2006
sn_curve_parameters_haibach = pd.Series({
    "ND": 1E6,
    "k_1": 4,
    "SD": 100
})
# parameters for the example given in Haibach2006
sn_curve_parameters_elementar = pd.Series({
    "ND": 2E6,
    "k_1": 7,
    "SD": 125
})


@pytest.fixture
def miner_elementar():
    return miner.MinerElementar.from_parameters(ND=1e6, k_1=6, SD=200.)

@pytest.fixture
def miner_haibach():
    return miner.MinerHaibach.from_parameters(ND=10**6, k_1=6, SD=200)


def test_effective_damage_sum_limitations():
    # lower limit for d_m = 1
    assert miner.effective_damage_sum(16) == 1.0
    # upper limit for d_m = 0.3
    assert miner.effective_damage_sum(1975.308641975309) == 0.3


@pytest.mark.usefixtures('collective')
class TestMinerElementar():

    def test_lifetime_factor(self, miner_elementar):
        z = 1.3180413239445 # = (ND / H0)**(1. / k)
        np.testing.assert_almost_equal(miner_elementar.calc_zeitfestigkeitsfaktor(190733.0), z)

    def test_lifetime_multiple(self, miner_elementar):
        # test requires k = 6

        A = miner_elementar.lifetime_multiple(self.coll)
        np.testing.assert_almost_equal(A, 862.99608075)

    def test_d_m(self, miner_elementar):
        d_m = miner_elementar.effective_damage_sum(self.coll)
        d_m_test = 0.36900120961677296
        np.testing.assert_almost_equal(d_m, d_m_test)

    def test_N_predict(self, miner_elementar):
        load_level = 400
        expected_cycles = 13484313.761758052
        gassner = miner_elementar.gassner(self.coll)
        np.testing.assert_almost_equal(gassner.cycles(load_level), expected_cycles)

        foo_collective = pd.Series({
            'amplitude': load_level,
            'cycles': expected_cycles
        })
        expected_damage = pd.Series(1.0, name='damage')
        pd.testing.assert_series_equal(gassner.damage(foo_collective), expected_damage)

    def test_damage_accumulation_validation(self):
        """The test uses data from the book Haibach2006

        See miner.py for reference.
        The examples can be found on page 271.
        """
        coll = make_collective_from_raw_data(data.coll_elementar_acc)
        load_level = coll.rainflow.amplitude.max()
        expected_N = 2167330
        miner_elementar = sn_curve_parameters_elementar.miner_elementar
        # some rounding is assumed in the example from the book
        # so in respect to millions of cycles a small neglectable tolerance is accepted
        gassner = miner_elementar.gassner(coll)
        np.testing.assert_approx_equal(expected_N, gassner.cycles(load_level), significant=6)


@pytest.mark.usefixtures('collective')
class TestMinerHaibach:

    def test_lifetime_multiple_split_damage_regions(self, miner_haibach):
        load_level = 400
        miner_haibach.lifetime_multiple(self.coll, load_level)

    def test_lifetime_multiple(self, miner_haibach):
        load_level = 400
        A = miner_haibach.lifetime_multiple(self.coll, load_level)
        np.testing.assert_almost_equal(A, 1000.342377197)

    def test_lifetime_multiple_no_load_level(self, miner_haibach):
        A = miner_haibach.lifetime_multiple(self.coll.rainflow.scale(200.).to_pandas())
        np.testing.assert_almost_equal(A, 1061.21644181784)

    def test_N_predict(self, miner_haibach):
        load_level = 400
        N_woehler_load_level = (
            miner_haibach.ND * (load_level / miner_haibach.SD)**(-miner_haibach.k_1)
        )
        A = miner_haibach.lifetime_multiple(self.coll, load_level)
        N_predict = N_woehler_load_level * A

        gassner = miner_haibach.gassner(self.coll, load_level)

        np.testing.assert_almost_equal(gassner.cycles(load_level), N_predict)

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
        coll = make_collective_from_raw_data(data.coll_haibach_mod_acc)

        miner_haibach = sn_curve_parameters_haibach.miner_haibach
        # some rounding is assumed in the example from the book
        # so in respect to millions of cycles a small neglectable tolerance is accepted
        gassner = miner_haibach.gassner(coll, load_level)
        np.testing.assert_approx_equal(predicted_N, gassner.cycles(load_level), significant=5)

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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import pytest
import numpy as np
import pandas as pd

import pylife.materialdata.woehler.accessors


wc_data = pd.Series({
    'k_1': 7.,
    'TN': 1. / 1.75,
    'ND_50': 1e6,
    'SD_50': 300.0
})


def test_woehler_accessor():
    wc = wc_data.drop('TN')

    for key in wc.index:
        wc_miss = wc.drop(key)
        with pytest.raises(AttributeError):
            wc_miss.woehler


def test_woehler_basquin_cycles_50():
    load = [200., 300., 400., 500.]

    cycles = wc_data.woehler.basquin_cycles(load)
    expected_cycles = [np.inf, 1e6,  133484,    27994]

    np.testing.assert_allclose(cycles, expected_cycles, rtol=1e-4)


def test_woehler_basquin_cycles_50_same_k():
    load = [200., 300., 400., 500.]

    wc = wc_data.copy()
    wc['k_2'] = wc['k_1']
    cycles = wc.woehler.basquin_cycles(load)

    calculated_k = - (np.log(cycles[-1]) - np.log(cycles[0])) / (np.log(load[-1]) - np.log(load[0]))
    np.testing.assert_approx_equal(calculated_k, wc.k_1)


def test_woehler_basquin_cycles_10_90():
    load = [200., 300., 400., 500.]

    cycles_10 = wc_data.woehler.basquin_cycles(load, 0.1)[1:]
    cycles_90 = wc_data.woehler.basquin_cycles(load, 0.9)[1:]

    expected = [0., 1. / 1.75, 1. / 1.75]
    np.testing.assert_allclose(cycles_90/cycles_10, expected)


def test_woehler_basquin_load_50():
    cycles = [np.inf, 1e6,  133484,    27994]

    load = wc_data.woehler.basquin_load(cycles)
    expected_load = [300., 300., 400., 500.]

    np.testing.assert_allclose(load, expected_load, rtol=1e-4)


def test_woehler_basquin_load_50_same_k():
    cycles = [1e7, 1e6, 1e5, 1e4]

    wc = wc_data.copy()
    wc['k_2'] = wc['k_1']

    load = wc.woehler.basquin_load(cycles)
    calculated_k = - (np.log(cycles[-1]) - np.log(cycles[0])) / (np.log(load[-1]) - np.log(load[0]))
    np.testing.assert_approx_equal(calculated_k, wc.k_1)


def test_woehler_basquin_load_10_90():
    cycles = [1e2, 1e7]

    load_10 = wc_data.woehler.basquin_load(cycles, 0.1)
    load_90 = wc_data.woehler.basquin_load(cycles, 0.9)

    expected = np.full_like(cycles, 1. / 1.75 ** (1./7.))

    np.testing.assert_allclose(load_90/load_10, expected, rtol=1e-4)


def test_woehler_TS_and_TN_guessed():
    wc = pd.Series({
        'k_1': 0.5,
        'SD_50': 300,
        'ND_50': 1e6
    })
    assert wc.woehler.TN == 1.0
    assert wc.woehler.TS == 1.0


def test_woehler_TS_guessed():
    wc = wc_data.copy()
    wc['k_1'] = 0.5
    wc['TN'] = 1. / 1.5

    assert wc.woehler.TS == 1. / (1.5 * 1.5)


def test_woehler_TN_guessed():
    wc = wc_data.copy()
    wc['k_1'] = 0.5
    wc['TS'] = 1. / (1.5 * 1.5)

    assert wc.woehler.TN == 1. / 1.5


def test_woehler_TS_given():
    wc_full = wc_data.copy()
    wc_full['TS'] = 1. / 1.25
    assert wc_full.woehler.TS == 1. / 1.25


def test_woehler_miner_original():
    assert wc_data.woehler.k_2 == np.inf


def test_woehler_miner():
    assert wc_data.woehler.miner_elementary().k_2 == wc_data.k_1


def test_woehler_miner_haibach():
    assert wc_data.woehler.miner_haibach().k_2 == 13.0

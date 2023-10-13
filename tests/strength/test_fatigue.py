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



import pytest
import numpy as np
import pandas as pd

import pylife.strength.fatigue
import pylife.stress


def load_index():
    nCode_Xbinsize = 62.375
    nCode_XMax = 468.8125
    nCode_XMin = 32.1875

    return pd.interval_range(start=nCode_XMin-nCode_Xbinsize/2,
                             end=nCode_XMax+nCode_Xbinsize/2, periods=8, name="range")


expected_elementary_1 = pd.Series([
    2.057849E-06,
    3.039750E-05,
    1.507985E-05,
    6.434856E-06,
    6.296737E-06,
    1.751881E-06,
    0.000000E+00,
    0.000000E+00
], name='damage', index=load_index())
expected_elementary_2 = pd.Series([
    1.868333E-06,
    3.894329E-05,
    4.829155E-05,
    1.543643E-05,
    1.357734E-05,
    1.379607E-05,
    1.023415E-05,
    7.548524E-07
], name='damage', index=load_index())
expected_elementary_3 = pd.Series([
    3.067487E-06,
    1.999014E-08,
    0.000000E+00,
    0.000000E+00,
    0.000000E+00,
    0.000000E+00,
    0.000000E+00,
    0.000000E+00
], name='damage', index=load_index())

expected_haibach_1 = pd.Series([
    2.293973E-08,
    7.414563E-05,
    4.631857E-04,
    4.779392E-04,
    6.006944E-04,
    2.041326E-04,
    0.000000E+00,
    0.000000E+00
], name='damage', index=load_index())
expected_haibach_2 = pd.Series([
    2.083222E-08,
    9.499395E-05,
    1.483300E-03,
    1.146517E-03,
    1.295247E-03,
    1.607545E-03,
    1.408691E-03,
    1.198482E-04
], name='damage', index=load_index())
expected_haibach_3 = pd.Series([
    3.419927E-08,
    4.876799E-08,
    0.000000E+00,
    0.000000E+00,
    0.000000E+00,
    0.000000E+00,
    0.000000E+00,
    0.000000E+00
], name='damage', index=load_index())

expected_original_1 = pd.Series([
    0.00000000E+00,
    0.00000000E+00,
    2.08681840E-05,
    1.73881920E-05,
    2.80698130E-05,
    1.16511320E-05,
    0.00000000E+00,
    0.00000000E+00,
], name='damage', index=load_index())
expected_original_2 = pd.Series([
    0.00000000E+00,
    0.00000000E+00,
    6.68280320E-05,
    4.17121220E-05,
    6.05255350E-05,
    9.17526660E-05,
    9.49790820E-05,
    9.32071420E-06
], name='damage', index=load_index())
expected_original_3 = pd.Series([
    0.00000000E+00,
    0.00000000E+00,
    0.00000000E+00,
    0.00000000E+00,
    0.00000000E+00,
    0.00000000E+00,
    0.00000000E+00,
    0.00000000E+00
], name='damage', index=load_index())


material = pd.DataFrame(
    index=['k_1', 'ND', 'SD'],
    columns=['elementary', 'haibach', 'original'],
    data=[[4., 5., 6.], [4e7, 1e6, 1e8], [100., 90., 75.]]
)

load_hist_1 = pd.Series([
    1.227E5,
    2.433E4,
    1591.0,
    178.0,
    64.0,
    8.0,
    0.0,
    0.0,
], name='cycles', index=load_index())

load_hist_2 = pd.Series([
    1.114E5,
    3.117E4,
    5095.0,
    427.0,
    138.0,
    63.0,
    24.0,
    1.0,
], name='cycles', index=load_index())

load_hist_3 = pd.Series([
    1.829E5,
    16.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
], name='cycles', index=load_index())


@pytest.mark.parametrize('load_hist, expected', [
    (load_hist_1, expected_elementary_1),
    (load_hist_2, expected_elementary_2),
    (load_hist_3, expected_elementary_3),
])
def test_damage_fatigue_elementary(load_hist, expected):
    fatigue = material['elementary'].fatigue.miner_elementary().damage(load_hist.load_collective)
    pd.testing.assert_series_equal(fatigue, expected, rtol=1e-3)


@pytest.mark.parametrize('load_hist, expected', [
    (load_hist_1, expected_haibach_1),
    (load_hist_2, expected_haibach_2),
    (load_hist_3, expected_haibach_3),
])
def test_damage_fatigue_haibach(load_hist, expected):
    fatigue = material['haibach'].fatigue.miner_haibach().damage(load_hist.load_collective)
    pd.testing.assert_series_equal(fatigue, expected, rtol=1e-3)


@pytest.mark.parametrize('load_hist, expected', [
    (load_hist_1, expected_original_1),
    (load_hist_2, expected_original_2),
    (load_hist_3, expected_original_3),
])
def test_damage_fatigue_original(load_hist, expected):
    fatigue = material['original'].fatigue.damage(load_hist.load_collective)
    pd.testing.assert_series_equal(fatigue, expected, rtol=1e-3)


@pytest.mark.parametrize('TS, allowed_pf, expected', [
    (2., 0.5, 5.0),
    (1.0000001, 0.1, 5.0),
    (1.25, 1e-6, 3.3055576111)
])
def test_security_load_single(TS, allowed_pf, expected):
    wc = pd.Series({
        'SD': 500.,
        'k_1': 6.0,
        'TS': TS,
        'ND': 1e6
    })
    load_collective = pd.DataFrame({'mean': [0.0], 'amplitude': [100.0], 'cycles': 1e7})
    security_factor = wc.fatigue.security_load(load_collective, allowed_pf)
    np.testing.assert_allclose(security_factor, expected)


def test_security_load_multiple():
    idx = pd.Index([1, 2, 3], name='element_id')
    wc = pd.DataFrame({
        'SD': [500., 500., 300.],
        'k_1': [6., 6., 6.],
        'TS': [2., 1.0000001, 1.25],
        'ND': [1e6, 1e6, 1e6]
    }, index=idx)
    load_signal = pd.DataFrame({
        'mean': [0., 0., 0.],
        'amplitude': [250., 125.,  100.],
        'cycles': [1e7, 1e7, 1e7]
    }, index=idx)

    expected = pd.Series([2.0, 4.0, 3.0], index=idx, name='security_factor')
    result = wc.fatigue.security_load(load_signal, 0.5)

    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize('TN, allowed_pf, cycles, expected', [
    (4., 0.5, 1e5, 10.0),
    (4., 0.5, 1e4, 100.0),
    (4., 0.1, 1e4, 50.0),
    (4., 1e-6, 1e4, 7.65),
    (4., 1e-6, 1e3, 76.5),
])
def test_security_cycles_single(TN, allowed_pf, cycles, expected):
    wc = pd.Series({
        'SD': 500.,
        'k_1': 0.5,
        'TN': TN,
        'ND': 1e6
    })
    load_collective = pd.DataFrame({'mean': [0.0], 'amplitude': [500.0], 'cycles': [cycles]})
    security_factor = wc.fatigue.security_cycles(load_collective, allowed_pf)
    np.testing.assert_allclose(security_factor, expected, rtol=1e-3)


def test_security_cycles_multiple():
    idx = pd.Index([1, 2, 3], name='element_id')
    wc = pd.DataFrame({
        'SD': [500., 500., 500.],
        'k_1': [.5, .5, .5],
        'TN': [4., 4., 4.],
        'ND': [1e6, 1e6, 1e6],
    }, index=idx)

    load_collective = pd.DataFrame({
        'amplitude': [500., 500., 500.],
        'cycles': [1e6, 1e5, 1e4],
    }, index=idx)

    expected = pd.Series([1., 10., 100.], index=idx, name='security_factor')
    result = wc.fatigue.security_cycles(load_collective, 0.5)

    pd.testing.assert_series_equal(result, expected)

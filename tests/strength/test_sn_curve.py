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


import pandas as pd
import pytest
from pylife.strength import sn_curve


def load_index():
    nCode_Xbinsize = 62.375
    nCode_XMax = 468.8125
    nCode_XMin = 32.1875

    return pd.interval_range(start=nCode_XMin-nCode_Xbinsize/2,
                             end=nCode_XMax+nCode_Xbinsize/2, periods=8, name="range")


def nCode_reference_results():
    elementar = pd.DataFrame([[2.057849E-06, 1.868333E-06, 3.067487E-06],
                              [3.039750E-05, 3.894329E-05, 1.999014E-08],
                              [1.507985E-05, 4.829155E-05, 0.000000E+00],
                              [6.434856E-06, 1.543643E-05, 0.000000E+00],
                              [6.296737E-06, 1.357734E-05, 0.000000E+00],
                              [1.751881E-06, 1.379607E-05, 0.000000E+00],
                              [0.000000E+00, 1.023415E-05, 0.000000E+00],
                              [0.000000E+00, 7.548524E-07, 0.000000E+00]],
                             index=load_index()).stack()
    miner_haibach = pd.DataFrame([[2.293973E-08, 2.083222E-08, 3.419927E-08],
                                  [7.414563E-05, 9.499395E-05, 4.876799E-08],
                                  [4.631857E-04, 1.483300E-03, 0.000000E+00],
                                  [4.779392E-04, 1.146517E-03, 0.000000E+00],
                                  [6.006944E-04, 1.295247E-03, 0.000000E+00],
                                  [2.041326E-04, 1.607545E-03, 0.000000E+00],
                                  [0.000000E+00, 1.408691E-03, 0.000000E+00],
                                  [0.000000E+00, 1.198482E-04, 0.000000E+00]],
                                 index=load_index()).stack()
    original = pd.DataFrame([[0.00000000E+00, 0.00000000E+00, 0.00000000E+00],
                             [0.00000000E+00, 0.00000000E+00, 0.00000000E+00],
                             [2.08681840E-05, 6.68280320E-05, 0.00000000E+00],
                             [1.73881920E-05, 4.17121220E-05, 0.00000000E+00],
                             [2.80698130E-05, 6.05255350E-05, 0.00000000E+00],
                             [1.16511320E-05, 9.17526660E-05, 0.00000000E+00],
                             [0.00000000E+00, 9.49790820E-05, 0.00000000E+00],
                             [0.00000000E+00, 9.32071420E-06, 0.00000000E+00]],
                            index=load_index()).stack()
    elementar.name = 'damage'
    miner_haibach.name = 'damage'
    original.name = 'damage'
    return [elementar, miner_haibach, original]


material = pd.DataFrame(index=['k_1', 'ND_50', 'SD_50'],
                        columns=['elementar', 'MinerHaibach', 'original'],
                        data=[[4, 5, 6], [4e7, 1e6, 1e8], [100, 90, 75]])

loads = pd.DataFrame([[1.227E5, 1.114E5, 1.829E5], [2.433E4, 3.117E4, 16],
                      [1591, 5095, 0], [178, 427, 0], [64, 138, 0], [8, 63, 0],
                      [0, 24, 0], [0, 1, 0]],
                     index=load_index()).stack()


@pytest.mark.parametrize('method, expected', zip(material, nCode_reference_results()))
def test_calc_damage(method, expected):
    with pytest.warns(DeprecationWarning):
        damage_calc = sn_curve.FiniteLifeCurve(**material[method])
    damage = damage_calc.calc_damage(loads, method=method)
    pd.testing.assert_series_equal(damage, expected, rtol=0.001, atol=0)


def test_calc_damage_index_name():
    foo_loads = loads.copy()
    foo_loads.index.name = 'foo'
    with pytest.warns(DeprecationWarning):
        damage = sn_curve.FiniteLifeCurve(**material['elementar']).calc_damage(foo_loads,
                                                                               method='elementar',
                                                                               index_name='foo')

    assert damage.index.names[0] == 'foo'

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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import pytest
import numpy as np
import pandas as pd

from hypothesis import given, settings, strategies as st

from pylife.materiallaws import WoehlerCurve


@pytest.fixture
def wc_data():
    return pd.Series({
    'k_1': 7.,
    'TN': 1.75,
    'ND': 1e6,
    'SD': 300.0
})


def test_woehler_accessor(wc_data):
    wc = wc_data.drop('TN')

    for key in wc.index:
        wc_miss = wc.drop(key)
        with pytest.raises(AttributeError):
            wc_miss.woehler


def test_woehler_transform_probability():
    wc_50 = pd.Series({
        'k_1': 2,
        'k_2': np.inf,
        'TS': 2.,
        'TN': 9.,
        'ND': 3e6,
        'SD': 300 * np.sqrt(2.),
        'failure_probability': 0.5
    }).sort_index()

    transformed_90 = wc_50.woehler.transform_to_failure_probability(0.9).to_pandas()
    pd.testing.assert_series_equal(transformed_90[['SD', 'ND', 'failure_probability']],
                                   pd.Series({'SD': 600.0, 'ND': 4.5e6, 'failure_probability': 0.9}))
    transformed_back = transformed_90.woehler.transform_to_failure_probability(0.5).to_pandas()
    pd.testing.assert_series_equal(transformed_back, wc_50)

    transformed_10 = wc_50.woehler.transform_to_failure_probability(0.1).to_pandas()
    pd.testing.assert_series_equal(transformed_10[['SD', 'ND', 'failure_probability']],
                                   pd.Series({'SD': 300.0, 'ND': 2e6, 'failure_probability': 0.1}))
    transformed_back = transformed_10.woehler.transform_to_failure_probability(0.5).to_pandas()
    pd.testing.assert_series_equal(transformed_back, wc_50)


def test_woehler_transform_probability_multiple():
    wc_50 = pd.Series({
        'k_1': 2,
        'k_2': np.inf,
        'TS': 2.,
        'TN': 9.,
        'ND': 3e6,
        'SD': 300 * np.sqrt(2.),
        'failure_probability': 0.5
    }).sort_index()

    transformed = wc_50.woehler.transform_to_failure_probability([.1, .9]).to_pandas()
    expected = pd.DataFrame({
        'k_1': [2., 2.],
        'k_2': [np.inf, np.inf],
        'TS': [2., 2.],
        'TN': [9., 9.],
        'ND': [2e6, 4.5e6],
        'SD': [300., 600.],
        'failure_probability': [0.1, 0.9]
    })

    pd.testing.assert_frame_equal(transformed, expected, check_like=True)

    transformed_back = transformed.woehler.transform_to_failure_probability([0.5, 0.5]).to_pandas()
    expected = pd.DataFrame({
        'k_1': [2., 2.],
        'k_2': [np.inf, np.inf],
        'TS': [2., 2.],
        'TN': [9., 9.],
        'ND': [3e6, 3e6],
        'SD': [300. * np.sqrt(2.), 300. * np.sqrt(2.)],
        'failure_probability': [0.5, 0.5]
    })

    pd.testing.assert_frame_equal(transformed_back, expected, check_like=True)


def test_woehler_transform_probability_SD_0():
    wc_50 = pd.Series({
        'k_1': 2,
        'k_2': np.inf,
        'TS': 2.,
        'TN': 9.,
        'ND': 3e6,
        'SD': 0.0,
        'failure_probability': 0.5
    }).sort_index()

    transformed_90 = wc_50.woehler.transform_to_failure_probability(0.9).to_pandas()
    pd.testing.assert_series_equal(transformed_90[['SD', 'ND', 'failure_probability']],
                                   pd.Series({'SD': 0, 'ND': 9e6, 'failure_probability': 0.9}))
    transformed_back = transformed_90.woehler.transform_to_failure_probability(0.5).to_pandas()
    pd.testing.assert_series_equal(transformed_back, wc_50)

    transformed_10 = wc_50.woehler.transform_to_failure_probability(0.1).to_pandas()
    pd.testing.assert_series_equal(transformed_10[['SD', 'ND', 'failure_probability']],
                                   pd.Series({'SD': 0.0, 'ND': 1e6, 'failure_probability': 0.1}))
    transformed_back = transformed_10.woehler.transform_to_failure_probability(0.5).to_pandas()
    pd.testing.assert_series_equal(transformed_back, wc_50)


def test_woehler_basquin_cycles_50_single_load_single_wc(wc_data):
    load = 500.

    cycles = wc_data.woehler.basquin_cycles(load)
    expected_cycles = 27994

    np.testing.assert_allclose(cycles, expected_cycles, rtol=1e-4)


def test_woehler_basquin_cycles_50_multiple_load_single_wc(wc_data):
    load = [200., 300., 400., 500.]

    cycles = wc_data.woehler.basquin_cycles(load)
    expected_cycles = [np.inf, 1e6,  133484,    27994]

    np.testing.assert_allclose(cycles, expected_cycles, rtol=1e-4)


def test_woehler_basquin_cycles_50_single_load_multiple_wc():
    load = 400.

    wc = pd.DataFrame({
        'k_1': [1., 2., 2.],
        'SD': [300., 400., 500.],
        'ND': [1e6, 1e6, 1e6]
    })

    cycles = wc.woehler.basquin_cycles(load)
    expected_cycles = [7.5e5, 1e6, np.inf]

    np.testing.assert_allclose(cycles, expected_cycles, rtol=1e-4)


def test_woehler_basquin_cycles_50_multiple_load_multiple_wc():
    load = [3000., 400., 500.]

    wc = pd.DataFrame({
        'k_1': [1., 2., 2.],
        'SD': [300., 400., 500.],
        'ND': [1e6, 1e6, 1e6]
    })

    cycles = wc.woehler.basquin_cycles(load)
    expected_cycles = [1e5, 1e6, 1e6]

    np.testing.assert_allclose(cycles, expected_cycles, rtol=1e-4)


def test_woehler_basquin_cycles_50_multiple_load_multiple_wc_aligned_index():
    index = pd.Index([1, 2, 3], name='element_id')
    load = pd.Series([3000., 400., 500.], index=index)

    wc = pd.DataFrame({
        'k_1': [1., 2., 2.],
        'SD': [300., 400., 500.],
        'ND': [1e6, 1e6, 1e6]
    }, index=index)

    cycles = wc.woehler.basquin_cycles(load)
    expected_cycles = pd.Series([1e5, 1e6, 1e6], index=index)

    pd.testing.assert_series_equal(cycles, expected_cycles, rtol=1e-4)


def test_woehler_basquin_cycles_50_multiple_load_multiple_wc_cross_index():
    load = pd.Series([3000., 400., 500.], index=pd.Index([1, 2, 3], name='scenario'))

    wc = pd.DataFrame({
        'k_1': [1., 2., 2.],
        'SD': [300., 400., 500.],
        'ND': [1e6, 1e6, 1e6]
    }, pd.Index([1, 2, 3], name='element_id'))

    cycles = wc.woehler.basquin_cycles(load)
    expected_index = pd.MultiIndex.from_tuples([
        (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3),
        (3, 1), (3, 2), (3, 3),
    ], names=['element_id', 'scenario'])

    pd.testing.assert_index_equal(cycles.index, expected_index)


def test_woehler_basquin_cycles_50_same_k(wc_data):
    load = [200., 300., 400., 500.]

    wc = wc_data.copy()
    wc['k_2'] = wc['k_1']
    cycles = wc.woehler.basquin_cycles(load)

    calculated_k = - (np.log(cycles[-1]) - np.log(cycles[0])) / (np.log(load[-1]) - np.log(load[0]))
    np.testing.assert_approx_equal(calculated_k, wc.k_1)


def test_woehler_basquin_cycles_10_90(wc_data):
    load = [200., 300., 400., 500.]

    cycles_10 = wc_data.woehler.basquin_cycles(load, 0.1)[1:]
    cycles_90 = wc_data.woehler.basquin_cycles(load, 0.9)[1:]

    expected = [np.inf, 1.75, 1.75]
    np.testing.assert_allclose(cycles_90/cycles_10, expected)


def test_woehler_basquin_load_50_single_cycles_single_wc(wc_data):
    cycles = 27994

    load = wc_data.woehler.basquin_load(cycles)
    expected_load = 500.

    np.testing.assert_allclose(load, expected_load, rtol=1e-4)


def test_woehler_basquin_load_50_multiple_cycles_single_wc(wc_data):
    cycles = [np.inf, 1e6,  133484,    27994]

    load = wc_data.woehler.basquin_load(cycles)
    expected_load = [300., 300., 400., 500.]

    np.testing.assert_allclose(load, expected_load, rtol=1e-4)


def test_woehler_basquin_load_single_cycles_multiple_wc():
    cycles = 1e6
    wc = pd.DataFrame({
        'k_1': [1., 2., 2.],
        'SD': [300., 400., 500.],
        'ND': [1e5, 1e6, 1e6]
    })

    load = wc.woehler.basquin_load(cycles)
    expected_load = [300., 400., 500.]
    np.testing.assert_allclose(load, expected_load)


def test_woehler_basquin_load_multiple_cycles_multiple_wc():
    cycles = [1e5, 1e6, 1e7]
    wc = pd.DataFrame({
        'k_1': [1., 2., 2.],
        'SD': [300., 400., 500.],
        'ND': [1e6, 1e6, 1e6]
    })

    load = wc.woehler.basquin_load(cycles)
    expected_load = [3000., 400., 500.]
    np.testing.assert_allclose(load, expected_load)


def test_woehler_basquin_load_multiple_cycles_multiple_wc_aligned_index():
    index = pd.Index([1, 2, 3], name='element_id')
    cycles = pd.Series([1e5, 1e6, 1e7], index=index)

    wc = pd.DataFrame({
        'k_1': [1., 2., 2.],
        'SD': [300., 400., 500.],
        'ND': [1e6, 1e6, 1e6],
    }, index=index)

    expected_load = pd.Series([3000., 400., 500.], index=cycles.index)
    load = wc.woehler.basquin_load(cycles)

    pd.testing.assert_series_equal(load, expected_load)


def test_woehler_basquin_load_multiple_cycles_multiple_wc_cross_index():
    cycles = pd.Series([1e5, 1e6, 1e7], index=pd.Index([1, 2, 3], name='scenario'))

    wc = pd.DataFrame({
        'k_1': [1., 2., 2.],
        'SD': [300., 400., 500.],
        'ND': [1e6, 1e6, 1e6],
    }, index=pd.Index([1, 2, 3], name='element_id'))

    expected_index = pd.MultiIndex.from_tuples([
        (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3),
        (3, 1), (3, 2), (3, 3),
    ], names=['element_id', 'scenario'])

    load = wc.woehler.basquin_load(cycles)

    pd.testing.assert_index_equal(load.index, expected_index)


def test_woehler_basquin_load_50_same_k(wc_data):
    cycles = [1e7, 1e6, 1e5, 1e4]

    wc = wc_data.copy()
    wc['k_2'] = wc['k_1']

    load = wc.woehler.basquin_load(cycles)
    calculated_k = - (np.log(cycles[-1]) - np.log(cycles[0])) / (np.log(load[-1]) - np.log(load[0]))
    np.testing.assert_approx_equal(calculated_k, wc.k_1)


def test_woehler_basquin_load_10_90(wc_data):
    cycles = [1e2, 1e7]

    load_10 = wc_data.woehler.basquin_load(cycles, 0.1)
    load_90 = wc_data.woehler.basquin_load(cycles, 0.9)

    expected = np.full_like(cycles, 1.75 ** (1./7.))

    np.testing.assert_allclose(load_90/load_10, expected, rtol=1e-4)


def test_woehler_basquin_load_integer_cycles(wc_data):
    wc_data.woehler.basquin_load(1000)


def test_woehler_basquin_cycles_integer_load(wc_data):
    wc_data.woehler.basquin_cycles(200)


@pytest.fixture
def wc_int_overflow():
    return pd.Series({
        'k_1': 4.,
        'SD': 100.,
        'ND': 1e6
    })


def test_woehler_integer_overflow_scalar(wc_int_overflow):
    assert wc_int_overflow.woehler.basquin_cycles(50) > 0.0


def test_woehler_integer_overflow_list(wc_int_overflow):
    assert (wc_int_overflow.woehler.basquin_cycles([50, 50]) > 0.0).all()


def test_woehler_integer_overflow_series(wc_int_overflow):
    load = pd.Series([50, 50], index=pd.Index(['foo', 'bar']))
    cycles = wc_int_overflow.woehler.basquin_cycles(load)
    assert (cycles > 0.0).all()
    pd.testing.assert_index_equal(cycles.index, load.index)


def test_woehler_ND(wc_data):
    assert wc_data.woehler.ND == 1e6


def test_woehler_SD(wc_data):
    assert wc_data.woehler.SD == 300


def test_woehler_k_1(wc_data):
    assert wc_data.woehler.k_1 == 7.


def test_woehler_TS_and_TN_guessed():
    wc = pd.Series({
        'k_1': 0.5,
        'SD': 300,
        'ND': 1e6
    })
    assert wc.woehler.TN == 1.0
    assert wc.woehler.TS == 1.0


def test_woehler_TS_guessed(wc_data):
    wc = wc_data.copy()
    wc['k_1'] = 0.5
    wc['TN'] = 1.5

    assert wc.woehler.TS == (1.5 * 1.5)


def test_woehler_TN_guessed():
    wc = pd.Series({
        'k_1': 0.5,
        'SD': 300,
        'ND': 1e6,
        'TS': 1.5 * 1.5
    })

    assert wc.woehler.TN == 1.5


def test_woehler_TS_given(wc_data):
    wc_full = wc_data.copy()
    wc_full['TS'] = 1.25
    assert wc_full.woehler.TS == 1.25


def test_woehler_TN_given(wc_data):
    wc_full = wc_data.copy()
    wc_full['TN'] = 1.75
    assert wc_full.woehler.TN == 1.75


def test_woehler_pf_guessed(wc_data):
    assert wc_data.woehler.failure_probability == 0.5


def test_woehler_pf_given(wc_data):
    wc = wc_data.copy()
    wc['failure_probability'] = 0.1
    assert wc.woehler.failure_probability == 0.1


def test_woehler_miner_original_as_default(wc_data):
    assert wc_data.woehler.k_2 == np.inf


def test_woehler_miner_original_as_request(wc_data):
    wc_data['k_2'] = wc_data.k_1
    assert wc_data.woehler.miner_original().k_2 == np.inf


def test_woehler_miner_original_new_object(wc_data):
    orig = wc_data.woehler
    assert orig.miner_original() is not orig


def test_woehler_miner_elementary(wc_data):
    assert wc_data.woehler.miner_elementary().k_2 == wc_data.k_1
    assert wc_data.woehler.miner_elementary().to_pandas().k_2 == wc_data.k_1


def test_woehler_miner_elementary_new_object(wc_data):
    orig = wc_data.woehler
    assert orig.miner_elementary() is not orig


def test_woehler_miner_haibach(wc_data):
    assert wc_data.woehler.miner_haibach().k_2 == 13.0
    assert wc_data.woehler.miner_haibach().to_pandas().k_2 == 13.0


def test_woehler_miner_haibach_new_object(wc_data):
    orig = wc_data.woehler
    assert orig.miner_haibach() is not orig


def test_woehler_to_pandas(wc_data):
    expected = pd.Series({
        'k_1': 0.5,
        'k_2': np.inf,
        'TN': 1.75,
        'TS': 1.75 * 1.75,
        'ND': 1e6,
        'SD': 300.0,
        'failure_probability': 0.5,
    }).sort_index()
    wc = wc_data.copy()
    wc['k_1'] = 0.5

    pd.testing.assert_series_equal(wc.woehler.to_pandas().sort_index(), expected)

    wc = wc_data.copy()
    wc['k_1'] = 0.5
    del wc['TN']

    expected['TN'] = 1.
    expected['TS'] = 1.

    pd.testing.assert_series_equal(wc.woehler.to_pandas().sort_index(), expected)


@pytest.mark.parametrize('pf', [0.1, 0.5, 0.9])
def test_woehler_miner_original_homogenious_load(pf):
    cycles = np.logspace(3., 7., 50)
    wc = pd.Series({
        'k_1': 7.,
        'TN': 1.75,
        'ND': 1e6,
        'SD': 300.0
    })
    load = wc.woehler.basquin_load(cycles, failure_probability=pf)
    assert (np.diff(load) <= 0.).all()
    assert (np.diff(np.diff(load)) >= 0.).all()


@pytest.mark.parametrize('pf', [0.1, 0.5, 0.9])
def test_woehler_miner_original_homogenious_cycles(pf, wc_data):
    load = np.logspace(3., 2., 50)
    cycles = wc_data.woehler.basquin_cycles(load, failure_probability=pf)
    assert (np.diff(cycles[np.isfinite(cycles)]) > 0.).all()


def test_broadcast_load_cycles_clashing_index():
    """Don't know exactly what's going on there."""
    wc = pd.DataFrame({
        'SD': [500., 500., 300.],
        'k_1': [6., 6., 6.],
        'TS': [2., 1.0000001, 1.25],
        'ND': [1e6, 1e6, 1e6]
    })

    cycles = pd.Series([1e6, 1e6, 1e6])

    result = wc.woehler.basquin_load(cycles)
    expected = pd.Series(
        [500., 500., 500., 500., 500., 500., 300., 300., 300],
        index=pd.MultiIndex.from_product([[0, 1, 2], [0, 1, 2]])
    )

    pd.testing.assert_series_equal(result, expected)


@settings(deadline=None)
@given(st.floats(min_value=10., max_value=500.),
       st.floats(min_value=1.0, max_value=10.0),
       st.floats(min_value=1e2, max_value=1e7),
       st.floats(min_value=1.0, max_value=1e9))
def test_load_is_basquin_load(SD, k_1, ND, cycles):
    wc = WoehlerCurve.from_parameters(SD=SD, k_1=k_1, ND=ND)
    assert wc.load(cycles) == wc.basquin_load(cycles)


@settings(deadline=None)
@given(st.floats(min_value=10., max_value=500.),
       st.floats(min_value=1.0, max_value=10.0),
       st.floats(min_value=1e2, max_value=1e7),
       st.floats(min_value=1.0, max_value=1000.0))
def test_cycles_is_basquin_cycles(SD, k_1, ND, load):
    wc = WoehlerCurve.from_parameters(SD=SD, k_1=k_1, ND=ND)
    assert wc.cycles(load) == wc.basquin_cycles(load)


def test_changed_TN():
    wc = pd.Series({
        'k_1': 7.,
        'TN': 1.75,
        'ND': 1e6,
        'SD': 300.0
    })

    _ = wc.woehler

    wc['TN'] = 1.48
    new_wc = wc.woehler.to_pandas()

    assert new_wc['TN'] == 1.48

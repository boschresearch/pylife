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
import pytest

from pylife.materialdata import woehler

data = pd.DataFrame(np.array([
        [4.50e+02, 3.40e+04],
        [4.50e+02, 5.40e+04],
        [4.50e+02, 6.00e+04],
        [4.50e+02, 7.60e+04],
        [4.00e+02, 5.30e+04],
        [4.00e+02, 9.40e+04],
        [4.00e+02, 2.07e+05],
        [4.00e+02, 2.27e+05],
        [3.75e+02, 6.80e+04],
        [3.75e+02, 2.34e+05],
        [3.75e+02, 3.96e+05],
        [3.75e+02, 5.00e+05],
        [3.75e+02, 6.00e+05],
        [3.75e+02, 7.09e+05],
        [3.50e+02, 1.70e+05],
        [3.50e+02, 1.87e+05],
        [3.50e+02, 2.20e+05],
        [3.50e+02, 2.89e+05],
        [3.50e+02, 3.09e+05],
        [3.50e+02, 1.00e+07],
        [3.25e+02, 6.75e+05],
        [3.25e+02, 7.51e+05],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.00e+02, 8.95e+05],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07]
]), columns=['load', 'cycles']).sample(frac=1)

load_sorted = pd.Series(np.array([
        4.50e+02, 4.50e+02, 4.50e+02, 4.50e+02, 4.00e+02, 4.00e+02, 4.00e+02, 4.00e+02, 3.75e+02, 3.75e+02,
        3.75e+02, 3.75e+02, 3.75e+02, 3.75e+02, 3.50e+02, 3.50e+02, 3.50e+02, 3.50e+02, 3.50e+02, 3.50e+02,
        3.25e+02, 3.25e+02, 3.25e+02, 3.25e+02, 3.25e+02, 3.25e+02, 3.25e+02, 3.25e+02, 3.25e+02, 3.25e+02,
        3.00e+02, 3.00e+02, 3.00e+02, 3.00e+02, 3.00e+02, 3.00e+02, 3.00e+02, 3.00e+02, 3.00e+02, 3.00e+02]), name='load').sort_values()

cycles_sorted = pd.Series(np.array([
        3.40e+04, 5.40e+04, 6.00e+04, 7.60e+04, 5.30e+04, 9.40e+04, 2.07e+05, 2.27e+05, 6.80e+04, 2.34e+05,
        3.96e+05, 5.00e+05, 6.00e+05, 7.09e+05, 1.70e+05, 1.87e+05, 2.20e+05, 2.89e+05, 3.09e+05, 1.00e+07,
        6.75e+05, 7.51e+05, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07,
        8.95e+05, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07]), name='cycles').sort_values()

data_finite_sorted = pd.DataFrame(np.array([
        [4.50e+02, 3.40e+04],
        [4.50e+02, 5.40e+04],
        [4.50e+02, 6.00e+04],
        [4.50e+02, 7.60e+04],
        [4.00e+02, 5.30e+04],
        [4.00e+02, 9.40e+04],
        [4.00e+02, 2.07e+05],
        [4.00e+02, 2.27e+05],
        [3.75e+02, 6.80e+04],
        [3.75e+02, 2.34e+05],
        [3.75e+02, 3.96e+05],
        [3.75e+02, 5.00e+05],
        [3.75e+02, 6.00e+05],
        [3.75e+02, 7.09e+05],
]), columns=['load', 'cycles']).sort_values(by='load').reset_index(drop=True)

data_infinite_sorted = pd.DataFrame(np.array([
        [3.50e+02, 1.70e+05],
        [3.50e+02, 1.87e+05],
        [3.50e+02, 2.20e+05],
        [3.50e+02, 2.89e+05],
        [3.50e+02, 3.09e+05],
        [3.50e+02, 1.00e+07],
        [3.25e+02, 6.75e+05],
        [3.25e+02, 7.51e+05],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.00e+02, 8.95e+05],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07]
]), columns=['load', 'cycles']).sort_values(by='load').reset_index(drop=True)

def test_woehler_accessor_missing_keys():
    wc = pd.Series({'k_1': 7, '1/TN': 12.0, 'ND_50': 1e6, 'SD_50': 350., '1/TS': 1.23})
    for k in wc.keys():
        _wc = wc.copy()
        del _wc[k]
        with pytest.raises(AttributeError, match=r"^.*Missing %s\." % k):
            _wc.woehler

def test_fatigue_data_property_methods():
    fd = woehler.determine_fractures(data, 1e7).sort_index().fatigue_data
    pd.testing.assert_series_equal(fd.load.sort_values(), load_sorted)
    pd.testing.assert_series_equal(fd.cycles.sort_values(), cycles_sorted)

    assert fd.num_runouts == 18
    assert fd.num_fractures == 22
    print(fd.finite_zone)
    print(data_finite_sorted)
    pd.testing.assert_frame_equal(fd.finite_zone.sort_values(by='load').reset_index(drop=True)[['load', 'cycles']], data_finite_sorted)
    fd = woehler.determine_fractures(data, 1e7).sort_index().fatigue_data
    pd.testing.assert_frame_equal(fd.infinite_zone.sort_values(by='load').reset_index(drop=True)[['load', 'cycles']], data_infinite_sorted)

def test_woehler_fracture_determination_given():
    df = pd.DataFrame({
        'load': [1, 2, 3],
        'cycles': [1e6, 1e7, 1e4]
    })

    expected = pd.DataFrame({
        'load': [1, 2, 3],
        'cycles': [1e6, 1e7, 1e4],
        'fracture': [True, False, True]
    })

    expected_runouts = pd.DataFrame({
        'load': [2],
        'cycles': [1e7],
        'fracture': [False]
    }, index=[1])

    expected_fractures = pd.DataFrame({
        'load': [1, 3],
        'cycles': [1e6, 1e4],
        'fracture': [True, True]
    }, index=[0, 2])

    test = woehler.determine_fractures(df, 1e7).sort_index()
    pd.testing.assert_frame_equal(test, expected)

    fd = test.fatigue_data
    pd.testing.assert_frame_equal(fd.fractures, expected_fractures)
    pd.testing.assert_frame_equal(fd.runouts, expected_runouts)


def test_woehler_fracture_determination_infered():
    df = pd.DataFrame({
        'load': [1, 2, 3],
        'cycles': [1e6, 1e7, 1e4]
    })

    expected = pd.DataFrame({
        'load': [1, 2, 3],
        'cycles': [1e6, 1e7, 1e4],
        'fracture': [True, False, True]
    })

    test = woehler.determine_fractures(df).sort_index()
    pd.testing.assert_frame_equal(test, expected)


def test_woehler_endur_zones():
    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    assert fd.fatigue_limit == 362.5


def test_woehler_endure_zones_no_runouts():
    df = data[data.cycles < 1e7]
    fd = woehler.determine_fractures(df, 1e7).fatigue_data
    assert fd.fatigue_limit == 0.0


def test_woehler_elementary():
    expected = pd.Series({
        'SD_50': 362.5,
        'k_1': 7.0,
        'ND_50': 3e5,
        '1/TN': 5.3,
        '1/TS': 1.27
    }).sort_index()

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    wc = woehler.Elementary(fd).analyze().sort_index()
    pd.testing.assert_index_equal(wc.index, expected.index)
    np.testing.assert_allclose(wc.to_numpy(), expected.to_numpy(), rtol=1e-1)


def test_woehler_probit():
    expected = pd.Series({
        'SD_50': 335,
        '1/TS': 1.19,
        'k_1': 6.94,
        'ND_50': 463000.,
        '1/TN': 5.26
    }).sort_index()

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    wc = woehler.Probit(fd).analyze().sort_index()
    pd.testing.assert_index_equal(wc.index, expected.index)
    np.testing.assert_allclose(wc.to_numpy(), expected.to_numpy(), rtol=1e-1)


def test_woehler_max_likelihood_inf_limit():
    expected = pd.Series({
        'SD_50': 335,
        '1/TS': 1.19,
        'k_1': 6.94,
        'ND_50': 463000.,
        '1/TN': 5.26
    }).sort_index()

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    wc = woehler.MaxLikeInf(fd).analyze().sort_index()
    pd.testing.assert_index_equal(wc.index, expected.index)
    np.testing.assert_allclose(wc.to_numpy(), expected.to_numpy(), rtol=1e-1)


def test_woehler_max_likelihood_full_without_fixed_params():
    expected = pd.Series({
        'SD_50': 335,
        '1/TS': 1.19,
        'k_1': 6.94,
        'ND_50': 463000.,
        '1/TN': 4.7
    }).sort_index()

    bic = 45.35256860035525

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    we = woehler.MaxLikeFull(fd)
    wc = we.analyze().sort_index()
    pd.testing.assert_index_equal(wc.index, expected.index)
    np.testing.assert_allclose(wc.to_numpy(), expected.to_numpy(), rtol=1e-1)
    np.testing.assert_almost_equal(we.bayesian_information_criterion(), bic, decimal=2)


def test_max_likelihood_full_with_fixed_params():
    expected = pd.Series({
        'SD_50': 335,
        '1/TS': 1.19,
        'k_1': 8.0,
        'ND_50': 520000.,
        '1/TN': 6.0
    }).sort_index()

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    wc = (
        woehler.MaxLikeFull(fd)
        .analyze(fixed_parameters={'1/TN': 6.0, 'k_1': 8.0})
        .sort_index()
    )
    pd.testing.assert_index_equal(wc.index, expected.index)
    np.testing.assert_allclose(wc.to_numpy(), expected.to_numpy(), rtol=1e-1)
    assert wc['1/TN'] == 6.0
    assert wc['k_1'] == 8.0


def test_max_likelihood_full_method_with_all_fixed_params():
    """
    Test of woehler curve parameters evaluation with the maximum likelihood method
    """
    fp = {'k_1': 15.7, '1/TN': 1.2, 'SD_50': 280, '1/TS': 1.2, 'ND_50': 10000000}
    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    with pytest.raises(AttributeError, match=r'You need to leave at least one parameter empty!'):
        (
            woehler.MaxLikeFull(fd)
            .analyze(fixed_parameters=fp)
        )


def test_bayesian_pseudo():
    expected = pd.Series({
        'SD_50': 340.,
        '1/TS': 1.12,
        'k_1': 7.0,
        'ND_50': 400000.,
        '1/TN': 5.3
    }).sort_index()

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    wc = woehler.Bayesian(fd).analyze(nsamples=100, tune=10, random_seed=4223, progressbar=False).sort_index()
    pd.testing.assert_index_equal(wc.index, expected.index)
    np.testing.assert_allclose(wc.to_numpy(), expected.to_numpy(), rtol=1e-0)


@pytest.mark.slow_acceptance
def test_bayesian_full():
    expected = pd.Series({
        'SD_50': 340.,
        '1/TS': 1.12,
        'k_1': 7.0,
        'ND_50': 400000.,
        '1/TN': 5.3
    }).sort_index()

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    wc = woehler.Bayesian(fd).analyze(random_seed=4223, progressbar=False).sort_index()
    pd.testing.assert_index_equal(wc.index, expected.index)
    np.testing.assert_allclose(wc.to_numpy(), expected.to_numpy(), rtol=1e-1)

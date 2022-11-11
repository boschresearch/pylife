# Copyright (c) 2019-2022 - for information on the respective copyright owner
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

import numpy as np
import pandas as pd
import pytest
import unittest.mock as mock


from pylife.materialdata import woehler

from .data import *


def sort_fatigue_data(fd):
    return fd.sort_values(by=['load', 'cycles']).reset_index(drop=True)


def test_fatigue_data_simple_properties():
    fd = woehler.determine_fractures(data, 1e7).sort_index().fatigue_data
    pd.testing.assert_series_equal(fd.load.sort_values(), load_sorted)
    pd.testing.assert_series_equal(fd.cycles.sort_values(), cycles_sorted)

    assert fd.num_runouts == 18
    assert fd.num_fractures == 22


@pytest.mark.parametrize("data, finite_zone_expected, infinite_zone_expected", [
    (data,
     finite_expected,
     infinite_expected),
    (data_no_mixed_horizons,
     no_mixed_horizons_finite_expected,
     no_mixed_horizons_infinite_expected),
    (data_pure_runout_horizon_and_mixed_horizons,
     pure_runout_horizon_and_mixed_horizons_finite_expected,
     pure_runout_horizon_and_mixed_horizons_infinite_expected),
    (data_no_runouts,
     no_runouts_finite_expected,
     no_runouts_infinite_expected),
    (data_only_runout_levels,
     only_runout_levels_finite_expected,
     only_runout_levels_infinite_expected)
])
def test_fatigue_data_finite_infinite_zone(data, finite_zone_expected, infinite_zone_expected):
    fd = woehler.determine_fractures(data, 1e7).sort_index().fatigue_data
    pd.testing.assert_frame_equal(sort_fatigue_data(fd.finite_zone)[['load', 'cycles']],
                                  sort_fatigue_data(finite_zone_expected))
    fd = woehler.determine_fractures(data, 1e7).sort_index().fatigue_data
    pd.testing.assert_frame_equal(sort_fatigue_data(fd.infinite_zone)[['load', 'cycles']],
                                  sort_fatigue_data(infinite_zone_expected))

@pytest.mark.parametrize("data, finite_zone_expected, infinite_zone_expected", [
    (data,
     finite_expected_conservative,
     infinite_expected_conservative),
    (data_no_mixed_horizons,
     no_mixed_horizons_finite_expected,
     no_mixed_horizons_infinite_expected),
    (data_pure_runout_horizon_and_mixed_horizons,
     pure_runout_horizon_and_mixed_horizons_finite_expected_conservative,
     pure_runout_horizon_and_mixed_horizons_infinite_expected_conservative),
    (data_no_runouts,
     no_runouts_finite_expected,
     no_runouts_infinite_expected)
])
def test_fatigue_data_finite_infinite_zone_conservative(data, finite_zone_expected, infinite_zone_expected):
    fd = woehler.determine_fractures(data, 1e7).sort_index().fatigue_data.conservative_fatigue_limit()
    print(sort_fatigue_data(fd.finite_zone))
    print(sort_fatigue_data(finite_zone_expected))
    print(fd.fatigue_limit)
    print(sort_fatigue_data(fd.infinite_zone))
    print(sort_fatigue_data(infinite_zone_expected))
    pd.testing.assert_frame_equal(sort_fatigue_data(fd.finite_zone)[['load', 'cycles']],
                                  sort_fatigue_data(finite_zone_expected))
    fd = woehler.determine_fractures(data, 1e7).sort_index().fatigue_data.conservative_fatigue_limit()
    pd.testing.assert_frame_equal(sort_fatigue_data(fd.infinite_zone)[['load', 'cycles']],
                                  sort_fatigue_data(infinite_zone_expected))


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


@pytest.mark.parametrize("data, fatigue_limit_expected", [
    (data, 362.5),
    (data_no_mixed_horizons, 362.5),
    (data_pure_runout_horizon_and_mixed_horizons, 362.5),
    (data_no_runouts, 0.0),
    (data_only_runout_levels, 362.5)
])
def test_woehler_endur_zones(data, fatigue_limit_expected):
    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    assert fd.fatigue_limit == fatigue_limit_expected


@pytest.mark.parametrize("data, fatigue_limit_expected", [
    (data, 325.0),
    (data_no_mixed_horizons, 350.0),
    (data_pure_runout_horizon_and_mixed_horizons, 325.0),
    (data_no_runouts, 0.0),
    (data_only_runout_levels, 325.0)
])
def test_woehler_endur_zones_conservative(data, fatigue_limit_expected):
    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    print(fd.fatigue_limit)
    fd = fd.conservative_fatigue_limit()
    print(fd.fatigue_limit)
    assert fd.fatigue_limit == fatigue_limit_expected


def test_woehler_endure_zones_no_runouts():
    df = data[data.cycles < 1e7]
    fd = woehler.determine_fractures(df, 1e7).fatigue_data
    assert fd.fatigue_limit == 0.0


def test_woehler_elementary():
    expected = pd.Series({
        'SD': 362.5,
        'k_1': 7.0,
        'ND': 3e5,
        'TN': 5.3,
        'TS': 1.27,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    wc = woehler.Elementary(fd).analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)


def test_woehler_elementary_initialize_with_determined_fractures():
    expected = pd.Series({
        'SD': 362.5,
        'k_1': 7.0,
        'ND': 3e5,
        'TN': 5.3,
        'TS': 1.27,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data, 1e7)
    wc = woehler.Elementary(fd).analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)


def test_woehler_elementary_initialize_with_pandas_dataframe():
    expected = pd.Series({
        'SD': 362.5,
        'k_1': 7.0,
        'ND': 3e5,
        'TN': 5.3,
        'TS': 1.27,
        'failure_probability': 0.5
    }).sort_index()

    wc = woehler.Elementary(data).analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)


def test_woehler_elementary_no_runouts():
    expected = pd.Series({
        'SD': 0.0,
        'k_1': 7.0,
        'TN': 5.3,
        'TS': 1.27,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data_no_runouts, 1e7).fatigue_data
    wc = woehler.Elementary(fd).analyze().sort_index().drop('ND')
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)


def test_woehler_elementary_only_one_load_level():
    data = pd.DataFrame(np.array([[350.0, 1e7], [350.0, 1e6]]), columns=['load', 'cycles'])
    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    with pytest.raises(ValueError, match=r"Need at least two load levels to do a Wöhler analysis."):
        woehler.Elementary(fd).analyze().sort_index()


def test_woehler_probit():
    expected = pd.Series({
        'SD': 335,
        'TS': 1.19,
        'k_1': 6.94,
        'ND': 463000.,
        'TN': 5.26,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    wc = woehler.Probit(fd).analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)


def test_woehler_probit_one_runout_load_level():
    fd = woehler.determine_fractures(data_one_runout_load_level, 1e7).fatigue_data
    expected = woehler.Elementary(fd).analyze()
    with pytest.warns(UserWarning, match=r"Probit needs at least two runout load levels. Falling back to Elementary."):
        wc = woehler.Probit(fd).analyze()
    pd.testing.assert_series_equal(wc, expected)


@pytest.mark.filterwarnings("error:invalid")
def test_woehler_probit_data01():
    expected = pd.Series({
        'SD': 490,
        'TS': 1.1,
        'k_1': 8.0,
        'ND': 530e3,
        'TN': 3.0,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data_01, 1e7).fatigue_data
    pb = woehler.Probit(fd)
    wc = pb.analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)


def test_woehler_probit_no_runouts():
    expected = pd.Series({
        'SD': 0.,
        'TS': 1.27,
        'k_1': 6.94,
        'ND': 4.4e30,
        'TN': 5.26,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data_no_runouts, 1e7).fatigue_data
    pb = woehler.Probit(fd)
    with pytest.warns(UserWarning):
        wc = pb.analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)


def test_woehler_max_likelihood_inf_limit():
    expected = pd.Series({
        'SD': 335,
        'TS': 1.19,
        'k_1': 6.94,
        'ND': 463000.,
        'TN': 5.26,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    wc = woehler.MaxLikeInf(fd).analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)


def test_bic_without_analysis():
    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    we = woehler.MaxLikeFull(fd)
    with pytest.raises(ValueError, match="^.*BIC.*"):
        we.bayesian_information_criterion()


def test_woehler_max_likelihood_inf_limit_no_runouts():
    expected = pd.Series({
        'SD': 0.,
        'TS': 1.19,
        'k_1': 6.94,
        'ND': 4.4e30,
        'TN': 5.26,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data_no_runouts, 1e7).fatigue_data
    wc = woehler.MaxLikeInf(fd).analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)


def test_woehler_max_likelihood_full_without_fixed_params():
    expected = pd.Series({
        'SD': 335,
        'TS': 1.19,
        'k_1': 6.94,
        'ND': 463000.,
        'TN': 4.7,
        'failure_probability': 0.5
    }).sort_index()

    bic = 45.35256860035525

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    we = woehler.MaxLikeFull(fd)
    wc = we.analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)
    np.testing.assert_almost_equal(we.bayesian_information_criterion(), bic, decimal=2)


def test_woehler_max_likelihood_full_without_fixed_params_no_runouts():
    expected = pd.Series({
        'SD': 0,
        'TS': 1.,
        'k_1': 6.94,
        'ND': 4.4e30,
        'TN': 5.7,
        'failure_probability': 0.5
    }).sort_index()

    bic = np.inf

    fd = woehler.determine_fractures(data_no_runouts, 1e7).fatigue_data
    we = woehler.MaxLikeFull(fd)
    with pytest.warns(UserWarning, match=r"^.*no runouts are present.*" ):
        wc = we.analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)
    np.testing.assert_almost_equal(we.bayesian_information_criterion(), bic, decimal=2)


def test_max_likelihood_full_with_fixed_params():
    expected = pd.Series({
        'SD': 335,
        'TS': 1.19,
        'k_1': 8.0,
        'ND': 520000.,
        'TN': 6.0,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    wc = (
        woehler.MaxLikeFull(fd)
        .analyze(fixed_parameters={'TN': 6.0, 'k_1': 8.0})
        .sort_index()
    )
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)
    assert wc['TN'] == 6.0
    assert wc['k_1'] == 8.0


def test_max_likelihood_full_method_with_all_fixed_params():
    """
    Test of woehler curve parameters evaluation with the maximum likelihood method
    """
    fp = {'k_1': 15.7, 'TN': 1.2, 'SD': 280, 'TS': 1.2, 'ND': 10000000}
    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    with pytest.raises(AttributeError, match=r'You need to leave at least one parameter empty!'):
        (
            woehler.MaxLikeFull(fd)
            .analyze(fixed_parameters=fp)
        )


@pytest.mark.parametrize("data,no", [(d, i) for i, d in enumerate(all_data)])
def test_max_likelihood_parameter_sign(data, no):
    def _modify_initial_parameters_mock(fd):
        return fd

    load_cycle_limit = 1e6
    if hasattr(data, "N_threshold"):
        load_cycle_limit = data.N_threshold
    fatdat = woehler.determine_fractures(data, load_cycle_limit=load_cycle_limit)
    ml = woehler.MaxLikeFull(fatigue_data=fatdat.fatigue_data)
    wl = ml.analyze()

    print("Data set number {}".format(no))
    print("Woehler parameters: {}".format(wl))

    def assert_positive_or_nan_but_not_zero(x):
        if np.isfinite(x):
            assert x >= 0
            assert not np.isclose(x, 0.0)

    assert_positive_or_nan_but_not_zero(wl['SD'])
    assert_positive_or_nan_but_not_zero(wl['TS'])
    assert_positive_or_nan_but_not_zero(wl['k_1'])
    assert_positive_or_nan_but_not_zero(wl['ND'])
    assert_positive_or_nan_but_not_zero(wl['TN'])


@pytest.mark.parametrize("invalid_data", [data_01_one_fracture_level, data_01_two_fractures])
def test_max_likelihood_min_three_fractures_on_two_load_levels(invalid_data):
    fd = woehler.determine_fractures(invalid_data, 1e7).fatigue_data
    ml = woehler.MaxLikeFull(fatigue_data=fd)
    with pytest.raises(ValueError, match=r"^.*[N|n]eed at least.*" ):
        ml.analyze()


def test_max_likelihood_one_mixed_horizon():
    expected = pd.Series({
        'SD': 489.3,
        'TS': 1.147,
        'k_1': 7.99,
        'ND': 541e3,
        'TN': 2.51,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data_01, 1e7).fatigue_data
    ml = woehler.MaxLikeFull(fatigue_data=fd)
    with pytest.warns(UserWarning, match=r"^.*less than two mixed load levels.*"):
        wc = ml.analyze().sort_index()
    bic = ml.bayesian_information_criterion()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)

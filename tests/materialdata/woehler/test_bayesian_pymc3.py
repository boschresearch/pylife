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

import sys
import importlib
import pytest
import pandas as pd
import numpy as np
import unittest.mock as mock

from .data import *

try:
    import theano
    import pymc3
    HAVE_PYMC3_AND_THEANO = True
except ModuleNotFoundError:
    HAVE_PYMC3_AND_THEANO = False

from pylife.materialdata import woehler


@pytest.mark.skipif(not HAVE_PYMC3_AND_THEANO, reason="Don't have pymc3")
@mock.patch('pylife.materialdata.woehler.bayesian.pm')
def test_bayesian_slope_trace(pm):
    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    bayes = woehler.Bayesian(fd)
    bayes._nsamples = 1000
    bayes._slope_trace()

    formula, data_dict = pm.glm.GLM.from_formula.call_args[0]
    assert formula == 'y ~ x'
    pd.testing.assert_series_equal(data_dict['x'], np.log10(fd.fractures.load))
    np.testing.assert_array_equal(data_dict['y'], np.log10(fd.fractures.cycles.to_numpy()))
    family = pm.glm.GLM.from_formula.call_args[1]['family']  # Consider switch to kwargs property when py3.7 is dropped
    assert family is pm.glm.families.StudentT()

    pm.sample.assert_called_with(1000, target_accept=0.99, random_seed=None, chains=2, tune=1000)


@pytest.mark.skipif(not HAVE_PYMC3_AND_THEANO, reason="Don't have pymc3")
@mock.patch('pylife.materialdata.woehler.bayesian.pm')
def test_bayesian_TN_trace(pm):
    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    bayes = woehler.Bayesian(fd)
    bayes._common_analysis()
    bayes._nsamples = 1000
    bayes._TN_trace()

    pm.HalfNormal.assert_called_with('stdev', sigma=1.3)

    assert pm.Normal.call_count == 2

    expected_mu = 5.294264482012933
    expected_sigma = 0.2621494419382026

    assert pm.Normal.call_args_list[0][0] == ('mu',)
    np.testing.assert_almost_equal(pm.Normal.call_args_list[0][1]['mu'], expected_mu, decimal=9)
    np.testing.assert_almost_equal(pm.Normal.call_args_list[0][1]['sigma'], expected_sigma, decimal=9)

    assert pm.Normal.call_args_list[1][0] == ('y',)
    observed = pm.Normal.call_args_list[1][1]['observed']  # Consider switch to kwargs property when py3.7 is dropped
    np.testing.assert_almost_equal(observed.mean(), expected_mu, decimal=9)
    np.testing.assert_almost_equal(observed.std(), expected_sigma, decimal=9)

    pm.sample.assert_called_with(1000, target_accept=0.99, random_seed=None, chains=3, tune=1000)


@pytest.mark.skipif(not HAVE_PYMC3_AND_THEANO, reason="Don't have pymc3")
@mock.patch('pylife.materialdata.woehler.bayesian.tt')
@mock.patch('pylife.materialdata.woehler.bayesian.pm')
def test_bayesian_SD_TS_trace_mock(pm, tt):
    def check_likelihood(l, var):
        assert var == tt.as_tensor_variable.return_value
        assert isinstance(l.likelihood, woehler.likelihood.Likelihood)
        np.testing.assert_array_equal(l.likelihood._fd, fd)
        return 'foovar'

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    inf_load_mean = fd.infinite_zone.load.mean()
    inf_load_std = fd.infinite_zone.load.std()

    with mock.patch.object(woehler.Bayesian._LogLike, '__call__', autospec=True) as loglike_call:
        loglike_call.side_effect = check_likelihood

        bayes = woehler.Bayesian(fd)
        bayes._nsamples = 1000
        bayes._SD_TS_trace()

    pm.Normal.assert_called_once_with('SD', mu=inf_load_mean, sigma=inf_load_std * 5)
    pm.Lognormal.assert_called_once()
    np.testing.assert_approx_equal(pm.Lognormal.call_args_list[0][1]['mu'], np.log10(1. / 1.1))
    np.testing.assert_approx_equal(pm.Lognormal.call_args_list[0][1]['sigma'], np.log10(0.5))

    tt.as_tensor_variable.assert_called_once_with([pm.Normal.return_value, pm.Lognormal.return_value])

    pm.Potential.assert_called_once_with('likelihood', 'foovar')

    pm.sample.assert_called_with(1000, cores=1,
                                 chains=3,
                                 random_seed=None,
                                 discard_tuned_samples=True,
                                 tune=1000)


@pytest.mark.skipif(not HAVE_PYMC3_AND_THEANO, reason="Don't have pymc3")
@mock.patch('pylife.materialdata.woehler.bayesian.Bayesian._SD_TS_trace')
@mock.patch('pylife.materialdata.woehler.bayesian.Bayesian._TN_trace')
@mock.patch('pylife.materialdata.woehler.bayesian.Bayesian._slope_trace')
def test_bayesian_mock(_slope_trace, _TN_trace, _SD_TS_trace):
    expected = pd.Series({
        'SD': 100.,
        'TS': 1.12,
        'k_1': 7.0,
        'ND': 1e6,
        'TN': 5.3,
        'failure_probability': 0.5
    }).sort_index()

    expected_slope_trace = {
        'x': np.array([0.0, -8.0, -6.0]),
        'Intercept': np.array([0.0, 19., 21.])
    }

    expected_SD_TS_trace = {
        'SD': np.array([0.0, 150., 50]),
        'TS': np.array([0.0, 1.22, 1.02])
    }

    _slope_trace.__call__().get_values.side_effect = lambda key: expected_slope_trace[key]
    _TN_trace.__call__().get_values.return_value = np.array([0.0, 5.4, 5.2])
    _SD_TS_trace.__call__().get_values.side_effect = lambda key: expected_SD_TS_trace[key]

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    wc = woehler.Bayesian(fd).analyze(nsamples=10).sort_index()

    pd.testing.assert_series_equal(wc, expected)


@pytest.mark.skipif(not HAVE_PYMC3_AND_THEANO, reason="Don't have pymc3")
@pytest.mark.slow_acceptance
def test_bayesian_full():
    expected = pd.Series({
        'SD': 340.,
        'TS': 1.12,
        'k_1': 7.0,
        'ND': 400000.,
        'TN': 5.3,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    wc = woehler.Bayesian(fd).analyze(random_seed=4223, progressbar=False).sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)

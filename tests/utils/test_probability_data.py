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

import numpy as np
import pytest

import pylife.utils.probability_data as PD
from pylife.utils.functions import rossow_cumfreqs


@pytest.fixture
def pd():
    occurrences = [0.19144096, 2.32066184, 0.07425225, 1.32781569, 0.77430177,
                   4.65758492, 9.35734404, 0.72244567, 0.29089061, 1.57460074] # randomly generated
    probs = rossow_cumfreqs(len(occurrences))
    return PD.ProbabilityFit(probs, np.sort(occurrences))


def test_probability_fit_inconsistent():
    with pytest.raises(ValueError, match="probs and occurrence arrays must have the same 1D shape."):
        _ = PD.ProbabilityFit([1, 2], [1, 3, 4])


def test_probability_fit_occurrences():
    pd = PD.ProbabilityFit([1, 2, 4], [1, 3, 4])
    np.testing.assert_array_equal(pd.occurrences, np.array([1., 3., 4.]))


def test_probability_fit_slope(pd):
    np.testing.assert_approx_equal(pd.slope, 1.440756470107515)


def test_probability_fit_intecept(pd):
    np.testing.assert_approx_equal(pd.intercept, 0.04474656548145248)


def test_precentiles(pd):
    expected = np.array([-1.517929, -0.989169, -0.649324, -0.372289, -0.121587,
                         0.121587, 0.372289,  0.649324,  0.989169,  1.517929])
    np.testing.assert_allclose(pd.percentiles, expected, rtol=1e-4)


def test_insufficient_data():
    occurrences = [100.0]
    probs = [-1.1]
    with pytest.raises(ValueError, match=r'Need at least two datapoints for probabilities and occurrences.'):
        _ = PD.ProbabilityFit(probs, occurrences)

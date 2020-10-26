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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import numpy as np
import pytest
from matplotlib.pyplot import axis

import pylife.utils.probability_data as PD
from pylife.utils.functions import rossow_cumfreqs
from pylife.utils.diagrams.probability_data import ProbabilityDataDiagram

@pytest.fixture
def pd():
    occurrences = [0.19144096, 2.32066184, 0.07425225, 1.32781569, 0.77430177,
                   4.65758492, 9.35734404, 0.72244567, 0.29089061, 1.57460074] # randomly generated
    probs = rossow_cumfreqs(len(occurrences))
    return PD.ProbabilityFit(probs, np.sort(occurrences))


def test_title_no_kw():
    diag = ProbabilityDataDiagram(None)
    assert diag.occurrences_name == 'Occurrences'
    assert diag.probability_name == 'Probability'
    assert diag.title == 'Probability plot'


def test_title_kw():
    diag = ProbabilityDataDiagram(None, occurrences_name='foo', probability_name='bar', title='baz')
    assert diag.occurrences_name == 'foo'
    assert diag.probability_name == 'bar'
    assert diag.title == 'baz'


def test_setters():
    diag = ProbabilityDataDiagram(None)
    diag.occurrences_name = 'foo'
    diag.probability_name = 'bar'
    diag.title = 'baz'
    assert diag.occurrences_name == 'foo'
    assert diag.probability_name == 'bar'
    assert diag.title == 'baz'


def test_plot_numerical_data(pd):
    diag = ProbabilityDataDiagram(pd)
    ax = diag.plot()
    xydata = ax.lines[0].get_xydata().T
    occurrences = xydata[0, :]
    percentiles = xydata[1, :]
    np.testing.assert_allclose(occurrences, pd.occurrences)
    np.testing.assert_allclose(percentiles, pd.percentiles)


def test_plot_axis(pd):
    diag = ProbabilityDataDiagram(pd)
    ax = diag.plot()
    assert ax.get_xscale() == 'log'
    assert ax.get_yscale() == 'linear'


def test_plot_titles(pd):
    diag = ProbabilityDataDiagram(pd)
    ax = diag.plot()
    assert ax.get_title() == "Probability plot"
    assert ax.get_ylabel() == "Probability"


def test_plot_ticks(pd):
    diag = ProbabilityDataDiagram(pd)
    ax = diag.plot()
    np.testing.assert_array_equal(ax.get_xticks(), np.logspace(-3., 3., 7))
    expected_yticks = np.array([-2.32634787, -1.64485363, -1.28155157, -0.84162123, 0.,
                                0.84162123, 1.28155157, 1.64485363, 2.32634787])
    np.testing.assert_allclose(ax.get_yticks(), expected_yticks, rtol=1e-4)

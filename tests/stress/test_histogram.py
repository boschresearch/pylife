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

import pytest

import numpy as np
import pandas as pd

from pylife.stress import histogram as hi


def test_combine_hist():
    hist1 = pd.Series(np.array([5, 10]), index=pd.interval_range(start=0, end=2))
    hist2 = pd.Series(np.array([12, 3, 20]), index=pd.interval_range(start=1, periods=3))

    expect_sum = pd.Series(np.array([5, 22, 3, 20]), index=pd.interval_range(start=0, periods=4))
    test_sum = hi.combine_hist([hist1, hist2], method='sum', nbins=4, histtype="range")
    pd.testing.assert_series_equal(test_sum, expect_sum, check_names=False)

    expect_min = pd.Series(np.array([5, 3]), index=pd.interval_range(
                                  start=0, end=4, periods=2))
    test_min = hi.combine_hist([hist1, hist2], method='min', nbins=2, histtype="range")
    pd.testing.assert_series_equal(test_min, expect_min, check_names=False)

    expect_max = pd.Series(np.array([12, 20]), index=pd.interval_range(
                                  start=0, end=4, periods=2))
    test_max = hi.combine_hist([hist1, hist2], method='max', nbins=2, histtype="range")
    pd.testing.assert_series_equal(test_max, expect_max, check_names=False)

    expect_mean = pd.Series(np.array([9., 11.5]), index=pd.interval_range(start=0, end=4, periods=2))
    test_mean = hi.combine_hist([hist1, hist2], method='mean', nbins=2, histtype="range")
    pd.testing.assert_series_equal(test_mean, expect_mean, check_names=False)

    expect_std = pd.Series(np.array([np.std(np.array([5, 10, 12])),
                                     np.std(np.array([3, 20]))]),
                           index=pd.interval_range(start=0, end=4, periods=2))
    test_std = hi.combine_hist([hist1, hist2], method='std', nbins=2, histtype="range")
    pd.testing.assert_series_equal(test_std, expect_std, check_names=False)


@pytest.fixture
def empty_histogram():
    return pd.Series(dtype=np.float64, index=pd.IntervalIndex.from_tuples([]))


@pytest.fixture
def regular_binning():
    return pd.IntervalIndex.from_tuples([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)])


def test_rebin_histogram_invalid_binning(empty_histogram):
    binning = None
    with pytest.raises(TypeError, match=r"binning argument must be a pandas.IntervalIndex."):
        hi.rebin_histogram(empty_histogram, binning)


def test_rebin_invalid_without_interval_index():
    binning = pd.IntervalIndex.from_tuples([(0.0, 1.0), (1.0, 2.0), (0.0, 1.0), (1.0, 2.0)])
    hist = pd.Series(dtype=np.float64)
    with pytest.raises(TypeError, match=r"histogram needs to have an IntervalIndex."):
        hi.rebin_histogram(hist, binning)


def test_rebin_histogram_non_monotonic_binning(empty_histogram):
    binning = pd.IntervalIndex.from_tuples([(0.0, 1.0), (1.0, 2.0), (0.0, 1.0), (1.0, 2.0)])
    with pytest.raises(ValueError, match=r"binning index must be monotonic increasing without overlaps."):
        hi.rebin_histogram(empty_histogram, binning)


def test_rebin_histogram_binning_decreasing(empty_histogram, regular_binning):
    with pytest.raises(ValueError, match=r"binning index must be monotonic increasing without overlaps."):
        hi.rebin_histogram(empty_histogram, regular_binning[::-1])


def test_rebin_histogram_binning_with_overlaps(empty_histogram):
    binning = pd.IntervalIndex.from_tuples([(0.0, 1.0), (1.0, 2.0), (2.5, 3.0), (2.5, 3.5)])
    with pytest.raises(ValueError, match=r"binning index must be monotonic increasing without overlaps."):
        hi.rebin_histogram(empty_histogram, binning)


def test_rebin_histogram_gapped_binning(empty_histogram):
    binning = pd.IntervalIndex.from_tuples([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (4.0, 5.0)])
    with pytest.raises(ValueError, match=r"binning index must not have gaps."):
        hi.rebin_histogram(empty_histogram, binning)


@pytest.mark.parametrize('hist', [
    (pd.Series([1.0, 1.0], index=pd.IntervalIndex.from_tuples([(0.3, 0.7), (5.6, 6.7)]))),
    (pd.Series([1.0, 1.0], index=pd.IntervalIndex.from_tuples([(0.3, 0.7), (-6.6, -5.7)]))),
])
def test_rebin_histogram_insufficient_binning(hist):
    binning = pd.IntervalIndex.from_tuples([(0.0, 1.0), (1.0, 2.0)])
    with pytest.warns(RuntimeWarning, match=r"histogram is partly out of binning. "
                      "This information will be lost!"):
        hi.rebin_histogram(hist, binning)


def test_rebin_empty_histogram(empty_histogram, regular_binning):
    rebinned = hi.rebin_histogram(empty_histogram, regular_binning)

    pd.testing.assert_series_equal(rebinned, pd.Series(0.0, index=regular_binning))
    pd.testing.assert_index_equal(rebinned.index, regular_binning)


@pytest.mark.parametrize('histogram, expected',[
    (
        pd.Series([1.0], index=pd.IntervalIndex.from_tuples([(0.2, 0.4)])),
        [1.0, 0.0, 0.0, 0.0]
    ),
    (
        pd.Series([2.0], index=pd.IntervalIndex.from_tuples([(0.2, 0.4)])),
        [2.0, 0.0, 0.0, 0.0]
    ),
    (
        pd.Series([2.0], index=pd.IntervalIndex.from_tuples([(1.2, 1.4)])),
        [0.0, 2.0, 0.0, 0.0]
    ),
    (
        pd.Series([2.0, 1.0], index=pd.IntervalIndex.from_tuples([(1.2, 1.4), (0.2, 0.4)])),
        [1.0, 2.0, 0.0, 0.0]
    ),
    (
        pd.Series([2.0, 1.0], index=pd.IntervalIndex.from_tuples([(0.5, 0.9), (0.2, 0.4)])),
        [3.0, 0.0, 0.0, 0.0]
    ),
    (
        pd.Series([2.0], index=pd.IntervalIndex.from_tuples([(0.5, 1.5)])),
        [1.0, 1.0, 0.0, 0.0]
    ),
    (
        pd.Series([2.0, 3.0], index=pd.IntervalIndex.from_tuples([(0.5, 1.5), (3.4, 3.6)])),
        [1.0, 1.0, 0.0, 3.0]
    ),
    (
        pd.Series([2.0, 4.0], index=pd.IntervalIndex.from_tuples([(0.5, 1.5), (1.5, 3.5)])),
        [1.0, 2.0, 2.0, 1.0]
    ),
])
def test_rebin_histogram(histogram, expected, regular_binning):
    rebinned = hi.rebin_histogram(histogram, regular_binning)

    pd.testing.assert_series_equal(rebinned, pd.Series(expected, index=regular_binning))

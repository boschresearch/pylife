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

from pylife.utils import histogram as hi


@pytest.fixture
def empty_histogram():
    return pd.Series(dtype=np.float64, index=pd.IntervalIndex.from_tuples([]))


@pytest.fixture
def some_histogram_1d():
    return pd.Series(np.array([5., 10.]), index=pd.interval_range(start=0, end=2))


@pytest.fixture
def another_histogram_1d():
    return pd.Series(np.array([12., 3., 20.]), index=pd.interval_range(start=1, periods=3))


@pytest.fixture
def some_histogram_2d():
    return pd.Series(
        np.array([5., 10., 10., 20.]),
        index=pd.MultiIndex.from_product([
            pd.interval_range(start=0, end=2),
            pd.interval_range(start=0, end=2)
        ], names=['foo', 'bar'])
    )


@pytest.fixture
def another_histogram_2d():
    return pd.Series(
        np.array([
            12., 3., 20.,
            24., 6., 40.,
            36., 9., 60.
        ]),
        index=pd.MultiIndex.from_product([
            pd.interval_range(start=0, periods=3),
            pd.interval_range(start=0, periods=3)
        ], names=['foo', 'bar'])
    )


@pytest.fixture
def some_histogram_3d():
    return pd.Series(
        np.array([
            5., 10.,
            10., 20.,
            15., 30.,
            20., 40.
        ]),
        index=pd.MultiIndex.from_product([
            pd.interval_range(start=0, end=2),
            pd.interval_range(start=2, end=4),
            pd.interval_range(start=0, periods=2)
        ], names=['foo', 'bar', 'baz'])
    )


@pytest.fixture
def hists_to_combine_1d(some_histogram_1d, another_histogram_1d):
    return [some_histogram_1d, another_histogram_1d]


def test_combine_zero_histograms(empty_histogram):
    result = hi.combine_histogram([])
    pd.testing.assert_series_equal(result, empty_histogram)


@pytest.mark.parametrize('method', ['sum', 'min', 'max', 'mean'])
def test_combine_two_empty_histograms_1d(empty_histogram, method):
    result = hi.combine_histogram([empty_histogram, empty_histogram], method=method)
    pd.testing.assert_series_equal(result, empty_histogram)


@pytest.mark.parametrize('method', ['sum', 'max'])
def test_combine_non_empty_with_empty(empty_histogram, some_histogram_1d, method):
    result = hi.combine_histogram([some_histogram_1d, empty_histogram], method=method)
    pd.testing.assert_series_equal(result, some_histogram_1d)


@pytest.mark.parametrize('method', ['sum', 'min', 'max', 'mean'])
def test_combine_empty_with_non_empty(empty_histogram, another_histogram_1d, method):
    result = hi.combine_histogram([empty_histogram, another_histogram_1d], method=method)
    pd.testing.assert_series_equal(result, another_histogram_1d)


@pytest.mark.parametrize('method, expected', [
    ('sum', [5., 22., 3., 20.]),
    ('min', [5., 10., 3., 20.]),
    ('max', [5., 12., 3., 20.]),
    ('mean', [5., 11., 3., 20.])
])
def test_combine_two_histograms_1d(some_histogram_1d, another_histogram_1d, method, expected):
    result = hi.combine_histogram([some_histogram_1d, another_histogram_1d], method=method)
    expected = pd.Series(expected, index=pd.interval_range(start=0, end=4, periods=4))
    pd.testing.assert_series_equal(result, expected)


def test_combine_1d_with_2d_histogram(some_histogram_1d, another_histogram_2d):
    with pytest.raises(ValueError, match=r"Histograms must have identical dimensions to be combined."):
        hi.combine_histogram([some_histogram_1d, another_histogram_2d])


def test_combine_1d_with_2d_histogram_duplicate_index_names(some_histogram_1d, another_histogram_2d):
    some = some_histogram_1d.copy(deep=True)
    other = another_histogram_2d.copy(deep=True)
    some.index.name = 'foo'
    other.index.names = ['foo', 'foo']
    with pytest.raises(ValueError, match=r"Histograms must have identical dimensions to be combined."):
        hi.combine_histogram([some, other])


def test_combine_2d_with_3d_histogram(some_histogram_2d, some_histogram_3d):
    with pytest.raises(ValueError, match=r"Histograms must have identical dimensions to be combined."):
        hi.combine_histogram([some_histogram_2d, some_histogram_3d])


def test_combine_2d_with_2d_histogram_inconsistent_names(some_histogram_2d):
    other = some_histogram_2d.copy(deep=True)
    other.index.names = ['foo', 'baz']
    with pytest.raises(ValueError, match=r"Histograms must have identical dimensions to be combined."):
        hi.combine_histogram([some_histogram_2d, other])


@pytest.mark.parametrize('method, expected', [
    ('sum', [17., 13., 20., 34., 26., 40., 36., 9., 60.]),
    ('min', [5., 3., 20., 10., 6., 40., 36., 9., 60.]),
    ('max', [12., 10., 20., 24., 20., 40., 36., 9., 60.])
])
def test_combine_2d_with_2d_histogram(some_histogram_2d, another_histogram_2d, method, expected):
    result = hi.combine_histogram([some_histogram_2d, another_histogram_2d], method=method)

    expected_index = pd.MultiIndex.from_product(
        [pd.interval_range(start=0, end=3, periods=3), pd.interval_range(start=0, end=3, periods=3)],
        names=['foo', 'bar']
    )
    expected = pd.Series(expected, index=expected_index)

    pd.testing.assert_series_equal(result, expected)


def test_combine_hist_sum(hists_to_combine_1d):
    expect_sum = pd.Series(np.array([5., 22., 3., 20.]), index=pd.interval_range(start=0, periods=4))
    test_sum = hi.combine_histogram(hists_to_combine_1d, method='sum')
    pd.testing.assert_series_equal(test_sum, expect_sum, check_names=False)


def test_combine_hist_min(hists_to_combine_1d):
    expect_min = pd.Series(
        np.array([5., 10., 3., 20.]),
        index=pd.interval_range(start=0, end=4, periods=4)
    )
    test_min = hi.combine_histogram(hists_to_combine_1d, method='min')
    pd.testing.assert_series_equal(test_min, expect_min, check_names=False)


def test_combine_hist_max(hists_to_combine_1d):
    expect_max = pd.Series(
        np.array([5., 12., 3., 20.]),
        index=pd.interval_range(start=0, end=4, periods=4)
    )
    test_max = hi.combine_histogram(hists_to_combine_1d, method='max')
    pd.testing.assert_series_equal(test_max, expect_max, check_names=False)


def test_combine_hist_mean(hists_to_combine_1d):
    expect_mean = pd.Series(
        np.array([5., 11., 3., 20.]),
        index=pd.interval_range(start=0, end=4, periods=4)
    )
    test_mean = hi.combine_histogram(hists_to_combine_1d, method='mean')
    pd.testing.assert_series_equal(test_mean, expect_mean, check_names=False)


def test_combine_hist_std(hists_to_combine_1d):
    expect_std = pd.Series(
        np.array([0, np.std([10, 12]), 0., 0.]),
        index=pd.interval_range(start=0, end=4, periods=4)
    )
    test_std = hi.combine_histogram(hists_to_combine_1d, method=lambda x: np.std(x))
    pd.testing.assert_series_equal(test_std, expect_std, check_names=False)


def test_combine_test_big_bin():
    hist1 = pd.Series([1, 2], index=pd.IntervalIndex.from_tuples([(0.0, 0.5), (0.5, 1.0)]))
    hist2 = pd.Series([4], index=pd.IntervalIndex.from_tuples([(0.0, 1.0)]))

    result = hi.combine_histogram([hist1, hist2], method="sum")

    expected = pd.Series(
        [1, 4, 2],
        index=pd.IntervalIndex.from_tuples([(0.0, 0.5), (0.0, 1.0), (0.5, 1.0)])
    )

    pd.testing.assert_series_equal(result, expected)


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


def test_rebin_histogram_name_empty_histogram(empty_histogram, regular_binning):
    empty_histogram.name = "empty_histogram"
    rebinned = hi.rebin_histogram(empty_histogram, regular_binning)

    assert rebinned.name == "empty_histogram"


def test_rebin_histogram_name_non_empty_histogram(regular_binning):
    histogram = pd.Series([1.0], index=pd.IntervalIndex.from_tuples([(0.2, 0.4)]), name="foo")
    rebinned = hi.rebin_histogram(histogram, regular_binning)

    assert rebinned.name == "foo"


def test_rebin_histogram_index_name_empty_histogram(empty_histogram, regular_binning):
    empty_histogram.index.name = "index_name"
    rebinned = hi.rebin_histogram(empty_histogram, regular_binning)

    assert rebinned.index.name == "index_name"


def test_rebin_histogram_index_name_non_empty_histogram(regular_binning):
    histogram = pd.Series(
        [1.0], index=pd.IntervalIndex.from_tuples([(0.2, 0.4)], name="index_name")
    )
    rebinned = hi.rebin_histogram(histogram, regular_binning)

    assert rebinned.index.name == "index_name"


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


def test_rebin_histogram_nan_default(regular_binning):
    histogram = pd.Series([1.0], index=pd.IntervalIndex.from_tuples([(0.2, 0.4)]))
    expected = pd.Series([1.0, np.nan, np.nan, np.nan], index=regular_binning)

    rebinned = hi.rebin_histogram(histogram, regular_binning, nan_default=True)

    pd.testing.assert_series_equal(rebinned, expected)


@pytest.mark.parametrize('original_binning, binnum, expected', [
    ([(0.0, 1.0)], 4, [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]),
    ([(0.0, 1.0)], 2, [(0.0, 0.5), (0.5, 1.0)]),
    ([(1.0, 2.0)], 4, [(1.0, 1.25), (1.25, 1.5), (1.5, 1.75), (1.75, 2.0)]),
    ([(1.0, 2.0)], 2, [(1.0, 1.5), (1.5, 2.0)]),
])
def test_rebin_histogram_n_bins(original_binning, binnum, expected):
    histogram = pd.Series([1.0], pd.IntervalIndex.from_tuples(original_binning))
    rebinned = hi.rebin_histogram(histogram, binnum)

    expected = pd.IntervalIndex.from_tuples(expected)
    pd.testing.assert_index_equal(rebinned.index, expected)


def test_rebin_histogram_additional_index():
    hi_index = pd.IntervalIndex.from_tuples([(0.0, 0.5), (0.5, 1.0)], name='hi')
    other_index = pd.Index([1, 2], name='foo')
    index = pd.MultiIndex.from_product([hi_index, other_index])

    histogram = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)

    expected_hi_index = pd.Index(
        pd.interval_range(start=0, end=1, periods=4),
        name='hi'
    )
    expected_index = pd.MultiIndex.from_product([expected_hi_index, other_index])
    expected_histogram = pd.Series([0.5, 1.0, 0.5, 1.0, 1.5, 2.0, 1.5, 2.0], index=expected_index)

    result = hi.rebin_histogram(histogram, 4)

    pd.testing.assert_series_equal(result.sort_index(), expected_histogram)


def test_rebin_histogram_additional_index_given_rebin():
    hi_index = pd.IntervalIndex.from_tuples([(0.0, 0.5), (0.5, 1.0)], name='hi')
    other_index = pd.Index([1, 2], name='foo')
    index = pd.MultiIndex.from_product([hi_index, other_index])

    histogram = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)

    new_hi_index = pd.Index(
        pd.interval_range(start=0, end=1, periods=4),
        name='hi'
    )
    expected_index = pd.MultiIndex.from_product([new_hi_index, other_index])
    expected_histogram = pd.Series([0.5, 1.0, 0.5, 1.0, 1.5, 2.0, 1.5, 2.0], index=expected_index)

    result = hi.rebin_histogram(histogram, new_hi_index)

    pd.testing.assert_series_equal(result.sort_index(), expected_histogram)


def test_rebin_histogram_2d():
    idx_x = pd.interval_range(0, 1, 2)
    idx_y = pd.interval_range(0, 10, 2)
    histogram = pd.Series(1.0, index=pd.MultiIndex.from_product([idx_x, idx_y], names=['foo', 'bar']))

    idx_x = pd.interval_range(0, 1, 4)
    idx_y = pd.interval_range(0, 10, 4)

    target_idx = pd.MultiIndex.from_product([idx_x, idx_y], names=['foo', 'bar'])

    expected = pd.Series(0.25, index=target_idx)

    result = hi.rebin_histogram(histogram, 4)

    assert result.sum() == histogram.sum()
    pd.testing.assert_series_equal(result, expected)


def test_rebin_histogram_2d_not_rebin_fill():
    idx_x = pd.interval_range(0., 1., 2)
    idx_y = pd.interval_range(0., 10., 2)
    histogram = pd.Series([1., 2., 3., 4.], index=pd.MultiIndex.from_product([idx_x, idx_y], names=['foo', 'bar']))

    idx_x = pd.interval_range(0., 2., 4)
    idx_y = pd.interval_range(0., 20., 4)

    target_idx = pd.MultiIndex.from_product([idx_x, idx_y], names=['foo', 'bar'])

    expected = pd.Series([1.0, 2.0, np.nan, np.nan, 3.0, 4.0] + [np.nan]*10, index=target_idx)

    result = hi.rebin_histogram(histogram, target_idx, nan_default=True)
    pd.testing.assert_series_equal(result, expected)


def test_rebin_histogram_2d_given_rebin():
    idx_x = pd.interval_range(0, 1, 2)
    idx_y = pd.interval_range(0, 10, 2)
    histogram = pd.Series(1.0, index=pd.MultiIndex.from_product([idx_x, idx_y], names=['foo', 'bar']))

    idx_x = pd.interval_range(0, 1, 4)
    idx_y = pd.interval_range(0, 10, 4)

    target_idx = pd.MultiIndex.from_product([idx_x, idx_y], names=['foo', 'bar'])

    expected = pd.Series(0.25, index=target_idx)

    result = hi.rebin_histogram(histogram, target_idx)

    assert result.sum() == histogram.sum()
    pd.testing.assert_series_equal(result, expected)


def test_rebin_histogram_3d():
    idx_x = pd.interval_range(0., 1., 2)
    idx_y = pd.interval_range(0., 10., 2)
    idx_z = pd.interval_range(0., 100., 2)
    histogram = pd.Series(2.0, index=pd.MultiIndex.from_product(
        [idx_x, idx_y, idx_z],
        names=['foo', 'bar', 'baz']
    ))

    idx_x = pd.interval_range(0., 1., 4)
    idx_y = pd.interval_range(0., 10., 4)
    idx_z = pd.interval_range(0., 100., 4)

    target_idx = pd.MultiIndex.from_product(
        [idx_x, idx_y, idx_z],
        names=['foo', 'bar', 'baz']
    )

    expected = pd.Series(0.25, index=target_idx)

    result = hi.rebin_histogram(histogram, 4)

    assert result.sum() == histogram.sum()
    pd.testing.assert_series_equal(result, expected)

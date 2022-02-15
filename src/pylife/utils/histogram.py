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

__author__ = "Daniel Christopher Kreuter"
__maintainer__ = "Johannes Mueller"

import warnings

import numpy as np
import pandas as pd


def combine_histogram(hist_list, method='sum'):
    """Combine a list of histograms to one.

    Parameters
    ----------
    hist_list: list
        list of histograms with all histograms as interval indexed :class:`pandas.Series`
    method: str or aggregating function
        method used for the aggregation, e.g. 'sum', 'min', 'max', 'mean', 'std'
        default is 'sum'

    Returns
    -------
    histogram : pd.Series
        The resulting histogram
    """
    def dimensions_are_consistent():
        for h in hist_list[1:]:
            if len(h.index.names) != len(hist_list[0].index.names):
                return False
            if set(h.index.names) != set(hist_list[0].index.names):
                return False

        return True

    hist_list = list(filter(lambda h: len(h) > 0, hist_list))
    if len(hist_list) == 0:
        return pd.Series(dtype=np.float64, index=pd.IntervalIndex.from_tuples([]))

    if not dimensions_are_consistent():
        raise ValueError("Histograms must have identical dimensions to be combined.")

    names = hist_list[0].index.names

    concat = pd.concat(hist_list)
    combined = concat.groupby(concat.index).agg(method)

    if isinstance(concat.index, pd.MultiIndex):
        combined.index = pd.MultiIndex.from_tuples(combined.index, names=names)

    return combined


def rebin_histogram(histogram, binning, nan_default=False):
    """Rebin a histogram to a given binning.

    Parameters
    ----------
    histogram : :class:`pandas.Series` with :class:`pandas.IntervalIndex`
        The histogram data to be rebinned

    binning : :class:`pandas.IntervalIndex` or int
        The given binning or number of bins

    nan_default : bool
        If True non occupied bins will be occupied with ``np.nan``, else 0.0
        Default False


    Returns
    -------
    rebinned : :class:`pandas.Series` with :class:`pandas.IntervalIndex`
        The rebinned histogram


    Raises
    ------
    TypeError : if the ``histogram`` or the ``binning`` do not have an ``IntervalIndex``.
    ValueError : if the binning is not monotonic increasing or has gaps.
    """
    default_value = np.nan if nan_default else 0.0

    if not isinstance(histogram.index, pd.MultiIndex):
        return _do_rebin_histogram(histogram, binning, default_value)

    original_names = histogram.index.names
    for name in histogram.index.names:
        if not isinstance(histogram.index.get_level_values(name), pd.IntervalIndex):
            continue

        this_binning = binning.levels[binning.names.index(name)] if isinstance(binning, pd.MultiIndex) else binning

        remaining_names = list(filter(lambda m: m != name, original_names))
        histogram = (histogram
                     .groupby(remaining_names)
                     .apply(lambda h: _do_rebin_histogram(h.droplevel(remaining_names), this_binning, default_value)))

    return histogram.reorder_levels(original_names)


def _do_rebin_histogram(histogram, binning, default_value):
    def interval_overlap(reference_interval, test_interval):
        overlap = min(reference_interval.right, test_interval.right) - max(reference_interval.left, test_interval.left)
        return overlap / test_interval.length

    def aggregate_hist(interval):
        occupied = hist.loc[hist.index.overlaps(interval)].dropna()
        if len(occupied) == 0:
            return default_value

        return occupied.apply(lambda v: v.iloc[0] * interval_overlap(interval, v.name), axis=1).sum()

    def binning_of_n_bins(index, binnum):
        start = index.left.min()
        end = index.right.max()

        if np.isnan(start) or np.isnan(end):
            return pd.interval_range(0., 0., 0)

        return pd.interval_range(start, end, binnum)

    def binning_does_not_cover_histogram():
        return (
            histogram.index.right.max() > binning.right.max() or
            histogram.index.left.min() < binning.left.min()
        )

    if not isinstance(histogram.index, pd.IntervalIndex):
        raise TypeError("histogram needs to have an IntervalIndex.")

    if isinstance(binning, int):
        binning = binning_of_n_bins(histogram.index, binning)
    else:
        _fail_if_binning_invalid(binning)

    if binning_does_not_cover_histogram():
        warnings.warn("histogram is partly out of binning. This information will be lost!", RuntimeWarning)

    if len(histogram) == 0:
        rebinned = pd.Series(0.0, index=binning)
    else:
        hist = histogram.to_frame()
        rebinned = binning.to_series().apply(aggregate_hist)

    rebinned.name = histogram.name
    rebinned.index.name = histogram.index.name
    return rebinned


def _fail_if_binning_invalid(binning):
    def binning_is_overlapping_or_non_monotonic_increasing():
        return (
            len(binning) > 0
            and (
                not binning.is_non_overlapping_monotonic or binning.is_monotonic_decreasing
            )
        )

    def binning_has_gaps():
        if len(binning) == 0:
            return False
        left = binning.left[1:]
        right = binning.right[:-1]
        return pd.DataFrame({'l': left, 'r': right}).apply(lambda r: r.l != r.r, axis=1).any()

    if not isinstance(binning, pd.IntervalIndex):
        raise TypeError("binning argument must be a pandas.IntervalIndex.")

    if binning_is_overlapping_or_non_monotonic_increasing():
        raise ValueError("binning index must be monotonic increasing without overlaps.")

    if binning_has_gaps():
        raise ValueError("binning index must not have gaps.")

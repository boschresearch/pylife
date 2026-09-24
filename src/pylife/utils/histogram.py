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

"""Provide helpers for interval-indexed histograms."""

__author__ = "Daniel Christopher Kreuter"
__maintainer__ = "Johannes Mueller"

import warnings

import numpy as np
import pandas as pd


def combine_histogram(hist_list, method='sum'):
    r"""Combine several interval-indexed histograms into one histogram.

    Histograms are represented by :class:`pandas.Series` objects whose index is
    either a :class:`pandas.IntervalIndex` or a :class:`pandas.MultiIndex` with
    interval-valued histogram dimensions.  Equal bins are grouped and
    aggregated; bins that do not occur in an input histogram are not filled
    before aggregation.

    Parameters
    ----------
    hist_list : list of pandas.Series
        Histograms to combine.  Each non-empty series must use compatible
        interval-indexed dimensions.
    method : str or callable, optional
        Aggregation passed to ``pandas`` group-by aggregation, for example
        ``'sum'``, ``'min'``, ``'max'``, ``'mean'``, ``'std'`` or a callable
        accepting a :class:`pandas.Series`.  Default is ``'sum'``.

    Returns
    -------
    pandas.Series
        Combined histogram with the grouped interval bins as its index.

    Raises
    ------
    ValueError
        Raised if the index levels of the histograms do not match.

    See Also
    --------
    pylife.utils.histogram.rebin_histogram : Rebin a histogram before or after
        combining histograms.

    Notes
    -----
    For every unique bin :math:`b`, the combined value is computed from all
    values with that exact bin:

    .. math::

        h_\mathrm{combined}(b) =
        \operatorname{agg}\{h_k(b) \mid b \in \operatorname{index}(h_k)\}

    No rebinning is performed before or after the aggregation.  Rebin the
    inputs explicitly with :func:`pylife.utils.histogram.rebin_histogram` when
    histograms use different but geometrically overlapping bins.

    Limitations: additional dimensions that are not histogram bins are only
    valid if their index level names and values are compatible with pandas
    grouping.  This operation does not align values onto missing combinations
    of non-histogram dimensions.

    Examples
    --------
    >>> h1 = pd.Series([5., 10.], index=pd.interval_range(start=0, end=2))
    >>> h2 = pd.Series([12., 3., 20.], index=pd.interval_range(start=1, periods=3))
    >>> combine_histogram([h1, h2])
    (0, 1]     5.0
    (1, 2]    22.0
    (2, 3]     3.0
    (3, 4]    20.0
    dtype: float64
    >>> combine_histogram([h1, h2], method='min')
    (0, 1]     5.0
    (1, 2]    10.0
    (2, 3]     3.0
    (3, 4]    20.0
    dtype: float64
    >>> combine_histogram([h1, h2], method='max')
    (0, 1]     5.0
    (1, 2]    12.0
    (2, 3]     3.0
    (3, 4]    20.0
    dtype: float64
    >>> combine_histogram([h1, h2], method='mean')
    (0, 1]     5.0
    (1, 2]    11.0
    (2, 3]     3.0
    (3, 4]    20.0
    dtype: float64
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
    r"""Rebin an interval-indexed histogram to a target binning.

    The function redistributes bin contents by geometric overlap.  It works
    with a one-dimensional :class:`pandas.IntervalIndex` and with
    :class:`pandas.MultiIndex` histograms that contain interval-valued
    dimensions.

    Parameters
    ----------
    histogram : pandas.Series
        Histogram data to rebin.  The index must be a
        :class:`pandas.IntervalIndex` or a :class:`pandas.MultiIndex` that
        contains interval-valued histogram dimensions.
    binning : pandas.IntervalIndex or pandas.MultiIndex or int
        Target binning.  If an integer is given, equally spaced bins spanning
        the original histogram range are created for each rebinned interval
        dimension.
    nan_default : bool, optional
        Fill unoccupied target bins with ``numpy.nan`` instead of ``0.0``.
        Default is ``False``.

    Returns
    -------
    pandas.Series
        Rebinned histogram with the target interval bins as its index.

    Raises
    ------
    TypeError
        Raised if ``histogram`` does not use an interval-indexed histogram
        dimension or if ``binning`` is not an interval index when explicit
        bins are supplied.
    ValueError
        Raised if the target binning is not monotonic increasing, overlaps, or
        has gaps.

    Warns
    -----
    RuntimeWarning
        Raised if the target binning does not cover the full histogram range
        and values outside the target range are discarded.

    See Also
    --------
    pylife.utils.histogram.combine_histogram : Combine histograms that already
        share compatible bins.

    Notes
    -----
    Each source bin value is distributed proportionally to its overlap with a
    target bin.  For source bins :math:`s_i`, target bins :math:`t_j`, source
    values :math:`h_i`, and interval length :math:`|s_i|`, the rebinned value
    is

    .. math::

        h'_j = \sum_i h_i
        \frac{|s_i \cap t_j|}{|s_i|}

    This preserves the total sum when the target bins cover the complete
    source range and have no gaps.

    Limitations: additional non-interval index levels are preserved and the
    operation is applied independently for their combinations.  The function
    does not interpolate within those non-histogram dimensions.

    Examples
    --------
    >>> h = pd.Series([10.0, 20.0, 30.0, 40.0], index=pd.interval_range(0.0, 4.0, 4))
    >>> h
    (0.0, 1.0]    10.0
    (1.0, 2.0]    20.0
    (2.0, 3.0]    30.0
    (3.0, 4.0]    40.0
    dtype: float64

    Rebin to a finer binning:

    >>> target_binning = pd.interval_range(0.0, 4.0, 8)
    >>> rebin_histogram(h, target_binning)
    (0.0, 0.5]     5.0
    (0.5, 1.0]     5.0
    (1.0, 1.5]    10.0
    (1.5, 2.0]    10.0
    (2.0, 2.5]    15.0
    (2.5, 3.0]    15.0
    (3.0, 3.5]    20.0
    (3.5, 4.0]    20.0
    dtype: float64

    Rebin to a coarser binning:

    >>> target_binning = pd.interval_range(0.0, 4.0, 2)
    >>> rebin_histogram(h, target_binning)
    (0.0, 2.0]    30.0
    (2.0, 4.0]    70.0
    dtype: float64

    Define the target bin just by an int:

    >>> rebin_histogram(h, 8)
    (0.0, 0.5]     5.0
    (0.5, 1.0]     5.0
    (1.0, 1.5]    10.0
    (1.5, 2.0]    10.0
    (2.0, 2.5]    15.0
    (2.5, 3.0]    15.0
    (3.0, 3.5]    20.0
    (3.5, 4.0]    20.0
    dtype: float64
    """
    default_value = np.nan if nan_default else 0.0

    if not isinstance(histogram.index, pd.MultiIndex):
        return _do_rebin_histogram(histogram, binning, default_value)

    original_names = histogram.index.names
    for name in histogram.index.names:
        if not isinstance(histogram.index.get_level_values(name), pd.IntervalIndex):
            continue

        if isinstance(binning, pd.MultiIndex):
            this_binning = binning.levels[binning.names.index(name)]
        elif isinstance(binning, int):
            index_to_rebin = histogram.index.get_level_values(name)
            lower = index_to_rebin.left.min()
            upper = index_to_rebin.right.max()
            binnum = binning if upper > lower else 1
            this_binning = pd.IntervalIndex(pd.interval_range(lower, upper, binnum))
        else:
            this_binning = binning

        remaining_names = list(filter(lambda m: m != name, original_names))
        remaining_index = histogram.index.droplevel(name)

        histogram = (
            _with_range_index(histogram, remaining_names)
            .groupby(remaining_names)
            .apply(
                lambda h: _do_rebin_histogram(
                    h.droplevel(remaining_names), this_binning, default_value
                )
            )
        )
        histogram.index = _restore_old_index(histogram, remaining_index)

    return histogram.reorder_levels(original_names)


def _with_range_index(hist, levels_to_remap):
    new_hist = hist.copy().reset_index(drop=False)
    for level in levels_to_remap:
        new_hist[level] = (
            hist.index.get_level_values(level)
            .unique()
            .get_indexer_for(hist.index.get_level_values(level))
        )

    res = new_hist.set_index(hist.index.names, drop=True).iloc[:, 0]
    res.name = hist.name
    return res


def _restore_old_index(hist, old_index):
    new_hist = hist.copy().reset_index(drop=False)

    for level in old_index.names:
        new_hist[level] = old_index.get_level_values(level).unique()[new_hist[level]]

    return new_hist.set_index(hist.index.names, drop=True).index


def _do_rebin_histogram(histogram, binning, default_value):
    def interval_overlap(reference_interval, test_interval):
        if test_interval == reference_interval:
            return 1.0
        overlap = min(reference_interval.right, test_interval.right) - max(reference_interval.left, test_interval.left)
        return overlap / test_interval.length

    def aggregate_hist(interval):
        equal_bins = hist.index == interval
        overlapping_bins = hist.index.overlaps(interval)

        occupied = hist.loc[overlapping_bins | equal_bins]
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
        hist = histogram.to_frame().dropna()
        rebinned = binning.to_series().apply(aggregate_hist)

    rebinned.name = histogram.name
    rebinned.index.name = histogram.index.name

    return rebinned


def _fail_if_binning_invalid(binning):
    def binning_is_overlapping_or_non_monotonic_increasing():
        return (
            len(binning) > 1
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

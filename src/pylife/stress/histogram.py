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


def combine_histogram(hist_list, binning=None, method='sum'):
    hist_list = list(filter(lambda h: len(h) > 1, hist_list))
    if len(hist_list) == 0:
        return pd.Series(dtype=np.float64, index=pd.IntervalIndex.from_tuples([]))

    left = min(filter(lambda v: not np.isnan(v), map(lambda h: h.index.left.min(), hist_list)))
    right = max(filter(lambda v: not np.isnan(v), map(lambda h: h.index.right.max(), hist_list)))

    binning = binning or int(np.ceil(np.sqrt(sum(map(lambda h: len(h)**2, hist_list)))))
    if isinstance(binning, int):
        binning = pd.interval_range(left, right, binning)

    hist_concat = pd.concat(map(lambda h: rebin_histogram(h, binning), hist_list))

    return hist_concat.groupby(hist_concat.index).agg(method)


def combine_hist(hist_list, method='sum', nbins=64, histtype="rf"):
    """
    Performs the combination of multiple Histograms.

    Parameters
    ----------

    hist_list: list
        list of histograms with all histograms (saved as DataFrames in pyLife format)
    method: str
        method: 'sum', 'min', 'max', 'mean', 'std'  default is 'sum'
    nbins: int
        number of bins of the combined histogram
    histtype: str
       histogram type: 'rf', 'spectrum'  default is 'rf'


    Returns
    -------

    DataFrame:
        Combined histogram
    list:
        list with the reindexed input histograms

    """

    hist_combined = pd.concat(hist_list)
    if histtype == "rf":
        from_max = np.max(hist_combined.index.get_level_values("from").right)
        from_min = np.min(hist_combined.index.get_level_values("from").left)
        to_max = np.max(hist_combined.index.get_level_values("to").right)
        to_min = np.min(hist_combined.index.get_level_values("to").left)

        index_from = pd.cut(hist_combined.index.get_level_values("from").mid,
                            np.linspace(from_min, from_max, nbins+1))
        index_to = pd.cut(hist_combined.index.get_level_values("to").mid,
                            np.linspace(to_min, to_max, nbins+1))

        categorial_index = [index_from, index_to]
    else:
        index_min = hist_combined.index.left.min()
        index_max = hist_combined.index.right.max()
        categorial_index = pd.cut(hist_combined.index.mid.values,
               np.linspace(index_min, index_max, nbins+1))

    kwargs = {'ddof': 0} if method == 'std' else {}
    result = hist_combined.groupby(categorial_index).agg(method, **kwargs)

    if histtype == "rf":
        result_index = pd.MultiIndex.from_arrays([pd.IntervalIndex(result.index.get_level_values(0)),
                                       pd.IntervalIndex(result.index.get_level_values(1))],
                                      names = ["from", "to"])
    else:
        result_index = pd.interval_range(index_min, index_max, nbins, name='range')

    result.index = result_index

    return result


def rebin_histogram(histogram, binning):
    """Rebin a histogram to a given binning.

    Parameters
    ----------
    histogram : :class:`pandas.Series` with :class:`pandas.IntervalIndex`
        The histogram data to be rebinned

    binning : :class:`pandas.IntervalIndex` or int
        The given binning or number of bins


    Returns
    -------
    rebinned : :class:`pandas.Series` with :class:`pandas.IntervalIndex`
        The rebinned histogram


    Raises
    ------
    TypeError : if the ``histogram`` or the ``binning`` do not have an ``IntervalIndex``.
    ValueError : if the binning is not monotonic increasing or has gaps.
    """
    def interval_overlap(reference_interval, test_interval):
        if not reference_interval.overlaps(test_interval):
            return 0.0

        overlap = min(reference_interval.right, test_interval.right) - max(reference_interval.left, test_interval.left)
        return overlap / test_interval.length

    def aggregate_hist(interval):
        return hist.apply(lambda v: v.iloc[0] * interval_overlap(interval, v.name), axis=1).sum()

    def binning_has_gaps():
        if len(binning) == 0:
            return False
        left = binning.left[1:]
        right = binning.right[:-1]
        return pd.DataFrame({'l': left, 'r': right}).apply(lambda r: r.l != r.r, axis=1).any()

    def binning_does_not_cover_histogram():
        return (
            histogram.index.right.max() > binning.right.max() or
            histogram.index.left.min() < binning.left.min()
        )

    if not isinstance(histogram.index, pd.IntervalIndex):
        raise TypeError("histogram needs to have an IntervalIndex.")

    if isinstance(binning, int):
        binning = _binning_of_n_bins(histogram, binning)

    if not isinstance(binning, pd.IntervalIndex):
        raise TypeError("binning argument must be a pandas.IntervalIndex.")

    if (
            len(binning) > 0
            and (
                not binning.is_non_overlapping_monotonic or binning.is_monotonic_decreasing
            )
    ):
        raise ValueError("binning index must be monotonic increasing without overlaps.")

    if binning_has_gaps():
        raise ValueError("binning index must not have gaps.")

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


def _binning_of_n_bins(histogram, binnum):
    start = histogram.index.left.min()
    end = histogram.index.right.max()

    if np.isnan(start) or np.isnan(end):
        return pd.interval_range(0., 0., 0)

    return pd.interval_range(start, end, binnum)

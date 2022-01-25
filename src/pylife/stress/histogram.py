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

__author__ = "Daniel Christopher Kreuter"
__maintainer__ = "Johannes Mueller"


import numpy as np
import pandas as pd


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

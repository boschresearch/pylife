# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 08:04:24 2020

@author: KRD2RNG
"""

import numpy as np
import pandas as pd

def combine_hist(hist_list,method = 'sum',nbins = 64):
    """
    Performes the combination of multiple Histograms.
       
    Parameters:
    ----------
    
    hist_list: list 
        list of histogramswith all histograms (saved as Dataframes in pyLife format)
    method: str
        method: 'sum', 'min', 'max', 'mean', 'std'  default is 'sum'
    nbins: int
        number of bins of the combined histogram
    
    Returns:
    --------
    
    DataFrame: 
        Combined histogram
    list: 
        list with the reindexed input histograms
        
    """
    for ii in range(len(hist_list)): 
        if ii == 0:
            hist_combined = hist_list[ii]
        else:
            hist_combined = pd.concat([hist_combined,hist_list[ii]]).sort_index()
    index_min =  hist_combined.index.left.min()
    index_max =  hist_combined.index.right.max()        
    hist_combined = hist_combined.groupby(
            pd.cut(hist_combined.index.mid.values,
                   np.linspace(index_min,index_max,nbins))).agg(method)
    hist_combined.index = pd.interval_range(index_min,index_max,
                                         nbins-1,name = 'range')
    return hist_combined
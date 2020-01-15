# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 08:04:24 2020

@author: KRD2RNG
"""

import numpy as np
import pandas as pd
import pickle

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
    minValueIndex = min(list(map(lambda k: hist_list[k].index.left.values.min(), range(len(hist_list)))))
    maxValueIndex = max(list(map(lambda k: hist_list[k].index.right.values.max(), range(len(hist_list)))))
    binsize = (maxValueIndex - minValueIndex)/nbins
    left = np.linspace(minValueIndex,maxValueIndex-binsize,nbins)
    right = np.linspace(minValueIndex+binsize,maxValueIndex,nbins)
    index_new = pd.IntervalIndex.from_arrays(left,right)
    midvalues = index_new.mid
    Hist_reindex = []
    for df_act in hist_list:
        # reindexing of intervallindex not possible, here is the workaround
        df_new = df_act.copy()
        df_new.index = df_act.index.mid
        df_reindex = df_new.reindex(midvalues,method = 'nearest')
        df_reindex.index = index_new
        Hist_reindex.append(df_reindex)
    Hist_combined = Hist_reindex[0]#.join(map(lambda k: Hist_reindex[k],range (1,len(Hist_reindex))))        
    for ii in range (1,len(Hist_reindex)):
        Hist_combined = Hist_combined.join(Hist_reindex[ii],rsuffix= "add") 
        
    if method == 'sum':
        Hist_combined = Hist_combined.sum(axis = 'columns')
    if method == 'max':    
        Hist_combined = Hist_combined.max(axis = 'columns')
    if method == 'min':    
        Hist_combined = Hist_combined.min(axis = 'columns')
    if method == 'mean':
        Hist_combined = Hist_combined.mean(axis = 'columns')
    if method == 'std':
        Hist_combined = Hist_combined.std(axis = 'columns')
    Hist_combined = pd.DataFrame(data = Hist_combined, index = index_new, columns = hist_list[0].columns)
    Hist_combined.index.name = hist_list[0].index.name            
    return Hist_combined#, Hist_reindex
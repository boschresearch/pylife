# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 07:32:53 2020

@author: KRD2RNG
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as op
import sys
from pylife.stress import frequencysignal as freqsig
psd = pd.DataFrame(pd.read_csv("PSD_values.csv",index_col = 0).iloc[5:1500,0])
values = psd.values[:,0]
ind = psd.index.values
integral = np.trapz(psd.values[:,0],x = psd.index.values)
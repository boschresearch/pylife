# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 07:28:32 2020

@author: KRD2RNG
"""


import numpy as np
import pandas as pd
from scipy import signal as sg
from pylife.stress import timesignal as tsig

def test_resample_acc():
    # sine
    omega = 10*2*np.pi # Hz
    ts_sin = pd.DataFrame(np.sin(
        omega*np.arange(0,2,1/1024)),index = np.arange(0,2,1/1024))
    expected_sin = ts_sin.describe().drop( ['count','mean','50%','25%','75%'])
    test_sin = tsig.TimeSignalPrep(ts_sin).resample_acc(int(12*omega)).describe().drop(
        ['count','mean','50%','25%','75%'])
    pd.testing.assert_frame_equal(test_sin,expected_sin, check_less_precise = 2)
    # white noise
    ts_wn = pd.DataFrame(np.random.randn(129),index = np.linspace(0,1,129))
    expected_wn = ts_wn.describe()
    test_wn = tsig.TimeSignalPrep(ts_wn).resample_acc(128).describe()
    pd.testing.assert_frame_equal(test_wn,expected_wn,check_exact=True)
    # SoR
    t = np.arange(0,20,1/4096)
    f =  np.logspace(np.log10(1),np.log10(10),num = len(t))
    df = pd.DataFrame(data = np.multiply(
                np.interp(f,np.array([1,10]),np.array([0.5,5]),left = 0, right = 0),
                sg.chirp(t,1,20,10,method = "logarithmic",phi = -90)),
                index = t,columns = ["Sweep"])
    df['sor'] = df['Sweep'] + 0.1*np.random.randn(len(df))
    expected_sor = df.describe().drop(['count','mean','50%','25%','75%'])
    test_sor = tsig.TimeSignalPrep(df).resample_acc(2048).describe().drop(['count','mean','50%','25%','75%'])
    pd.testing.assert_frame_equal(test_sor,expected_sor, check_less_precise = 1)

def test_running_stats_filt():
    t =np.linspace(0,1,2048+1)
    sin1 = -abs(150*np.sin(2*np.pi*10*t)+152)
    sin2 = 300*np.sin(2*np.pi*10*t)
    wn1 = np.ones(len(t))
    wn1[500] = -250
    wn2 = np.zeros(len(t))
    wn2[0] = 350
    df = pd.DataFrame(np.hstack((
            sin1,
            sin2,
            wn1,
            wn2)),columns = ['data'])  
    df_sep =pd.DataFrame(np.vstack((
            sin1,
            sin2,
            wn1,
            wn2))).T
    df_sep.columns = ['sin1','sin2','wn1','wn2']
    df_stats = df_sep.describe() 
    df.plot()
    
    test_rms = tsig.TimeSignalPrep(df).running_stats_filt(col = 'data',window_length = 2049,buffer_overlap = 0.0,limit = 0.95, method = "rms")
    test_rms.columns = ['sin2']
    pd.testing.assert_frame_equal(test_rms,pd.DataFrame(df_sep['sin2']),
                                  check_index_type = False)
    
    test_max = tsig.TimeSignalPrep(df).running_stats_filt(col = 'data',window_length = 2049,buffer_overlap = 0.0,limit = 0.95, method = "max")
    test_max.columns = ['wn2']
    pd.testing.assert_frame_equal(test_max,pd.DataFrame(df_sep['wn2']),
                                  check_index_type = False)
    
    test_min = tsig.TimeSignalPrep(df).running_stats_filt(col = 'data',window_length = 2049,buffer_overlap = 0.0,limit = 1, method = "min")
    test_min.columns = ['sin1']
    pd.testing.assert_frame_equal(test_min,pd.DataFrame(df_sep['sin1']),
                                  check_index_type = False)
    
    test_abs = tsig.TimeSignalPrep(df).running_stats_filt(col = 'data',window_length = 2049,buffer_overlap = 0.0,limit = 0.1, method = "abs")
    pd.testing.assert_frame_equal(test_abs,df,
                                  check_index_type = False)
    


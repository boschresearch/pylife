# -*- coding: utf-8 -*-

import sys, os, copy
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import numpy.testing as testing

import pylife.strength.meanstress as MST

def create_cyclic_signal(Sm, Sa):
    return pd.DataFrame( {'sigma_m': Sm, 'sigma_a': Sa } )

def create_series_goodman_signal(M, M2):
    return pd.Series({ 'M':M, 'M2':M2 })

def create_dataframe_goodman_signal(M, M2, MCount):
    return pd.DataFrame({ 'M':[M]*MCount, 'M2':[M2]*MCount, })

def test_FKM_goodman_plain_different_m_a():
    Sm = np.array([-4., -2.,   -1., 0., 0.4, 2./3., 7./6.])
    Sa = np.array([ 2.,  2., 3./2., 1., 0.8, 2./3., 7./12.])
    goodman_signal = create_series_goodman_signal(0.5, 0.5 / 3)
    R_goal = -1.

    res = goodman_signal.FKM_Goodman.FKM_goodman(Sa, Sm, R_goal)
    assert np.array_equal(res, np.ones_like(res))

def test_FKM_goodman_plain_same_m_a():
    Sm = np.array([5])
    Sa = np.array([0])
    goodman_signal = create_series_goodman_signal(0.5, 0.5 / 3)
    R_goal = -1.

    res = goodman_signal.FKM_Goodman.FKM_goodman(Sa, Sm, R_goal)
    assert np.equal(res,0.)


def test_FKM_goodman_single_M():
    Sm = np.array([-4., -2.,   -1., 0., 0.4, 2./3., 7./6.])
    Sa = np.array([ 2.,  2., 3./2., 1., 0.8, 2./3., 7./12.])
    goodman_signal = create_series_goodman_signal(0.5, 0.5 / 3)
    cyclic_signal = create_cyclic_signal(Sm, Sa)
    R_goal = -1.

    res = cyclic_signal.meanstress_mesh.FKM_goodman(goodman_signal, R_goal).sigma_a
    assert np.array_equal(res, np.ones_like(res))

def test_FKM_goodman_multiple_M():
    Sm = np.array([-4., -2.,   -1., 0., 0.4, 2./3., 7./6.])
    Sa = np.array([ 2.,  2., 3./2., 1., 0.8, 2./3., 7./12.])   
    goodman_signal = create_dataframe_goodman_signal(0.5, 0.5 / 3, 7)
    cyclic_signal = create_cyclic_signal(Sm, Sa)
    R_goal = -1.

    res = cyclic_signal.meanstress_mesh.FKM_goodman(goodman_signal, R_goal).sigma_a
    print(res)
    assert np.array_equal(res, np.ones_like(res))
 
def test_5_segment():
    Sm = np.array([-12./5., -2., -1., 0., 2./5., 2./3., 7./6., 1.+23./75., 2.+1./150., 3.+11./25., 3.+142./225.])
    Sa = np.array([ 6./5., 2., 3./2., 1., 4./5., 2./3., 7./12., 14./25., 301./600., 86./225., 43./225.])

    M0 = 0.5
    M1 = M0/3.
    M2 = M0/6.
    M3 = 1.
    M4 = -2.

    R_12 = 2./5.
    R_23 = 4./5.

    R_goal = 1.
    testing.assert_raises(ValueError, MST.Five_Segment_Correction,Sa, Sm, R_goal, M0, M1, M2, M3, M4, R_12, R_23)

    calc = MST.MeanstressFiveSlope(M0=M0, M1=M1, M2=M2, M3=M3, M4=M4, R12=R_12, R23=R_23, R_goal=-1)
    res = calc(Sa, Sm)
    assert np.allclose(res, np.ones_like(res))

    R_goal = -1.
    res = MST.Five_Segment_Correction(Sa, Sm, R_goal, M0, M1, M2, M3, M4, R_12, R_23)
    assert np.allclose(res, np.ones_like(res))

    Sm = np.array([5])
    Sa = np.array([0])
    res = MST.Five_Segment_Correction(Sa, Sm, R_goal, M0, M1, M2, M3, M4, R_12, R_23)
    assert np.equal(res,0.)

# -*- coding: utf-8 -*-

import sys, os, copy
import warnings

import numpy as np
import pandas as pd
import numpy.testing as testing

import pylife.strength.meanstress as MST


def goodman_signal_sm():
    Sm = np.array([-4., -2.,   -1., 0., 0.4, 2./3., 7./6.])
    Sa = np.array([ 2.,  2., 3./2., 1., 0.8, 2./3., 7./12.])
    return pd.DataFrame({'sigma_m': Sm, 'sigma_a': Sa })

def goodman_signal_r():
    Sm = np.array([-4., -2.,   -1., 0., 0.4, 2./3., 7./6.])
    Sa = np.array([ 2.,  2., 3./2., 1., 0.8, 2./3., 7./12.])
    warnings.simplefilter('ignore', RuntimeWarning)
    R = (Sm-Sa)/(Sm+Sa)
    warnings.simplefilter('default', RuntimeWarning)
    return pd.DataFrame({'sigma_a': Sa, 'R': R})


def test_FKM_goodman_plain_sm():
    cyclic_signal = goodman_signal_sm()
    Sa = cyclic_signal.sigma_a.to_numpy()
    Sm = cyclic_signal.sigma_m.to_numpy()
    M = 0.5

    R_goal = 1.
    testing.assert_raises(ValueError, MST.FKM_goodman, Sa, Sm, M, M/3, R_goal)

    R_goal = -1.
    res = MST.FKM_goodman(Sa, Sm, M, M/3, R_goal)
    np.testing.assert_array_equal(res, np.ones_like(res))

    Sm = np.array([5])
    Sa = np.array([0])
    res = MST.FKM_goodman(Sa, Sm, M, M/3, R_goal)
    assert np.equal(res,0.)



def test_FKM_goodman_single_M_sm():
    cyclic_signal = goodman_signal_sm()
    M = 0.5

    R_goal = -1.

    res = cyclic_signal.meanstress_mesh.FKM_goodman(pd.Series({ 'M':M, 'M2':M/3 }), R_goal).sigma_a
    np.testing.assert_array_equal(res, np.ones_like(res))


def test_FKM_goodman_single_M_R():
    cyclic_signal = goodman_signal_r()
    M = 0.5

    R_goal = -1.

    res = cyclic_signal.meanstress_mesh.FKM_goodman(pd.Series({ 'M':M, 'M2':M/3 }), R_goal).sigma_a
    np.testing.assert_array_equal(res, np.ones_like(res))



def test_FKM_goodman_multiple_M_sm():
    cyclic_signal = goodman_signal_sm()
    M = 0.5

    R_goal = -1.
    res = cyclic_signal.meanstress_mesh.FKM_goodman(pd.DataFrame({ 'M':[M]*7, 'M2':[M/3]*7, }), R_goal).sigma_a
    np.testing.assert_array_equal(res, np.ones_like(res))


def test_FKM_goodman_multiple_M_sm():
    cyclic_signal = goodman_signal_r()
    M = 0.5

    R_goal = -1.
    res = cyclic_signal.meanstress_mesh.FKM_goodman(pd.DataFrame({ 'M':[M]*7, 'M2':[M/3]*7, }), R_goal).sigma_a
    np.testing.assert_array_equal(res, np.ones_like(res))


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

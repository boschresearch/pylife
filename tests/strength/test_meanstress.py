# Copyright (c) 2019-2020 - for information on the respective copyright owner
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
# -*- coding: utf-8 -*-

import sys, os, copy
import warnings
import pytest

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


def five_segment_signal_sm():
    Sm = np.array([-12./5., -2., -1., 0., 2./5., 2./3., 7./6., 1.+23./75., 2.+1./150., 3.+11./25., 3.+142./225.])
    Sa = np.array([ 6./5., 2., 3./2., 1., 4./5., 2./3., 7./12., 14./25., 301./600., 86./225., 43./225.])
    return pd.DataFrame({'sigma_m': Sm, 'sigma_a': Sa })

def five_segment_signal_r():
    Sm = np.array([-12./5., -2., -1., 0., 2./5., 2./3., 7./6., 1.+23./75., 2.+1./150., 3.+11./25., 3.+142./225.])
    Sa = np.array([ 6./5., 2., 3./2., 1., 4./5., 2./3., 7./12., 14./25., 301./600., 86./225., 43./225.])
    warnings.simplefilter('ignore', RuntimeWarning)
    R = (Sm-Sa)/(Sm+Sa)
    warnings.simplefilter('default', RuntimeWarning)
    return pd.DataFrame({'sigma_a': Sa, 'R': R })


def test_FKM_goodman_plain_sm():
    cyclic_signal = goodman_signal_sm()
    Sa = cyclic_signal.sigma_a.to_numpy()
    Sm = cyclic_signal.sigma_m.to_numpy()
    M = 0.5

    R_goal = 1.
    testing.assert_raises(ValueError, MST.FKM_goodman, Sa, Sm, M, M/3, R_goal)

    R_goal = -1.
    res = MST.FKM_goodman(Sa, Sm, M, M/3, R_goal)
    np.testing.assert_array_almost_equal(res, np.ones_like(res))

    Sm = np.array([5])
    Sa = np.array([0])
    res = MST.FKM_goodman(Sa, Sm, M, M/3, R_goal)
    assert np.equal(res,0.)


def test_FKM_goodman_single_M_sm():
    cyclic_signal = goodman_signal_sm()
    M = 0.5

    R_goal = -1.

    res = cyclic_signal.meanstress_mesh.FKM_goodman(pd.Series({ 'M':M, 'M2':M/3 }), R_goal).sigma_a
    np.testing.assert_array_almost_equal(res, np.ones_like(res))


def test_FKM_goodman_single_M_R():
    cyclic_signal = goodman_signal_r()
    M = 0.5

    R_goal = -1.

    res = cyclic_signal.meanstress_mesh.FKM_goodman(pd.Series({ 'M':M, 'M2':M/3 }), R_goal).sigma_a
    np.testing.assert_array_almost_equal(res, np.ones_like(res))


def test_FKM_goodman_multiple_M_sm():
    cyclic_signal = goodman_signal_sm()
    M = 0.5

    R_goal = -1.
    res = cyclic_signal.meanstress_mesh.FKM_goodman(pd.DataFrame({ 'M':[M]*7, 'M2':[M/3]*7, }), R_goal).sigma_a
    np.testing.assert_array_almost_equal(res, np.ones_like(res))


def test_FKM_goodman_multiple_M_sm():
    cyclic_signal = goodman_signal_r()
    M = 0.5

    R_goal = -1.
    res = cyclic_signal.meanstress_mesh.FKM_goodman(pd.DataFrame({ 'M':[M]*7, 'M2':[M/3]*7, }), R_goal).sigma_a
    np.testing.assert_array_almost_equal(res, np.ones_like(res))


def test_five_segment_plain_sm():
    cyclic_signal = five_segment_signal_sm()
    Sa = cyclic_signal.sigma_a.to_numpy()
    Sm = cyclic_signal.sigma_m.to_numpy()

    M0= 0.5
    M1 = M0/3.
    M2 = M0/6.
    M3 = 1.
    M4 = -2.

    R12 = 2./5.
    R23 = 4./5.

    R_goal = 1.
    testing.assert_raises(ValueError, MST.five_segment_correction, Sa, Sm, M0, M1, M2, M3, M4, R12, R23, R_goal)

    res = MST.five_segment_correction(Sa, Sm, M0=M0, M1=M1, M2=M2, M3=M3, M4=M4, R12=R12, R23=R23, R_goal=-1)
    np.testing.assert_allclose(res, np.ones_like(res))

    R_goal = -1.
    res = MST.five_segment_correction(Sa, Sm, M0, M1, M2, M3, M4, R12, R23, R_goal)
    np.testing.assert_array_almost_equal(res, np.ones_like(res))

    Sm = np.array([5])
    Sa = np.array([0])
    res = MST.five_segment_correction(Sa, Sm, M0, M1, M2, M3, M4, R12, R23, R_goal)
    assert np.equal(res,0.)


def test_five_segment_single_M_sm():
    cyclic_signal = five_segment_signal_sm()
    M0= 0.5
    M1 = M0/3.
    M2 = M0/6.
    M3 = 1.
    M4 = -2.

    R12 = 2./5.
    R23 = 4./5.

    R_goal = -1.

    res = cyclic_signal.meanstress_mesh.five_segment(pd.Series({
        'M0': M0, 'M1': M1, 'M2': M2, 'M3': M3, 'M4': M4,
        'R12': R12, 'R23': R23
    }), R_goal).sigma_a
    np.testing.assert_array_almost_equal(res, np.ones_like(res))


@pytest.mark.parametrize("Sm, Sa", [(np.array([row.sigma_m]), np.array([row.sigma_a])) for _, row in five_segment_signal_sm().iterrows()])
def test_five_segment_single_M_backwards(Sm, Sa):
    cyclic_signal = pd.DataFrame({'sigma_a': [1.0], 'sigma_m': [0.0]})
    M0= 0.5
    M1 = M0/3.
    M2 = M0/6.
    M3 = 1.
    M4 = -2.

    R12 = 2./5.
    R23 = 4./5.

    warnings.simplefilter('ignore', RuntimeWarning)

    R_goal = (Sm-Sa)/(Sm+Sa)


    warnings.simplefilter('default', RuntimeWarning)

    res = cyclic_signal.meanstress_mesh.five_segment(pd.Series({
        'M0': M0, 'M1': M1, 'M2': M2, 'M3': M3, 'M4': M4,
        'R12': R12, 'R23': R23
    }), R_goal)
    np.testing.assert_array_almost_equal(res.sigma_a, Sa)



def test_five_segment_single_M_R():
    cyclic_signal = five_segment_signal_r()
    M0= 0.5
    M1 = M0/3.
    M2 = M0/6.
    M3 = 1.
    M4 = -2.

    R12 = 2./5.
    R23 = 4./5.

    R_goal = -1.

    res = cyclic_signal.meanstress_mesh.five_segment(pd.Series({
        'M0': M0, 'M1': M1, 'M2': M2, 'M3': M3, 'M4': M4,
        'R12': R12, 'R23': R23
    }), R_goal).sigma_a
    np.testing.assert_array_almost_equal(res, np.ones_like(res))


def test_five_segment_multiple_M_sm():
    cyclic_signal = five_segment_signal_sm()
    M0= 0.5
    M1 = M0/3.
    M2 = M0/6.
    M3 = 1.
    M4 = -2.

    R12 = 2./5.
    R23 = 4./5.

    R_goal = -1.
    res = cyclic_signal.meanstress_mesh.five_segment(pd.DataFrame({
        'M0': [M0]*11, 'M1': [M1]*11, 'M2': [M2]*11, 'M3': [M3]*11, 'M4': [M4]*11,
        'R12': [R12]*11, 'R23': [R23]*11
    }), R_goal).sigma_a
    np.testing.assert_array_almost_equal(res, np.ones_like(res))


@pytest.mark.parametrize("R_goal, expected", [ # all calculated by pencil on paper
    (-1., 2.0),
    (0., 4./3.),
    (-1./3., 8./5.),
    (1./3., 14./12.)
])
def test_FKM_goodman_hist_range_mean(R_goal, expected):
    rg = pd.IntervalIndex.from_breaks(np.linspace(0, 2, 25), closed='left')
    mn = pd.IntervalIndex.from_breaks(np.linspace(0, 2, 25), closed='left')

    df = pd.DataFrame({'frequency': np.zeros(24*24)}, index=pd.MultiIndex.from_product([rg,mn], names=['range', 'mean']))
    df.loc[(7./6. - 1./24., 7./6.)] = 1.
    df.loc[(4./3. - 1./24., 2./3.)] = 3.
    df.loc[(2. - 1./24., 0.)] = 5.

    haigh = pd.Series({'M': 0.5, 'M2': 0.5/3.})
    res = df.meanstress_hist.FKM_goodman(haigh, R_goal)

    test_interval = pd.Interval(expected-1./96., expected+1./96.)
    assert res.loc[res.index.overlaps(test_interval), 'frequency'].sum() == 9
    assert res.loc[np.logical_not(res.index.overlaps(test_interval)), 'frequency'].sum() == 0


@pytest.mark.parametrize("R_goal, expected", [ # all calculated by pencil on paper
    (-1., 2.0),
    (0., 4./3.),
    (-1./3., 8./5.),
    (1./3., 14./12.)
])
def test_FKM_goodman_hist_from_to(R_goal, expected):
    fr = pd.IntervalIndex.from_breaks(np.linspace(-1., 1., 49), closed='left')
    to = pd.IntervalIndex.from_breaks(np.linspace(0, 2., 49), closed='left')

    df = pd.DataFrame({'frequency': np.zeros(48*48)}, index=pd.MultiIndex.from_product([fr,to], names=['from', 'to']))
    df.loc[(14./24., 21./12.)] = 1
    df.loc[(0., 4./3.)] = 3
    df.loc[(-1., 1.)] = 5

    haigh = pd.Series({'M': 0.5, 'M2': 0.5/3.})
    res = df.meanstress_hist.FKM_goodman(haigh, R_goal)

    test_interval = pd.Interval(expected-1./96., expected+1./96.)
    assert res.loc[res.index.overlaps(test_interval), 'frequency'].sum() == 9
    assert res.loc[np.logical_not(res.index.overlaps(test_interval)), 'frequency'].sum() == 0


@pytest.mark.parametrize("R_goal, expected", [ # all calculated by pencil on paper
    (-1., 2.0),
    (0., 4./3.),
    (-1./3., 8./5.),
    (1./3., 14./12.)
])
def test_FKM_goodman_hist_range_mean_nonzero(R_goal, expected):
    rg = pd.IntervalIndex.from_breaks(np.linspace(0, 2, 25), closed='left')
    mn = pd.IntervalIndex.from_breaks(np.linspace(0, 2, 25), closed='left')

    df = pd.DataFrame({'frequency': np.zeros(24*24)}, index=pd.MultiIndex.from_product([rg,mn], names=['range', 'mean']))
    df.loc[(7./6., 7./6.)] = 1.
    df.loc[(4./3., 2./3.)] = 3.
    df.loc[(2.-1./96., 0.)] = 5.

    haigh = pd.Series({'M': 0.5, 'M2': 0.5/3.})
    res = df[df.values > 0].meanstress_hist.FKM_goodman(haigh, R_goal)

    test_interval = pd.Interval(expected-1./96., expected+1./96.)
    print(df[df.values > 0])
    print(df.meanstress_hist.FKM_goodman(haigh, R_goal))
    print(res)
    assert res.loc[res.index.overlaps(test_interval), 'frequency'].sum() == 9
    assert res.loc[np.logical_not(res.index.overlaps(test_interval)), 'frequency'].sum() == 0

    binsize = res.index.get_level_values('range').length.min()
    np.testing.assert_approx_equal(binsize, 2./24., significant=1)


def test_null_histogram():
    rg = pd.IntervalIndex.from_breaks(np.linspace(0, 2, 25), closed='left')
    mn = pd.IntervalIndex.from_breaks(np.linspace(0, 2, 25), closed='left')

    df = pd.DataFrame({'frequency': np.zeros(24*24, dtype=np.int)}, index=pd.MultiIndex.from_product([rg,mn], names=['from', 'to']))
    haigh = pd.Series({'M': 0.5, 'M2': 0.5/3.})
    res = df.meanstress_hist.FKM_goodman(haigh, -1)

    assert not res['frequency'].any()


def test_full_histogram():
    rg = pd.IntervalIndex.from_breaks(np.linspace(0, 2, 25), closed='left')
    mn = pd.IntervalIndex.from_breaks(np.linspace(0, 2, 25), closed='left')

    df = pd.DataFrame({'frequency': np.linspace(1, 576, 576, dtype=np.int)}, index=pd.MultiIndex.from_product([rg,mn], names=['from', 'to']))
    haigh = pd.Series({'M': 0.5, 'M2': 0.5/3.})
    res = df.meanstress_hist.FKM_goodman(haigh, -1)

    print(df)
    print(res)
    print(res['frequency'].sum() - df['frequency'].sum())
    assert res['frequency'].sum() == df['frequency'].sum()

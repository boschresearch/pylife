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

import warnings
import pytest

import numpy as np
import pandas as pd
import numpy.testing as testing

import pylife.strength.meanstress as MST


def goodman_signal_sm():
    Sm = np.array([-4., -2.,   -1., 0., 0.4, 2./3., 7./6.])
    Sa = np.array([ 2.,  2., 3./2., 1., 0.8, 2./3., 7./12.])
    return pd.DataFrame({'mean': Sm, 'range': 2.*Sa })


def goodman_signal_r():
    Sm = np.array([-4., -2.,   -1., 0., 0.4, 2./3., 7./6.])
    Sa = np.array([ 2.,  2., 3./2., 1., 0.8, 2./3., 7./12.])
    warnings.simplefilter('ignore', RuntimeWarning)
    R = (Sm-Sa)/(Sm+Sa)
    warnings.simplefilter('default', RuntimeWarning)
    return pd.DataFrame({'range': 2.*Sa, 'R': R})


def five_segment_signal_sm():
    Sm = np.array([-12./5., -2., -1., 0., 2./5., 2./3., 7./6., 1.+23./75., 2.+1./150., 3.+11./25., 3.+142./225.])
    Sa = np.array([ 6./5., 2., 3./2., 1., 4./5., 2./3., 7./12., 14./25., 301./600., 86./225., 43./225.])
    return pd.DataFrame({'mean': Sm, 'range': 2.*Sa })


def five_segment_signal_r():
    Sm = np.array([-12./5., -2., -1., 0., 2./5., 2./3., 7./6., 1.+23./75., 2.+1./150., 3.+11./25., 3.+142./225.])
    Sa = np.array([ 6./5., 2., 3./2., 1., 4./5., 2./3., 7./12., 14./25., 301./600., 86./225., 43./225.])
    warnings.simplefilter('ignore', RuntimeWarning)
    R = (Sm-Sa)/(Sm+Sa)
    warnings.simplefilter('default', RuntimeWarning)
    return pd.DataFrame({'range': 2.*Sa, 'R': R })


def test_fkm_goodman_plain_sm():
    cyclic_signal = goodman_signal_sm()
    Sa = cyclic_signal['range'].to_numpy()/2.
    Sm = cyclic_signal['mean'].to_numpy()
    M = 0.5

#    R_goal = 1.
#    testing.assert_raises(ValueError, MST.fkm_goodman, Sa, Sm, M, M/3, R_goal)

    R_goal = -1.
    res = MST.fkm_goodman(Sa, Sm, M, M/3, R_goal)
    np.testing.assert_array_almost_equal(res, np.ones_like(res))

    Sm = np.array([5])
    Sa = np.array([0])
    res = MST.fkm_goodman(Sa, Sm, M, M/3, R_goal)
    assert np.equal(res, 0.)


def test_fkm_goodman_single_M_sm():
    cyclic_signal = goodman_signal_sm()
    M = 0.5

    R_goal = -1.

    res = cyclic_signal.meanstress_transform.fkm_goodman(pd.Series({'M': M, 'M2': M/3 }), R_goal).amplitude
    np.testing.assert_array_almost_equal(res, np.ones_like(res))


def test_fkm_goodman_multiple_M_sm():
    cyclic_signal = goodman_signal_sm()
    cyclic_signal.index.name = 'element_id'
    M = 0.5

    R_goal = -1.
    haigh = pd.DataFrame({'M': [M]*7, 'M2': [M/3]*7})
    res = cyclic_signal.meanstress_transform.fkm_goodman(haigh, R_goal).amplitude
    np.testing.assert_array_almost_equal(res, np.ones_like(res))


def test_five_segment_plain_sm():
    cyclic_signal = five_segment_signal_sm()
    Sa = cyclic_signal['range'].to_numpy()/2.
    Sm = cyclic_signal['mean'].to_numpy()

    M0 = 0.5
    M1 = M0/3.
    M2 = M0/6.
    M3 = 1.
    M4 = -2.

    R12 = 2./5.
    R23 = 4./5.

#    R_goal = 1.
#    testing.assert_raises(ValueError, MST.five_segment_correction, Sa, Sm, M0, M1, M2, M3, M4, R12, R23, R_goal)

    res = MST.five_segment_correction(Sa, Sm, M0=M0, M1=M1, M2=M2, M3=M3, M4=M4, R12=R12, R23=R23, R_goal=-1)
    np.testing.assert_allclose(res, np.ones_like(res))

    R_goal = -1.
    res = MST.five_segment_correction(Sa, Sm, M0, M1, M2, M3, M4, R12, R23, R_goal)
    np.testing.assert_array_almost_equal(res, np.ones_like(res))

    Sm = np.array([5])
    Sa = np.array([0])
    res = MST.five_segment_correction(Sa, Sm, M0, M1, M2, M3, M4, R12, R23, R_goal)
    assert np.equal(res, 0.)

    Sm = np.array([5, 5])
    Sa = np.array([0, 0])
    R_goal = 0.1
    res = MST.five_segment_correction(Sa, Sm, M0, M1, M2, M3, M4, R12, R23, R_goal)
    assert np.array_equal(res, np.array([0., 0.]))


def test_five_segment_single_M_sm():
    cyclic_signal = five_segment_signal_sm()
    M0 = 0.5
    M1 = M0/3.
    M2 = M0/6.
    M3 = 1.
    M4 = -2.

    R12 = 2./5.
    R23 = 4./5.

    R_goal = -1.

    res = cyclic_signal.meanstress_transform.five_segment(pd.Series({
        'M0': M0, 'M1': M1, 'M2': M2, 'M3': M3, 'M4': M4,
        'R12': R12, 'R23': R23
    }), R_goal).amplitude
    np.testing.assert_array_almost_equal(res, np.ones_like(res))


@pytest.mark.parametrize("Sm, Sa", [
    (np.array([row['mean']]), np.array([row['range']/2.])) for _, row in five_segment_signal_sm().iterrows()
])
def test_five_segment_single_M_backwards(Sm, Sa):
    cyclic_signal = pd.DataFrame({'range': [2.0], 'mean': [0.0]})
    M0 = 0.5
    M1 = M0/3.
    M2 = M0/6.
    M3 = 1.
    M4 = -2.

    R12 = 2./5.
    R23 = 4./5.

    R_goal = -np.inf if Sm+Sa == 0 else ((Sm-Sa)/(Sm+Sa))[0]

    res = cyclic_signal.meanstress_transform.five_segment(pd.Series({
        'M0': M0, 'M1': M1, 'M2': M2, 'M3': M3, 'M4': M4,
        'R12': R12, 'R23': R23
    }), R_goal)

    np.testing.assert_array_almost_equal(res.amplitude, Sa)


def test_five_segment_multiple_M_sm():
    cyclic_signal = five_segment_signal_sm()
    M0 = 0.5
    M1 = M0/3.
    M2 = M0/6.
    M3 = 1.
    M4 = -2.

    R12 = 2./5.
    R23 = 4./5.

    R_goal = -1.
    index = pd.MultiIndex.from_tuples([
        (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3)
    ], names=['element_id', 'node_id'])
    res = cyclic_signal.meanstress_transform.five_segment(pd.DataFrame({
        'M0': [M0]*6, 'M1': [M1]*6, 'M2': [M2]*6, 'M3': [M3]*6, 'M4': [M4]*6,
        'R12': [R12]*6, 'R23': [R23]*6
    }, index=index), R_goal)
    np.testing.assert_array_almost_equal(res.amplitude, np.ones_like(res))


#GH-105
@pytest.mark.parametrize("R_goal", [-1., 0., -1./3., 1./3.])
def test_fkm_goodman_hist_R_goal(R_goal):
    rg = pd.IntervalIndex.from_breaks(np.linspace(0., 2., 25), closed='left')
    mn = pd.IntervalIndex.from_breaks(np.linspace(-1./12., 23./12., 25), closed='left')

    mat = pd.Series(np.zeros(24*24), name='cycles',
                    index=pd.MultiIndex.from_product([rg, mn], names=['range', 'mean']))
    mat.loc[(7./6., 7./6.)] = 1.
    mat.loc[(4./3., 2./3.)] = 3.
    mat.loc[(2. - 1e-9, 0.)] = 5.

    haigh = pd.Series({'M': 0.5, 'M2': 0.5/3.})
    result = mat.meanstress_transform.fkm_goodman(haigh, R_goal).R

    expected = pd.Series(R_goal, index=result.index, name=result.name)

    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize("R_goal, expected", [  # all calculated by pencil on paper
    (-1., 2.0),
    (0., 4./3.),
    (-1./3., 8./5.),
    (1./3., 14./12.)
])
def test_fkm_goodman_hist_range_mean(R_goal, expected):
    rg = pd.IntervalIndex.from_breaks(np.linspace(0., 2., 25), closed='left')
    mn = pd.IntervalIndex.from_breaks(np.linspace(-1./12., 23./12., 25), closed='left')

    mat = pd.Series(np.zeros(24*24), name='cycles',
                    index=pd.MultiIndex.from_product([rg, mn], names=['range', 'mean']))
    mat.loc[(7./6., 7./6.)] = 1.
    mat.loc[(4./3., 2./3.)] = 3.
    mat.loc[(2. - 1e-9, 0.)] = 5.

    haigh = pd.Series({'M': 0.5, 'M2': 0.5/3.})
    res = mat.meanstress_transform.fkm_goodman(haigh, R_goal).to_pandas()
    test_interval = pd.Interval(expected-1./96., expected+1./96.)

    mask = res.index.get_level_values('range').overlaps(test_interval)
    assert res.loc[mask].sum() == 9
    assert res.loc[~mask].sum() == 0


@pytest.mark.parametrize("R_goal, expected", [  # all calculated by pencil on paper
    (-1., 2.0),
    (0., 4./3.),
    (-1./3., 8./5.),
    (1./3., 14./12.)
])
def test_fkm_goodman_hist_from_to(R_goal, expected):
    fr = pd.IntervalIndex.from_breaks(np.linspace(-25./24., 1., 49), closed='left')
    to = pd.IntervalIndex.from_breaks(np.linspace(-1./24., 2., 49), closed='left')

    mat = pd.Series(np.zeros(48*48), name='cycles',
                    index=pd.MultiIndex.from_product([fr, to], names=['from', 'to']))
    mat.loc[(14./24., 21./12.)] = 1
    mat.loc[(0., 4./3.)] = 3
    mat.loc[(-1., 1.)] = 5

    haigh = pd.Series({'M': 0.5, 'M2': 0.5/3.})
    res = mat.meanstress_transform.fkm_goodman(haigh, R_goal).to_pandas()

    test_interval = pd.Interval(expected-1./96., expected+1./96.)

    mask = res.index.get_level_values('range').overlaps(test_interval)
    assert res.loc[mask].sum() == 9
    assert res.loc[~mask].sum() == 0


def test_meanstress_transform_additional_index():
    fr = pd.IntervalIndex.from_breaks(np.linspace(-25./24., 1., 49), closed='left')
    to = pd.IntervalIndex.from_breaks(np.linspace(-1./24., 2., 49), closed='left')
    node = pd.Index([1, 2, 3], name='node_id')

    mat = pd.Series(
        0.0,
        index=pd.MultiIndex.from_product([fr, to, node], names=['from', 'to', 'node_id']),
        name='cycles'
    )

    mat.loc[(14./24., 21./12., 1)] = 1
    mat.loc[(0., 4./3., 1)] = 3
    mat.loc[(-1., 1., 1)] = 5

    mat.loc[(14./24., 21./12., 2)] = 2
    mat.loc[(0., 4./3., 2)] = 6
    mat.loc[(-1., 1., 2)] = 10

    mat.loc[(14./24., 21./12., 3)] = 4
    mat.loc[(0., 4./3., 3)] = 12
    mat.loc[(-1., 1., 3)] = 20

    haigh = pd.Series({'M': 0.5, 'M2': 0.5/3.})
    res = mat.meanstress_transform.fkm_goodman(haigh, -1.0).to_pandas()

    assert set(res.index.names) == {'range', 'mean', 'node_id'}

    null_itv = pd.Interval(0.0, 0.0)

    assert res.loc[(2.0, null_itv, 1)] == 9
    assert res.loc[(2.0, null_itv, 2)] == 18
    assert res.loc[(2.0, null_itv, 3)] == 36

    assert res.sum() == 9 + 18 + 36
    assert res.min() == 0


def test_meanstress_transform_two_additional_indices():
    fr = pd.IntervalIndex.from_breaks(np.linspace(-25./24., 1., 49), closed='left', name='from')
    to = pd.IntervalIndex.from_breaks(np.linspace(-1./24., 2., 49), closed='left', name='to')
    node = pd.Index([1, 2], name='node_id')
    element = pd.Index([1, 2], name='element_id')

    mat = pd.Series(
        0.0,
        index=pd.MultiIndex.from_product([fr, to, node, element]),
        name='cycles'
    )

    mat.loc[(14./24., 21./12., 1, 1)] = 1
    mat.loc[(0., 4./3., 1, 1)] = 3
    mat.loc[(-1., 1., 1, 1)] = 5

    mat.loc[(14./24., 21./12., 2, 1)] = 2
    mat.loc[(0., 4./3., 2, 1)] = 6
    mat.loc[(-1., 1., 2, 1)] = 10

    mat.loc[(14./24., 21./12., 1, 2)] = 4
    mat.loc[(0., 4./3., 1, 2)] = 12
    mat.loc[(-1., 1., 1, 2)] = 20

    mat.loc[(14./24., 21./12., 2, 2)] = 8
    mat.loc[(0., 4./3., 2, 2)] = 24
    mat.loc[(-1., 1., 2, 2)] = 40

    haigh = pd.Series({'M': 0.5, 'M2': 0.5/3.})
    res = mat.meanstress_transform.fkm_goodman(haigh, -1.0).to_pandas()

    assert set(res.index.names) == {'range', 'mean', 'node_id', 'element_id'}

    null_itv = pd.Interval(0.0, 0.0)

    assert res.loc[(2.0, null_itv, 1, 1)] == 9
    assert res.loc[(2.0, null_itv, 2, 1)] == 18

    assert res.loc[(2.0, null_itv, 1, 2)] == 36
    assert res.loc[(2.0, null_itv, 2, 2)] == 72

    assert res.sum() == 9 + 18 + 36 + 72
    assert res.min() == 0


@pytest.mark.parametrize("R_goal, expected", [  # all calculated by pencil on paper
    (-1., 2.0),
    (0., 4./3.),
    (-1./3., 8./5.),
    (1./3., 14./12.)
])
def test_fkm_goodman_hist_range_mean_nonzero(R_goal, expected):
    rg = pd.IntervalIndex.from_breaks(np.linspace(0, 2, 25), closed='left')
    mn = pd.IntervalIndex.from_breaks(np.linspace(-1./12., 23./12., 25), closed='left')

    mat = pd.Series(np.zeros(24*24), name='cycles',
                    index=pd.MultiIndex.from_product([rg, mn], names=['range', 'mean']))
    mat.loc[(7./6., 7./6.)] = 1.
    mat.loc[(4./3., 2./3.)] = 3.
    mat.loc[(2. - 1e-9, 0.)] = 5.

    haigh = pd.Series({'M': 0.5, 'M2': 0.5/3.})
    res = mat[mat.values > 0].meanstress_transform.fkm_goodman(haigh, R_goal).to_pandas()

    test_interval = pd.Interval(expected-1./96., expected+1./96.)

    mask = res.index.get_level_values('range').overlaps(test_interval)
    assert res.loc[mask].sum() == 9
    assert res.loc[~mask].sum() == 0

    binsize = res.index.get_level_values('range').length.min()
    np.testing.assert_approx_equal(binsize, 2./24., significant=1)


def test_null_histogram():
    rg = pd.IntervalIndex.from_breaks(np.linspace(0, 2, 25), closed='left')
    mn = pd.IntervalIndex.from_breaks(np.linspace(0, 2, 25), closed='left')

    mat = pd.Series(np.zeros(24*24, dtype=np.int32), name='cycles',
                    index=pd.MultiIndex.from_product([rg, mn], names=['from', 'to']))
    haigh = pd.Series({'M': 0.5, 'M2': 0.5/3.})
    res = mat.meanstress_transform.fkm_goodman(haigh, -1).to_pandas()

    assert not res.any()


def test_full_histogram():
    rg = pd.IntervalIndex.from_breaks(np.linspace(0, 2, 25), closed='left')
    mn = pd.IntervalIndex.from_breaks(np.linspace(0, 2, 25), closed='left')

    series = pd.Series(np.linspace(1, 576, 576, dtype=np.int32), name='cycles',
                       index=pd.MultiIndex.from_product([rg, mn], names=['from', 'to']))
    haigh = pd.Series({'M': 0.5, 'M2': 0.5/3.})
    res = series.meanstress_transform.fkm_goodman(haigh, -1).to_pandas()

    assert res.sum() == series.sum()


# import itertools

# @pytest.mark.parametrize('bincount_from, bincount_to', itertools.product(*[range(1, 25), range(1, 25)]))
# def test_full_histogram_varying_bins(bincount_from, bincount_to):
#     rg = pd.IntervalIndex.from_breaks(np.linspace(0, 2, bincount_from+1), closed='left')
#     mn = pd.IntervalIndex.from_breaks(np.linspace(0, 2, bincount_to+1), closed='left')

#     bincount_prod = bincount_from * bincount_to
#     series = pd.Series(np.linspace(1, bincount_prod, bincount_prod, dtype=np.int32), name='cycles',
#                        index=pd.MultiIndex.from_product([rg, mn], names=['range', 'mean']))
#     haigh = pd.Series({'M': 0.5, 'M2': 0.5/3.})
#     res = series.meanstress_transform.fkm_goodman(haigh, -1).to_pandas()

#     assert res.sum() == series.sum()


# @pytest.mark.parametrize('bincount_from, bincount_to', itertools.product(*[range(1, 25), range(1, 25)]))
# def test_full_histogram_range_mean_varying_bins(bincount_from, bincount_to):
#     fr = pd.IntervalIndex.from_breaks(np.linspace(0, 2, bincount_from+1), closed='left')
#     to = pd.IntervalIndex.from_breaks(np.linspace(0, 2, bincount_to+1), closed='left')

#     bincount_prod = bincount_from * bincount_to
#     series = pd.Series(np.linspace(1, bincount_prod, bincount_prod, dtype=np.int32), name='cycles',
#                        index=pd.MultiIndex.from_product([fr, to], names=['from', 'to']))
#     haigh = pd.Series({'M': 0.5, 'M2': 0.5/3.})
#     res = series.meanstress_transform.fkm_goodman(haigh, -1).to_pandas()

#     assert res.sum() == series.sum()


@pytest.mark.parametrize("N_c, M_sigma", [  # Calculated by pencil and paper
    (3e6, 0.2),
    (1e6, 0.3784),
    (1e5, 0.0140)
])
def test_experimental_mean_stress_sensitivity(N_c, M_sigma):
    sn_curve_R0 = pd.Series({'SD': 100, 'ND': 1e6, 'k_1': 3})
    sn_curve_Rn1 = pd.Series({'SD': 120, 'ND': 2e6, 'k_1': 5})

    testing.assert_almost_equal(
        actual=MST.experimental_mean_stress_sensitivity(sn_curve_R0, sn_curve_Rn1, N_c=N_c),
        desired=M_sigma,
        decimal=4)


def test_experimental_mean_stress_sensitivity_no_Nc():
    sn_curve_R0 = pd.Series({'SD': 100, 'ND': 1e6, 'k_1': 3})
    sn_curve_Rn1 = pd.Series({'SD': 120, 'ND': 2e6, 'k_1': 5})

    testing.assert_almost_equal(
        actual=MST.experimental_mean_stress_sensitivity(sn_curve_R0, sn_curve_Rn1),
        desired=0.2,
        decimal=4)


def test_experimental_mean_stress_sensitivity_plausible():
    sn_curve_R0 = pd.Series({'SD': 100, 'ND': 1e6, 'k_1': 3})
    sn_curve_Rn1 = pd.Series({'SD': 120, 'ND': 2e6, 'k_1': 5})
    with testing.assert_raises(ValueError):
        # Should lead to -0.27
        MST.experimental_mean_stress_sensitivity(sn_curve_R0, sn_curve_Rn1, N_c=10**4)

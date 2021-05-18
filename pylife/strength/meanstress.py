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

'''
Meanstress routines
===================

Mean stress transformation methods
----------------------------------

* FKM Goodman
* Five Segment Correction

'''

__author__ = "Johannes Mueller, Lena Rapp"
__maintainer__ = "Johannes Mueller"

import numpy as np
import pandas as pd

import pylife
from pylife.stress import stresssignal
from pylife.core import signal


@pd.api.extensions.register_dataframe_accessor("meanstress_mesh")
class MeanstressMesh(stresssignal.CyclicStressAccessor):

    def FKM_goodman(self, haigh, R_goal):
        haigh.FKM_Goodman
        Sa = self._obj.sigma_a.to_numpy()
        Sm = self._obj.sigma_m.to_numpy()
        Sa_transformed = FKM_goodman(Sa, Sm, haigh.M, haigh.M2, R_goal)
        return pd.DataFrame({'sigma_a': Sa_transformed, 'R': np.ones_like(Sa_transformed) * R_goal},
                            index=self._obj.index)

    def five_segment(self, haigh, R_goal):
        haigh.haigh_five_segment
        Sa = self._obj.sigma_a.to_numpy()
        Sm = self._obj.sigma_m.to_numpy()
        Sa_transformed = five_segment_correction(Sa, Sm,
                                                 haigh.M0, haigh.M1, haigh.M2, haigh.M3, haigh.M4,
                                                 haigh.R12, haigh.R23,
                                                 R_goal)
        return pd.DataFrame({'sigma_a': Sa_transformed, 'R': np.ones_like(Sa_transformed) * R_goal},
                            index=self._obj.index)

@pd.api.extensions.register_dataframe_accessor("meanstress_hist")
class MeanstressHist:

    def __init__(self, df):
        if df.index.names == ['from', 'to']:
            f = df.index.get_level_values('from').mid
            t = df.index.get_level_values('to').mid
            self._Sa = np.abs(f-t)/2.
            self._Sm = (f+t)/2.
            self._binsize_x = df.index.get_level_values('from').length.min()
            self._binsize_y = df.index.get_level_values('to').length.min()
        elif df.index.names == ['range', 'mean']:
            self._Sa = df.index.get_level_values('range').mid / 2.
            self._Sm = df.index.get_level_values('mean').mid
            self._binsize_x = df.index.get_level_values('range').length.min()
            self._binsize_y = df.index.get_level_values('mean').length.min()
        else:
            raise AttributeError("MeanstressHist needs index names either ['from', 'to'] or ['range', 'mean']")

        self._df = df

    def FKM_goodman(self, haigh, R_goal):
        haigh.FKM_Goodman
        Dsig = FKM_goodman(self._Sa, self._Sm, haigh.M, haigh.M2, R_goal) * 2.
        return self._rebin_results(Dsig)

    def five_segment(self, haigh, R_goal):
        haigh.haigh_five_segment
        Dsig = five_segment_correction(self._Sa, self._Sm,
                                       haigh.M0, haigh.M1, haigh.M2, haigh.M3, haigh.M4, haigh.R12,
                                       haigh.R23, R_goal) * 2.
        return self._rebin_results(Dsig)

    def _rebin_results(self, Dsig):
        Dsig_max = Dsig.max()
        binsize = np.hypot(self._binsize_x, self._binsize_y) / np.sqrt(2.)
        bincount = int(np.ceil(Dsig_max / binsize))
        new_idx = pd.IntervalIndex.from_breaks(np.linspace(0, Dsig_max, bincount), name="range")
        result = pd.DataFrame(data=np.zeros(bincount-1), index=new_idx, columns=['frequency'], dtype=np.int32)
        for i, intv in enumerate(new_idx):
            cond = np.logical_and(Dsig >= intv.left, Dsig < intv.right)
            result.loc[intv, 'frequency'] = np.int32(np.sum(self._df.values[cond]))
        result['frequency'].iloc[-1] += np.int32(np.sum(self._df.values[Dsig == Dsig_max]))
        return result

@pd.api.extensions.register_dataframe_accessor("FKM_Goodman")
@pd.api.extensions.register_series_accessor("FKM_Goodman")
class FKMGoodman:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    def _validate(self, obj):
        signal.DataValidator().fail_if_key_missing(obj, ['M', 'M2'])


@pd.api.extensions.register_dataframe_accessor("haigh_five_segment")
@pd.api.extensions.register_series_accessor("haigh_five_segment")
class FiveSegment:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    def _validate(self, obj):
        signal.DataValidator().fail_if_key_missing(obj, ['M0', 'M1', 'M2', 'M3', 'M4', 'R12', 'R23'])



def FKM_goodman(Sa, Sm, M, M2, R_goal):
    ''' Performs a mean stress transformation to R_goal according to the FKM-Goodman model

    :param Sa: the stress amplitude
    :param Sm: the mean stress
    :param M: the mean stress sensitivity between R=-inf and R=0
    :param M2: the mean stress sensitivity beyond R=0
    :param R_goal: the R-value to transform to

    :returns: the transformed stress range
    '''

    if R_goal == 1:
        raise ValueError('R_goal = 1 is invalid input')

    old_err_state = np.seterr(divide='ignore')

    R = np.divide(Sm-Sa, Sm+Sa)

    ignored_states = np.seterr(**old_err_state)

    c = np.where(R <= 0.)
    c2 = np.where((R > 0.) & (R < 1.))

    M = np.broadcast_to(M, Sa.shape)
    M2 = np.broadcast_to(M2, Sa.shape)

    Ma = np.zeros_like(Sa)
    Ma[c] = M[c]
    Ma[c2] = M2[c2]

    S0 = np.zeros_like(Sa)
    Sinf = np.zeros_like(Sa)

    r1 = np.where(R < 1.)
    r2 = np.where(R > 1.)

    S0[r1] = (Sa[r1]+Sm[r1]*Ma[r1])/(1.+Ma[r1])
    Sinf[r1] = S0[r1]*(1.+M[r1])/(1.-M[r1])

    Sinf[r2] = Sa[r2]
    S0[r2] = Sinf[r2]*(1.-M[r2])/(1.+M[r2])

    if R_goal == 0.0:
        return S0

    elif R_goal == -1.:
        return S0*(1.+M)

    elif R_goal == "-inf "or R_goal > 1.:
        return Sinf

    elif R_goal < 0.0:
        Mf = M

    else:
        Mf = M2

    return S0*(1.+Mf)*(1.-Mf/(Mf+(1.-R_goal)/(1.+R_goal)))


def five_segment_correction(Sa, Sm, M0, M1, M2, M3, M4, R12, R23, R_goal):
    ''' Performs a mean stress transformation to R_goal according to the
        Five Segment Mean Stress Correction

    :param Sa: the stress amplitude
    :param Sm: the mean stress
    :param Rgoal: the R-value to transform to
    :param M: the mean stress sensitivity between R=-inf and R=0
    :param M1: the mean stress sensitivity between R=0 and R=R12
    :param M2: the mean stress sensitivity betwenn R=R12 and R=R23
    :param M3: the mean stress sensitivity between R=R23 and R=1
    :param M4: the mean stress sensitivity beyond R=1
    :param R12: R-value between M1 and M2
    :param R23: R-value between M2 and M3

    :returns: the transformed stress range
    '''

    if R_goal == 1:
        raise ValueError('R_goal = 1 is invalid input')

    old_err_state = np.seterr(divide='ignore')

    R = np.divide(Sm-Sa, Sm+Sa)

    ignored_states = np.seterr(**old_err_state)

    c4 = np.where(R > 1.)
    c0 = np.where(R <= 0.)
    c1 = np.where((R > 0.) & (R <= R12))
    c2 = np.where((R > R12) & (R <= R23))
    c3 = np.where((R > R23) & (R < 1.))

    M0 = np.broadcast_to(M0, Sa.shape)
    M1 = np.broadcast_to(M1, Sa.shape)
    M2 = np.broadcast_to(M2, Sa.shape)
    M3 = np.broadcast_to(M3, Sa.shape)
    M4 = np.broadcast_to(M4, Sa.shape)

    Ma = np.zeros_like(Sa)
    Ma[c0] = M0[c0]
    Ma[c1] = M1[c1]
    Ma[c2] = M2[c2]
    Ma[c3] = M3[c3]
    Ma[c4] = M4[c4]

    S_inf = np.zeros_like(Sa)
    S_0 = np.zeros_like(Sa)
    S_12 = np.zeros_like(Sa)
    S_23 = np.zeros_like(Sa)

    B_12 = np.broadcast_to((1.+R12)/(1.-R12), Sa.shape)
    B_23 = np.broadcast_to((1.+R23)/(1.-R23), Sa.shape)

    r4 = c4
    r = np.append(c0, c1)
    r23 = np.append(c2, c3)

    S_inf[r4] = (Sa[r4]+Sm[r4]*Ma[r4])/(1.-Ma[r4])
    S_0[r4] = S_inf[r4]*(1.-M0[r4])/(1.+M0[r4])
    S_12[r4] = S_0[r4]*(1.+M1[r4])/(1.+M1[r4]*B_12[r4])
    S_23[r4] = S_12[r4]*(1.+M2[r4]*B_12[r4])/(1.+M2[r4]*B_23[r4])

    S_0[r] = (Sa[r]+Sm[r]*Ma[r])/(1.+Ma[r])
    S_inf[r] = S_0[r]*(1.+M0[r])/(1.-M0[r])
    S_12[r] = S_0[r]*(1.+M1[r])/(1.+M1[r]*B_12[r])
    S_23[r] = S_12[r]*(1.+M2[r]*B_12[r])/(1.+M2[r]*B_23[r])

    S_23[r23] = (Sa[r23]+Sm[r23]*Ma[r23])/(1.+Ma[r23]*B_23[r23])
    S_12[r23] = S_23[r23]*(1.+M2[r23]*B_23[r23])/(1.+M2[r23]*B_12[r23])
    S_0[r23] = S_12[r23]*(1.+M1[r23]*B_12[r23])/(1.+M1[r23])
    S_inf[r23] = S_0[r23]*(1.+M0[r23])/(1.-M0[r23])

    if R_goal == 0.0:
        return S_0

    if R_goal == -1.:
        return S_0*(1.+M0)

    if R_goal == -np.inf:
        return S_inf

    if R_goal == R12:
        return S_12

    if R_goal == R23:
        return S_23

    B_goal = (1.+R_goal)/(1.-R_goal)

    if R_goal <= 0.0:
        return S_0*(1.+M0)*(1.-M0/(M0+1./B_goal))

    if R_goal > 0.0 and R_goal < R12:
        return S_0*(1.+M1)*(1.-M1/(M1+1./B_goal))

    if R_goal > R12 and R_goal < R23:
        return S_23*(1.+M2*B_23)/(1.+M2*B_goal)

    if R_goal > R23 and R_goal < 1.:
        return S_23*(1.+M3*B_23)/(1.+M3*B_goal)

    if R_goal > 1.:
        return S_inf*(1.-M4)*(1.-M4/(M4+1./B_goal))


def experimental_mean_stress_sensitivity(sn_curve_R0, sn_curve_Rn1, N_c=np.inf):
    """
    Estimate the mean stress sensitivity from two `FiniteLifeCurve` objects for the same amount of cycles `N_c`.

    The formula for calculation is taken from: "Betriebsfestigkeit", Haibach, 3. Auflage 2006

    Formula (2.1-24):

    .. math::
        M_{\sigma} = {S_a}^{R=-1}(N_c) / {S_a}^{R=0}(N_c) - 1

    Alternatively the mean stress sensitivity is calculated based on both SD_50 values
    (if N_c is not given).

    Parameters
    ----------
    sn_curve_R0: pylife.strength.sn_curve.FiniteLifeCurve
        Instance of FiniteLifeCurve for R == 0
    sn_curve_Rn1: pylife.strength.sn_curve.FiniteLifeCurve
        Instance of FiniteLifeCurve for R == -1
    N_c: float, (default=np.inf)
        Amount of cycles where the amplitudes should be compared.
        If N_c is higher than a fatigue transition point (ND_50) for the SN-Curves, SD_50 is taken.
        If N_c is None, SD_50 values are taken as stress amplitudes instead.

    Returns
    -------
    float
        Mean stress sensitivity M_sigma

    Raises
    ------
    ValueError
        If the resulting M_sigma doesn't lie in the range from 0 to 1 a ValueError is raised, as this value would
        suggest higher strength with additional loads.
    """
    S_a_R0 = sn_curve_R0.calc_S(N_c) if N_c < sn_curve_R0.ND_50 else sn_curve_R0.SD_50
    S_a_Rn1 = sn_curve_Rn1.calc_S(N_c) if N_c < sn_curve_Rn1.ND_50 else sn_curve_Rn1.SD_50
    M_sigma = S_a_Rn1 / S_a_R0 - 1
    if not 0 <= M_sigma <= 1:
        raise ValueError("M_sigma: %.2f exceeds the interval [0, 1] which is not plausible." % M_sigma)
    return M_sigma

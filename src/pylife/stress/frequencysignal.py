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

"""Handle frequency-domain stress signals for pyLife workflows.

Provide helpers for smoothing frequency-indexed power spectral densities
before spectral comparison or frequency-domain fatigue assessment.

Warnings
--------
This module is not considered finalized even though it is part of
``pylife-2.0``. Breaking changes might occur in upcoming minor releases.
"""

import numpy as np
import pandas as pd
from scipy import optimize as op

class psdSignal:
    r"""Handle routines for frequency-indexed PSD signals.

    The class stores a pandas data frame schema where the index is a frequency
    index in Hz and each column contains one power spectral density, e.g. in
    MPa²/Hz for a stress signal. The current methods are also used as legacy
    unbound helpers with a :class:`pandas.DataFrame` as ``self``.

    Parameters
    ----------
    df : pandas.DataFrame
        Frequency-domain signal with frequency index in Hz and PSD columns.

    Notes
    -----
    The RMS value follows the one-sided PSD convention

    .. math::

        x_\mathrm{rms} = \sqrt{\int_0^\infty S_{xx}(f)\,df}.

    Examples
    --------
    >>> psd = pd.DataFrame({"stress": [1.0, 1.0]}, index=[1.0, 2.0])
    >>> round(float(psdSignal.rms_psd(psd).iloc[0]), 6)
    1.0
    """
    def __init__(self,df):

        self.df = df

    def rms_psd(self):
        f  = np.logspace(np.log10(self.index.values.min()),np.log10(
                              self.index.values.max()),2048)
        psd = pd.DataFrame()
        for colact in self.columns:
            psd[colact] = np.interp(f,self.index.values,self[colact])
        psd.index = f
        return ((psd.diff()+psd).dropna()).multiply(np.diff(psd.index.values),axis = 0).sum()**0.5

    def _intMinlog(self,psdin,fsel,factor_rms_nods):
        self_rms_df = pd.DataFrame(data = 10**np.interp(psdin.index.values, fsel,np.log10(self)),
                                                    index = psdin.index.values)
        ysel =  np.interp(fsel, psdin.index.values, psdin.values.flatten())
        rms_in = psdSignal.rms_psd(psdin).values
        rms_smooth = psdSignal.rms_psd(self_rms_df).values
        eps1 = (rms_in-rms_smooth)**2/rms_in**2
        eps2 =  np.dot(np.log10(ysel/self),np.log10(ysel/self))/np.dot(np.log10(ysel),np.log10(ysel))
        return factor_rms_nods*eps1+(1-factor_rms_nods)*eps2


    def psd_smoother(self,fsel,factor_rms_nodes = 0.5):
        r"""Smooth a PSD by optimizing values at selected frequency nodes.

        Replace a dense frequency-indexed PSD by a compact node-based
        representation. The optimization balances preservation of the RMS
        value against preservation of the PSD values at the selected nodes.

        Parameters
        ----------
        fsel : array_like
            Frequency nodes in Hz used for the smoothed PSD.
        factor_rms_nodes : float, optional
            Weighting factor between node-value error and RMS error. ``0``
            considers only the error of node PSD values, while ``1`` considers
            only the RMS error. Default is ``0.5``.

        Returns
        -------
        pandas.DataFrame
            Smoothed PSD with frequency index in Hz. The index contains the
            original minimum frequency, unique selected nodes, and the
            original maximum frequency.

        Notes
        -----
        For every input column, the optimized node PSD values minimize

        .. math::

            \alpha \frac{(r_\mathrm{in} - r_\mathrm{smooth})^2}
            {r_\mathrm{in}^2}
            + (1 - \alpha)
            \frac{\lVert \log_{10}(S_\mathrm{node}/H) \rVert^2}
            {\lVert \log_{10}(S_\mathrm{node}) \rVert^2},

        where :math:`\alpha` is ``factor_rms_nodes`` and :math:`H` are the
        optimized node values.
        """

        f  = np.logspace(np.log10(self.index.values.min()),np.log10(
                              self.index.values.max()),1024)
        fsel = np.unique(fsel)
        fout = np.append(np.append(self.index.values.min(),np.unique(fsel)),self.index.values.max())
        opt_df = pd.DataFrame()
        for colact in self.columns:
            df_in = pd.DataFrame(data = np.interp(f,self.index.values,self[colact]),
                                 index = f)
            Hi0 = 10**(np.interp(fsel,f,np.log10(df_in.values.flatten())))
            lim = np.array([df_in.values.min()*np.ones_like(fsel),
                            np.array(df_in.values.max()*np.ones_like(fsel))]).T

            Hi = op.minimize(psdSignal._intMinlog,x0 = Hi0,bounds = tuple(map(tuple, lim)),
                             args=(df_in,fsel,factor_rms_nodes))
            opt_df[colact] =  10**np.interp(fout,fsel,np.log10(Hi.x))
        opt_df.index = fout
        return opt_df

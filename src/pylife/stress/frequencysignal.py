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

"""A module for frequency signal handling

Warning
-------

This module is not considered finalized even though it is part of pylife-2.0.
Breaking changes might occur in upcoming minor releases.
"""

import numpy as np
import pandas as pd
from scipy import optimize as op

class psdSignal:
    '''Handles different routines for self signals

    Remark: We are using the pandas data frame schema. The index contains the
    discrete frequency step. Every single column one self.

    Some functions of these class:

    * psd_optimizer
    * ...

    '''
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
        ''' Smoothen a PSD using nodes and a penalty factor weighting the errors
        for the RMS and for the node PSD values


        Parameters
        ----------

        self: DataFrame
            unsmoothed PSD
        fsel: list or np.array
           nodes
        factor_rms_nodes: float (0 <= factor_rms_nods <= 1)
            penalty error weighting the errors:

            * 0: only error of node PSD values is considered
            * 1: only error of the RMS is considered


        Returns
        -------
        DataFrame
        '''

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

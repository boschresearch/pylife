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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import pandas as pd
import numpy as np

from pylife import PylifeSignal


@pd.api.extensions.register_dataframe_accessor("voigt")
class StressTensorVoigt(PylifeSignal):
    '''DataFrame accessor class for Voigt noted stress tensors

    Raises
    ------
    AttributeError
        if at least one of the needed columns is missing.

    Notes
    -----
    Base class to access :class:`pandas.DataFrame` objects containing
    Voigt noted stress tensors. The stress tensor components are assumed
    to be in the columns `S11`, `S22`, `S33`, `S12`, `S13`, `S23`.

    See also
    --------
    :func:`pandas.api.extensions.register_dataframe_accessor()`

    Examples
    --------
    For an example see :class:`equistress.StressTensorEquistress`.
    '''
    def _validate(self):
        self.fail_if_key_missing(['S11', 'S22', 'S33', 'S12', 'S13', 'S23'])


@pd.api.extensions.register_dataframe_accessor("cyclic_stress")
class CyclicStress(PylifeSignal):
    '''DataFrame accessor class for cyclic stress data

    Raises
    ------
    AttributeError
        if there is no stress amplitide `sigma_a` key in the data

    Notes
    -----
    Base class to access :class:`pandas.DataFrame` objects containing
    cyclic stress data consisting of at least a stress amplitude using
    the key `sigma_a`. Optionally an value for the stress ratio `R` or
    for meanstress `sigma_m` can be supplied. If neither `R` nor
    `sigma_a` are supplied a mean stress of zero i.e. `R=-1` is assumed.

    Todo
    ----
    Handle also input data with lower and upper stress.
    '''

    def __init__(self, pandas_obj):
        self._obj = pandas_obj.copy()
        self._validate()

    def _validate(self):
        if 'sigma_a' in self._obj.columns and 'R' in self._obj.columns:
            self._obj['sigma_m'] = self._obj['sigma_a']*(1.+self._obj.R)/(1.-self._obj.R)
            self._obj.loc[self._obj['R'] == -np.inf, 'sigma_m'] = -self._obj.sigma_a
            return True
        if 'sigma_a' in self._obj.columns and 'sigma_m' in self._obj.columns:
            return True
        if 'sigma_a' in self._obj.columns:
            self._obj['sigma_m'] = np.zeros_like(self._obj['sigma_a'].to_numpy())
        else:
            self.fail_if_key_missing(['sigma_m', 'sigma_a'])
        return True

    def constant_R(self, R):
        ''' Sets `sigma_m` in a way that `R` is a constant value.

        Parameters
        ----------
        R : float
            The value for `R`.

        Returns
        -------
        the accessed DataFrame : :class:`pandas.DataFrame`
        '''
        self._obj['sigma_m'] = self._obj['sigma_a']*(1.+R)/(1.-R)
        return self._obj

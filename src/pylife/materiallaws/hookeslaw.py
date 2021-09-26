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

__author__ = 'Alexander Maier'
__maintainer__ = __author__

class HookesLaw1D:
    '''
    Implementation of the Hooke's-Law in 1D

    Parameters
    ----------

    E : float
        Young's Modulus
    '''
    def __init__(self, E):
        self._E = E

    @property
    def E(self):
        '''Get Young's Modulus'''
        return self._E

    def stress(self, strain):
        '''Calculate the stress for a given strain

        Parameters
        ----------
        srain : array-like float
            The strain
        
        Returns
        -------
        stress : array-like float
            the resulting stress
        '''
        return self._E * strain

    def strain(self, stress):
        '''Calculate the strain for a given stress

        Parameters
        ----------
        stress : array-like float
            The stress
        
        Returns
        -------
        strain : array-like float
            the resulting strain
        '''
        return stress / self._E

    def __eq__(self, other):
        if (isinstance(other, HookesLaw1D)):
            return self._E == other._E
        return False

    def __ne__(self, other):
        return not self == other

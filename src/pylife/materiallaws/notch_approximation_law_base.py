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

__author__ = ["Benjamin Maier"]
__maintainer__ = __author__

import pylife.materiallaws.rambgood

class NotchApproximationLawBase:
    """This is a base class for any notch approximation law, e.g., the extended Neuber and the Seeger-Beste laws.

    It initializes the internal variables used by the derived classes and provides getters and setters.
    """

    def __init__(self, E, K, n, K_p=None):
        self._E = E
        self._K = K
        self._n = n
        self._K_p = K_p

        self._ramberg_osgood_relation = pylife.materiallaws.rambgood.RambergOsgood(E, K, n)

    @property
    def E(self):
        '''Get Young's Modulus'''
        return self._E

    @property
    def K(self):
        '''Get the strength coefficient'''
        return self._K

    @property
    def n(self):
        '''Get the strain hardening coefficient'''
        return self._n

    @property
    def K_p(self):
        '''Get the shape factor (de: Traglastformzahl)'''
        return self._K_p

    @property
    def ramberg_osgood_relation(self):
        '''Get the ramberg osgood relation object
        '''
        return self._ramberg_osgood_relation

    @K_p.setter
    def K_p(self, value):
        """Set the shape factor value K_p  (de: Traglastformzahl)"""
        self._K_p = value

    @K.setter
    def K_prime(self, value):
        """Set the strain hardening coefficient"""
        self._K = value
        self._ramberg_osgood_relation = pylife.materiallaws.rambgood.RambergOsgood(self._E, self._K, self._n)

    @K.setter
    def K(self, value):
        """Set the strain hardening coefficient"""
        self.K_prime = value


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

__author__ = ["Johannes Mueller", 'Alexander Maier']
__maintainer__ = __author__

import numpy as np
from scipy import optimize


class RambergOsgood:
    '''Simple implementation of the Ramberg-Osgood relation

    Parameters
    ----------
    E : float
        Young's Modulus
    K : float
        The strength coefficient, usually named ``K'`` or ``K_prime`` in FKM nonlinear related formulas.
    n : float
        The strain hardening coefficient, usually named ``n'`` or ``n_prime`` in FKM nonlinear related formulas.

    Notes
    -----
    The equation implemented is the one that `Wikipedia
    <https://en.wikipedia.org/wiki/Ramberg%E2%80%93Osgood_relationship#Alternative_Formulations>`__
    refers to as "Alternative Formulation". The parameters `n` and `k` in this
    are formulation are the Hollomon parameters.

    '''

    def __init__(self, E, K, n):
        self._E = E
        self._K = K
        self._n = n

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

    def strain(self, stress):
        '''Calculate the elastic plastic strain for a given stress

        Parameters
        ----------
        stress : array-like float
            The stress

        Returns
        -------
        strain : array-like float
            The resulting strain

        '''
        stress = np.asarray(stress)
        return self.elastic_strain(stress) + self.plastic_strain(stress)

    def elastic_strain(self, stress):
        '''Calculate the elastic strain for a given stress

        Parameters
        ----------
        stress : array-like float
            The stress

        Returns
        -------
        strain : array-like float
            The resulting elastic strain
        '''
        return stress/self._E

    def plastic_strain(self, stress):
        '''Calculate the plastic strain for a given stress

        Parameters
        ----------
        stress : array-like float
            The stress

        Returns
        -------
        strain : array-like float
            The resulting plastic strain
        '''
        absstress, signstress = self._get_abs_sign(stress)
        return signstress * np.power(absstress/self._K, 1./self._n)

    def _get_abs_sign(self, x):
        '''Calculate the absolute value and the sign for a given input

        Parameters
        ----------
        x : array-like float
            The input

        Returns
        -------
        abs_x : array-like float
            The resulting absolute value
        sign_x : array-like float
            The resulting sign of the input
        '''
        abs_x = np.fabs(x)
        sign_x = np.sign(x)
        return abs_x, sign_x

    def stress(self, strain, *, rtol=1e-5, tol=1e-6):
        '''Calculate the stress for a given strain

        Parameters
        ----------
        strain : array-like float
            The strain

        Returns
        -------
        stress : array-like float
            The resulting stress
        '''

        def residuum(stress):
            return self.strain(stress) - abs_strain

        def dresiduum(stress):
            return self.tangential_compliance(stress)

        strain = np.asarray(strain)
        abs_strain, sign_strain = self._get_abs_sign(strain)
        stress0 = self._E * abs_strain
        abs_stress = optimize.newton(
            func=residuum,
            x0=stress0,
            fprime=dresiduum,
            rtol=rtol, tol=tol
        )
        return abs_stress * sign_strain

    def tangential_compliance(self, stress):
        '''Calculate the derivative of the strain with respect to the stress for a given stress

        Parameters
        ----------
        stress : array-like float
            The stress

        Returns
        -------
        dstrain : array-like float
            The resulting derivative
        '''
        stress = np.abs(stress)
        return 1./self._E + 1./(self._n*self._K) * np.power(stress/self._K, 1./self._n - 1)

    def tangential_modulus(self, stress):
        '''Calculate the derivative of the stress with respect to the strain for a given stress

        Parameters
        ----------
        stress : array-like float
            The stress

        Returns
        -------
        dstress : array-like float
            The resulting derivative
        '''
        return 1. / self.tangential_compliance(stress)

    def delta_strain(self, delta_stress):
        '''Calculate the cyclic Masing strain span for a given stress span

        Parameters
        ----------
        delta_stress : array-like float
            The stress span

        Returns
        -------
        delta_strain : array-like float
            The corresponding strain span

        Notes
        -----
        A Masing like behavior is assumed for the material as described in
        `Kerbgrundkonzept <https://de.wikipedia.org/wiki/Kerbgrundkonzept#Masing-Verhalten_und_Werkstoffged%C3%A4chtnis>`__.
        '''
        return 2*self.strain(stress=delta_stress/2.)

    def delta_stress(self, delta_strain):
        '''Calculate the cyclic Masing stress span for a given strain span

        Parameters
        ----------
        delta_strain : array-like float
            The strain span

        Returns
        -------
        delta_stress : array-like float
            The corresponding stress span

        Notes
        -----
        A Masing like behavior is assumed for the material as described in
        `Kerbgrundkonzept <https://de.wikipedia.org/wiki/Kerbgrundkonzept#Masing-Verhalten_und_Werkstoffged%C3%A4chtnis>`__.
        '''
        return 2*self.stress(strain=delta_strain/2.)

    def lower_hysteresis(self, stress, max_stress):
        '''Calculate the lower (relaxation to compression) hysteresis starting from a given maximum stress

        Parameters
        ----------
        stress : array-like float
            The stress (must be below the maximum stress)
        max_stress : float
            The maximum stress of the hysteresis look

        Returns
        -------
        lower_hysteresis : array-like float
            The lower hysteresis branch from `max_stress` all the way to `stress`

        Raises
        ------
        ValueError
            if stress > max_stress
        '''
        stress = np.asarray(stress)
        if (stress > max_stress).any():
            raise ValueError("Value for 'stress' must not be higher than 'max_stress'.")
        return self.strain(max_stress) - self.delta_strain(max_stress-stress)

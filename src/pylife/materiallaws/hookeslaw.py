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

__author__ = 'Alexander Maier'
__maintainer__ = __author__

import numpy as np


class _Hookeslawcore:
    '''Parent class for the multidimensional Hooke's Law implementation. Defines the properties and checks for correct inputs'''

    def __init__(self, E, nu):
        '''Instantiate a multidimensional Hooke's Law implementation

        Parameters
        ----------

        E : float
            Young's modulus

        nu : float
            Poisson's ratio. Must be between -1 and 1./2.
        '''
        self._validateinit(nu)
        self._E = E
        self._nu = nu
        self._G = E / (2. * (1 + nu))
        self._K = E / (3. * (1 - 2 * nu))

    def _validateinit(self, nu):
        '''Validates the input of the Poisson\'s ratio
        '''
        if nu < - 1 or nu > 1./2:
            raise ValueError('Poisson\'s ratio nu is %.2f but must be -1 <= nu <= 1./2.' % nu)

    def _as_consistant_arrays(self, *args):
        '''Transforms the inputs into numpy arrays and checks the shape of the given inputs. If the shapes are note equal, a ValueError is raised
        '''
        transformed = tuple(np.asarray(arg) for arg in args)
        shape0 = transformed[0].shape
        shape = [shape0 == arg.shape for arg in transformed]
        if not all(shape):
            raise ValueError('Components\' shape is not consistent.')

        return transformed

    @property
    def E(self):
        '''Get Young's modulus'''
        return self._E

    @property
    def nu(self):
        '''Get Poisson's ratio'''
        return self._nu

    @property
    def G(self):
        '''Get the sheer modulus'''
        return self._G

    @property
    def K(self):
        '''Get the bulk modulus'''
        return self._K


class HookesLaw1d:
    '''Implementation of the one dimensional Hooke's Law

    Parameters
    ----------

    E : float
        Young's modulus
    '''

    def __init__(self, E):
        '''
        Instantiate a one dimensional Hooke's Law implementation with a given Young's modulus

        Parameters
        ----------

        E : float
            Young's modulus
        '''
        self._E = E

    @property
    def E(self):
        '''Get Young's modulus'''
        return self._E

    def stress(self, strain):
        '''Get the stress for a given elastic strain

        Parameters
        ----------
        strain : array-like float
            The elastic strain

        Returns
        -------
        strain : array-like float
            The resulting stress
        '''
        return np.asarray(strain) * self._E

    def strain(self, stress):
        '''Get the elastic strain for a given stress

        Parameters
        ----------
        stress : array-like float
            The stress

        Returns
        -------
        strain : array-like float
            The resulting elastic strain
        '''
        return np.asarray(stress) / self._E


class HookesLaw2dPlaneStress(_Hookeslawcore):
    '''Implementation of the Hooke's Law under plane stress conditions.

    Parameters
    ----------

    E : float
        Young's modulus

    nu : float
        Poisson's ratio. Must be between -1 and 1./2.

    Notes
    -----

    A cartesian coordinate system is assumed. The stress components in 3 direction are assumed to be zero, s33 = s13 = s23 = 0.'''

    def __init__(self, E, nu):
        super().__init__(E, nu)
        self._Et = E
        self._nut = self._nu

    def strain(self, s11, s22, s12):
        '''Get the elastic strain components for given stress components

        Parameters
        ----------
        s11 : array-like float
            The normal stress component with basis 1-1
        s22 : array-like float
            The normal stress component with basis 2-2
        s12 : array-like float
            The shear stress component with basis 1-2


        Returns
        -------
        e11 : array-like float
            The resulting elastic normal strain component with basis 1-1
        e22 : array-like float
            The resulting elastic normal strain component with basis 2-2
        e33 : array-like float
            The resulting elastic normal strain component with basis 3-3
        g12 : array-like float
            The resulting elastic engineering shear strain component with basis 1-2,
            (1. / 2 * g12 is the tensor component)
        '''
        s11, s22, s12 = self._as_consistant_arrays(s11, s22, s12)
        e11 = 1. / self._Et * (s11 - self._nut * s22)
        e22 = 1. / self._Et * (s22 - self._nut * s11)
        e33 = - self._nu / self._E * (s11 + s22)
        g12 = 1. / self._G * s12
        return e11, e22, e33, g12

    def stress(self, e11, e22, g12):
        '''Get the stress components for given elastic strain components

        Parameters
        ----------
        e11 : array-like float
            The elastic normal strain component with basis 1-1
        e22 : array-like float
            The elastic normal strain component with basis 2-2
        g12 : array-like float
            The elastic engineering shear strain component with basis 1-2,
            (1. / 2 * g12 is the tensor component)

        Returns
        -------
        s11 : array-like float
            The resulting normal stress component with basis 1-1
        s22 : array-like float
            The resulting normal stress component with basis 2-2
        s12 : array-like float
            The resulting shear stress component with basis 1-2
        '''
        e11, e22, g12 = self._as_consistant_arrays(e11, e22, g12)
        factor = self._Et / (1 - np.power(self._nut, 2.))
        s11 = factor * (e11 + self._nut * e22)
        s22 = factor * (e22 + self._nut * e11)
        s12 = self._G * g12
        return s11, s22, s12


class HookesLaw2dPlaneStrain(HookesLaw2dPlaneStress):
    '''Implementation of the Hooke's Law under plane strain conditions.

    Parameters
    ----------

    E : float
        Young's modulus

    nu : float
        Poisson's ratio. Must be between -1 and 1./2.

    Notes
    -----

    A cartesian coordinate system is assumed. The strain components in 3 direction are assumed to be zero, e33 = g13 = g23 = 0.
    '''

    def __init__(self, E, nu):
        super().__init__(E, nu)
        self._Et = self._E / (1 - np.power(self._nu, 2))
        self._nut = self._nu / (1 - self._nu)

    def strain(self, s11, s22, s12):
        '''Get the elastic strain components for given stress components

        Parameters
        ----------
        s11 : array-like float
            The normal stress component with basis 1-1
        s22 : array-like float
            The normal stress component with basis 2-2
        s12 : array-like float
            The shear stress component with basis 1-2


        Returns
        -------
        e11 : array-like float
            The resulting elastic normal strain component with basis 1-1
        e22 : array-like float
            The resulting elastic normal strain component with basis 2-2
        g12 : array-like float
            The resulting elastic engineering shear strain component with basis 1-2,
            (1. / 2 * g12 is the tensor component)
        '''
        e11, e22, _, g12 = super().strain(s11, s22, s12)
        return e11, e22, g12

    def stress(self, e11, e22, g12):
        '''Get the stress components for given elastic strain components

        Parameters
        ----------
        e11 : array-like float
            The elastic normal strain component with basis 1-1
        e22 : array-like float
            The elastic normal strain component with basis 2-2
        g12 : array-like float
            The elastic engineering shear strain component with basis 1-2,
            (1. / 2 * g12 is the tensor component)

        Returns
        -------
        s11 : array-like float
            The resulting normal stress component with basis 1-1
        s22 : array-like float
            The resulting normal stress component with basis 2-2
        s33 : array-like float
            The resulting normal stress component with basis 3-3
        s12 : array-like float
            The resulting shear stress component with basis 1-2
        '''
        s11, s22, s12 = super().stress(e11, e22, g12)
        s33 = self.nu * (s11 + s22)
        return s11, s22, s33, s12


class HookesLaw3d(_Hookeslawcore):
    '''Implementation of the Hooke's Law in three dimensions.

    Parameters
    ----------

    E : float
        Young's modulus

    nu : float
        Poisson's ratio. Must be between -1 and 1./2

    Notes
    -----

    A cartesian coordinate system is assumed.
    '''

    def __init__(self, E, nu):
        super().__init__(E, nu)

    def strain(self, s11, s22, s33, s12, s13, s23):
        '''Get the elastic strain components for given stress components

        Parameters
        ----------
        s11 : array-like float
            The resulting normal stress component with basis 1-1
        s22 : array-like float
            The resulting normal stress component with basis 2-2
        s33 : array-like float
            The resulting normal stress component with basis 3-3
        s12 : array-like float
            The resulting shear stress component with basis 1-2
        s13 : array-like float
            The resulting shear stress component with basis 1-3
        s23 : array-like float
            The resulting shear stress component with basis 2-3

        Returns
        -------
        e11 : array-like float
            The resulting elastic normal strain component with basis 1-1
        e22 : array-like float
            The resulting elastic normal strain component with basis 2-2
        e33 : array-like float
            The resulting elastic normal strain component with basis 3-3
        g12 : array-like float
            The resulting elastic engineering shear strain component with basis 1-2,
            (1. / 2 * g12 is the tensor component)
        g13 : array-like float
            The resulting elastic engineering shear strain component with basis 1-3,
            (1. / 2 * g13 is the tensor component)
        g23 : array-like float
            The resulting elastic engineering shear strain component with basis 2-3,
            (1. / 2 * g23 is the tensor component)
        '''
        s11, s22, s33, s12, s13, s23 = self._as_consistant_arrays(s11, s22, s33, s12, s13, s23)
        e11 = 1 / self._E * (s11 - self._nu * (s22 + s33))
        e22 = 1 / self._E * (s22 - self._nu * (s11 + s33))
        e33 = 1 / self._E * (s33 - self._nu * (s11 + s22))
        g12 = s12 / self._G
        g13 = s13 / self._G
        g23 = s23 / self._G
        return e11, e22, e33, g12, g13, g23

    def stress(self, e11, e22, e33, g12, g13, g23):
        '''Get the stress components for given elastic strain components

        Parameters
        ----------
        e11 : array-like float
            The elastic normal strain component with basis 1-1
        e22 : array-like float
            The elastic normal strain component with basis 2-2
        e33 : array-like float
            The elastic normal strain component with basis 3-3
        g12 : array-like float
            The elastic engineering shear strain component with basis 1-2,
            (1. / 2 * g12 is the tensor component)
        g13 : array-like float
            The elastic engineering shear strain component with basis 1-3,
            (1. / 2 * g13 is the tensor component)
        g23 : array-like float
            The elastic engineering shear strain component with basis 2-3,
            (1. / 2 * g23 is the tensor component)

        Returns
        -------
        s11 : array-like float
            The resulting normal stress component with basis 1-1
        s22 : array-like float
            The resulting normal stress component with basis 2-2
        s33 : array-like float
            The resulting normal stress component with basis 3-3
        s12 : array-like float
            The resulting shear stress component with basis 1-2
        s13 : array-like float
            The resulting shear stress component with basis 1-3
        s23 : array-like float
            The resulting shear stress component with basis 2-3
        '''
        e11, e22, e33, g12, g13, g23 = self._as_consistant_arrays(e11, e22, e33, g12, g13, g23)
        factor1 = self._E / ((1 + self._nu) * (1 - 2 * self._nu))
        factor2 = 1 - self._nu
        s11 = factor1 * (factor2 * e11 + self._nu * (e22 + e33))
        s22 = factor1 * (factor2 * e22 + self._nu * (e11 + e33))
        s33 = factor1 * (factor2 * e33 + self._nu * (e11 + e22))
        s12 = self._G * g12
        s13 = self._G * g13
        s23 = self._G * g23
        return s11, s22, s33, s12, s13, s23

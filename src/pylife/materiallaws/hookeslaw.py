# Copyright (c) 2019-2026 - for information on the respective copyright owner
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

r"""Provide isotropic linear elastic Hooke law relations.

The module converts stresses and elastic strains for one-dimensional,
plane-stress, plane-strain, and three-dimensional material states. Stresses
and moduli use the same unit, typically MPa, while strains are dimensionless.
"""

__author__ = 'Alexander Maier'
__maintainer__ = __author__

import numpy as np


class _Hookeslawcore:
    """Provide shared constants and validation for multidimensional Hooke laws.

    Parameters
    ----------
    E : float
        Young's modulus in MPa.
    nu : float
        Poisson's ratio, dimensionless. Must satisfy ``-1 <= nu <= 0.5``.
    """

    def __init__(self, E, nu):
        """Initialize isotropic elastic constants.

        Parameters
        ----------
        E : float
            Young's modulus in MPa.
        nu : float
            Poisson's ratio, dimensionless. Must satisfy ``-1 <= nu <= 0.5``.
        """
        self._validateinit(nu)
        self._E = E
        self._nu = nu
        self._G = E / (2. * (1 + nu))
        self._K = E / (3. * (1 - 2 * nu))

    def _validateinit(self, nu):
        """Validate Poisson's ratio.

        Parameters
        ----------
        nu : float
            Poisson's ratio, dimensionless.

        Raises
        ------
        ValueError
            Raised if ``nu`` is outside ``[-1, 0.5]``.
        """
        if nu < - 1 or nu > 1./2:
            raise ValueError('Poisson\'s ratio nu is %.2f but must be -1 <= nu <= 1./2.' % nu)

    def _as_consistant_arrays(self, *args):
        """Convert component arrays and require equal shapes.

        Parameters
        ----------
        *args : array_like
            Component arrays to convert to :class:`numpy.ndarray`.

        Returns
        -------
        tuple of numpy.ndarray
            Converted arrays with identical shapes.

        Raises
        ------
        ValueError
            Raised if the component arrays do not have identical shapes.
        """
        transformed = tuple(np.asarray(arg) for arg in args)
        shape0 = transformed[0].shape
        shape = [shape0 == arg.shape for arg in transformed]
        if not all(shape):
            raise ValueError('Components\' shape is not consistent.')

        return transformed

    @property
    def E(self):
        """Return Young's modulus.

        Returns
        -------
        float
            Young's modulus in MPa.
        """
        return self._E

    @property
    def nu(self):
        """Return Poisson's ratio.

        Returns
        -------
        float
            Poisson's ratio, dimensionless.
        """
        return self._nu

    @property
    def G(self):
        """Return the shear modulus.

        Returns
        -------
        float
            Shear modulus in MPa.
        """
        return self._G

    @property
    def K(self):
        """Return the bulk modulus.

        Returns
        -------
        float
            Bulk modulus in MPa.
        """
        return self._K


class HookesLaw1d:
    r"""Apply one-dimensional linear elastic Hooke's law.

    Parameters
    ----------
    E : float
        Young's modulus in MPa.

    Notes
    -----
    The implemented relation is

    .. math::

        \sigma = E \, \varepsilon .

    Examples
    --------
    >>> from pylife.materiallaws import HookesLaw1d
    >>> law = HookesLaw1d(210000.0)
    >>> float(law.stress(0.001))
    210.0
    >>> float(law.strain(210.0))
    0.001
    """

    def __init__(self, E):
        """Initialize one-dimensional Hooke's law.

        Parameters
        ----------
        E : float
            Young's modulus in MPa.
        """
        self._E = E

    @property
    def E(self):
        """Return Young's modulus.

        Returns
        -------
        float
            Young's modulus in MPa.
        """
        return self._E

    def stress(self, strain):
        """Calculate uniaxial stress from elastic strain.

        Parameters
        ----------
        strain : array_like
            Elastic normal strain, dimensionless.

        Returns
        -------
        numpy.ndarray
            Stress in MPa.
        """
        return np.asarray(strain) * self._E

    def strain(self, stress):
        """Calculate uniaxial elastic strain from stress.

        Parameters
        ----------
        stress : array_like
            Stress in MPa.

        Returns
        -------
        numpy.ndarray
            Elastic normal strain, dimensionless.
        """
        return np.asarray(stress) / self._E


class HookesLaw2dPlaneStress(_Hookeslawcore):
    """Apply isotropic Hooke's law for plane stress.

    Parameters
    ----------
    E : float
        Young's modulus in MPa.
    nu : float
        Poisson's ratio, dimensionless. Must satisfy ``-1 <= nu <= 0.5``.

    Notes
    -----
    The out-of-plane stress components are ``s33 = s13 = s23 = 0``.
    """

    def __init__(self, E, nu):
        super().__init__(E, nu)
        self._Et = E
        self._nut = self._nu

    def strain(self, s11, s22, s12):
        """Calculate elastic strain components for plane stress.

        Parameters
        ----------
        s11 : array_like
            Normal stress component in 1-direction in MPa.
        s22 : array_like
            Normal stress component in 2-direction in MPa.
        s12 : array_like
            Engineering shear stress component in the 1-2 plane in MPa.

        Returns
        -------
        e11 : numpy.ndarray
            Elastic normal strain component in 1-direction, dimensionless.
        e22 : numpy.ndarray
            Elastic normal strain component in 2-direction, dimensionless.
        e33 : numpy.ndarray
            Elastic normal strain component in 3-direction, dimensionless.
        g12 : numpy.ndarray
            Elastic engineering shear strain component in the 1-2 plane,
            dimensionless. The tensor shear strain is ``0.5 * g12``.
        """
        s11, s22, s12 = self._as_consistant_arrays(s11, s22, s12)
        e11 = 1. / self._Et * (s11 - self._nut * s22)
        e22 = 1. / self._Et * (s22 - self._nut * s11)
        e33 = - self._nu / self._E * (s11 + s22)
        g12 = 1. / self._G * s12
        return e11, e22, e33, g12

    def stress(self, e11, e22, g12):
        """Calculate stress components for plane stress.

        Parameters
        ----------
        e11 : array_like
            Elastic normal strain component in 1-direction, dimensionless.
        e22 : array_like
            Elastic normal strain component in 2-direction, dimensionless.
        g12 : array_like
            Elastic engineering shear strain component in the 1-2 plane,
            dimensionless.

        Returns
        -------
        s11 : numpy.ndarray
            Normal stress component in 1-direction in MPa.
        s22 : numpy.ndarray
            Normal stress component in 2-direction in MPa.
        s12 : numpy.ndarray
            Engineering shear stress component in the 1-2 plane in MPa.
        """
        e11, e22, g12 = self._as_consistant_arrays(e11, e22, g12)
        factor = self._Et / (1 - np.power(self._nut, 2.))
        s11 = factor * (e11 + self._nut * e22)
        s22 = factor * (e22 + self._nut * e11)
        s12 = self._G * g12
        return s11, s22, s12


class HookesLaw2dPlaneStrain(HookesLaw2dPlaneStress):
    """Apply isotropic Hooke's law for plane strain.

    Parameters
    ----------
    E : float
        Young's modulus in MPa.
    nu : float
        Poisson's ratio, dimensionless. Must satisfy ``-1 <= nu <= 0.5``.

    Notes
    -----
    The out-of-plane strain components are ``e33 = g13 = g23 = 0``.
    """

    def __init__(self, E, nu):
        super().__init__(E, nu)
        self._Et = self._E / (1 - np.power(self._nu, 2))
        self._nut = self._nu / (1 - self._nu)

    def strain(self, s11, s22, s12):
        """Calculate elastic strain components for plane strain.

        Parameters
        ----------
        s11 : array_like
            Normal stress component in 1-direction in MPa.
        s22 : array_like
            Normal stress component in 2-direction in MPa.
        s12 : array_like
            Engineering shear stress component in the 1-2 plane in MPa.

        Returns
        -------
        e11 : numpy.ndarray
            Elastic normal strain component in 1-direction, dimensionless.
        e22 : numpy.ndarray
            Elastic normal strain component in 2-direction, dimensionless.
        g12 : numpy.ndarray
            Elastic engineering shear strain component in the 1-2 plane,
            dimensionless. The tensor shear strain is ``0.5 * g12``.
        """
        e11, e22, _, g12 = super().strain(s11, s22, s12)
        return e11, e22, g12

    def stress(self, e11, e22, g12):
        """Calculate stress components for plane strain.

        Parameters
        ----------
        e11 : array_like
            Elastic normal strain component in 1-direction, dimensionless.
        e22 : array_like
            Elastic normal strain component in 2-direction, dimensionless.
        g12 : array_like
            Elastic engineering shear strain component in the 1-2 plane,
            dimensionless.

        Returns
        -------
        s11 : numpy.ndarray
            Normal stress component in 1-direction in MPa.
        s22 : numpy.ndarray
            Normal stress component in 2-direction in MPa.
        s33 : numpy.ndarray
            Normal stress component in 3-direction in MPa.
        s12 : numpy.ndarray
            Engineering shear stress component in the 1-2 plane in MPa.
        """
        s11, s22, s12 = super().stress(e11, e22, g12)
        s33 = self.nu * (s11 + s22)
        return s11, s22, s33, s12


class HookesLaw3d(_Hookeslawcore):
    """Apply isotropic Hooke's law in three dimensions.

    Parameters
    ----------
    E : float
        Young's modulus in MPa.
    nu : float
        Poisson's ratio, dimensionless. Must satisfy ``-1 <= nu <= 0.5``.

    Notes
    -----
    Engineering shear strains ``g12``, ``g13``, and ``g23`` are twice the
    corresponding tensor shear strains.
    """

    def __init__(self, E, nu):
        super().__init__(E, nu)

    def strain(self, s11, s22, s33, s12, s13, s23):
        """Calculate three-dimensional elastic strain components.

        Parameters
        ----------
        s11 : array_like
            Normal stress component in 1-direction in MPa.
        s22 : array_like
            Normal stress component in 2-direction in MPa.
        s33 : array_like
            Normal stress component in 3-direction in MPa.
        s12 : array_like
            Engineering shear stress component in the 1-2 plane in MPa.
        s13 : array_like
            Engineering shear stress component in the 1-3 plane in MPa.
        s23 : array_like
            Engineering shear stress component in the 2-3 plane in MPa.

        Returns
        -------
        e11 : numpy.ndarray
            Elastic normal strain component in 1-direction, dimensionless.
        e22 : numpy.ndarray
            Elastic normal strain component in 2-direction, dimensionless.
        e33 : numpy.ndarray
            Elastic normal strain component in 3-direction, dimensionless.
        g12 : numpy.ndarray
            Elastic engineering shear strain component in the 1-2 plane, dimensionless.
        g13 : numpy.ndarray
            Elastic engineering shear strain component in the 1-3 plane, dimensionless.
        g23 : numpy.ndarray
            Elastic engineering shear strain component in the 2-3 plane, dimensionless.
        """
        s11, s22, s33, s12, s13, s23 = self._as_consistant_arrays(s11, s22, s33, s12, s13, s23)
        e11 = 1 / self._E * (s11 - self._nu * (s22 + s33))
        e22 = 1 / self._E * (s22 - self._nu * (s11 + s33))
        e33 = 1 / self._E * (s33 - self._nu * (s11 + s22))
        g12 = s12 / self._G
        g13 = s13 / self._G
        g23 = s23 / self._G
        return e11, e22, e33, g12, g13, g23

    def stress(self, e11, e22, e33, g12, g13, g23):
        """Calculate three-dimensional stress components.

        Parameters
        ----------
        e11 : array_like
            Elastic normal strain component in 1-direction, dimensionless.
        e22 : array_like
            Elastic normal strain component in 2-direction, dimensionless.
        e33 : array_like
            Elastic normal strain component in 3-direction, dimensionless.
        g12 : array_like
            Elastic engineering shear strain component in the 1-2 plane, dimensionless.
        g13 : array_like
            Elastic engineering shear strain component in the 1-3 plane, dimensionless.
        g23 : array_like
            Elastic engineering shear strain component in the 2-3 plane, dimensionless.

        Returns
        -------
        s11 : numpy.ndarray
            Normal stress component in 1-direction in MPa.
        s22 : numpy.ndarray
            Normal stress component in 2-direction in MPa.
        s33 : numpy.ndarray
            Normal stress component in 3-direction in MPa.
        s12 : numpy.ndarray
            Engineering shear stress component in the 1-2 plane in MPa.
        s13 : numpy.ndarray
            Engineering shear stress component in the 1-3 plane in MPa.
        s23 : numpy.ndarray
            Engineering shear stress component in the 2-3 plane in MPa.
        """
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

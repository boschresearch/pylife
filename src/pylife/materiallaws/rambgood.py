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

"""Provide the Ramberg-Osgood stress-strain relation.

The module implements monotonic and cyclic Masing evaluations of the
Ramberg-Osgood relation for elastic-plastic material behavior.
"""

__author__ = ["Johannes Mueller", 'Alexander Maier']
__maintainer__ = __author__

import numpy as np
from scipy import optimize


class RambergOsgood:
    r"""Represent an elastic-plastic Ramberg-Osgood material law.

    The material law describes total strain as the sum of an elastic part and
    a plastic power-law part. Stresses and the Young's modulus must use the
    same stress unit, typically MPa.

    Parameters
    ----------
    E : float
        Young's modulus in MPa or another consistent stress unit.
    K : float
        Cyclic strength coefficient in the same stress unit as ``E``. This
        value is often written as ``K'`` or ``K_prime`` in FKM nonlinear
        formulas.
    n : float
        Cyclic strain hardening exponent, dimensionless. This value is often
        written as ``n'`` or ``n_prime`` in FKM nonlinear formulas.

    Notes
    -----
    The implemented relation follows the alternative Ramberg-Osgood form with
    Hollomon parameters:

    .. math::

        \varepsilon = \frac{\sigma}{E}
        + \operatorname{sign}(\sigma)
        \left|\frac{\sigma}{K}\right|^{1/n}
    """

    def __init__(self, E, K, n):
        self._E = E
        self._K = K
        self._n = n

    @property
    def E(self):
        """Return Young's modulus.
        """
        return self._E

    @property
    def K(self):
        """Return the cyclic strength coefficient.
        """
        return self._K

    @property
    def n(self):
        """Return the cyclic strain hardening exponent.
        """
        return self._n

    def strain(self, stress):
        """Calculate elastic-plastic strain for a stress.

        Parameters
        ----------
        stress : array_like
            Stress in MPa or another unit consistent with ``E`` and ``K``.

        Returns
        -------
        numpy.ndarray
            Total elastic-plastic strain, dimensionless.

        Examples
        --------
        >>> from pylife.materiallaws import RambergOsgood
        >>> rg = RambergOsgood(210000.0, 1000.0, 0.2)
        >>> round(float(rg.strain(300.0)), 6)
        0.003859
        """
        stress = np.asarray(stress)
        return self.elastic_strain(stress) + self.plastic_strain(stress)

    def elastic_strain(self, stress):
        """Calculate elastic strain for a stress.

        Parameters
        ----------
        stress : array_like
            Stress in MPa or another unit consistent with ``E``.

        Returns
        -------
        array_like
            Elastic strain, dimensionless.
        """
        return stress/self._E

    def plastic_strain(self, stress):
        """Calculate plastic strain for a stress.

        Parameters
        ----------
        stress : array_like
            Stress in MPa or another unit consistent with ``K``.

        Returns
        -------
        array_like
            Plastic strain, dimensionless.
        """
        absstress, signstress = self._get_abs_sign(stress)
        return signstress * np.power(absstress/self._K, 1./self._n)

    def _get_abs_sign(self, x):
        """Calculate absolute values and signs of an input.

        Parameters
        ----------
        x : array_like
            Input values.

        Returns
        -------
        abs_x : array_like
            Absolute values of ``x``.
        sign_x : array_like
            Signs of ``x``.
        """
        abs_x = np.fabs(x)
        sign_x = np.sign(x)
        return abs_x, sign_x

    def stress(self, strain, *, rtol=1e-5, tol=1e-6):
        """Calculate stress for an elastic-plastic strain.

        Parameters
        ----------
        strain : array_like
            Total elastic-plastic strain, dimensionless.
        rtol : float, optional
            Relative tolerance passed to :func:`scipy.optimize.newton`.
            Default is ``1e-5``.
        tol : float, optional
            Absolute tolerance passed to :func:`scipy.optimize.newton`.
            Default is ``1e-6``.

        Returns
        -------
        numpy.ndarray
            Stress in MPa or another unit consistent with ``E`` and ``K``.
        """

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
        """Calculate tangential compliance for a stress.

        Parameters
        ----------
        stress : array_like
            Stress in MPa or another unit consistent with ``E`` and ``K``.

        Returns
        -------
        array_like
            Derivative of strain with respect to stress, in reciprocal stress
            units.
        """
        stress = np.abs(stress)
        return 1./self._E + 1./(self._n*self._K) * np.power(stress/self._K, 1./self._n - 1)

    def tangential_modulus(self, stress):
        """Calculate tangential modulus for a stress.

        Parameters
        ----------
        stress : array_like
            Stress in MPa or another unit consistent with ``E`` and ``K``.

        Returns
        -------
        array_like
            Derivative of stress with respect to strain, in stress units.
        """
        return 1. / self.tangential_compliance(stress)

    def delta_strain(self, delta_stress):
        """Calculate cyclic Masing strain span for a stress span.

        Parameters
        ----------
        delta_stress : array_like
            Stress span in MPa or another unit consistent with ``E`` and ``K``.

        Returns
        -------
        numpy.ndarray
            Strain span, dimensionless.

        Notes
        -----
        The calculation assumes Masing material behavior as used in the notch
        strain concept (``Kerbgrundkonzept``). It evaluates twice the monotonic
        strain at half the stress span.
        """
        return 2*self.strain(stress=delta_stress/2.)

    def delta_stress(self, delta_strain):
        """Calculate cyclic Masing stress span for a strain span.

        Parameters
        ----------
        delta_strain : array_like
            Strain span, dimensionless.

        Returns
        -------
        numpy.ndarray
            Stress span in MPa or another unit consistent with ``E`` and ``K``.

        Notes
        -----
        The calculation assumes Masing material behavior as used in the notch
        strain concept (``Kerbgrundkonzept``). It inverts twice the monotonic
        strain at half the strain span.
        """
        return 2*self.stress(strain=delta_strain/2.)

    def lower_hysteresis(self, stress, max_stress):
        """Calculate the lower hysteresis branch from maximum stress.

        Parameters
        ----------
        stress : array_like
            Stress values on the lower branch in MPa or another unit
            consistent with ``E`` and ``K``. Values must not exceed
            ``max_stress``.
        max_stress : float
            Maximum stress of the hysteresis loop in the same unit as
            ``stress``.

        Returns
        -------
        numpy.ndarray
            Strain values on the lower hysteresis branch from ``max_stress``
            to ``stress``, dimensionless.

        Raises
        ------
        ValueError
            Raised if any value in ``stress`` is greater than ``max_stress``.
        """
        stress = np.asarray(stress)
        if (stress > max_stress).any():
            raise ValueError("Value for 'stress' must not be higher than 'max_stress'.")
        return self.strain(max_stress) - self.delta_strain(max_stress-stress)

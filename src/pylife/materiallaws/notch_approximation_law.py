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

r"""Provide base classes and extended Neuber notch approximation laws.

The module supports FKM nonlinear assessments by converting linear-elastic
local loads from finite-element calculations to elastic-plastic local
stress-strain paths following Ramberg-Osgood material behavior.
"""

__author__ = ["Benjamin Maier"]
__maintainer__ = __author__

from abc import ABC, abstractmethod

import numpy as np
from scipy import optimize
import pandas as pd

import pylife.materiallaws.rambgood

class NotchApproximationLawBase(ABC):
    """Define the interface for notch approximation laws.

    A notch approximation law maps a linear-elastic local load from an FE result
    to the elastic-plastic stress and strain used by the FKM nonlinear assessment.
    The primary path starts at the origin; secondary branches describe hysteresis
    increments. Use :class:`ExtendedNeuber` for P_RAM and
    :class:`~pylife.materiallaws.notch_approximation_law_seegerbeste.SeegerBeste`
    for P_RAJ.

    Parameters
    ----------
    E : float
        Young's modulus in MPa.
    K : float
        Ramberg-Osgood strength coefficient in MPa, also denoted ``K_prime``.
    n : float
        Ramberg-Osgood strain hardening exponent, dimensionless.
    K_p : float, optional
        Plastic shape factor, dimensionless.
    """

    def __init__(self, E, K, n, K_p=None):
        self._E = E
        self._K = K
        self._n = n
        self._K_p = K_p

        self._ramberg_osgood_relation = pylife.materiallaws.rambgood.RambergOsgood(E, K, n)

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
    def K(self):
        """Return the Ramberg-Osgood strength coefficient.

        Returns
        -------
        float
            Strength coefficient in MPa, also denoted ``K_prime``.
        """
        return self._K

    @property
    def n(self):
        """Return the Ramberg-Osgood strain hardening exponent.

        Returns
        -------
        float
            Strain hardening exponent, dimensionless.
        """
        return self._n

    @property
    def K_p(self):
        """Return the plastic shape factor.

        Returns
        -------
        float
            Plastic shape factor, dimensionless.
        """
        return self._K_p

    @property
    def ramberg_osgood_relation(self):
        """Return the Ramberg-Osgood material relation.

        Returns
        -------
        pylife.materiallaws.rambgood.RambergOsgood
            Material relation used to convert elastic-plastic stress and strain.
        """
        return self._ramberg_osgood_relation

    @K_p.setter
    def K_p(self, value):
        """Set the plastic shape factor.

        Parameters
        ----------
        value : float
            Plastic shape factor, dimensionless.
        """
        self._K_p = value

    @K.setter
    def K_prime(self, value):
        """Set the Ramberg-Osgood strength coefficient.

        Parameters
        ----------
        value : float
            Strength coefficient in MPa, also denoted ``K_prime``.
        """
        self._K = value
        self._ramberg_osgood_relation = pylife.materiallaws.rambgood.RambergOsgood(self._E, self._K, self._n)

    @K.setter
    def K(self, value):
        """Set the Ramberg-Osgood strength coefficient.

        Parameters
        ----------
        value : float
            Strength coefficient in MPa, also denoted ``K_prime``.
        """
        self.K_prime = value

    @abstractmethod
    def load(self, stress, *, rtol=1e-4, tol=1e-4):
        """Calculate linear-elastic load from elastic-plastic stress.

        Parameters
        ----------
        stress : array_like
            Elastic-plastic stress in MPa on the primary path.
        rtol : float, optional
            Relative tolerance for solving the implicit equation. Default is ``1e-4``.
        tol : float, optional
            Absolute tolerance for solving the implicit equation. Default is ``1e-4``.

        Returns
        -------
        float or numpy.ndarray
            Linear-elastic load in MPa that produces ``stress``.
        """
        ...

    @abstractmethod
    def stress(self, load, *, rtol=1e-4, tol=1e-4):
        """Calculate primary-path elastic-plastic stress from load.

        Parameters
        ----------
        load : array_like
            Linear-elastic von Mises stress from a scaled FE result in MPa, denoted
            as load ``L`` in the FKM nonlinear guideline.
        rtol : float, optional
            Relative tolerance for solving the implicit equation. Default is ``1e-4``.
        tol : float, optional
            Absolute tolerance for solving the implicit equation. Default is ``1e-4``.

        Returns
        -------
        float or numpy.ndarray
            Elastic-plastic stress in MPa on the primary path.
        """
        ...

    @abstractmethod
    def strain(self, load):
        """Calculate primary-path elastic-plastic strain from stress.

        Parameters
        ----------
        load : array_like
            Elastic-plastic stress in MPa on the primary path. The abstract
            base class uses the historical parameter name ``load``.

        Returns
        -------
        float or numpy.ndarray
            Elastic-plastic strain on the primary path, dimensionless.
        """
        ...

    @abstractmethod
    def load_secondary_branch(self, load, *, rtol=1e-4, tol=1e-4):
        """Calculate load increment from secondary-branch stress increment.

        Parameters
        ----------
        load : array_like
            Elastic-plastic stress increment in MPa on a secondary hysteresis
            branch. The abstract base class uses the historical parameter name
            ``load``.
        rtol : float, optional
            Relative tolerance for solving the implicit equation. Default is ``1e-4``.
        tol : float, optional
            Absolute tolerance for solving the implicit equation. Default is ``1e-4``.

        Returns
        -------
        float or numpy.ndarray
            Linear-elastic load increment in MPa that produces ``load``.
        """
        ...

    @abstractmethod
    def stress_secondary_branch(self, load, *, rtol=1e-4, tol=1e-4):
        """Calculate stress increment from secondary-branch load increment.

        Parameters
        ----------
        load : array_like
            Linear-elastic load increment in MPa for a hysteresis branch.
        rtol : float, optional
            Relative tolerance for solving the implicit equation. Default is ``1e-4``.
        tol : float, optional
            Absolute tolerance for solving the implicit equation. Default is ``1e-4``.

        Returns
        -------
        float or numpy.ndarray
            Elastic-plastic stress increment in MPa on the secondary branch.
        """
        ...

    @abstractmethod
    def strain_secondary_branch(self, load):
        """Calculate secondary-branch strain increment from stress increment.

        Parameters
        ----------
        load : array_like
            Elastic-plastic stress increment in MPa on a secondary hysteresis
            branch. The abstract base class uses the historical parameter name
            ``load``.

        Returns
        -------
        float or numpy.ndarray
            Elastic-plastic strain increment on the secondary branch, dimensionless.
        """
        ...

    def primary(self, load):
        """Calculate stress and strain for the primary path.

        Parameters
        ----------
        load : array_like
            Linear-elastic load in MPa.

        Returns
        -------
        numpy.ndarray
            Stress-strain array. Scalars return ``[stress, strain]``; arrays return
            stress values and strain values stacked along the last axis.
        """
        load = np.asarray(load)
        stress = self.stress(load)
        strain = self.strain(stress)
        return np.stack([stress, strain], axis=len(load.shape))

    def secondary(self, delta_load):
        """Calculate stress and strain increments for a secondary branch.

        Parameters
        ----------
        delta_load : array_like
            Linear-elastic load increment in MPa for the hysteresis branch.

        Returns
        -------
        numpy.ndarray
            Stress-strain increment array. Scalars return ``[delta_stress,
            delta_strain]``; arrays return increments stacked along the last axis.
        """
        delta_load = np.asarray(delta_load)
        delta_stress = self.stress_secondary_branch(delta_load)
        delta_strain = self.strain_secondary_branch(delta_stress)
        return np.stack([delta_stress, delta_strain], axis=len(delta_load.shape))


class ExtendedNeuber(NotchApproximationLawBase):
    r"""Apply the extended Neuber notch approximation law.

    Use this law for the P_RAM damage parameter in the FKM nonlinear assessment.
    It converts a linear-elastic FE load to an elastic-plastic local stress and
    strain following the Ramberg-Osgood material law.

    Parameters
    ----------
    E : float
        Young's modulus in MPa.
    K : float
        Ramberg-Osgood strength coefficient in MPa, also denoted ``K_prime``.
    n : float
        Ramberg-Osgood strain hardening exponent, dimensionless.
    K_p : float, optional
        Plastic shape factor, dimensionless.

    Notes
    -----
    The primary path follows extended Neuber's rule from section 2.5.7 of the
    FKM guideline nonlinear [FKM-Neuber]_:

    .. math::

        \varepsilon(\sigma) = \frac{L}{\sigma} K_p\,
        \varepsilon^*(L), \qquad
        \varepsilon^*(L) = \varepsilon\!\left(\frac{L}{K_p}\right).

    References
    ----------
    .. [FKM-Neuber] Forschungskuratorium Maschinenbau,
       ``FKM-Richtlinie nichtlinear``, 2019.
    """

    def stress(self, load, *, rtol=1e-4, tol=1e-4):
        """Calculate primary-path elastic-plastic stress from load.

        Parameters
        ----------
        load : array_like
            Linear-elastic von Mises stress from a scaled FE result in MPa, denoted
            as load ``L`` in the FKM nonlinear guideline.
        rtol : float, optional
            Relative tolerance for solving the implicit equation. Default is ``1e-4``.
        tol : float, optional
            Absolute tolerance for solving the implicit equation. Default is ``1e-4``.

        Returns
        -------
        float or numpy.ndarray
            Elastic-plastic stress in MPa on the primary path.
        """
        stress = optimize.newton(
            func=self._stress_implicit,
            x0=np.asarray(load),
            fprime=self._d_stress_implicit,
            args=([load]),
            rtol=rtol, tol=tol, maxiter=20
        )
        return stress

    def strain(self, stress):
        """Calculate primary-path elastic-plastic strain from stress.

        Parameters
        ----------
        stress : array_like
            Elastic-plastic stress in MPa on the primary path.

        Returns
        -------
        float or numpy.ndarray
            Elastic-plastic strain on the primary path, dimensionless.
        """

        return self._ramberg_osgood_relation.strain(stress)

    def load(self, stress, *, rtol=1e-4, tol=1e-4):
        """Calculate linear-elastic load from elastic-plastic stress.

        Parameters
        ----------
        stress : array_like
            Elastic-plastic stress in MPa on the primary path.
        rtol : float, optional
            Relative tolerance for solving the implicit equation. Default is ``1e-4``.
        tol : float, optional
            Absolute tolerance for solving the implicit equation. Default is ``1e-4``.

        Returns
        -------
        float or numpy.ndarray
            Linear-elastic load in MPa that produces ``stress``.
        """

        # self._stress_implicit(stress) = 0
        # f(sigma) = sigma/E + (sigma/K')^(1/n') - (L/sigma * K_p * e_star) = 0
        # =>   sigma/E + (sigma/K')^(1/n') =  (L/sigma * K_p * e_star)
        # =>   (sigma/E + (sigma/K')^(1/n')) /  K_p * sigma =  L *  e_star(L)
        # <=> self._ramberg_osgood_relation.strain(stress) / self._K_p * stress = L * e_star(L)
        load = optimize.newton(
            func=self._load_implicit,
            x0=np.asarray(stress),
            fprime=self._d_load_implicit,
            args=([stress]),
            rtol=rtol, tol=tol, maxiter=20
        )
        return load

    def stress_secondary_branch(self, delta_load, *, rtol=1e-4, tol=1e-4):
        """Calculate stress increment from secondary-branch load increment.

        Parameters
        ----------
        delta_load : array_like
            Linear-elastic load increment in MPa for a hysteresis branch.
        rtol : float, optional
            Relative tolerance for solving the implicit equation. Default is ``1e-4``.
        tol : float, optional
            Absolute tolerance for solving the implicit equation. Default is ``1e-4``.

        Returns
        -------
        float or numpy.ndarray
            Elastic-plastic stress increment in MPa on the secondary branch.
        """
        delta_stress = optimize.newton(
            func=self._stress_secondary_implicit,
            x0=np.asarray(delta_load),
            fprime=self._d_stress_secondary_implicit,
            args=([np.asarray(delta_load, dtype=np.float64)]),
            rtol=rtol, tol=tol, maxiter=20
        )
        return delta_stress

    def strain_secondary_branch(self, delta_stress):
        """Calculate secondary-branch strain increment from stress increment.

        Parameters
        ----------
        delta_stress : array_like
            Elastic-plastic stress increment in MPa on a secondary hysteresis branch.

        Returns
        -------
        float or numpy.ndarray
            Elastic-plastic strain increment on the secondary branch, dimensionless.
        """

        return self._ramberg_osgood_relation.delta_strain(delta_stress)

    def load_secondary_branch(self, delta_stress, *, rtol=1e-4, tol=1e-4):
        """Calculate load increment from secondary-branch stress increment.

        Parameters
        ----------
        delta_stress : array_like
            Elastic-plastic stress increment in MPa on a secondary hysteresis branch.
        rtol : float, optional
            Relative tolerance for solving the implicit equation. Default is ``1e-4``.
        tol : float, optional
            Absolute tolerance for solving the implicit equation. Default is ``1e-4``.

        Returns
        -------
        float or numpy.ndarray
            Linear-elastic load increment in MPa that produces ``delta_stress``.
        """

        # self._stress_implicit(stress) = 0
        # f(sigma) = sigma/E + (sigma/K')^(1/n') - (L/sigma * K_p * e_star) = 0
        # =>   sigma/E + (sigma/K')^(1/n') =  (L/sigma * K_p * e_star)
        # =>   (sigma/E + (sigma/K')^(1/n')) /  K_p * sigma =  L *  e_star(L)
        # <=> self._ramberg_osgood_relation.strain(stress) / self._K_p * stress = L * e_star(L)
        delta_load = optimize.newton(
            func=self._load_secondary_implicit,
            x0=np.asarray(delta_stress),
            fprime=self._d_load_secondary_implicit,
            args=([delta_stress]),
            rtol=rtol, tol=tol, maxiter=20
        )
        return delta_load

    def _e_star(self, load):
        """Calculate the Neuber-corrected primary strain term.
        """

        corrected_load = load / self._K_p
        return self._ramberg_osgood_relation.strain(corrected_load)

    def _d_e_star(self, load):
        """Calculate the derivative of the primary corrected strain term.
        """
        return 1/(self.K_p * self.E) \
            + self._ramberg_osgood_relation.tangential_compliance(load/self.K_p) / self.K_p

    def _neuber_strain(self, stress, load):
        """Calculate the primary Neuber strain term.
        """

        e_star = self._e_star(load)

        # bad conditioned problem for stress approximately 0 (divide by 0), use factor 1 instead
        # convert data from int to float
        if not isinstance(load, float):
            load = load.astype(float)
        # factor = load / stress, avoid division by 0
        factor = np.divide(load, stress, out=np.ones_like(load), where=stress!=0)

        return factor * self._K_p * e_star

    def _stress_implicit(self, stress, load):
        """Calculate the primary implicit stress residual.
        """

        return self._ramberg_osgood_relation.strain(stress) - self._neuber_strain(stress, load)

    def _d_stress_implicit(self, stress, load):
        """Calculate the derivative of the primary stress residual.
        """

        e_star = self._e_star(load)
        return self._ramberg_osgood_relation.tangential_compliance(stress) \
            - load * self._K_p * e_star \
            * -np.power(stress, -2, out=np.ones_like(stress), where=stress!=0)

    def _delta_e_star(self, delta_load):
        """Calculate the Neuber-corrected secondary strain term.
        """

        corrected_load = delta_load / self._K_p
        return self._ramberg_osgood_relation.delta_strain(corrected_load)

    def _d_delta_e_star(self, delta_load):
        """Calculate the derivative of the secondary corrected strain term.
        """
        return 1/(self.K_p * self.E) \
            + self._ramberg_osgood_relation.tangential_compliance(delta_load/(2*self.K_p)) / self.K_p

    def _neuber_strain_secondary(self, delta_stress, delta_load):
        """Calculate the secondary Neuber strain term.
        """

        delta_e_star = self._delta_e_star(delta_load)

        # bad conditioned problem for delta_stress approximately 0 (divide by 0), use factor 1 instead
        # convert data from int to float
        if not isinstance(delta_load, float):
            delta_load = delta_load.astype(float)
        # factor = load / stress, avoid division by 0
        factor = np.divide(delta_load, delta_stress, out=np.ones_like(delta_load), where=delta_stress!=0)

        return factor * self._K_p * delta_e_star

    def _stress_secondary_implicit(self, delta_stress, delta_load):
        """Calculate the secondary implicit stress residual.
        """

        return self._ramberg_osgood_relation.delta_strain(delta_stress) - self._neuber_strain_secondary(delta_stress, delta_load)

    def _d_stress_secondary_implicit(self, delta_stress, delta_load):
        """Calculate the derivative of the secondary stress residual.
        """

        delta_e_star = self._delta_e_star(delta_load)

        return self._ramberg_osgood_relation.tangential_compliance(delta_stress/2) \
            - delta_load * self._K_p * delta_e_star \
            * -np.power(delta_stress, -2, out=np.ones_like(delta_stress), where=delta_stress!=0)

    def _load_implicit(self, load, stress):
         """Calculate the primary implicit load residual.
         """

         return self._stress_implicit(stress, load)

    def _d_load_implicit(self, load, stress):
        """Calculate the derivative of the primary load residual.
        """

        return -1/stress * self.K_p * self._e_star(load) \
            - load/stress * self.K_p * self._d_e_star(load)

    def _load_secondary_implicit(self, delta_load, delta_stress):
        """Calculate the secondary implicit load residual.
        """

        return self._stress_secondary_implicit(delta_stress, delta_load)

    def _d_load_secondary_implicit(self, delta_load, delta_stress):
        """Calculate the derivative of the secondary load residual.
        """

        return -1/delta_stress * self.K_p * self._delta_e_star(delta_load) \
            - delta_load/delta_stress * self.K_p * self._d_delta_e_star(delta_load)



class NotchApproxBinner:
    """Cache a notch approximation law on lookup tables.

    Use this helper when repeated evaluations of the same notch approximation law
    would make nonlinear root finding too expensive. The binner precomputes the
    primary path and secondary hysteresis branches up to an expected maximum load.

    Parameters
    ----------
    notch_approximation_law : NotchApproximationLawBase
        Notch approximation law to tabulate.
    number_of_bins : int, optional
        Number of bins in the primary lookup table. Default is ``100``.
    """

    def __init__(self, notch_approximation_law, number_of_bins=100):
        self._n_bins = number_of_bins
        self._notch_approximation_law = notch_approximation_law
        self._ramberg_osgood_relation = notch_approximation_law.ramberg_osgood_relation
        self._max_load_rep = None
        self._max_load_index = None

    def initialize(self, max_load):
        """Initialize lookup tables up to the maximum expected load.

        Parameters
        ----------
        max_load : array_like
            Maximum linear-elastic load in MPa expected during the assessment.

        Returns
        -------
        NotchApproxBinner
            Initialized binner instance.
        """
        max_load = np.asarray(max_load)
        self._max_load_rep, _ = self._representative_value_and_sign(max_load)

        load = self._param_for_lut(self._n_bins, max_load)
        self._lut_primary = self._notch_approximation_law.primary(load)

        delta_load = self._param_for_lut(2 * self._n_bins, 2.0*max_load)
        self._lut_secondary = self._notch_approximation_law.secondary(delta_load)

        return self

    @property
    def ramberg_osgood_relation(self):
        """Return the Ramberg-Osgood material relation.

        Returns
        -------
        pylife.materiallaws.rambgood.RambergOsgood
            Material relation of the tabulated notch approximation law.
        """
        return self._ramberg_osgood_relation

    def primary(self, load):
        """Look up stress and strain on the primary path.

        Parameters
        ----------
        load : array_like
            Linear-elastic load in MPa.

        Returns
        -------
        numpy.ndarray
            Tabulated stress-strain array on the primary path.

        Raises
        ------
        RuntimeError
            Raised if :meth:`initialize` has not been called.
        ValueError
            Raised if ``load`` exceeds the initialized maximum load.
        """
        self._raise_if_uninitialized()
        load_rep, sign = self._representative_value_and_sign(load)

        if load_rep > self._max_load_rep:
            msg = f"Requested load `{load_rep}`, higher than initialized maximum load `{self._max_load_rep}`"
            raise ValueError(msg)

        idx = int(np.ceil(load_rep / self._max_load_rep * self._n_bins)) - 1
        return sign * self._lut_primary[idx, :]

    def secondary(self, delta_load):
        """Look up stress and strain increments on a secondary branch.

        Parameters
        ----------
        delta_load : array_like
            Linear-elastic load increment in MPa.

        Returns
        -------
        numpy.ndarray
            Tabulated stress-strain increment array on the secondary branch.

        Raises
        ------
        RuntimeError
            Raised if :meth:`initialize` has not been called.
        ValueError
            Raised if ``delta_load`` exceeds the initialized maximum load increment.
        """
        self._raise_if_uninitialized()
        delta_load_rep, sign = self._representative_value_and_sign(delta_load)

        if delta_load_rep > 2.0 * self._max_load_rep:
            msg = f"Requested load `{delta_load_rep}`, higher than initialized maximum delta load `{2.0*self._max_load_rep}`"
            raise ValueError(msg)

        idx = int(np.ceil(delta_load_rep / (2.0*self._max_load_rep) * 2*self._n_bins)) - 1
        return sign * self._lut_secondary[idx, :]

    def _raise_if_uninitialized(self):
        if self._max_load_rep is None:
            raise RuntimeError("NotchApproxBinner not initialized.")

    def _param_for_lut(self, number_of_bins, max_val):
        scale = np.linspace(0.0, 1.0, number_of_bins + 1)[1:]
        max_val, scale_m = np.meshgrid(max_val, scale)
        return (max_val * scale_m)

    def _representative_value_and_sign(self, value):
        value = np.asarray(value)

        single_point = len(value.shape) == 0

        if self._max_load_index is None and single_point:
            self._max_load_index = 0

        value_rep = value if single_point else self._first_or_maximum_load_of_mesh(value)

        return np.abs(value_rep), np.sign(value_rep)

    def _first_or_maximum_load_of_mesh(self, mesh_values):
        if self._max_load_index is None:
            if mesh_values[0] != 0.0:
                self._max_load_index = 0
            else:
                self._max_load_index = np.argmax(np.abs(mesh_values))
            if mesh_values[self._max_load_index] == 0.0:
                raise ValueError(
                    "NotchApproxBinner must have at least one non zero point in max_load."
                )
        return mesh_values[self._max_load_index]

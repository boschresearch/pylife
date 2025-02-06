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

from abc import ABC, abstractmethod

import numpy as np
from scipy import optimize
import pandas as pd

import pylife.materiallaws.rambgood

class NotchApproximationLawBase(ABC):
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
        """Young's Modulus"""
        return self._E

    @property
    def K(self):
        """the strength coefficient"""
        return self._K

    @property
    def n(self):
        """the strain hardening coefficient"""
        return self._n

    @property
    def K_p(self):
        """the shape factor (de: Traglastformzahl)"""
        return self._K_p

    @property
    def ramberg_osgood_relation(self):
        """the Ramberg-Osgood relation object, i.e., an object of type RambergOsgood
        """
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

    @abstractmethod
    def load(self, stress, *, rtol=1e-4, tol=1e-4):
        """Apply the notch-approximation law "backwards", i.e., compute the linear-elastic stress (called "load" or "L" in FKM nonlinear)

        This is to be reimplemented by derived classes

        Parameters
        ----------
        stress : array-like float
            The elastic-plastic stress as computed by the notch approximation
        rtol : float, optional
            The relative tolerance to which the implicit formulation of the load gets solved,
            by default 1e-4
        tol : float, optional
            The absolute tolerance to which the implicit formulation of the load gets solved,
            by default 1e-4

        Returns
        -------
        load : array-like float
            The resulting load or lienar-elastic stress.

        """
        ...

    @abstractmethod
    def stress(self, load, *, rtol=1e-4, tol=1e-4):
        r"""Calculate the stress of the primary path in the stress-strain diagram at a given

        This is to be reimplemented by derived classes

        Parameters
        ----------
        load : array-like float
            The elastic von Mises stress from a linear elastic FEA.
            In the FKM nonlinear document, this is also called load "L", because it is derived
            from a load-time series. Note that this value is scaled to match the actual loading
            in the assessment, it equals the FEM solution times the transfer factor.
        rtol : float, optional
            The relative tolerance to which the implicit formulation of the stress gets solved,
            by default 1e-4
        tol : float, optional
            The absolute tolerance to which the implicit formulation of the stress gets solved,
            by default 1e-4

        Returns
        -------
        stress : array-like float
            The resulting elastic-plastic stress according to the notch-approximation law.
        """        "Compute the local notch stress from the local nominal load."
        ...

    @abstractmethod
    def strain(self, load):
        """Calculate the strain of the primary path in the stress-strain diagram at a given stress and load.

        This is to be reimplemented by derived classes

        Parameters
        ----------
        stress : array-like float
            The stress
        load : array-like float
            The load

        Returns
        -------
        strain : array-like float
            The resulting strain
        """
        ...

    @abstractmethod
    def load_secondary_branch(self, load, *, rtol=1e-4, tol=1e-4):
        """Apply the notch-approximation law "backwards", i.e., compute the linear-elastic stress (called "load" or "L" in FKM nonlinear) from the elastic-plastic stress as from the notch approximation.

        This backward step is needed for the pfp FKM nonlinear surface layer & roughness.

        This is to be reimplemented by derived classes

        Parameters
        ----------
        delta_stress : array-like float
            The increment of the elastic-plastic stress as computed by the notch approximation
        rtol : float, optional
            The relative tolerance to which the implicit formulation of the stress gets solved,
            by default 1e-4
        tol : float, optional
            The absolute tolerance to which the implicit formulation of the stress gets solved,
            by default 1e-4

        Returns
        -------
        delta_load : array-like float
            The resulting load or lienar-elastic stress.

        """
        ...

    @abstractmethod
    def stress_secondary_branch(self, load, *, rtol=1e-4, tol=1e-4):
        """Calculate the stress on secondary branches in the stress-strain diagram at a given
        elastic-plastic stress (load), from a FE computation.

        This is to be reimplemented by derived classes

        Parameters
        ----------
        delta_load : array-like float
            The load increment of the hysteresis
        rtol : float, optional
            The relative tolerance to which the implicit formulation of the stress gets solved,
            by default 1e-4
        tol : float, optional
            The absolute tolerance to which the implicit formulation of the stress gets solved,
            by default 1e-4

        Returns
        -------
        delta_stress : array-like float
            The resulting stress increment within the hysteresis
        """
        ...

    @abstractmethod
    def strain_secondary_branch(self, load):
        """Calculate the strain on secondary branches in the stress-strain diagram at a given stress and load.

        This is to be reimplemented by derived classes

        Parameters
        ----------
        delta_sigma : array-like float
            The stress increment
        delta_load : array-like float
            The load increment

        Returns
        -------
        strain : array-like float
            The resulting strain
        """
        ...

    def primary(self, load):
        """Calculate stress and strain for primary branch.

        Parameters
        ----------
        load : array-like
            The load for which the stress and strain are to be calculated

        Returns
        -------
        stress strain : ndarray
            The resulting stress strain data.

            If the argument is scalar, the resulting array is of the strucuture
            ``[<σ>, <ε>]``

            If the argument is an 1D-array with length `n`the resulting array is of the
            structure ``[[<σ1>, <σ2>, <σ3>, ... <σn>], [<ε1>, <ε2>, <ε3>, ... <εn>]]``

        """
        load = np.asarray(load)
        stress = self.stress(load)
        strain = self.strain(stress)
        return np.stack([stress, strain], axis=len(load.shape))

    def secondary(self, delta_load):
        """Calculate stress and strain for secondary branch.

        Parameters
        ----------
        load : array-like
            The load for which the stress and strain are to be calculated

        Returns
        -------
        stress strain : ndarray
            The resulting stress strain data.

            If the argument is scalar, the resulting array is of the strucuture
            ``[<σ>, <ε>]``

            If the argument is an 1D-array with length `n`the resulting array is of the
            structure ``[[<σ1>, <σ2>, <σ3>, ... <σn>], [<ε1>, <ε2>, <ε3>, ... <εn>]]``

        """
        delta_load = np.asarray(delta_load)
        delta_stress = self.stress_secondary_branch(delta_load)
        delta_strain = self.strain_secondary_branch(delta_stress)
        return np.stack([delta_stress, delta_strain], axis=len(delta_load.shape))


class ExtendedNeuber(NotchApproximationLawBase):
    r"""Implementation of the extended Neuber notch approximation material relation.

    This notch approximation law is used for the P_RAM damage parameter in the FKM
    nonlinear guideline (2019). Given an elastic-plastic stress (and strain) from a linear FE
    calculation, it derives a corresponding elastic-plastic stress (and strain).

    Note, the input stress and strain follow a linear relationship :math:`\sigma = E \cdot \epsilon`.
    The output stress and strain follow the Ramberg-Osgood relation.

    Parameters
    ----------

    E : float
        Young's Modulus
    K : float
        The strain hardening coefficient, often also designated :math:`K'`, or ``K_prime``.
    n : float
        The strain hardening exponent, often also designated :math:`n'`, or ``n_prime``.
    K_p : float, optional
        The shape factor (de: Traglastformzahl)

    Notes
    -----
    The equation implemented is described in the FKM nonlinear reference, chapter 2.5.7.

    """

    def stress(self, load, *, rtol=1e-4, tol=1e-4):
        r"""Calculate the stress of the primary path in the stress-strain diagram at a given

        Parameters
        ----------
        load : array-like float
            The elastic von Mises stress from a linear elastic FEA.
            In the FKM nonlinear document, this is also called load "L", because it is derived
            from a load-time series. Note that this value is scaled to match the actual loading
            in the assessment, it equals the FEM solution times the transfer factor.
        rtol : float, optional
            The relative tolerance to which the implicit formulation of the stress gets solved,
            by default 1e-4
        tol : float, optional
            The absolute tolerance to which the implicit formulation of the stress gets solved,
            by default 1e-4

        Returns
        -------
        stress : array-like float
            The resulting elastic-plastic stress according to the notch-approximation law.
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
        """Calculate the strain of the primary path in the stress-strain diagram at a given stress and load.
        The formula is given by eq. 2.5-42 of FKM nonlinear.
        load / stress * self._K_p * e_star

        Parameters
        ----------
        stress : array-like float
            The stress
        load : array-like float
            The load

        Returns
        -------
        strain : array-like float
            The resulting strain
        """

        return self._ramberg_osgood_relation.strain(stress)

    def load(self, stress, *, rtol=1e-4, tol=1e-4):
        """Apply the notch-approximation law "backwards", i.e., compute the linear-elastic stress (called "load" or "L" in FKM nonlinear)
        from the elastic-plastic stress as from the notch approximation.
        This backward step is needed for the pfp FKM nonlinear surface layer & roughness.

        This method is the inverse operation of "stress", i.e., ``L = load(stress(L))`` and ``S = stress(load(stress))``.

        Parameters
        ----------
        stress : array-like float
            The elastic-plastic stress as computed by the notch approximation
        rtol : float, optional
            The relative tolerance to which the implicit formulation of the load gets solved,
            by default 1e-4
        tol : float, optional
            The absolute tolerance to which the implicit formulation of the load gets solved,
            by default 1e-4

        Returns
        -------
        load : array-like float
            The resulting load or lienar-elastic stress.

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
        """Calculate the stress on secondary branches in the stress-strain diagram at a given
        elastic-plastic stress (load), from a FE computation.
        This is done by solving for the root of f(sigma) in eq. 2.5-46 of FKM nonlinear.

        Parameters
        ----------
        delta_load : array-like float
            The load increment of the hysteresis
        rtol : float, optional
            The relative tolerance to which the implicit formulation of the stress gets solved,
            by default 1e-4
        tol : float, optional
            The absolute tolerance to which the implicit formulation of the stress gets solved,
            by default 1e-4

        Returns
        -------
        delta_stress : array-like float
            The resulting stress increment within the hysteresis
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
        """Calculate the strain on secondary branches in the stress-strain diagram at a given stress and load.
        The formula is given by eq. 2.5-46 of FKM nonlinear.

        Parameters
        ----------
        delta_sigma : array-like float
            The stress increment
        delta_load : array-like float
            The load increment

        Returns
        -------
        strain : array-like float
            The resulting strain
        """

        return self._ramberg_osgood_relation.delta_strain(delta_stress)

    def load_secondary_branch(self, delta_stress, *, rtol=1e-4, tol=1e-4):
        """Apply the notch-approximation law "backwards", i.e., compute the linear-elastic stress (called "load" or "L" in FKM nonlinear)
        from the elastic-plastic stress as from the notch approximation.
        This backward step is needed for the pfp FKM nonlinear surface layer & roughness.

        This method is the inverse operation of "stress", i.e., ``L = load(stress(L))`` and ``S = stress(load(stress))``.

        Parameters
        ----------
        delta_stress : array-like float
            The increment of the elastic-plastic stress as computed by the notch approximation
        rtol : float, optional
            The relative tolerance to which the implicit formulation of the stress gets solved,
            by default 1e-4
        tol : float, optional
            The absolute tolerance to which the implicit formulation of the stress gets solved,
            by default 1e-4

        Returns
        -------
        delta_load : array-like float
            The resulting load or lienar-elastic stress.

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
        """Compute the plastic corrected strain term e^{\ast} from the Neuber approximation
        (eq. 2.5-43 in FKM nonlinear)

        ``e_star = L/K_p / E + (L/K_p / K')^(1/n')``
        """

        corrected_load = load / self._K_p
        return self._ramberg_osgood_relation.strain(corrected_load)

    def _d_e_star(self, load):
        """Compute the first derivative of self._e_star(load)

        .. code::

          e_star = L/K_p / E + (L/K_p / K')^(1/n')

          de_star(L)/dL = d/dL[ L/K_p / E + (L/K_p / K')^(1/n') ]
             = 1/(K_p * E) + tangential_compliance(L/K_p) / K_p
        """
        return 1/(self.K_p * self.E) \
            + self._ramberg_osgood_relation.tangential_compliance(load/self.K_p) / self.K_p

    def _neuber_strain(self, stress, load):
        """Compute the additional strain term from the Neuber approximation
        (2nd summand in eq. 2.5-45 in FKM nonlinear)

        ``(L/sigma * K_p * e_star)``
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
        """Compute the implicit function of the stress, f(sigma),
        defined in eq.2.5-45 of FKM nonlinear

        ``f(sigma) = sigma/E + (sigma/K')^(1/n') - (L/sigma * K_p * e_star)``
        """

        return self._ramberg_osgood_relation.strain(stress) - self._neuber_strain(stress, load)

    def _d_stress_implicit(self, stress, load):
        """Compute the first derivative of self._stress_implicit

        ``df/dsigma``
        """

        e_star = self._e_star(load)
        return self._ramberg_osgood_relation.tangential_compliance(stress) \
            - load * self._K_p * e_star \
            * -np.power(stress, -2, out=np.ones_like(stress), where=stress!=0)

    def _delta_e_star(self, delta_load):
        """Compute the plastic corrected strain term e^{\ast} from the Neuber approximation
        (eq. 2.5-43 in FKM nonlinear), for secondary branches in the stress-strain diagram
        """

        corrected_load = delta_load / self._K_p
        return self._ramberg_osgood_relation.delta_strain(corrected_load)

    def _d_delta_e_star(self, delta_load):
        """Compute the first derivative of self._delta_e_star(load)

        .. code::

          delta_e_star = ΔL/K_p / E + 2*(ΔL/K_p / (2*K'))^(1/n')
                       = ΔL/K_p / E + 2*(ΔL/(2*K_p) / K')^(1/n')

          d_delta_e_star(ΔL)/dΔL = d/dΔL[ ΔL/K_p / E + 2*(ΔL/(2*K_p) / K')^(1/n') ]
             = 1/(K_p * E) + 2*tangential_compliance(ΔL/(2*K_p)) / (2*K_p)
             = 1/(K_p * E) + tangential_compliance(ΔL/(2*K_p)) / K_p
        """
        return 1/(self.K_p * self.E) \
            + self._ramberg_osgood_relation.tangential_compliance(delta_load/(2*self.K_p)) / self.K_p

    def _neuber_strain_secondary(self, delta_stress, delta_load):
        """Compute the additional strain term from the Neuber approximation (2nd summand in eq. 2.5-45 in FKM nonlinear)"""

        delta_e_star = self._delta_e_star(delta_load)

        # bad conditioned problem for delta_stress approximately 0 (divide by 0), use factor 1 instead
        # convert data from int to float
        if not isinstance(delta_load, float):
            delta_load = delta_load.astype(float)
        # factor = load / stress, avoid division by 0
        factor = np.divide(delta_load, delta_stress, out=np.ones_like(delta_load), where=delta_stress!=0)

        return factor * self._K_p * delta_e_star

    def _stress_secondary_implicit(self, delta_stress, delta_load):
        """Compute the implicit function of the stress, f(sigma), defined in eq.2.5-46 of FKM nonlinear"""

        return self._ramberg_osgood_relation.delta_strain(delta_stress) - self._neuber_strain_secondary(delta_stress, delta_load)

    def _d_stress_secondary_implicit(self, delta_stress, delta_load):
        """Compute the first derivative of self._stress_secondary_implicit
        Note, the derivative of `self._ramberg_osgood_relation.delta_strain` is:

        .. code::

          d/dΔsigma delta_strain(Δsigma) =  d/dΔsigma 2*strain(Δsigma/2)
            = 2*d/dΔsigma strain(Δsigma/2) = 2 * 1/2 * tangential_compliance(Δsigma/2)
            = self._ramberg_osgood_relation.tangential_compliance(delta_stress/2)
        """

        delta_e_star = self._delta_e_star(delta_load)

        return self._ramberg_osgood_relation.tangential_compliance(delta_stress/2) \
            - delta_load * self._K_p * delta_e_star \
            * -np.power(delta_stress, -2, out=np.ones_like(delta_stress), where=delta_stress!=0)

    def _load_implicit(self, load, stress):
         """Compute the implicit function of the stress, f(sigma),
         as a function of the load,
         defined in eq.2.5-45 of FKM nonlinear.
         This is needed to apply the notch approximation law "backwards", i.e.,
         to get from stress back to load. This is required for the FKM nonlinear roughness & surface layer.

         ``f(L) = sigma/E + (sigma/K')^(1/n') - (L/sigma * K_p * e_star(L))``
         """

         return self._stress_implicit(stress, load)

    def _d_load_implicit(self, load, stress):
        """Compute the first derivative of self._load_implicit

        .. code::

          f(L) = sigma/E + (sigma/K')^(1/n') - (L/sigma * K_p * e_star(L))

          df/dL = d/dL [ -(L/sigma * K_p * e_star(L))]
           = -1/sigma * K_p * e_star(L) - L/sigma * K_p * de_star/dL

        """

        return -1/stress * self.K_p * self._e_star(load) \
            - load/stress * self.K_p * self._d_e_star(load)

    def _load_secondary_implicit(self, delta_load, delta_stress):
        """Compute the implicit function of the stress, f(Δsigma),
        as a function of the load,
        defined in eq.2.5-46 of FKM nonlinear.
        This is needed to apply the notch approximation law "backwards", i.e.,
        to get from stress back to load. This is required for the FKM nonlinear roughness & surface layer.

        ``f(ΔL) = Δsigma/E + 2*(Δsigma/(2*K'))^(1/n') - (ΔL/Δsigma * K_p * Δe_star(ΔL))``

        """

        return self._stress_secondary_implicit(delta_stress, delta_load)

    def _d_load_secondary_implicit(self, delta_load, delta_stress):
        """Compute the first derivative of self._load_secondary_implicit

        .. code::

          f(ΔL) = Δsigma/E + 2*(Δsigma/(2*K'))^(1/n') - (ΔL/Δsigma * K_p * Δe_star(ΔL))

          df/dΔL = d/dΔL [ -(ΔL/Δsigma * K_p * Δe_star(ΔL))]
           = -1/Δsigma * K_p * Δe_star(ΔL) - ΔL/Δsigma * K_p * dΔe_star/dΔL

        """

        return -1/delta_stress * self.K_p * self._delta_e_star(delta_load) \
            - delta_load/delta_stress * self.K_p * self._d_delta_e_star(delta_load)



class NotchApproxBinner:
    """Binning for notch approximation laws, as described in FKM nonlinear 2.5.8.2, p.55.
    The implicitly defined stress function of the notch approximation law is precomputed
    for various loads at a fixed number of equispaced `bins`. The values are stored in two
    look-up tables for the primary and secondary branches of the stress-strain hysteresis
    curves. When stress and strain values are needed for a given load, the nearest value
    of the corresponding bin is retrived. This is faster than invoking the nonlinear
    root finding algorithm for every new load.

    There are two variants of the data structure.

    * First, for a single assessment point, the lookup-table contains one load,
      strain and stress value in every bin.
    * Second, for vectorized assessment of multiple nodes at once, the lookup-table
      contains at every load bin an array with stress and strain values for every node.
      The representative load, stored in the lookup table and used for the lookup
      is the first element of the given load array.

    Parameters
    ----------
    notch_approximation_law : NotchApproximationLawBase
       The law for the notch approximation to be used.

    number_of_bins : int, optional
       The number of bins in the lookup table, default 100
    """

    def __init__(self, notch_approximation_law, number_of_bins=100):
        self._n_bins = number_of_bins
        self._notch_approximation_law = notch_approximation_law
        self._ramberg_osgood_relation = notch_approximation_law.ramberg_osgood_relation
        self._max_load_rep = None
        self._max_load_index = None

    def initialize(self, max_load):
        """Initialize with a maximum expected load.

        Parameters
        ----------
        max_load : array_like
            The state of the maximum nominal load that is expected.  The first
            element is chosen as representative to calculate the lookup table.

        Returns
        -------
        self
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
        """the Ramberg-Osgood relation object, i.e., an object of type RambergOsgood
        provided by the notch approximation law.
        """
        return self._ramberg_osgood_relation

    def primary(self, load):
        """Lookup the stress strain of the primary branch.

        Parameters
        ----------
        load : array-like
            The load as argument for the stress strain laws.
            If non-scalar, the first element will be used to look it up in the
            lookup table.

        Returns
        -------
        stress strain : ndarray
            The resulting stress strain data.

            If the argument is scalar, the resulting array is of the strucuture
            ``[<σ>, <ε>]``

            If the argument is an 1D-array with length `n`the resulting array is of the
            structure ``[[<σ1>, <σ2>, <σ3>, ... <σn>], [<ε1>, <ε2>, <ε3>, ... <εn>]]``

        """
        self._raise_if_uninitialized()
        load_rep, sign = self._representative_value_and_sign(load)

        if load_rep > self._max_load_rep:
            msg = f"Requested load `{load_rep}`, higher than initialized maximum load `{self._max_load_rep}`"
            raise ValueError(msg)

        idx = int(np.ceil(load_rep / self._max_load_rep * self._n_bins)) - 1
        return sign * self._lut_primary[idx, :]

    def secondary(self, delta_load):
        """Lookup the stress strain of the secondary branch.

        Parameters
        ----------
        load : array-like
            The load as argument for the stress strain laws.
            If non-scalar, the first element will be used to look it up in the
            lookup table.

        Returns
        -------
        stress strain : ndarray
            The resulting stress strain data.

            If the argument is scalar, the resulting array is of the strucuture
            ``[<σ>, <ε>]``

            If the argument is an 1D-array with length `n`the resulting array is of the
            structure ``[[<σ1>, <σ2>, <σ3>, ..., <σn>], [<ε1>, <ε2>, <ε3>, ..., <εn>]]``

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

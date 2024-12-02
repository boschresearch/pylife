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

import numpy as np
from scipy import optimize
import pandas as pd

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

    def strain(self, stress, load):
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

    def strain_secondary_branch(self, delta_stress, delta_load):
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


class Binned:
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
      contains at every load bin a list with stress and strain values for every node.
      The DataFrame has a multi-index over class_index and node_id.

    """

    def __init__(self, notch_approximation_law, maximum_absolute_load, number_of_bins=100):
        self._notch_approximation_law = notch_approximation_law
        self._maximum_absolute_load = maximum_absolute_load
        self._number_of_bins = number_of_bins
        self._create_bins()

    @property
    def ramberg_osgood_relation(self):
        """The ramberg osgood relation object
        """
        return self._notch_approximation_law.ramberg_osgood_relation

    def stress(self, load, *, rtol=1e-5, tol=1e-6):
        """The stress of the primary path in the stress-strain diagram at a given load
        by using the value of the look-up table.

        .. note::

            The exact value would be computed by ``self._notch_approximation_law.stress(load)``.

        Parameters
        ----------
        load : array-like float
            The load, either a scalar value or a pandas DataFrame with RangeIndex (no named index)
        rtol : float, optional
            The relative tolerance to which the implicit formulation of the stress gets solved.
            In this case for the `Binning` class, the parameter is not used.
        tol : float, optional
            The absolute tolerance to which the implicit formulation of the stress gets solved.
            In this case for the `Binning` class, the parameter is not used.

        Returns
        -------
        stress : array-like float
            The resulting stress
        """


        # FIXME consolidate theese methods (duplicated code)
        sign = np.sign(load)

        # if the assessment is performed for multiple points at once, i.e. load is a DataFrame with values for every node
        if isinstance(load, pd.Series) and isinstance(self._lut_primary_branch.index, pd.MultiIndex):

            # the lut is a DataFrame with MultiIndex with levels class_index and node_id

            # find the corresponding class only for the first node, use the result for all nodes
            first_node_id = self._lut_primary_branch.index.get_level_values("node_id")[0]
            lut_for_first_node = self._lut_primary_branch.load[self._lut_primary_branch.index.get_level_values("node_id")==first_node_id]
            first_abs_load = abs(load.iloc[0])

            # get the class index of the corresponding bin/class
            class_index = lut_for_first_node.searchsorted(first_abs_load)

            max_class_index = max(self._lut_primary_branch.index.get_level_values("class_index"))

            # raise error if requested load is higher than initialized maximum absolute load
            if class_index+1 > max_class_index:
                raise ValueError(f"Binned class is initialized with a maximum absolute load of {self._maximum_absolute_load}, "\
                                 f" but a higher absolute load value of {first_abs_load} is requested (in stress()).")

            # get stress from matching class, "+1", because the next higher class is used
            stress = self._lut_primary_branch[self._lut_primary_branch.index.get_level_values("class_index") == class_index+1].stress

            # multiply with sign
            return sign * stress.to_numpy()

        # if the assessment is performed for multiple points at once, but only one lookup-table is used
        elif isinstance(load, pd.Series):

            load = load.fillna(0)
            sign = sign.fillna(0)

            index = self._lut_primary_branch.load.searchsorted(np.abs(load.values))-1   # "-1", transform to zero-based indices

            # raise error if requested load is higher than initialized maximum absolute load
            if np.any(index+1 >= len(self._lut_primary_branch)):
                raise ValueError(f"Binned class is initialized with a maximum absolute load of {self._maximum_absolute_load}, "\
                                 f" but a higher absolute load value of |{load.max()}| is requested (in stress()).")

            return sign.values.flatten() * self._lut_primary_branch.iloc[(index+1).flatten()].stress.reset_index(drop=True)    # "+1", because the next higher class is used

        # if the assessment is done only for one value, i.e. load is a scalar
        else:

            index = self._lut_primary_branch.load.searchsorted(np.abs(load))-1   # "-1", transform to zero-based indices

            # raise error if requested load is higher than initialized maximum absolute load
            if np.any(index+1 >= len(self._lut_primary_branch)):
                raise ValueError(f"Binned class is initialized with a maximum absolute load of {self._maximum_absolute_load}, "\
                                 f" but a higher absolute load value of |{load}| is requested (in stress()).")

            return sign * self._lut_primary_branch.iloc[index+1].stress    # "+1", because the next higher class is used

    def strain(self, stress, load):
        """Get the strain of the primary path in the stress-strain diagram at a given stress and load
        by using the value of the look-up table.

        This method performs the task for for multiple points at once,
        i.e. delta_load is a DataFrame with values for every node.

        Parameters
        ----------
        load : array-like float
            The load

        Returns
        -------
        strain : array-like float
            The resulting strain
        """
        sign = np.sign(load)

        # if the assessment is performed for multiple points at once, i.e. load is a DataFrame with values for every node
        if isinstance(load, pd.Series) and isinstance(self._lut_primary_branch.index, pd.MultiIndex):

            # the lut is a DataFrame with MultiIndex with levels class_index and node_id

            # find the corresponding class only for the first node, use the result for all nodes
            first_node_id = self._lut_primary_branch.index.get_level_values("node_id")[0]
            lut_for_first_node = self._lut_primary_branch.load[self._lut_primary_branch.index.get_level_values("node_id")==first_node_id]
            first_abs_load = abs(load.iloc[0])

            # get the class index of the corresponding bin/class
            class_index = lut_for_first_node.searchsorted(first_abs_load)

            max_class_index = max(self._lut_primary_branch.index.get_level_values("class_index"))

            # raise error if requested load is higher than initialized maximum absolute load
            if class_index+1 > max_class_index:
                raise ValueError(f"Binned class is initialized with a maximum absolute load of {self._maximum_absolute_load}, "\
                                 f" but a higher absolute load value of {first_abs_load} is requested (in strain()).")

            # get strain from matching class, "+1", because the next higher class is used
            strain = self._lut_primary_branch[self._lut_primary_branch.index.get_level_values("class_index") == class_index+1].strain

            # multiply with sign
            return sign * strain.to_numpy()

        # if the assessment is performed for multiple points at once, but only one lookup-table is used
        elif isinstance(load, pd.Series):

            load = load.fillna(0)
            sign = sign.fillna(0)

            index = self._lut_primary_branch.load.searchsorted(np.abs(load.values))-1   # "-1", transform to zero-based indices

            # raise error if requested load is higher than initialized maximum absolute load
            if np.any(index+1 >= len(self._lut_primary_branch)):
                raise ValueError(f"Binned class is initialized with a maximum absolute load of {self._maximum_absolute_load}, "\
                                 f" but a higher absolute load value of |{load.max()}| is requested (in strain()).")

            return sign.values.flatten() * self._lut_primary_branch.iloc[(index+1).flatten()].strain.reset_index(drop=True)    # "+1", because the next higher class is used

        # if the assessment is done only for one value, i.e. load is a scalar
        else:

            index = self._lut_primary_branch.load.searchsorted(np.abs(load))-1  # "-1", transform to zero-based indices

            # raise error if requested load is higher than initialized maximum absolute load
            if np.any(index+1 >= len(self._lut_primary_branch)):
                raise ValueError(f"Binned class is initialized with a maximum absolute load of {self._maximum_absolute_load}, "\
                                 f" but a higher absolute load value of |{load}| is requested (in strain()).")

            return sign * self._lut_primary_branch.iloc[index+1].strain     # "+1", because the next higher class is used

    def stress_secondary_branch(self, delta_load, *, rtol=1e-5, tol=1e-6):
        """Get the stress on secondary branches in the stress-strain diagram at a given load
        by using the value of the look-up table (lut).

        This method performs the task for for multiple points at once,
        i.e. delta_load is a DataFrame with values for every node.

        Parameters
        ----------
        delta_load : array-like float
            The load increment of the hysteresis
        rtol : float, optional
            The relative tolerance to which the implicit formulation of the stress gets solved.
            In this case for the `Binning` class, the parameter is not used.
        tol : float, optional
            The absolute tolerance to which the implicit formulation of the stress gets solved.
            In this case for the `Binning` class, the parameter is not used.

        Returns
        -------
        delta_stress : array-like float
            The resulting stress increment within the hysteresis
        """

        sign = np.sign(delta_load)

        # if the assessment is performed for multiple points at once, i.e. load is a DataFrame with values for every node
        if isinstance(delta_load, pd.Series) and isinstance(self._lut_primary_branch.index, pd.MultiIndex):

            # the lut is a DataFrame with MultiIndex with levels class_index and node_id

            # find the corresponding class only for the first node, use the result for all nodes
            first_node_id = self._lut_primary_branch.index.get_level_values("node_id")[0]
            lut_for_first_node = self._lut_secondary_branch.delta_load[self._lut_secondary_branch.index.get_level_values("node_id")==first_node_id]
            first_abs_load = abs(delta_load.iloc[0])

            # get the class index of the corresponding bin/class
            class_index = lut_for_first_node.searchsorted(first_abs_load)

            max_class_index = max(self._lut_secondary_branch.index.get_level_values("class_index"))

            # raise error if requested load is higher than initialized maximum absolute load
            if class_index+1 > max_class_index:
                raise ValueError(f"Binned class is initialized with a maximum absolute delta_load load of {2*self._maximum_absolute_load}, "\
                                 f" but a higher absolute delta_load value of {first_abs_load} is requested (in stress_secondary_branch()).")

            # get stress from matching class, "+1", because the next higher class is used
            delta_stress = self._lut_secondary_branch[self._lut_secondary_branch.index.get_level_values("class_index") == class_index+1].delta_stress

            # multiply with sign
            return sign * delta_stress.to_numpy()

        # if the assessment is performed for multiple points at once, but only one lookup-table is used
        elif isinstance(delta_load, pd.Series):

            delta_load = delta_load.fillna(0)
            sign = sign.fillna(0)

            index = self._lut_secondary_branch.delta_load.searchsorted(np.abs(delta_load.values))-1   # "-1", transform to zero-based indices

            # raise error if requested load is higher than initialized maximum absolute load
            if np.any(index+1 >= len(self._lut_secondary_branch)):
                raise ValueError(f"Binned class is initialized with a maximum absolute load of {self._maximum_absolute_load}, "\
                                 f" but a higher absolute load value of |{delta_load.max()}| is requested (in stress_secondary_branch()).")

            return sign.values.flatten() * self._lut_secondary_branch.iloc[(index+1).flatten()].delta_stress.reset_index(drop=True)    # "+1", because the next higher class is used

        # if the assessment is done only for one value, i.e. load is a scalar
        else:

            index = self._lut_secondary_branch.delta_load.searchsorted(np.abs(delta_load))-1   # "-1", transform to zero-based indices

            # raise error if requested load is higher than initialized maximum absolute load
            if np.any(index+1 >= len(self._lut_secondary_branch)):
                raise ValueError(f"Binned class is initialized for a maximum absolute delta_load of {2*self._maximum_absolute_load}, "\
                                 f" but a higher absolute delta_load value of |{delta_load}| is requested (in stress_secondary_branch()).")

            return sign * self._lut_secondary_branch.iloc[index+1].delta_stress     # "+1", because the next higher class is used

    def strain_secondary_branch(self, delta_stress, delta_load):
        """Get the strain on secondary branches in the stress-strain diagram at a given stress and load
        by using the value of the look-up table (lut).
        The lut is a DataFrame with MultiIndex with levels class_index and node_id.

        This method performs the task for for multiple points at once,
        i.e. delta_load is a DataFrame with values for every node.

        Parameters
        ----------
        delta_load : array-like float
            The load increment

        Returns
        -------
        strain : array-like float
            The resulting strain
        """
        #return self._notch_approximation_law.strain_secondary_branch(delta_stress, delta_load)

        sign = np.sign(delta_load)

        # if the assessment is performed for multiple points at once, i.e. load is a DataFrame with values for every node
        if isinstance(delta_load, pd.Series) and isinstance(self._lut_primary_branch.index, pd.MultiIndex):

            # the lut is a DataFrame with MultiIndex with levels class_index and node_id

            # find the corresponding class only for the first node, use the result for all nodes
            first_node_id = self._lut_secondary_branch.index.get_level_values("node_id")[0]
            lut_for_first_node = self._lut_secondary_branch.delta_load[self._lut_secondary_branch.index.get_level_values("node_id")==first_node_id]
            first_abs_load = abs(delta_load.iloc[0])

            # get the class index of the corresponding bin/class
            class_index = lut_for_first_node.searchsorted(first_abs_load)

            max_class_index = max(self._lut_secondary_branch.index.get_level_values("class_index"))

            # raise error if requested load is higher than initialized maximum absolute load
            if class_index+1 > max_class_index:
                raise ValueError(f"Binned class is initialized with a maximum absolute delta_load of {2*self._maximum_absolute_load}, "\
                                 f" but a higher absolute delta_load value of {first_abs_load} is requested (in strain_secondary_branch()).")

            # get strain from matching class, "+1", because the next higher class is used
            delta_strain = self._lut_secondary_branch[self._lut_secondary_branch.index.get_level_values("class_index") == class_index+1].delta_strain

            # multiply with sign
            return sign * delta_strain.to_numpy()

        # if the assessment is performed for multiple points at once, but only one lookup-table is used
        elif isinstance(delta_load, pd.Series):

            delta_load = delta_load.fillna(0)
            sign = sign.fillna(0)

            index = self._lut_secondary_branch.delta_load.searchsorted(np.abs(delta_load.values))-1   # "-1", transform to zero-based indices

            # raise error if requested load is higher than initialized maximum absolute load
            if np.any(index+1 >= len(self._lut_secondary_branch)):
                raise ValueError(f"Binned class is initialized with a maximum absolute load of {self._maximum_absolute_load}, "\
                                 f" but a higher absolute load value of |{delta_load.max()}| is requested (in strain_secondary_branch()).")

            return sign.values.flatten() * self._lut_secondary_branch.iloc[(index+1).flatten()].delta_strain.reset_index(drop=True)    # "+1", because the next higher class is used

        # if the assessment is done only for one value, i.e. load is a scalar
        else:

            index = self._lut_secondary_branch.delta_load.searchsorted(np.abs(delta_load))-1   # "-1", transform to zero-based indices

            # raise error if requested load is higher than initialized maximum absolute load
            if np.any(index+1 >= len(self._lut_secondary_branch)):
                raise ValueError(f"Binned class is initialized for a maximum absolute delta_load of {2*self._maximum_absolute_load}, "\
                                 f" but a higher absolute delta_load value of |{delta_load}| is requested (in strain_secondary_branch()).")

            return sign * self._lut_secondary_branch.iloc[index+1].delta_strain     # "-1", transform to zero-based indices

    def _create_bins(self):
        """Initialize the lookup tables by precomputing the notch approximation law values.
        """
        # for multiple assessment points at once use a Series with MultiIndex
        if isinstance(self._maximum_absolute_load, pd.Series):
            assert self._maximum_absolute_load.index.name == "node_id"

            self._create_bins_multiple_assessment_points()

        # for a single assessment point use the standard data structure
        else:
            self._create_bins_single_assessment_point()

    def _create_bins_single_assessment_point(self):
        """Initialize the lookup tables by precomputing the notch approximation law values,
        for the case of scalar variables, i.e., only a single assessment point."""

        # create look-up table (lut) for the primary branch values, named PFAD in FKM nonlinear
        self._lut_primary_branch = pd.DataFrame(0,
            index=pd.Index(np.arange(1, self._number_of_bins+1), name="class_index"),
            columns=["load", "strain", "stress"])

        self._lut_primary_branch.load \
            = self._lut_primary_branch.index/self._number_of_bins * self._maximum_absolute_load

        self._lut_primary_branch.stress \
            = self._notch_approximation_law.stress(self._lut_primary_branch.load)

        self._lut_primary_branch.strain \
            = self._notch_approximation_law.strain(
                self._lut_primary_branch.stress, self._lut_primary_branch.load)

        # create look-up table (lut) for the secondary branch values, named AST in FKM nonlinear
        # Note that this time, we used twice the number of entries with the same bin width.
        self._lut_secondary_branch = pd.DataFrame(0,
            index=pd.Index(np.arange(1, 2*self._number_of_bins+1), name="class_index"),
            columns=["delta_load", "delta_strain", "delta_stress"])

        self._lut_secondary_branch.delta_load \
            = self._lut_secondary_branch.index/self._number_of_bins * self._maximum_absolute_load

        self._lut_secondary_branch.delta_stress \
            = self._notch_approximation_law.stress_secondary_branch(self._lut_secondary_branch.delta_load)

        self._lut_secondary_branch.delta_strain \
            = self._notch_approximation_law.strain_secondary_branch(
                self._lut_secondary_branch.delta_stress, self._lut_secondary_branch.delta_load)

    def _create_bins_multiple_assessment_points(self):
        """Initialize the lookup tables by precomputing the notch approximation law values,
        for the case of vector-valued variables caused by an assessment on multiple points at once."""

        self._maximum_absolute_load.name = "max_abs_load"

        # create look-up table (lut) for the primary branch values, named PFAD in FKM nonlinear
        index = pd.MultiIndex.from_product([np.arange(1, self._number_of_bins+1), self._maximum_absolute_load.index],
                                           names = ["class_index", "node_id"])

        self._lut_primary_branch = pd.DataFrame(0, index=index, columns=["load", "strain", "stress"])

        # create cartesian product of class index and max load
        a = pd.DataFrame({"class_index": np.arange(1, self._number_of_bins+1)})
        class_index_with_max_load = a.merge(self._maximum_absolute_load, how='cross')

        # calculate load of each bin
        load = class_index_with_max_load.class_index.astype(float)/self._number_of_bins * class_index_with_max_load.max_abs_load
        load.index = index
        self._lut_primary_branch.load = load

        self._lut_primary_branch.stress \
            = self._notch_approximation_law.stress(self._lut_primary_branch.load)

        self._lut_primary_branch.strain \
            = self._notch_approximation_law.strain(
                self._lut_primary_branch.stress, self._lut_primary_branch.load)

        # ----------------
        # create look-up table (lut) for the secondary branch values, named AST in FKM nonlinear
        # Note that this time, we used twice the number of entries with the same bin width.
        index = pd.MultiIndex.from_product([np.arange(1, 2*self._number_of_bins+1), self._maximum_absolute_load.index],
                                           names = ["class_index", "node_id"])

        self._lut_secondary_branch = pd.DataFrame(0, index=index, columns=["delta_load", "delta_strain", "delta_stress"])

        # create cartesian product of class index and max load
        a = pd.DataFrame({"class_index": np.arange(1, 2*self._number_of_bins+1)})
        class_index_with_max_load = a.merge(self._maximum_absolute_load, how='cross')

        # calculate load of each bin
        delta_load = class_index_with_max_load.class_index.astype(float)/self._number_of_bins * class_index_with_max_load.max_abs_load
        delta_load.index = index
        self._lut_secondary_branch.delta_load = delta_load

        self._lut_secondary_branch.delta_stress \
            = self._notch_approximation_law.stress_secondary_branch(self._lut_secondary_branch.delta_load)

        self._lut_secondary_branch.delta_strain \
            = self._notch_approximation_law.strain_secondary_branch(
                self._lut_secondary_branch.delta_stress, self._lut_secondary_branch.delta_load)

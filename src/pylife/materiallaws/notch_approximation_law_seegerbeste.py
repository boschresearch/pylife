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

__author__ = ["Sebastian Bucher", "Benjamin Maier"]
__maintainer__ = __author__

import numpy as np
from scipy import optimize
import warnings

import pylife.materiallaws.rambgood
import pylife.materiallaws.notch_approximation_law

class SeegerBeste(pylife.materiallaws.notch_approximation_law.NotchApproximationLawBase):
    r'''Implementation of the Seeger-Beste notch approximation material relation.

    This notch approximation law is used for the P_RAJ damage parameter in the FKM
    nonlinear guideline (2019). Given an elastic-plastic stress (and strain) from a linear FE
    calculation, it derives a corresponding elastic-plastic stress (and strain).

    Note, the input stress and strain follow a linear relationship :math:`\sigma = E \cdot \epsilon`.
    The output stress and strain follow the Ramberg-Osgood relation.

    Parameters
    ----------

    E : float
        Young's Modulus
    K : float
        The strength coefficient, often also designated :math:`K'`, or ``K_prime``.
    n : float
        The strain hardening coefficient, often also designated :math:`n'`, or ``n_prime``.
    K_p : float
        The shape factor (de: Traglastformzahl)

    Notes
    -----
    The equation implemented is described in the FKM nonlinear reference, chapter 2.8.7.
    '''

    def stress(self, load, *, rtol=1e-4, tol=1e-4):
        '''Calculate the stress of the primary path in the stress-strain diagram at a given
        elastic-plastic stress (load), from a FE computation.
        This is done by solving for the root of f(sigma) in eq. 2.8-42 of FKM nonlinear.

        The secant method is used which does not rely on a derivative and has good numerical stability properties,
        but is slower than Newton's method. The algorithm is implemented in scipy for multiple values at once.
        The documentation states that this is faster for more than ~100 entries than a simple loop over the
        individual values.

        We employ the scipy function on all items in the given array at once.
        Usually, some of them fail and we recompute the value of the failed items afterwards.
        Calling the Newton method on a scalar function somehow always converges, while calling
        the Newton method with same initial conditions on the same values, but with multiple at once, fails sometimes.

        Parameters
        ----------
        load : array-like float
            The load

        Returns
        -------
        stress : array-like float
            The resulting stress
        '''
        # initial value as given by correction document to FKM nonlinear
        x0 = np.asarray(load * (1 - (1 - 1/self._K_p)/1000))

        # suppress the divergence warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            stress = optimize.newton(
                func=self._stress_implicit,
                x0=np.asarray(x0),
                args=([load]),
                full_output=True,
                rtol=rtol, tol=tol, maxiter=50
            )

            # Now, `stress` is a tuple, either
            #    (value, info_object) for scalar values,
            # or (value, converged, zero_der) for vector-valued invocation

        # only for multiple points at once, if some points diverged
        multidim = len(x0.shape) > 1 and x0.shape[1] > 1
        if multidim and not stress[1].all():
            stress = self._stress_fix_not_converged_values(stress, load, x0, rtol, tol)

        return stress[0]

    def strain(self, stress):
        '''Calculate the strain of the primary path in the stress-strain diagram at a given stress and load.
        The formula is given by eq. 2.8-39 of FKM nonlinear.
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
        '''

        if not isinstance(stress, float):
            stress = stress.astype(float)

        return self._ramberg_osgood_relation.strain(stress)

    def load(self, stress, *, rtol=1e-4, tol=1e-4):
        '''Apply the notch-approximation law "backwards", i.e., compute the linear-elastic stress (called "load" or "L" in FKM nonlinear)
        from the elastic-plastic stress as from the notch approximation.
        This backward step is needed for the pfp FKM nonlinear surface layer & roughness.

        This method is the inverse operation of "stress", i.e., ``L = load(stress(L))`` and ``S = stress(load(stress))``.

        Note that this method is only implemented for the scalar case, as the  FKM nonlinear surface layer & roughness
        also only handles the scalar case with one assessment point at once, not with entire meshes.

        Parameters
        ----------
        stress : array-like float
            The elastic-plastic stress as computed by the notch approximation

        Returns
        -------
        load : array-like float
            The resulting load or linear elastic stress.

        '''

        x0 = stress / (1 - (1 - 1/self._K_p)/1000)

        # suppress the divergence warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            load = optimize.newton(
                func=self._load_implicit,
                x0=x0,
                args=([stress]),
                rtol=rtol, tol=tol, maxiter=50
            )

        return load

    def stress_secondary_branch(self, delta_load, *, rtol=1e-4, tol=1e-4):
        '''Calculate the stress on secondary branches in the stress-strain diagram at a given
        elastic-plastic stress (load), from a FE computation.
        This is done by solving for the root of f(sigma) in eq. 2.8-43 of FKM nonlinear.

        Parameters
        ----------
        delta_load : array-like float
            The load increment of the hysteresis

        Returns
        -------
        delta_stress : array-like float
            The resulting stress increment within the hysteresis

        Todo
        ----

        In the future, we can evaluate the runtime performance and try a Newton method instead
        of the currently used secant method to speed up the computation.

        .. code::

            fprime=self._d_stress_secondary_implicit_numeric

        '''

        # initial value as given by correction document to FKM nonlinear
        delta_load = np.asarray(delta_load)
        x0 = delta_load * (1 - (1 - 1/self._K_p)/1000)

        # suppress the divergence warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            delta_stress = optimize.newton(
                func=self._stress_secondary_implicit,
                x0=x0,
                args=([delta_load]),
                full_output=True,
                rtol=rtol, tol=tol, maxiter=50
            )

            # Now, `delta_stress` is a tuple, either
            #    (value, info_object) for scalar values,
            # or (value, converged, zero_der) for vector-valued invocation

        # only for multiple points at once, if some points diverged

        multidim = len(x0.shape) > 1 and x0.shape[1] > 1
        if multidim and x0.shape[1] > 1 and not delta_stress[1].all():
            delta_stress = self._stress_secondary_fix_not_converged_values(delta_stress, delta_load, x0, rtol, tol)

        return delta_stress[0]

    def strain_secondary_branch(self, delta_stress):
        '''Calculate the strain on secondary branches in the stress-strain diagram at a given stress and load.
        The formula is given by eq. 2.8-43 of FKM nonlinear.

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
        '''

        if not isinstance(delta_stress, float):
            delta_stress = delta_stress.astype(float)

        return self._ramberg_osgood_relation.delta_strain(delta_stress)

    def load_secondary_branch(self, delta_stress, *, rtol=1e-4, tol=1e-4):
        '''Apply the notch-approximation law "backwards", i.e., compute the linear-elastic stress (called "load" or "L" in FKM nonlinear)
        from the elastic-plastic stress as from the notch approximation.
        This backward step is needed for the pfp FKM nonlinear surface layer & roughness.

        This method is the inverse operation of "stress", i.e., ``L = load(stress(L))`` and ``S = stress(load(stress))``.

        Note that this method is only implemented for the scalar case, as the  FKM nonlinear surface layer & roughness
        also only handles the scalar case with one assessment point at once, not with entire meshes.

        Parameters
        ----------
        delta_stress : array-like float
            The increment of the elastic-plastic stress as computed by the notch approximation

        Returns
        -------
        delta_load : array-like float
            The resulting load or linear elastic stress.

        '''

        x0 = delta_stress / (1 - (1 - 1/self._K_p)/1000)

        # suppress the divergence warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            delta_load = optimize.newton(
                func=self._load_secondary_implicit,
                x0=x0,
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

    def _u_term(self, stress, load):
        '''
        Compute the "u-term" from equation 2.8.40

        ``(pi/2*(L/Sigma-1/k_p-1))``

        '''
        if not isinstance(load, float):
            load = load.astype(float)
        factor = np.divide(load, stress, out=np.ones_like(load), where=stress!=0)
        return (np.pi/2)*((factor-1)/(self._K_p-1))

    def _middle_term(self, stress, load):
        '''
        Compute the middle term of euqation 2.8.42

        (2/u^2)*ln(1/cos(u))+(Sigma/L)^2-(Sigma/L)

        Note, this is only possible for

        .. code::

          1/cos(u) > 0
          <=>  0 <= u < pi/2
          <=>  0 <= L/Sigma - 1/k_p - 1 < 1
          <=>  1 <= L/Sigma - 1/k_p < 2
          <=>  1/(L/Sigma - 2) < k_p <= 1/(L/Sigma - 1)

        '''
        # convert stress value to float
        if not isinstance(stress, float):
            stress = stress.astype(float)
        factor = np.divide(stress, load, out=np.ones_like(stress), where=load!=0)

        factor1 = np.divide(2, self._u_term(stress, load)**2, out=np.ones_like(stress), where=self._u_term(stress, load)!=0)
        factor2 = np.divide(1, np.cos(self._u_term(stress, load)), out=np.ones_like(stress), where=np.cos(self._u_term(stress, load))>0)

        return (factor1)*np.log(factor2)+(factor)**2-(factor)

    def _stress_implicit(self, stress, load):
        """Compute the implicit function of the stress, f(sigma),
        defined in eq.2.8-42 of FKM nonlinear

        f(sigma) = sigma/E + (sigma/K')^(1/n') - ((2/u^2)*ln(1/cos(u))+(Sigma/L)^2-(Sigma/L)) * (L/sigma * K_p * e_star)
        """

        return self._ramberg_osgood_relation.strain(stress) / ((self._middle_term(stress, load))*(self._neuber_strain(stress, load))) - 1

    def _delta_e_star(self, delta_load):
        """Compute the plastic corrected strain term e^{\ast} from the Neuber approximation
        (eq. 2.5-43 in FKM nonlinear), for secondary branches in the stress-strain diagram
        """

        corrected_load = delta_load / self._K_p
        return self._ramberg_osgood_relation.delta_strain(corrected_load)

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

    def _u_term_secondary(self, delta_stress, delta_load):
        '''
        Compute the "u-term" from equation 2.8.45 for the secondary branch

        ``(pi/2*(delta_L/delta_Sigma-1/k_p-1))``

        '''
        if not isinstance(delta_load, float):
            delta_load = delta_load.astype(float)
        factor = np.divide(delta_load, delta_stress, out=np.ones_like(delta_load), where=delta_stress!=0)
        return (np.pi/2)*((factor-1)/(self._K_p-1))

    def _middle_term_secondary(self, delta_stress, delta_load):
        '''
        Compute the middle term of euqation 2.8.42 for the secondary branch

        ``(2/u^2)*ln(1/cos(u))+(Sigma/L)^2-(Sigma/L)``

        Note, this is only possible for
        .. code::

          1/cos(u) > 0
          <=>  0 <= u < pi/2
          <=>  0 <= delta_L/delta_Sigma - 1/k_p - 1 < 1
          <=>  1 <= delta_L/delta_Sigma - 1/k_p < 2
          <=>  1/(delta_L/delta_Sigma - 2) < k_p <= 1/(delta_L/delta_Sigma - 1)

        '''
        if not isinstance(delta_stress, float):
            delta_stress = delta_stress.astype(float)
        factor = np.divide(delta_stress, delta_load, out=np.ones_like(delta_stress), where=delta_load!=0)

        factor1 = np.divide(2, self._u_term_secondary(delta_stress, delta_load)**2, out=np.ones_like(delta_stress), where=self._u_term_secondary(delta_stress, delta_load)!=0)
        factor2 = np.divide(1, np.cos(self._u_term_secondary(delta_stress, delta_load)), out=np.ones_like(delta_stress), where=np.cos(self._u_term_secondary(delta_stress, delta_load))>0)

        return (factor1)*np.log(factor2)+(factor)**2-(factor)

    def _stress_secondary_implicit(self, delta_stress, delta_load):
        """Compute the implicit function of the stress, f(sigma), defined in eq.2.8-43 of FKM nonlinear.
        There are in principal two different approaches:

        * find root of ``f(sigma)-epsilon``
        * find root of ``f(sigma)/epsilon - 1``

        The second approach is numerically more stable and is used here.
        The code for the first approach would be:

        .. code::

            return self._ramberg_osgood_relation.delta_strain(delta_stress) \
                - (self._middle_term_secondary(delta_stress, delta_load))*(self._neuber_strain_secondary(delta_stress, delta_load))
        """

        return self._ramberg_osgood_relation.delta_strain(delta_stress) \
            / ((self._middle_term_secondary(delta_stress, delta_load))*(self._neuber_strain_secondary(delta_stress, delta_load))) - 1

    def _d_stress_secondary_implicit_numeric(self, delta_stress, delta_load):
        """Compute the first derivative of self._stress_secondary_implicit
        df/dsigma
        """

        h = 1e-4
        return (self._stress_secondary_implicit(delta_stress+h, delta_load) - self._stress_secondary_implicit(delta_stress-h, delta_load)) / (2*h)

    def _load_implicit(self, load, stress):
         """Compute the implicit function of the stress, f(sigma),
         as a function of the load,
         defined in eq.2.8-42 of FKM nonlinear.
         This is needed to apply the notch approximation law "backwards", i.e.,
         to get from stress back to load. This is required for the FKM nonlinear roughness & surface layer.
         """

         return self._stress_implicit(stress, load)

    def _load_secondary_implicit(self, delta_load, delta_stress):
        """Compute the implicit function of the stress, f(Δsigma),
        as a function of the load,
        defined in eq.2.8-43 of FKM nonlinear.
        This is needed to apply the notch approximation law "backwards", i.e.,
        to get from stress back to load. This is required for the FKM nonlinear roughness & surface layer.
        """

        return self._stress_secondary_implicit(delta_stress, delta_load)

    def _stress_fix_not_converged_values(self, stress, load, x0, rtol, tol):
        '''For the values that did not converge in the previous vectorized call to optimize.newton,
        call optimize.newton again on the scalar value. This usually finds the correct solution.'''

        indices_diverged = np.where(~stress[1].all(axis=1))[0]
        x0_array = np.asarray(x0)
        load_array = np.asarray(load)

        # recompute previously failed points individually
        for index_diverged in indices_diverged:
            x0_diverged = x0_array[index_diverged]
            load_diverged = load_array[index_diverged]
            result = optimize.newton(
                func=self._stress_implicit,
                x0=np.asarray(x0_diverged),
                args=([load_diverged]),
                full_output=True,
                rtol=rtol, tol=tol, maxiter=50
            )

            if result.converged.all():
                stress[0][index_diverged] = result[0]
        return stress

    def _stress_secondary_fix_not_converged_values(self, delta_stress, delta_load, x0, rtol, tol):
        '''For the values that did not converge in the previous vectorized call to optimize.newton,
        call optimize.newton again on the scalar value. This usually finds the correct solution.'''

        indices_diverged = np.where(~delta_stress[1].all(axis=1))[0]
        x0_array = np.asarray(x0)
        delta_load_array = np.asarray(delta_load)

        # recompute previously failed points individually
        for index_diverged in indices_diverged:
            x0_diverged = x0_array[index_diverged, 0]
            delta_load_diverged = delta_load_array[index_diverged, 0]
            result = optimize.newton(
                func=self._stress_secondary_implicit,
                x0=np.asarray(x0_diverged),
                args=([delta_load_diverged]),
                full_output=True,
                rtol=rtol, tol=tol, maxiter=50
            )
            if result[1].converged:
                delta_stress[0][index_diverged] = result[0]
        return delta_stress

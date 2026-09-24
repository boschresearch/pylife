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

r"""Provide the Seeger-Beste notch approximation law.

The module supports FKM nonlinear assessments by converting linear-elastic
local loads from finite-element calculations to elastic-plastic local
stress-strain paths using the Seeger-Beste approximation.
"""

__author__ = ["Sebastian Bucher", "Benjamin Maier"]
__maintainer__ = __author__

import numpy as np
from scipy import optimize
import warnings

import pylife.materiallaws.rambgood
import pylife.materiallaws.notch_approximation_law

class SeegerBeste(pylife.materiallaws.notch_approximation_law.NotchApproximationLawBase):
    """Apply the Seeger-Beste notch approximation law.

    Use this law for the P_RAJ damage parameter in the FKM nonlinear assessment.
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
    K_p : float
        Plastic shape factor, dimensionless.

    Notes
    -----
    The implementation follows section 2.8.7 of the FKM guideline nonlinear
    [FKM-SeegerBeste]_ and solves the Seeger-Beste implicit equations for
    primary and secondary paths.

    References
    ----------
    .. [FKM-SeegerBeste] Forschungskuratorium Maschinenbau,
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

        if not isinstance(stress, float):
            stress = stress.astype(float)

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

        if not isinstance(delta_stress, float):
            delta_stress = delta_stress.astype(float)

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
        """Calculate the Neuber-corrected primary strain term.
        """

        corrected_load = load / self._K_p
        return self._ramberg_osgood_relation.strain(corrected_load)

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

    def _u_term(self, stress, load):
        """Calculate the Seeger-Beste primary geometry term.
        """
        if not isinstance(load, float):
            load = load.astype(float)
        factor = np.divide(load, stress, out=np.ones_like(load), where=stress!=0)
        return (np.pi/2)*((factor-1)/(self._K_p-1))

    def _middle_term(self, stress, load):
        """Calculate the Seeger-Beste primary correction factor.
        """
        # convert stress value to float
        if not isinstance(stress, float):
            stress = stress.astype(float)
        factor = np.divide(stress, load, out=np.ones_like(stress), where=load!=0)

        factor1 = np.divide(2, self._u_term(stress, load)**2, out=np.ones_like(stress), where=self._u_term(stress, load)!=0)
        factor2 = np.divide(1, np.cos(self._u_term(stress, load)), out=np.ones_like(stress), where=np.cos(self._u_term(stress, load))>0)

        return (factor1)*np.log(factor2)+(factor)**2-(factor)

    def _stress_implicit(self, stress, load):
        """Calculate the primary implicit stress residual.
        """

        return self._ramberg_osgood_relation.strain(stress) / ((self._middle_term(stress, load))*(self._neuber_strain(stress, load))) - 1

    def _delta_e_star(self, delta_load):
        """Calculate the Neuber-corrected secondary strain term.
        """

        corrected_load = delta_load / self._K_p
        return self._ramberg_osgood_relation.delta_strain(corrected_load)

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

    def _u_term_secondary(self, delta_stress, delta_load):
        """Calculate the Seeger-Beste secondary geometry term.
        """
        if not isinstance(delta_load, float):
            delta_load = delta_load.astype(float)
        factor = np.divide(delta_load, delta_stress, out=np.ones_like(delta_load), where=delta_stress!=0)
        return (np.pi/2)*((factor-1)/(self._K_p-1))

    def _middle_term_secondary(self, delta_stress, delta_load):
        """Calculate the Seeger-Beste secondary correction factor.
        """
        if not isinstance(delta_stress, float):
            delta_stress = delta_stress.astype(float)
        factor = np.divide(delta_stress, delta_load, out=np.ones_like(delta_stress), where=delta_load!=0)

        factor1 = np.divide(2, self._u_term_secondary(delta_stress, delta_load)**2, out=np.ones_like(delta_stress), where=self._u_term_secondary(delta_stress, delta_load)!=0)
        factor2 = np.divide(1, np.cos(self._u_term_secondary(delta_stress, delta_load)), out=np.ones_like(delta_stress), where=np.cos(self._u_term_secondary(delta_stress, delta_load))>0)

        return (factor1)*np.log(factor2)+(factor)**2-(factor)

    def _stress_secondary_implicit(self, delta_stress, delta_load):
        """Calculate the secondary implicit stress residual.
        """

        return self._ramberg_osgood_relation.delta_strain(delta_stress) \
            / ((self._middle_term_secondary(delta_stress, delta_load))*(self._neuber_strain_secondary(delta_stress, delta_load))) - 1

    def _d_stress_secondary_implicit_numeric(self, delta_stress, delta_load):
        """Calculate the numerical derivative of the secondary stress residual.
        """

        h = 1e-4
        return (self._stress_secondary_implicit(delta_stress+h, delta_load) - self._stress_secondary_implicit(delta_stress-h, delta_load)) / (2*h)

    def _load_implicit(self, load, stress):
         """Calculate the primary implicit load residual.
         """

         return self._stress_implicit(stress, load)

    def _load_secondary_implicit(self, delta_load, delta_stress):
        """Calculate the secondary implicit load residual.
        """

        return self._stress_secondary_implicit(delta_stress, delta_load)

    def _stress_fix_not_converged_values(self, stress, load, x0, rtol, tol):
        """Recompute non-converged primary stress values scalar-wise.
        """

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
        """Recompute non-converged secondary stress values scalar-wise.
        """

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

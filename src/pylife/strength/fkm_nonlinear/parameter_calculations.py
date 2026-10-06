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

r"""Derive FKM nonlinear assessment parameters from user inputs.

This module implements the material-group dependent formulas used by the
FKM nonlinear guideline for cyclic material data, material and component
Wöhler curves, roughness factors, nonlocal support factors, and statistical
failure-probability factors.  The functions take a
:class:`pandas.Series` of assessment parameters, add derived keys to a copy,
and return that copy for use by
``pylife.strength.fkm_nonlinear.assessment_nonlinear_standard``.
"""
__author__ = "Benjamin Maier"
__maintainer__ = __author__

import numpy as np
import pandas as pd
import scipy

# pylife
import pylife
import pylife.vmap
import pylife.stress.equistress
import pylife.strength.fkm_load_distribution
import pylife.strength.damage_parameter
import pylife.strength.woehler_fkm_nonlinear
import pylife.materiallaws
import pylife.stress.rainflow
import pylife.stress.rainflow.recorders
import pylife.stress.rainflow.fkm_nonlinear
import pylife.materiallaws.notch_approximation_law
from pylife.strength.fkm_nonlinear.constants import FKMNLConstants

''' Collection of functions for computational proof of the strength
    for machine elements considering their non-linear material deformation
    (FKM non-linear guideline 2019)
'''


def calculate_cyclic_assessment_parameters(assessment_parameters_):
    r"""Calculate cyclic Ramberg-Osgood material parameters.

    Parameters
    ----------
    assessment_parameters_ : pandas.Series
        Assessment parameters containing ``MatGroupFKM`` and ultimate tensile
        strength ``R_m`` in MPa.

    Returns
    -------
    pandas.Series
        Copy of ``assessment_parameters_`` with Young's modulus ``E`` in MPa,
        cyclic hardening exponent ``n_prime``, and cyclic hardening coefficient
        ``K_prime`` in MPa added.

    Notes
    -----
    Implements FKM nonlinear guideline Section 2.5.3, in particular equation
    (2.5-13).  ``E`` and ``n_prime`` are taken from the material group table,
    while ``K_prime`` is estimated from ``R_m``.
    """
    assessment_parameters = assessment_parameters_.copy()
    assert "R_m" in assessment_parameters

    # select set of constants according to given material group
    constants = FKMNLConstants().for_material_group(assessment_parameters)

    # use constant values for n' and E
    assessment_parameters["n_prime"] = constants.n_prime
    assessment_parameters["E"] = constants.E

    # for FKM nonlinear, R_m is used to estimate material data
    # compute K' according to eq. (2.5-13)
    assessment_parameters["K_prime"] = constants.a_sigma * assessment_parameters.R_m ** constants.b_sigma \
        / (np.minimum(constants.epsilon_grenz, constants.a_epsilon * assessment_parameters.R_m ** constants.b_epsilon)) \
            ** constants.n_prime

    return assessment_parameters


def calculate_material_woehler_parameters_P_RAM(assessment_parameters_):
    r"""Calculate material Wöhler parameters for the ``P_RAM`` path.

    Parameters
    ----------
    assessment_parameters_ : pandas.Series
        Assessment parameters containing ``MatGroupFKM``, ultimate tensile
        strength ``R_m`` in MPa, and failure probability ``P_A`` as a
        dimensionless probability.

    Returns
    -------
    pandas.Series
        Copy of ``assessment_parameters_`` with ``P_RAM_Z_WS`` at
        ``N = 1e3`` cycles, fatigue limit ``P_RAM_D_WS``, first slope ``d_1``,
        and second slope ``d_2`` of the material damage Wöhler curve added.

    Notes
    -----
    Implements FKM nonlinear guideline Section 2.5.5, equations (2.5-22) and
    (2.5-23).  For ``P_A`` values other than ``0.5`` the curve is shifted by
    the material-group dependent 2.5 % factor from the guideline table.
    """
    assessment_parameters = assessment_parameters_.copy()

    assert "P_A" in assessment_parameters
    assert "R_m" in assessment_parameters

    # select set of constants according to given material group
    constants = FKMNLConstants().for_material_group(assessment_parameters)

    # add an empty "notes" entry in assessment_parameters
    if "notes" not in assessment_parameters:
        assessment_parameters["notes"] = ""

    # for standard FKM nonlinear, R_m is used to estimate material data

    # computations for P_RAM
    # compute sampling point "Z" according to eq. (2.5-22)
    assessment_parameters["P_RAM_Z_WS"] = constants.a_PZ_RAM \
        * assessment_parameters.R_m ** constants.b_PZ_RAM

    # compute sampling point "D" according to eq. (2.5-23)
    assessment_parameters["P_RAM_D_WS"] = constants.a_PD_RAM \
        * assessment_parameters.R_m ** constants.b_PD_RAM

    # depending on P_A, add the factor f_2.5%, as described in eqs. (2.5-22), (2.5-23)
    if np.isclose(assessment_parameters.P_A, 0.5):

        # add a note
        assessment_parameters["notes"] += "P_A is 0.5: no scaling of P_RAM woehler curve to 2.5%.\n"

    else:
        # rescale woehler curve with f_2.5%, eqs. (2.5-22), (2.5-23)
        assessment_parameters.P_RAM_Z_WS = constants.f_25percent_material_woehler_RAM * assessment_parameters.P_RAM_Z_WS
        assessment_parameters.P_RAM_D_WS = constants.f_25percent_material_woehler_RAM * assessment_parameters.P_RAM_D_WS

        # add a note
        assessment_parameters["notes"] += f"P_A not 0.5 (but {assessment_parameters.P_A}): scale P_RAM woehler curve by f_2.5% = {constants.f_25percent_material_woehler_RAM}.\n"

    # use constant values for d_1 and d_2
    assessment_parameters["d_1"] = constants.d_1
    assessment_parameters["d_2"] = constants.d_2

    return assessment_parameters


def calculate_material_woehler_parameters_P_RAJ(assessment_parameters_):
    r"""Calculate material Wöhler parameters for the ``P_RAJ`` path.

    Parameters
    ----------
    assessment_parameters_ : pandas.Series
        Assessment parameters containing ``MatGroupFKM``, ultimate tensile
        strength ``R_m`` in MPa, and failure probability ``P_A`` as a
        dimensionless probability.

    Returns
    -------
    pandas.Series
        Copy of ``assessment_parameters_`` with ``P_RAJ_Z_WS`` at ``N = 1``
        cycle, fatigue limit ``P_RAJ_D_WS``, and material curve slope
        ``d_RAJ`` added.

    Notes
    -----
    Implements FKM nonlinear guideline Sections 2.8.5 and 2.9.4, equations
    (2.8-20), (2.8-21), and (2.9-12).  The implementation uses the corrected
    ``P_RAJ,D,WS`` relation where the printed equation (2.9-13) is ambiguous.
    """
    assessment_parameters = assessment_parameters_.copy()

    assert "P_A" in assessment_parameters
    assert "R_m" in assessment_parameters

    # select set of constants according to given material group
    constants = FKMNLConstants().for_material_group(assessment_parameters)

    # add an empty "notes" entry in assessment_parameters
    if "notes" not in assessment_parameters:
        assessment_parameters["notes"] = ""

    # for standard FKM nonlinear, R_m is used to estimate material data

    # computations for P_RAJ
    # compute first sampling point for N=1 according to eq. (2.8-20), (2.9-12)
    assessment_parameters["P_RAJ_Z_WS"] = constants.a_PZ_RAJ \
        * assessment_parameters.R_m ** constants.b_PZ_RAJ

    # compute second sampling point, the infinite life threshold according to eq. (2.8-21), note the error in (2.9-13) (should be P_RAJ,D,WS)
    assessment_parameters["P_RAJ_D_WS"] = constants.a_PD_RAJ \
        * assessment_parameters.R_m ** constants.b_PD_RAJ


    # depending on P_A, add the factor f_2.5%, as described in eqs. (2.5-22), (2.5-23)
    if np.isclose(assessment_parameters.P_A, 0.5):

        # add a note
        assessment_parameters["notes"] += "P_A is 0.5: no scaling of P_RAJ woehler curve to 2.5%.\n"

    else:
        # rescale woehler curve with f_2.5%, eqs. (2.8-20), (2.8-21)
        assessment_parameters.P_RAJ_Z_WS = constants.f_25percent_material_woehler_RAJ * assessment_parameters.P_RAJ_Z_WS
        assessment_parameters.P_RAJ_D_WS = constants.f_25percent_material_woehler_RAJ * assessment_parameters.P_RAJ_D_WS

        # add a note
        assessment_parameters["notes"] += f"P_A not 0.5 (but {assessment_parameters.P_A}): scale P_RAJ woehler curve by f_2.5% = {constants.f_25percent_material_woehler_RAJ}.\n"

    # use constant value for d
    assessment_parameters["d_RAJ"] = constants.d_RAJ

    return assessment_parameters


def calculate_roughness_material_woehler_parameters_P_RAM(assessment_parameters_):
    r"""Calculate roughness-adjusted material parameters for ``P_RAM``.

    Parameters
    ----------
    assessment_parameters_ : pandas.Series
        Assessment parameters containing ``P_RAM_Z_WS``, ``P_RAM_D_WS``,
        material slope ``d_2``, and roughness factor ``K_RP``.

    Returns
    -------
    pandas.Series
        Copy of ``assessment_parameters_`` with roughness-adjusted fatigue limit
        ``P_RAM_D_WS_rau``, adjusted second slope ``d2_RAM_rau``, and the
        check value ``d2_RAM_rau_alternative`` added.

    Notes
    -----
    Implements the roughness and surface-layer extension to FKM nonlinear
    Section 2.5.6.  The fatigue-limit ordinate is multiplied by ``K_RP`` and
    the second finite-life slope is recomputed so that the original transition
    cycle number remains unchanged.
    """
    assessment_parameters = assessment_parameters_.copy()

    assessment_parameters["P_RAM_D_WS_rau"] = assessment_parameters["P_RAM_D_WS"] * assessment_parameters["K_RP"]

    # log(f(N)) = d_2 * log(N-1e3) + log(P_RAM_Z_WS)
    # log(f(N_D)) = d_2 * [log(N_D)-log(1e3)] + log(P_RAM_Z_WS) = log(P_RAM_D_WS)
    #  => (log(P_RAM_D_WS) - log(P_RAM_Z_WS)) / d_2 + log(1e3) = log(N_D)
    #  => N_D = 1e3 * (P_RAM_D_WS/P_RAM_Z_WS)**(1/d_2)

    # d2_RAM_rau = log(P_RAM_D_WS_rau / P_RAM_Z_WS) / log(N_D/1e3)
    # d2_RAM_rau = d_2 * log(P_RAM_D_WS_rau / P_RAM_Z_WS) / log(P_RAM_D_WS/P_RAM_Z_WS)

    assessment_parameters["d2_RAM_rau"] = assessment_parameters["d_2"] \
        * (np.log(assessment_parameters["P_RAM_Z_WS"])-np.log(assessment_parameters["P_RAM_D_WS_rau"])) \
        / (np.log(assessment_parameters["P_RAM_Z_WS"])-np.log(assessment_parameters["P_RAM_D_WS"]))

    # alternative calculation via N_D
    # (N_D: Eckschwingspielzahl zur Dauerfestigkeit)
    # this equation does the same as fatigue_life_limit in woehler_fkm_nonlinear
    N_D = 1e3 * (assessment_parameters["P_RAM_D_WS"] / assessment_parameters["P_RAM_Z_WS"]) ** (1/assessment_parameters["d_2"])

    assessment_parameters["d2_RAM_rau_alternative"] = np.log(assessment_parameters["P_RAM_D_WS_rau"] \
        / assessment_parameters["P_RAM_Z_WS"]) / np.log(N_D/1e3)

    return assessment_parameters


def calculate_roughness_material_woehler_parameters_P_RAJ(assessment_parameters_):
    r"""Calculate roughness-adjusted material parameters for ``P_RAJ``.

    Parameters
    ----------
    assessment_parameters_ : pandas.Series
        Assessment parameters containing ``P_RAJ_Z_WS``, ``P_RAJ_D_WS``,
        material slope ``d_RAJ``, and roughness factor ``K_RP``.

    Returns
    -------
    pandas.Series
        Copy of ``assessment_parameters_`` with ``P_RAJ_Z_1e3``,
        roughness-adjusted fatigue limit ``P_RAJ_D_WS_rau``, adjusted slope
        ``d_RAJ_2_rau``, and the check value ``d_RAJ_2_rau_alternative`` added.

    Notes
    -----
    Implements the roughness and surface-layer extension to FKM nonlinear
    Sections 2.8.6 and 2.9.6.  The ``P_RAJ`` roughness correction acts with
    ``K_RP ** 2`` because ``P_RAJ`` is an energy-like damage parameter.
    """
    assessment_parameters = assessment_parameters_.copy()

    assessment_parameters["P_RAJ_Z_1e3"] = assessment_parameters["P_RAJ_Z_WS"]*np.power(1e3, assessment_parameters["d_RAJ"])
    assessment_parameters["P_RAJ_D_WS_rau"] = assessment_parameters["P_RAJ_D_WS"] * assessment_parameters["K_RP"]**2.

    assessment_parameters["d_RAJ_2_rau"] = assessment_parameters["d_RAJ"] \
        * (np.log(assessment_parameters["P_RAJ_Z_1e3"])-np.log(assessment_parameters["P_RAJ_D_WS_rau"])) \
        / (np.log(assessment_parameters["P_RAJ_Z_1e3"])-np.log(assessment_parameters["P_RAJ_D_WS"]))

    # alternative calculation via N_D
    # N_D: Eckschwingspielzahl zur Dauerfestigkeit
    # this equation does the same as fatigue_life_limit in woehler_fkm_nonlinear
    N_D = (assessment_parameters["P_RAJ_D_WS"] / assessment_parameters["P_RAJ_Z_WS"]) ** (1/assessment_parameters["d_RAJ"])

    # P_RAJ_D*K**2=P_RAJ_Z_1e3 * (N_D/1e3)**r_rau
    assessment_parameters["d_RAJ_2_rau_alternative"] = np.log(assessment_parameters["P_RAJ_D_WS_rau"] \
        / assessment_parameters["P_RAJ_Z_1e3"]) / np.log(N_D/1e3)

    return assessment_parameters


def calculate_roughness_component_woehler_parameters_P_RAM(assessment_parameters_, include_n_P):
    r"""Calculate roughness-adjusted component parameters for ``P_RAM``.

    Parameters
    ----------
    assessment_parameters_ : pandas.Series
        Assessment parameters containing ``gamma_M_RAM``, ``P_RAM_Z_WS``,
        ``P_RAM_D_WS_rau``, and optionally nonlocal support factor ``n_P``.
    include_n_P : bool
        Whether to include ``n_P`` in the component curve shift.  Set to
        ``True`` for the surface point with notch support and ``False`` when the
        roughness extension shall omit that support factor.

    Returns
    -------
    pandas.Series
        Copy of ``assessment_parameters_`` with component curve knee
        ``P_RAM_Z`` and component fatigue limit ``P_RAM_D`` added.

    Notes
    -----
    Implements FKM nonlinear Section 2.5.6 together with the roughness and
    surface-layer extension.  The material curve ordinates are divided by the
    material safety factor ``gamma_M_RAM`` and optionally multiplied by ``n_P``.
    """
    assessment_parameters = assessment_parameters_.copy()

    assert "gamma_M_RAM" in assessment_parameters
    assert "P_RAM_Z_WS" in assessment_parameters
    assert "P_RAM_D_WS_rau" in assessment_parameters

    # set n_P only if it should be added (for the surface point in FKM nonlinear roughness & surface layer)
    n_P = 1
    if include_n_P:
        assert "n_P" in assessment_parameters
        n_P = assessment_parameters.n_P

    # calculate first knee point of component Woehler curve, eq. (2.5-25) in the FKM nonlinear guideline without roughness
    assessment_parameters["P_RAM_Z"] = n_P / assessment_parameters.gamma_M_RAM * assessment_parameters.P_RAM_Z_WS

    # calculate fatigue strength limit of the component, i.e., the P_RAM value below which we have infinite life, rhs of eq. (2.6-88)
    assessment_parameters["P_RAM_D"] = n_P / assessment_parameters.gamma_M_RAM * assessment_parameters.P_RAM_D_WS_rau

    return assessment_parameters


def calculate_roughness_component_woehler_parameters_P_RAJ(assessment_parameters_, include_n_P):
    r"""Calculate roughness-adjusted component parameters for ``P_RAJ``.

    Parameters
    ----------
    assessment_parameters_ : pandas.Series
        Assessment parameters containing ``gamma_M_RAJ``, ``P_RAJ_Z_WS``,
        ``P_RAJ_D_WS_rau``, ``P_RAJ_Z_1e3``, and optionally nonlocal support
        factor ``n_P``.
    include_n_P : bool
        Whether to include ``n_P`` in the component curve shift.  Set to
        ``True`` for the surface point with notch support and ``False`` when the
        roughness extension shall omit that support factor.

    Returns
    -------
    pandas.Series
        Copy of ``assessment_parameters_`` with ``P_RAJ_Z``, shifted
        ``P_RAJ_Z_1e3``, initial fatigue limit ``P_RAJ_D_0``, and component
        fatigue limit ``P_RAJ_D`` added.

    Notes
    -----
    Implements FKM nonlinear Sections 2.8.6 and 2.9.6 together with the
    roughness and surface-layer extension.  ``n_P`` enters squared for
    ``P_RAJ`` because this damage parameter is energy based.
    """
    assessment_parameters = assessment_parameters_.copy()

    assert "gamma_M_RAJ" in assessment_parameters
    assert "P_RAJ_Z_WS" in assessment_parameters
    assert "P_RAJ_D_WS_rau" in assessment_parameters

    # set n_P only if it should be added (for the surface point in FKM nonlinear roughness & surface layer)
    n_P = 1
    if include_n_P:
        assert "n_P" in assessment_parameters
        n_P = assessment_parameters.n_P

    # calculations for P_RAJ of component Woehler curve
    # eq. (2.8-23), (2.9-25)
    assessment_parameters["P_RAJ_Z"] = n_P**2 / assessment_parameters.gamma_M_RAJ * assessment_parameters.P_RAJ_Z_WS

    # also shift the knee point of the roughness P_RAJ woehler curve
    assessment_parameters["P_RAJ_Z_1e3"] = n_P**2 / assessment_parameters.gamma_M_RAJ * assessment_parameters["P_RAJ_Z_1e3"]

    # eq. (2.8-24), (2.9-26). Note that there is also eq. 2.9-13, but this is errorneous and not relevant here.
    assessment_parameters["P_RAJ_D_0"] = n_P**2 / assessment_parameters.gamma_M_RAJ * assessment_parameters.P_RAJ_D_WS_rau

    # eq. (2.8-25), (2.9-27)
    assessment_parameters["P_RAJ_D"] = assessment_parameters.P_RAJ_D_0

    return assessment_parameters


def calculate_nonlocal_parameters(assessment_parameters_):
    r"""Calculate nonlocal notch support factors.

    Parameters
    ----------
    assessment_parameters_ : pandas.Series
        Assessment parameters containing ``MatGroupFKM``, reference highly
        stressed surface ``A_ref`` in mm², component highly stressed surface
        ``A_sigma`` in mm², relative stress gradient ``G`` in 1/mm, and ultimate
        tensile strength ``R_m`` in MPa.

    Returns
    -------
    pandas.Series
        Copy of ``assessment_parameters_`` with statistical support factor
        ``n_st``, unclipped fracture-mechanical support factor ``n_bm_``,
        clipped fracture-mechanical support factor ``n_bm``, and total support
        factor ``n_P`` added.

    Notes
    -----
    Implements FKM nonlinear Section 2.5.6.1, equations (2.5-27) to (2.5-32).
    The same factors are used by the ``P_RAM`` and ``P_RAJ`` assessment paths;
    for ``P_RAJ`` the corresponding formulas are stated in Section 2.8.6.1.
    """
    assessment_parameters = assessment_parameters_.copy()

    assert "A_ref" in assessment_parameters
    assert "A_sigma" in assessment_parameters
    assert "G" in assessment_parameters
    assert "R_m" in assessment_parameters

    # select set of constants according to given material group
    constants = FKMNLConstants().for_material_group(assessment_parameters)

    # calculate statistic coefficient, eq. (2.5-28)
    assessment_parameters["n_st"] = (assessment_parameters.A_ref / assessment_parameters.A_sigma) \
        ** (1 / constants.k_st)

    # eq. (2.5-32)
    k_ = 5 * assessment_parameters.n_st + assessment_parameters.R_m / constants.R_m_bm \
        * np.sqrt((7.5 + np.sqrt(assessment_parameters.G)) / (1 + 0.2*np.sqrt(assessment_parameters.G)))

    # eq. (2.5-31)
    assessment_parameters["n_bm_"] = (5 + np.sqrt(assessment_parameters.G)) / k_

    # eq. (2.5-30)
    assessment_parameters["n_bm"] = np.maximum(assessment_parameters.n_bm_, 1)

    # calculate total coefficient, eq. (2.5-27)
    assessment_parameters["n_P"] =  assessment_parameters.n_bm * assessment_parameters.n_st

    return assessment_parameters


def calculate_roughness_parameter(assessment_parameters_):
    r"""Calculate the roughness factor ``K_RP``.

    Parameters
    ----------
    assessment_parameters_ : pandas.Series
        Assessment parameters containing ``MatGroupFKM`` and ultimate tensile
        strength ``R_m`` in MPa.  If ``K_RP`` is absent, roughness ``R_z`` in µm
        is also required.

    Returns
    -------
    pandas.Series
        Copy of ``assessment_parameters_`` with roughness factor ``K_RP`` added,
        or the unchanged copy when ``K_RP`` was already supplied.

    Notes
    -----
    Implements FKM nonlinear Section 2.5.6.2, equation (2.5-37).  For the
    ``P_RAJ`` path the corresponding formulas are stated in Section 2.8.6.2.
    A roughness of ``R_z <= 1`` µm gives ``K_RP = 1``.
    """
    assessment_parameters = assessment_parameters_.copy()

    # if K_RP is already set (e.g., manually set to 1), do nothing
    if "K_RP" in assessment_parameters:
        print(f"The parameter `K_RP` is already set to {assessment_parameters.K_RP}, not using the FKM formula.")
        return assessment_parameters

    assert "R_m" in assessment_parameters
    assert "R_z" in assessment_parameters

    # select set of constants according to given material group
    constants = FKMNLConstants().for_material_group(assessment_parameters)

    # calculate roughness factor, eq. (2.5-37)
    if assessment_parameters.R_z > 1:
        assessment_parameters["K_RP"] = (1 - constants.a_RP * np.log10(assessment_parameters.R_z) \
            * np.log10(2 * assessment_parameters.R_m / constants.R_m_N_min)) ** constants.b_RP

    else:
        assessment_parameters["K_RP"] = 1.

    return assessment_parameters


def compute_beta(P_A):
    r"""Calculate the reliability index ``beta`` from a failure probability.

    Parameters
    ----------
    P_A : float
        Failure probability for the assessment as a dimensionless probability.

    Returns
    -------
    float
        Reliability index ``beta`` corresponding to ``P_A`` for a standard
        normal distribution.

    Raises
    ------
    RuntimeError
        Raised if the numerical root search does not converge.

    Notes
    -----
    The FKM nonlinear guideline uses ``beta`` in the material safety factor but
    does not provide this conversion formula.  This helper solves
    ``Phi(-beta) = P_A`` for the standard normal distribution.

    Examples
    --------
    >>> from pylife.strength.fkm_nonlinear.parameter_calculations import compute_beta
    >>> round(float(abs(compute_beta(0.5))), 6)
    0.0
    """
    sigma = 1
    result = scipy.optimize.root(lambda x: abs(scipy.stats.norm.cdf(x, 0, sigma)-P_A), x0=-0.6, tol=1e-10)

    if not result.success:
        raise RuntimeError(f"Could not compute the value of beta for P_A={P_A}, "
                           "the optimizer did not find a solution.")

    return -result.x[0] / sigma


def calculate_failure_probability_factor_P_RAM(assessment_parameters_):
    r"""Calculate the material safety factor for ``P_RAM``.

    Parameters
    ----------
    assessment_parameters_ : pandas.Series
        Assessment parameters containing failure probability ``P_A`` or an
        already computed reliability index ``beta``.  ``P_A = 0.5`` disables the
        safety-factor shift and yields ``gamma_M_RAM = 1``.

    Returns
    -------
    pandas.Series
        Copy of ``assessment_parameters_`` with reliability index ``beta`` when
        needed and material safety factor ``gamma_M_RAM`` added.

    Notes
    -----
    Implements FKM nonlinear Section 2.5.6.3, equation (2.5-38).  The guideline
    clips ``gamma_M_RAM`` to at least ``1.1`` except for the explicit
    ``P_A = 0.5`` no-statistics case used for experiment-like assessments.
    """
    assessment_parameters = assessment_parameters_.copy()

    if "beta" not in assessment_parameters:
        assert "P_A" in assessment_parameters

        P_A = assessment_parameters.P_A
        assert P_A > 0

        assessment_parameters["beta"] = compute_beta(assessment_parameters.P_A)

    if "beta" in assessment_parameters:
        # eq. (2.5-38)
        assessment_parameters["gamma_M_RAM"] = np.max([10**((0.8*assessment_parameters.beta - 2)*0.08), 1.1])

    # set to 1 for P_A = 0.5
    if np.isclose(assessment_parameters.P_A, 0.5):
        assessment_parameters["gamma_M_RAM"] = 1

    return assessment_parameters


def calculate_failure_probability_factor_P_RAJ(assessment_parameters_):
    r"""Calculate the material safety factor for ``P_RAJ``.

    Parameters
    ----------
    assessment_parameters_ : pandas.Series
        Assessment parameters containing failure probability ``P_A`` or an
        already computed reliability index ``beta``.  ``P_A = 0.5`` disables the
        safety-factor shift and yields ``gamma_M_RAJ = 1``.

    Returns
    -------
    pandas.Series
        Copy of ``assessment_parameters_`` with reliability index ``beta`` when
        needed and material safety factor ``gamma_M_RAJ`` added.

    Notes
    -----
    Implements FKM nonlinear Section 2.8.6.3, equation (2.8-38).  The guideline
    clips ``gamma_M_RAJ`` to at least ``1.2`` except for the explicit
    ``P_A = 0.5`` no-statistics case used for experiment-like assessments.
    """
    assessment_parameters = assessment_parameters_.copy()

    if "beta" not in assessment_parameters:
        assert "P_A" in assessment_parameters

        P_A = assessment_parameters.P_A
        assert P_A > 0

        assessment_parameters["beta"] = compute_beta(assessment_parameters.P_A)

    if "beta" in assessment_parameters:
        # eq. (2.8-38)
        assessment_parameters["gamma_M_RAJ"] = np.max([10**((0.8*assessment_parameters.beta - 2)*0.155), 1.2])

    # set to 1 for P_A = 0.5
    if np.isclose(assessment_parameters.P_A, 0.5):
        assessment_parameters["gamma_M_RAJ"] = 1

    return assessment_parameters


def calculate_component_woehler_parameters_P_RAM(assessment_parameters_):
    r"""Calculate component Wöhler parameters for the ``P_RAM`` path.

    Parameters
    ----------
    assessment_parameters_ : pandas.Series
        Assessment parameters containing ``gamma_M_RAM``, support factor
        ``n_P``, roughness factor ``K_RP``, material knee ``P_RAM_Z_WS``, and
        material fatigue limit ``P_RAM_D_WS``.

    Returns
    -------
    pandas.Series
        Copy of ``assessment_parameters_`` with component shift factor
        ``f_RAM``, component knee ``P_RAM_Z``, and component fatigue limit
        ``P_RAM_D`` added.

    Notes
    -----
    Implements FKM nonlinear Section 2.5.6, equations (2.5-24) and (2.5-25),
    and the fatigue-limit ordinate used by equation (2.6-88).  ``f_RAM`` maps
    the material Wöhler curve to the assessed component.
    """
    assessment_parameters = assessment_parameters_.copy()

    assert "gamma_M_RAM" in assessment_parameters
    assert "n_P" in assessment_parameters
    assert "K_RP" in assessment_parameters
    assert "P_RAM_Z_WS" in assessment_parameters

    # eq. (2.5-24)
    assessment_parameters["f_RAM"] = assessment_parameters.gamma_M_RAM / (assessment_parameters.n_P * assessment_parameters.K_RP)

    # calculate knee point of component Woehler curve, eq. (2.5-25)
    assessment_parameters["P_RAM_Z"] = 1 / assessment_parameters.f_RAM * assessment_parameters.P_RAM_Z_WS

    # calculate fatigue strength limit of the component, i.e., the P_RAM value below which we have infinite life, rhs of eq. (2.6-88)
    assessment_parameters["P_RAM_D"] = 1 / assessment_parameters.f_RAM * assessment_parameters.P_RAM_D_WS

    return assessment_parameters


def calculate_component_woehler_parameters_P_RAJ(assessment_parameters_):
    r"""Calculate component Wöhler parameters for the ``P_RAJ`` path.

    Parameters
    ----------
    assessment_parameters_ : pandas.Series
        Assessment parameters containing ``gamma_M_RAJ``, support factor
        ``n_P``, roughness factor ``K_RP``, material start point
        ``P_RAJ_Z_WS``, and material fatigue limit ``P_RAJ_D_WS``.

    Returns
    -------
    pandas.Series
        Copy of ``assessment_parameters_`` with component shift factor
        ``f_RAJ``, component start point ``P_RAJ_Z``, initial fatigue limit
        ``P_RAJ_D_0``, and component fatigue limit ``P_RAJ_D`` added.

    Notes
    -----
    Implements FKM nonlinear Sections 2.8.6 and 2.9.6, equations (2.8-22) to
    (2.8-25) and (2.9-24) to (2.9-27).  ``n_P`` and ``K_RP`` enter squared for
    ``P_RAJ`` because this damage parameter is energy based.
    """
    assessment_parameters = assessment_parameters_.copy()

    assert "gamma_M_RAJ" in assessment_parameters
    assert "n_P" in assessment_parameters
    assert "K_RP" in assessment_parameters
    assert "P_RAJ_Z_WS" in assessment_parameters
    assert "P_RAJ_D_WS" in assessment_parameters

    # eq. (2.9-24) or eq. (2.8-22)
    assessment_parameters["f_RAJ"] = assessment_parameters.gamma_M_RAJ / (assessment_parameters.n_P**2 * assessment_parameters.K_RP**2)

    # calculations for P_RAJ of component Woehler curve
    # eq. (2.8-23), (2.9-25)
    assessment_parameters["P_RAJ_Z"] = 1 / assessment_parameters.f_RAJ * assessment_parameters.P_RAJ_Z_WS

    # eq. (2.8-24), (2.9-26). Note that there is also eq. 2.9-13, but this is errorneous and not relevant here.
    assessment_parameters["P_RAJ_D_0"] = 1 / assessment_parameters.f_RAJ * assessment_parameters.P_RAJ_D_WS

    # eq. (2.8-25), (2.9-27)
    assessment_parameters["P_RAJ_D"] = assessment_parameters.P_RAJ_D_0

    return assessment_parameters

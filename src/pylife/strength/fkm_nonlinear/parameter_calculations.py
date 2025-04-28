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
    """Calculate the values of :math:`n', K'`, and :math:`E`, used to
    describe the cyclic material behavior (Sec. 2.5.3 of FKM nonlinear).

    The calculated values will be set in a copy of the input series. The intended
    use is as follows:

    .. code::

        assessment_parameters = calculate_cyclic_assessment_parameters(assessment_parameters)

    Parameters
    ----------
    assessment_parameters : pandas Series
        The named material parameters. This Series has to include at least the following values:

        * ``MatGroupFKM``: Which material group, one of ``Steel``, ``SteelCast``, ``Al_wrought``
        * ``R_m``: The ultimate tensile strength of the material, :math:`R_m`.

    Returns
    -------
    pandas DataFrame
        A copy of ``assessment_parameters`` with the following additional items set:

        * ``E``: Young's modulus, constant estimated according to the material group
        * ``n_prime``: parameter for the Ramberg-Osgood material law
        * ``K_prime``: parameter for the Ramberg-Osgood material law

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
    """Calculate the parameters of the material damage Woehler curve for the P_RAM damage parameter
    (Sec. 2.5.5 of FKM nonlinear).

    The calculated values will be set in a copy of the input series. The intended
    use is as follows:

    .. code::

        assessment_parameters = calculate_material_woehler_parameters_P_RAM(assessment_parameters)

    Parameters
    ----------
    assessment_parameters : pandas Series
        The named material parameters. This Series has to include at least the following values:

        * ``MatGroupFKM``: Which material group, one of ``Steel``, ``SteelCast``, ``Al_wrought``
        * ``R_m``: The ultimate tensile strength of the material, :math:`R_m`.
        * ``P_A``: The failure probability

    Returns
    -------
    pandas DataFrame
        A copy of ``assessment_parameters`` with the following additional items set:

        * ``P_RAM_Z_WS``: First sampling point ("knee") of the material damage Woehler curve at N=1e3
        * ``P_RAM_D_WS``: Damage threshold for infinite life, the second "knee" of the material damage Woehler curve
        * ``d_1``: first slope of the material damage Woehler curve
        * ``d_2``: second slope of the material damage Woehler curve

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
    """Calculate the parameters of the material damage Woehler curve for the P_RAJ damage parameter
    (Sec. 2.9.4 of FKM nonlinear).

    The calculated values will be set in a copy of the input series. The intended
    use is as follows:

    .. code::

        assessment_parameters = calculate_material_woehler_parameters_P_RAJ(assessment_parameters)

    Parameters
    ----------
    assessment_parameters : pandas Series
        The named material parameters. This Series has to include at least the following values:

        * ``MatGroupFKM``: Which material group, one of ``Steel``, ``SteelCast``, ``Al_wrought``
        * ``R_m``: The ultimate tensile strength of the material, :math:`R_m`.
        * ``P_A``: The failure probability

    Returns
    -------
    pandas DataFrame
        A copy of ``assessment_parameters`` with the following additional items set:

        * ``P_RAJ_Z_WS``: First sampling point of the material damage Woehler curve for N=1 (not N=1e3 as for P_RAM!)
        * ``P_RAJ_D_WS``: Damage threshold for infinite life, the "knee" of the material damage Woehler curve
        * ``d_RAJ``: slope of the material damage Woehler curve

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
    """
    For FKM nonlinear roughness & surface layer, calculate the additional parameters in the P_RAM Woehler curve
    that model the dependency on the roughness. The resulting woehler curve is still the material woehler curve
    but with roughness effects. The other assessment factors are yet missing.

    The calculated values will be set in a copy of the input series. The intended
    use is as follows:

    .. code::

        assessment_parameters = calculate_roughness_material_woehler_parameters_P_RAM(assessment_parameters)
        assessment_parameters = calculate_roughness_component_woehler_parameters_P_RAM(assessment_parameters)

    Parameters
    ----------
    assessment_parameters : pandas Series
        The named material parameters. This Series has to include at least the following values:

        * ``P_RAM_Z_WS``: First sampling point ("knee") of the material damage Woehler curve at N=1e3
        * ``P_RAM_D_WS``: Damage threshold for infinite life, the second "knee" of the material damage Woehler curve
        * ``d_2``: The second slope of the material damage Woehler curve without roughness effect
        * ``K_RP``: The roughness factor K_R,P

    Returns
    -------
    pandas DataFrame
        A copy of ``assessment_parameters`` with the following additional items set:

        * ``P_RAM_D_WS_rau``: Damage threshold for infinite life, the lower "knee" of the material damage Woehler curve
        * ``d2_RAM_rau``: second slope of the material damage Woehler curve, adjusted by roughness factor K_R,P

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
    """
    For FKM nonlinear roughness & surface layer, calculate the additional parameters in the P_RAJ Woehler curve
    that model the dependency on the roughness. The resulting woehler curve is still the material woehler curve
    but with roughness effects. The other assessment factors are yet missing.

    The calculated values will be set in a copy of the input series. The intended
    use is as follows:

    .. code::

        assessment_parameters = calculate_roughness_material_woehler_parameters_P_RAJ(assessment_parameters)
        assessment_parameters = calculate_roughness_component_woehler_parameters_P_RAJ(assessment_parameters)

    Parameters
    ----------
    assessment_parameters : pandas Series
        The named material parameters. This Series has to include at least the following values:

        * ``P_RAJ_Z_WS``: Point at N=1 of the material damage Woehler curve (not N=1e3 as for P_RAM!)
        * ``P_RAJ_D_WS``: Damage threshold for infinite life, the second "knee" of the material damage Woehler curve
        * ``d_RAJ``: The slope of the material damage Woehler curve without roughness effect
        * ``K_RP``: The roughness factor K_R,P

    Returns
    -------
    pandas DataFrame
        A copy of ``assessment_parameters`` with the following additional items set:

        * ``P_RAJ_Z_1e3``: The upper "knee" of the material damage Woehler curve at N=1e3.
        * ``P_RAJ_D_WS_rau``: Damage threshold for infinite life, the lower "knee" of the material damage Woehler curve
        * ``d_RAJ_2_rau``: The second slope of the material damage Woehler curve adjusted by roughness factor K_R,P

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
    """Calculate the component woehler curve from the material woehler curve
    (Sec. 2.5.6 of FKM nonlinear), but with the special roughness consideration
    described in the extension surface layer & roughness.
    This involves multiplying the appropriate factors to the point P_RAM_Z_WS and P_RAM_D_WS_rau
    in the material woehler curve to obtain the points P_RAM_Z and P_RAM_D in the component Woehler curve.

    The "appropriate factors" consist of ``gamma_M`` which relates to the failure probability and optionally
    ``n_P``, which describes the supporting effect of the notch (fracture mechanical number and statistic number).
    The roughness has already been incorporated during `calculate_roughness_material_woehler_parameters_P_RAM`.

    The calculated values will be set in a copy of the input series. The intended
    use is as follows:

    .. code::

        assessment_parameters = calculate_roughness_parameter(assessment_parameters)
        assessment_parameters = calculate_roughness_material_woehler_parameters_P_RAM(assessment_parameters)
        assessment_parameters = calculate_nonlocal_parameters(assessment_parameters)    # include this line if include_n_P is True
        assessment_parameters = calculate_failure_probability_factor_P_RAM(assessment_parameters)
        assessment_parameters = calculate_roughness_component_woehler_parameters_P_RAM(assessment_parameters, True)

    Parameters
    ----------
    assessment_parameters : pandas Series
        The named material parameters. This Series has to include at least the following values:

        * ``gamma_M_RAM``: The factor for the standard deviation of the capacity to withstand stresses of the component.
        * ``n_P``: The factor for nonlocal effects, can be computed by ``calculate_nonlocal_parameters``.
            This is only needed if ``include_n_P`` is `True`.
        * ``P_RAM_Z_WS``: The point at N=1e3 in the material Woehler curve.
        * ``P_RAM_D_WS_rau``: The point in the material Woehler curve.

    include_n_P : bool
        Whether the supporting effect of the notch should be included, i.e., the factor ``n_P`` should be used.

    Returns
    -------
    pandas DataFrame
        A copy of ``assessment_parameters`` with the following additional items set:

        * ``P_RAM_Z``: The "Zeitfestigkeit" point in the component Woehler curve.
        * ``P_RAM_D``: The "Dauerfestigkeit" point in the component Woehler curve, i.e., the fatigue strength limit, the P_RAM value below which we have infinite life

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
    """Calculate the component woehler curve from the material woehler curve
    (Sec. 2.8.6 of FKM nonlinear), but with the special roughness consideration
    described in the extension surface layer & roughness.
    This involves multiplying the appropriate factors to the points P_RAJ_Z_WS
    and P_RAJ_D_WS_rau in the material woehler curve to obtain the
    points P_RAJ_Z and P_RAJ_D in the component Woehler curve.

    The "appropriate factors" consist of ``gamma_M`` which relates to the failure probability and optionally
    ``n_P``, which describes the supporting effect of the notch (fracture mechanical number and statistic number).
    The roughness has already been incorporated during `calculate_roughness_material_woehler_parameters_P_RAJ`.

    The calculated values will be set in a copy of the input series. The intended
    use is as follows:

    .. code::

        assessment_parameters = calculate_roughness_parameter(assessment_parameters)
        assessment_parameters = calculate_roughness_material_woehler_parameters_P_RAJ(assessment_parameters)
        assessment_parameters = calculate_nonlocal_parameters(assessment_parameters)    # include this line if include_n_P is True
        assessment_parameters = calculate_failure_probability_factor_P_RAJ(assessment_parameters)
        assessment_parameters = calculate_roughness_component_woehler_parameters_P_RAJ(assessment_parameters, True)

    Parameters
    ----------
    assessment_parameters : pandas Series
        The named material parameters. This Series has to include at least the following values:

        * ``gamma_M_RAJ``: The factor for the standard deviation of the capacity to withstand stresses of the component.
        * ``n_P``: The factor for nonlocal effects, can be computed by ``calculate_nonlocal_parameters``.
            This is only needed if ``include_n_P`` is `True`.
        * ``P_RAJ_Z_WS``: The point at N=1 in the material Woehler curve (note that for P_RAJ it is not N=1e3 as for P_RAM!)
        * ``P_RAJ_D_WS_rau``: The threshold for infinite life in the material Woehler curve.

    include_n_P : bool
        Whether the supporting effect of the notch should be included, i.e., the factor ``n_P`` should be used.

    Returns
    -------
    pandas DataFrame
        A copy of ``assessment_parameters`` with the following additional items set:

        * ``P_RAJ_Z``: The first point in the component Woehler curve at N=1 (not N=1e3 as for P_RAM!).
        * ``P_RAJ_D_0`` and ``P_RAJ_D``: The infinite life threshold of the component Woehler curve.

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
    """Calculate the factors for the nonlocal effects on the component lifetime.
     (Sec. 2.5.6.1 of FKM nonlinear). This includes the statistic factor and the
     fracture mechanics factor.

     The calculation procedure is the same for P_RAM and P_RAJ damage parameters.
     For P_RAJ, the equivalent formulas are presented in chapter 2.8.6.1 of FKM nonlinear.

    The calculated values will be set in a copy of the input series. The intended
    use is as follows:

    .. code::

        assessment_parameters = calculate_nonlocal_parameters(assessment_parameters)

    Parameters
    ----------
    assessment_parameters : pandas Series
        The named material parameters. This Series has to include at least the following values:

        * ``MatGroupFKM``: Which material group, one of ``Steel``, ``SteelCast``, ``Al_wrought``
        * ``A_ref``: Reference surface area of the highly loaded area, usually set to 500 [mm^2].
        * ``A_sigma``: Surface area of the highly loaded area of the component (in [mm^2]).
        * ``R_m``: The ultimate tensile strength of the material, :math:`R_m`.
        * ``G``: Relative stress gradient [mm]

    Returns
    -------
    pandas DataFrame
        A copy of ``assessment_parameters`` with the following additional items set:

        * ``n_st``: The statistic factor (de: Statistische Stützzahl)
        * ``n_bm``: The fracture mechanic factor (de: bruchmechanische Stützzahl)
        * ``n_P``: The total material factor as the product of ``n_st`` and ``n_bm`` (de: werkstoffmechanische Stützzahl)

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
    """Calculate the roughness factor K_R,P (Sec. 2.5.6.2 of FKM nonlinear).

    If the factor assessment_parameters["K_RP"] is already set, this function does nothing.

    The calculation is the same for P_RAM and P_RAJ damage parameters.
    For P_RAJ, the equivalent formulas are presented in chapter 2.8.6.2 of FKM nonlinear.

    The calculated values will be set in a copy of the input series. The intended
    use is as follows:

    .. code::

        assessment_parameters = calculate_roughness_parameter(assessment_parameters)

    Parameters
    ----------
    assessment_parameters : pandas Series
        The named material parameters. This Series has to include at least the following values:

        * ``MatGroupFKM``: Which material group, one of ``Steel``, ``SteelCast``, ``Al_wrought``
        * ``R_m``: The ultimate tensile strength of the material, :math:`R_m`.
        * ``R_z``: Only if K_RP is not yet given: The surface roughness of the component, :math:`R_z`.

    Returns
    -------
    pandas DataFrame
        A copy of ``assessment_parameters`` with the following additional items set:

        * ``K_RP``: The roughness factor K_R,P.

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
    """Calculates the beta parameter ("damage index"),
    which is an intermediate value for the lifetime assessment factor gamma_M.
    Note that the FKM nonlinear guideline does not list the formula for beta,
    they assume that the beta value is known.

    Parameters
    ----------
    P_A : float
        Failure probability for the assessment

    Returns
    -------
    float
        The parameter beta.

    """
    sigma = 1
    result = scipy.optimize.root(lambda x: abs(scipy.stats.norm.cdf(x, 0, sigma)-P_A), x0=-0.6, tol=1e-10)

    if not result.success:
        raise RuntimeError(f"Could not compute the value of beta for P_A={P_A}, "
                           "the optimizer did not find a solution.")

    return -result.x[0] / sigma


def calculate_failure_probability_factor_P_RAM(assessment_parameters_):
    """Calculate the factor for the failure probability of the component, i.e., the factor
    for the standard deviation of the capacity to withstand stresses of the component.
    This calculation is for use with the P_RAM damage parameter.

    If assessment_parameters.beta is set, the safety factor gamma_M for the capacity to withstand stresses of the component (de: Beanspruchbarkeit)
    is computed from the damage index ``beta``. Otherwise, it is derived from the failure probability assessment_parameters.P_A

    The calculated values will be set in a copy of the input series. The intended
    use is as follows:

    .. code::

        assessment_parameters = calculate_failure_probability_factor_P_RAM(assessment_parameters)

    Parameters
    ----------
    assessment_parameters : pandas Series
        The named material parameters. This Series has to include at least the following values:

        * Either the failure probability, ``P_A``, or the damage index, ``beta``.

    Returns
    -------
    pandas DataFrame
        A copy of ``assessment_parameters`` with the following additional items set:

        * ``gamma_M_RAM``: The factor for the standard deviation of the capacity to withstand stresses of the component.

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
    """Calculate the factor for the failure probability of the component, i.e., the factor
    for the standard deviation of the capacity to withstand stresses of the component.
    This calculation is for use with the P_RAJ damage parameter.

    If assessment_parameters.beta is set, the safety factor gamma_M for the capacity to withstand stresses of the component (de: Beanspruchbarkeit)
    is computed from the damage index ``beta``. Otherwise, it is derived from the failure probability assessment_parameters.P_A

    The calculated values will be set in a copy of the input series. The intended
    use is as follows:

    .. code::

        assessment_parameters = calculate_failure_probability_factor_P_RAJ(assessment_parameters)

    Parameters
    ----------
    assessment_parameters : pandas Series
        The named material parameters. This Series has to include at least the following values:

        * Either the failure probability, ``P_A``, or the damage index, ``beta``.

    Returns
    -------
    pandas DataFrame
        A copy of ``assessment_parameters`` with the following additional items set:

        * ``gamma_M_RAJ``: The factor for the standard deviation of the capacity to withstand stresses of the component.

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
    """Calculate the component woehler curve from the material woehler curve
    (Sec. 2.5.6 of FKM nonlinear). This involves multiplying the appropriate
    factors to the point P_RAM_Z_WS in the material woehler curve to obtain the
    point P_RAM_Z in the component Woehler curve.

    If assessment_parameters.beta is set, the safety factor gamma_M for the capacity to withstand stresses of the component (de: Beanspruchbarkeit)
    is computed from the damage index ``beta``. Otherwise, it is derived from the failure probability assessment_parameters.P_A

    The calculated values will be set in a copy of the input series. The intended
    use is as follows:

    .. code::

        assessment_parameters = calculate_material_woehler_parameters_P_RAM(assessment_parameters)
        assessment_parameters = calculate_nonlocal_parameters(assessment_parameters)
        assessment_parameters = calculate_roughness_parameter(assessment_parameters)
        assessment_parameters = calculate_failure_probability_factor_P_RAM(assessment_parameters)
        assessment_parameters = calculate_component_woehler_parameters_P_RAM(assessment_parameters)

    Parameters
    ----------
    assessment_parameters : pandas Series
        The named material parameters. This Series has to include at least the following values:

        * ``gamma_M_RAM``: The factor for the standard deviation of the capacity to withstand stresses of the component.
        * ``n_P``: The factor for nonlocal effects, can be computed by ``calculate_nonlocal_parameters``.
        * ``K_RP``: The roughness factor K_R,P.
        * ``P_RAM_Z_WS``: The point for N=1e3 in the material Woehler curve.
        * ``P_RAM_D_WS``: The threshold for infinite life in the material Woehler curve.
        * Either the failure probability, ``P_A``, or the damage index, ``beta``.

    Returns
    -------
    pandas DataFrame
        A copy of ``assessment_parameters`` with the following additional items set:

        * ``f_RAM``: The factor to map between component and material Woehler curves.
        * ``P_RAM_Z``: The "Zeitfestigkeit" point in the component Woehler curve.
        * ``P_RAM_D``: The "Dauerfestigkeit" point in the component Woehler curve, i.e., the fatigue strength limit, the P_RAM value below which we have infinite life

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
    """Calculate the component woehler curve from the material woehler curve
    (Sec. 2.8.6 of FKM nonlinear). This involves multiplying the appropriate
    factors to the points P_RAJ_Z_WS and P_RAJ_D_WS in the material woehler curve to obtain the
    points P_RAJ_Z and P_RAJ_D in the component Woehler curve.

    If assessment_parameters.beta is set, the safety factor gamma_M for the capacity to withstand stresses of the component (de: Beanspruchbarkeit)
    is computed from the damage index ``beta``. Otherwise, it is derived from the failure probability assessment_parameters.P_A

    The calculated values will be set in a copy of the input series. The intended
    use is as follows:

    .. code::

        assessment_parameters = calculate_material_woehler_parameters_P_RAJ(assessment_parameters)
        assessment_parameters = calculate_nonlocal_parameters(assessment_parameters)
        assessment_parameters = calculate_roughness_parameter(assessment_parameters)
        assessment_parameters = calculate_failure_probability_factor_P_RAJ(assessment_parameters)
        assessment_parameters = calculate_component_woehler_parameters_P_RAJ(assessment_parameters)

    Parameters
    ----------
    assessment_parameters : pandas Series
        The named material parameters. This Series has to include at least the following values:

        * ``gamma_M_RAJ``: The factor for the standard deviation of the capacity to withstand stresses of the component.
        * ``n_P``: The factor for nonlocal effects, can be computed by ``calculate_nonlocal_parameters``.
        * ``K_RP``: The roughness factor K_R,P.
        * ``P_RAJ_Z_WS``: The point at N=1 in the material Woehler curve.
        * ``P_RAJ_D_WS``: The threshold for infinite life in the material Woehler curve.
        * Either the failure probability, ``P_A``, or the damage index, ``beta``.

    Returns
    -------
    pandas DataFrame
        A copy of ``assessment_parameters`` with the following additional items set:

        * ``f_RAJ``: The factor to map between component and material Woehler curves.
        * ``P_RAJ_Z``: The first point in the component Woehler curve.
        * ``P_RAJ_D_0`` and ``P_RAJ_D``: The infinite life threshold of the component Woehler curve.

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

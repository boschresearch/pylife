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

import copy
import numpy as np
import pandas as pd

# pylife
import pylife
import pylife.strength.fkm_load_distribution
import pylife.strength.damage_parameter
import pylife.strength.woehler_fkm_nonlinear
import pylife.strength.fkm_nonlinear.damage_calculator
import pylife.materiallaws
import pylife.stress.rainflow
import pylife.stress.rainflow.recorders
import pylife.stress.rainflow.fkm_nonlinear
import pylife.materiallaws.notch_approximation_law
import pylife.materiallaws.notch_approximation_law_seegerbeste
import pylife.strength.fkm_nonlinear.damage_calculator_praj_miner

import pylife.strength.fkm_nonlinear.parameter_calculations as parameter_calculations

''' Collection of functions for computational proof of the strength
    for machine elements considering their non-linear material deformation
    (FKM non-linear guideline 2019)
'''


def perform_fkm_nonlinear_assessment(assessment_parameters, load_sequence, calculate_P_RAM=True, calculate_P_RAJ=True):
    r"""Perform the lifetime assessment according to FKM nonlinear, using the damage parameters P_RAM and/or P_RAJ.
    The assessment can be done for a load sequence on a single point or for multiple points at once, e.g., a FEM mesh.
    If multiple points at once are used, it is assumed that the load sequences at all nodes are scaled versions of each
    other.

    For an assessment with multiple points at once, the relative stress gradient G can be either specified to be constant
    or it can have a different value at every point.

    The FKM nonlinear guideline defines three possible methods to consider the statistical distribution of the load:

        1. a normal distribution with given standard deviation, :math:`s_L`
        2. a logarithmic-normal distribution with given standard deviation :math:`LSD_s`
        3. an unknown distribution, use the constant factor :math:`\gamma_L=1.1` for :math:`P_L = 2.5\%`
            or :math:`\gamma_L=1` for :math:`P_L = 50\%` or

    If the ``assessment_parameters`̀`  contain a value for ``s_L``, the first approach is used (normal distribution).
    Else, if the ``assessment_parameters`̀  contain a value for ``LSD_s``, the second approach is used (log-normal distribution).
    Else, if only ``P_L``̀  is given a scaling with the according factor is used. The statistical assessment can be skipped
    by settings ``P_A = 0.5`` and ``P_L = 50``.

    Parameters
    ----------
    assessment_parameters : pandas Series
        The parameters that specify the material and the assessment problem. The following parameters are required:

        * ``MatGroupFKM``: string, one of {``Steel``, ``SteelCast``, ``Al_wrought``}. Specifies the considered material group.
        * ``FinishingFKM``: string, one of {``none``}, the type of surface finishing (Surface finishing types are not implemented for FKM nonlinear).
        * ``R_m``: float [MPa], ultimate tensile strength (de: Zugfestigkeit).
            Note that this value can also be estimated from a pre-product nominal value, as described in the FKM document.
        * ``K_RP``: float, [-], surface roughness factor, set to 1 for polished surfaces or determine from the given diagrams included in the FKM document.
        * ``R_z``: float [um], average roughness (de: mittlere Rauheit), only required if K_RP is not specified directly
        * ``P_A``: float. Specifies the failure probability for the assessment (de: auszulegende Ausfallwahrscheinlichkeit).
            Note that any value for P_A in (0,1) is possible, not just the fixed values that are defined in the FKM nonlinear
            guideline
            Set to 0.5 to disable statistical assessment, e.g., to simulate cyclic experiments.
        * ``beta``: float, damage index, specify this as an alternative to ``P_A``.
        * ``P_L``: float, [%],  one of {̀ `2.5``%, ``50``%}, probability of occurence of the specified load sequence
            (de: Auftretenswahrscheinlilchkeit der Lastfolge). Usually set to 50 to disable statistical assessment for the
            load.
        * ``s_L``: float (optional), [MPa] standard deviation of Gaussian distribution for the statistical distribution of the load
        * ``LSD_s``: float (optional), [MPa] standard deviation of the lognormal distribution for the statistical distribution of the load
        * `̀ c``, float, [MPa/N] factor from reference load with which FE simulation was obtained to computed equivalent stress
            (de: Übertragungsfaktor Vergleichsspannung zu Referenzlast im Nachweispunkt) c = sigma_V / L_REF
        * ̀ `A_sigma``: float, [mm^2] highly loaded surface area of the component (de: Hochbeanspruchte Oberfläche des Bauteils)
        * ``A_ref``: float, [mm^2] (de: Hochbeanspruchte Oberfläche eines Referenzvolumens), usually set to 500
        * ``G``: float, [mm^-1] relative stress gradient (de: bezogener Spannungsgradient).
            This value can be either a constant value or a pandas Series with different values for every node.
            If a Series is used, the order of the G values in the Series has to match the order of the assessment points in the load sequence.
            The actual values of the index are irrelevant.

            Note that the relative stress gradient can be computed as follows:

                .. code::

                    grad = pyLife_mesh.gradient_3D.gradient_of('mises')

                    # compute the absolute stress gradient
                    grad["abs_grad"] = np.linalg.norm(grad, axis = 1)
                    pylife_mesh = pylife_mesh.join(grad, sort=False)

                    # compute scaled stress gradient (bezogener Spannungsgradient)
                    pylife_mesh["G"] = pylife_mesh.abs_grad / pylife_mesh.mises

            To add the value of G to the ``assessment_parameters``, do the following:

                .. code::

                    # remove element_id
                    G = pylife_mesh['G'].droplevel("element_id")

                    # remove duplicate node entries
                    G = G[~G.index.duplicated(keep="first")].sort_index()

                    assessment_parameters["G"] = G

        * ``K_p``: float, [-] (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1).
            Note that Seeger-Beste and P_RAJ only work for K_p > 1.
        * ``n_bins``: int, optional (default: 200) number of bins or classes for P_RAJ computation. A larger value gives more accurate results but longer runtimes.
    load_sequence : pandas Series
        A sequential list of loads that are applied to the component. If the assessment should be done for
        a single points, this is simply a pandas Series. For multiple points at once, it should be a pandas
        DataFrame with a two-level MultiIndex with fields "load_step" and "node_id".
        The load_step describes the point in time of the sequence and must be consecutive starting from 0.
        The node_id identifies the assessment point or mesh node in every load step. The data frame contains
        only one column with the stress at every node. The relation between the loads at every nodes
        has to be constant over the load steps, i.e., the load sequences at the nodes are scaled versions
        of each other.

        An example is given below:

        .. code::

                                  S_v
            load_step   node_id
            0           1         -51.135208
                        2         28.023306
                        3         30.012435
                        4         -11.698302
                        5         287.099222
            ...         ...       ...
                        14614     287.099222
            1           1         -51.135208
            ...         ...       ...
            7           1         -51.135208
            ...         ...       ...
                        14610     -113.355076
                        14611     -43.790024
                        14612     -99.422582
                        14613     -77.195496
                        14614     -90.303717

    calculate_P_RAM : bool (optional)
        Whether to use the P_RAM damage parameter for the assessment. Default: True.
    calculate_P_RAJ : bool (optional)
        Whether to use the P_RAJ damage parameter for the assessment. Default: True.

    Returns
    -------
    result : pandas Series
        The asssessment result containing at least the following items:

        * ``P_RAM_is_life_infinite``: (bool) whether we have infinite life (de: Dauerfestigkeit)
        * ``P_RAM_lifetime_n_cycles``: (float) lifetime in number of cycles
        * ``P_RAM_lifetime_n_times_load_sequence``: (float) lifetime how often the full load sequence can be applied
        * ``P_RAJ_is_life_infinite`` (bool) whether we have infinite life (de: Dauerfestigkeit)
        * ``P_RAJ_lifetime_n_cycles``: (float) lifetime in number of cycles
        * ``P_RAJ_lifetime_n_times_load_sequence``: (float) lifetime how often the full load sequence can be applied

        The result dict contains even more entries which are for further information and debugging purposes, such as
        woehler curve objects and collective tables.

        If P_A is set to 0.5 and P_L is set to 50, i.e., no statistical assessment is specified, and if the load sequence
        is scalar (i.e., not for an entire FEM mesh), the result contains the following additional values:

        * ``P_RAM_lifetime_N_1ppm``, ``P_RAM_lifetime_N_10``, ``P_RAM_lifetime_N_50̀ `, ``P_RAM_lifetime_N_90``: (float)
            lifetimes in numbers of cycles,
            for P_A = 1ppm = 1e-6, 10%, 50%, and 90%, according to the assessment defined in the FKM nonlinear
            guideline. Note that the guideline does not yield a log-normal distributed lifetime.
            Furthermore, the value of ``P_RAM_lifetime_N_50̀ ` is lower than the calculated lifetime
            ``P_RAM_lifetime_n_cycles``, because it contains a safety factor even for P_A = 50%.

        * ``P_RAM_N_max_bearable``: (function) A python function
            ``N_max_bearable(P_A, clip_gamma=False)``
            that calculates the maximum number of cycles
            the component can withstand with the given failure probability.
            The parameter ``clip_gamma`` specifies whether the scaling factor gamma_M
            will be at least 1.1 (P_RAM) or 1.2 (P_RAJ), as defined
            in eq. (2.5-38) (PRAM) / eq. (2.8-38) (PRAJ).

            Note that it holds ``P_RAM_lifetime_N_10`` = P_RAM_N_max_bearable(0.1),
            and analogously for the variables for 1ppm, 50%, and 90%.

        * ``P_RAM_failure_probability``: (function) A python function,
            ``failure_probability(N)`` that calculates the failure probability for a
            given number of cycles.
    """

    # check that gradient G is in the correct format
    _assert_G_is_in_correct_format(assessment_parameters)
    _check_K_p_is_in_range(assessment_parameters)

    scaled_load_sequence = _scale_load_sequence_according_to_probability(assessment_parameters, load_sequence)
    scaled_load_sequence = _scale_load_sequence_by_c_factor(assessment_parameters, scaled_load_sequence)

    assessment_parameters = _calculate_local_parameters(assessment_parameters)

    assessment_parameters, component_woehler_curve_P_RAM, component_woehler_curve_P_RAJ \
        = _compute_component_woehler_curves(assessment_parameters)

    result = {}

    # HCM rainflow counting and damage computation for P_RAM
    if calculate_P_RAM:
        result = _compute_lifetimes_P_RAM(assessment_parameters, result, scaled_load_sequence, component_woehler_curve_P_RAM)

    # HCM rainflow counting and damage computation for P_RAJ
    if calculate_P_RAJ:
        result = _compute_lifetimes_P_RAJ(assessment_parameters, result, scaled_load_sequence, component_woehler_curve_P_RAJ)

    # additional quantities
    result["assessment_parameters"] = assessment_parameters

    return result


def _assert_G_is_in_correct_format(assessment_parameters):
    """Check that the related stress gradient G is given in the correct format,
    either as a single float or as a pandas Series with values for each node
    of the mesh. The check is performed by an assertion.
    """

    # check that gradient G is in the correct format
    assert isinstance(assessment_parameters.G, float) \
        or (isinstance(assessment_parameters.G, pd.Series) and not isinstance(assessment_parameters.G.index, pd.MultiIndex)), \
        "stress gradient G is in a wrong format (should be either float or pd.Series indexed by node)"


def _check_K_p_is_in_range(assessment_parameters):
    """Check that the load shape factor (de: Traglastformzahl) K_p = F_plastic / F_yield (3.1.1)
    is larger than 1.
    """

    # check that gradient G is in the correct format
    assert assessment_parameters.K_p >= 1, \
        "K_p should be at least 1"

    if assessment_parameters.K_p == 1:
        print("Note, K_p is set to 1 which means only P_RAM can be calculated. "
            f"To use P_RAJ set K_p > 1, e.g. try K_p = 1.001.")


def _scale_load_sequence_according_to_probability(assessment_parameters, load_sequence):
    r"""Scales the given load sequence according to one of three methods defined in the FKM nonlinear guideline.

    The FKM nonlinear guideline defines three possible methods to consider the statistical distribution of the load:

        1. a normal distribution with given standard deviation, :math:`s_L`
        2. a logarithmic-normal distribution with given standard deviation :math:`LSD_s`
        3. an unknown distribution, use the constant factor :math:`\gamma_L=1.1` for :math:`P_L = 2.5\%`
            or :math:`\gamma_L=1` for :math:`P_L = 50\%` or

    If the ``assessment_parameters`̀`  contain a value for ``s_L``, the first approach is used (normal distribution).
    Else, if the ``assessment_parameters``̀  contain a value for ``LSD_s``, the second approach is used (log-normal distribution).
    Else, if only ``P_L`̀  is given a scaling with the according factor is used. The statistical assessment can be skipped
    by settings ``P_A = 0.5`` and ``P_L = 50``.

    Parameters
    ----------
    assessment_parameters : :class:`pandas.Series`
        All parameters to the FKM algorithm, given in a series.
    load_sequence : :class:`pandas.Series`
        The load-time series for the assessment.

    Returns
    -------
    :class:`pandas.Series`
        The scaled load sequence.
    """

    # add an empty "notes" entry in assessment_parameters
    if "notes" not in assessment_parameters:
        assessment_parameters["notes"] = ""

    # FKMLoadDistributionNormal, uses assessment_parameters.s_L, assessment_parameters.P_L, assessment_parameters.P_A
    if "s_L" in assessment_parameters:
        scaled_load_sequence = load_sequence.fkm_safety_normal_from_stddev.scaled_load_sequence(assessment_parameters)

        # add a note
        assessment_parameters["notes"] += f"s_L was defined (s_L={assessment_parameters.s_L}), P_L={assessment_parameters.P_L}, "\
            f" P_A={assessment_parameters.P_A}, using normal distribution "\
            f"for load, factor gamma_L={load_sequence.fkm_safety_normal_from_stddev.gamma_L(assessment_parameters)}.\n"

    elif "LSD_s" in assessment_parameters:
        # FKMLoadDistributionLognormal, uses assessment_parameters.LSD_s, assessment_parameters.P_L, assessment_parameters.P_A
        scaled_load_sequence = load_sequence.fkm_safety_lognormal_from_stddev.scaled_load_sequence(assessment_parameters)

        # add a note
        assessment_parameters["notes"] += f"LSD_s was defined (LSD_s={assessment_parameters.LSD_s}), "\
            f" P_L={assessment_parameters.P_L}, P_A={assessment_parameters.P_A}, using lognormal distribution "\
            f"for load, factor gamma_L={load_sequence.fkm_safety_lognormal_from_stddev.gamma_L(assessment_parameters)}.\n"

    else:
        # FKMLoadDistributionBlanket, uses input_parameters.P_L
        scaled_load_sequence = load_sequence.fkm_safety_blanket.scaled_load_sequence(assessment_parameters)

        # add a note
        assessment_parameters["notes"] += f"none of s_L, LSD_s was defined, P_L={assessment_parameters.P_L}, "\
            f"factor gamma_L={load_sequence.fkm_safety_blanket.gamma_L(assessment_parameters)}.\n"

    return scaled_load_sequence


def _scale_load_sequence_by_c_factor(assessment_parameters, scaled_load_sequence):
    """Scale the load sequence by the given transfer factor c from the
    linear elastic FE result to the given magnitude.
    The factor c in defined as :math:`1/L_{REF}` with the reference load :math:`L_{REF}`.
    """

    # scale load sequence by reference load
    c = assessment_parameters.c
    scaled_load_sequence = scaled_load_sequence.fkm_load_sequence.scaled_by_constant(c)

    return scaled_load_sequence


def _calculate_local_parameters(assessment_parameters):
    r"""Calculate several intermediate parameters as described in the FKM nonlinear guideline:

    * The cyclic parameters :math:`n', K'`, and :math:`E`, used to
        describe the cyclic material behavior (Sec. 2.5.3 of FKM nonlinear)
    * The material woehler curve parameters for both the P_RAM and P_RAJ woehler curves
        (Sec. 2.5.5 of FKM nonlinear)
    * The factor for non-local influences, :math:`n_P = n_{bm}(R_m, G) \cdot n_{st}(A_\sigma)`,
        where :math:`n_{bm}` is the fracture mechanics factor (de: bruchmechanische Stützzahl)
        and :math:`n_{st}` is the statistic factor (de: statistische Stützzahl).
        The factors depend on the stress gradient, :math:`G`, and the highly loaded surface,
        :math:`A_\sigma`, respectively.
    * The roughness factor :math:`K_{R,P}` which is estimated based on the ultimate tensile strength.
    """

    # compute intermediate values
    assessment_parameters = parameter_calculations.calculate_cyclic_assessment_parameters(assessment_parameters)

    # calculate the parameters for the material woehler curve
    # (for both P_RAM and P_RAJ, the variable names do not interfere)
    assessment_parameters = parameter_calculations.calculate_material_woehler_parameters_P_RAM(assessment_parameters)
    assessment_parameters = parameter_calculations.calculate_material_woehler_parameters_P_RAJ(assessment_parameters)

    # Size and geometry factor $n_P$, Spannungsgradient $G$, $A_\sigma$
    assessment_parameters = parameter_calculations.calculate_nonlocal_parameters(assessment_parameters)

    # Roughness factor $K_{R,P}$
    assessment_parameters = parameter_calculations.calculate_roughness_parameter(assessment_parameters)

    return assessment_parameters


def _compute_component_woehler_curves(assessment_parameters):
    r"""Compute the PRAM and PRAJ component woehler curves.
    At first, the safety factors :math:`\gamma_M` and :math:`f_\text{RAM}, f_\text{RAJ}`
    are calculated. Then, the woehler curve objects are created.
    """

    # Compute the safety factors to derive the component Woehler curve from the material Woehler curve.
    # Compute gamma_M
    assessment_parameters = parameter_calculations.calculate_failure_probability_factor_P_RAM(assessment_parameters)
    assessment_parameters = parameter_calculations.calculate_failure_probability_factor_P_RAJ(assessment_parameters)

    # Compute the component woehler curve parameters
    assessment_parameters = parameter_calculations.calculate_component_woehler_parameters_P_RAM(assessment_parameters)
    assessment_parameters = parameter_calculations.calculate_component_woehler_parameters_P_RAJ(assessment_parameters)

    # Wöhler curve for P_RAM
    component_woehler_curve_parameters = assessment_parameters[["P_RAM_Z", "P_RAM_D", "d_1", "d_2"]]
    component_woehler_curve_P_RAM = component_woehler_curve_parameters.woehler_P_RAM

    # Wöhler curve for P_RAJ
    component_woehler_curve_parameters = assessment_parameters[["P_RAJ_Z", "P_RAJ_D_0", "d_RAJ"]]
    component_woehler_curve_P_RAJ = component_woehler_curve_parameters.woehler_P_RAJ

    return assessment_parameters, component_woehler_curve_P_RAM, component_woehler_curve_P_RAJ


def _compute_hcm_RAM(assessment_parameters, scaled_load_sequence):
    """Perform the HCM rainflow counting with the extended Neuber notch approximation.
    The HCM algorithm is executed twice, as described in the FKM nonlinear guideline."""

    # initialize notch approximation law
    E, K_prime, n_prime, K_p = assessment_parameters[["E", "K_prime", "n_prime", "K_p"]]
    extended_neuber = pylife.materiallaws.notch_approximation_law.ExtendedNeuber(E, K_prime, n_prime, K_p)

    # create recorder object
    recorder = pylife.stress.rainflow.recorders.FKMNonlinearRecorder()

    # create detector object
    detector = pylife.stress.rainflow.fkm_nonlinear.FKMNonlinearDetector(
        recorder=recorder, notch_approximation_law=extended_neuber
    )

    # perform HCM algorithm, first run
    detector.process_hcm_first(scaled_load_sequence)
    detector_1st = copy.deepcopy(detector)

    # perform HCM algorithm, second run
    detector.process_hcm_second(scaled_load_sequence)

    return detector_1st, detector, extended_neuber, recorder


def _compute_damage_and_lifetimes_RAM(assessment_parameters, recorder, component_woehler_curve_P_RAM, result):
    """For P_RAM, calculate the damage and the lifetime and store in result dict."""

    # define damage parameter
    damage_parameter = pylife.strength.damage_parameter.P_RAM(recorder.collective, assessment_parameters)

    # compute the effect of the damage parameter with the woehler curve
    damage_calculator = pylife.strength.fkm_nonlinear.damage_calculator\
        .DamageCalculatorPRAM(damage_parameter.collective, component_woehler_curve_P_RAM)

    result["P_RAM_damage_parameter"] = damage_parameter

    # Infinite life assessment
    result["P_RAM_is_life_infinite"] = damage_calculator.is_life_infinite

    # finite life assessment
    result["P_RAM_lifetime_n_cycles"] = damage_calculator.lifetime_n_cycles
    result["P_RAM_lifetime_n_times_load_sequence"] = damage_calculator.lifetime_n_times_load_sequence

    return result, damage_calculator


def _compute_lifetimes_for_failure_probabilities_RAM(assessment_parameters, result, damage_calculator):
    """If P_A is set to 0.5, i.e., no explicit statistical assessment is performed, do
    some statistical assessment as post-processing.

    The lifetimes for 1ppm, 10%, 50%, and 90% are calculated using the given assessment concept
    defined by the FKM nonlinear guideline. Further, two python functions for arbitrary lifetimes
    and failure probabilities are created. Everything is stored in the result dict."""

    if "P_A" in assessment_parameters and np.isclose(assessment_parameters.P_A, 0.5):

        N_max_bearable, failure_probability = damage_calculator.get_lifetime_functions(assessment_parameters)

        N_1ppm = N_max_bearable(1e-6)
        N_10 = N_max_bearable(0.1)
        N_50 = N_max_bearable(0.5)
        N_90 = N_max_bearable(0.9)

        # add lifetime and failure probability results
        result["P_RAM_lifetime_N_1ppm"] = N_1ppm
        result["P_RAM_lifetime_N_10"] = N_10
        result["P_RAM_lifetime_N_50"] = N_50
        result["P_RAM_lifetime_N_90"] = N_90
        result["P_RAM_N_max_bearable"] = N_max_bearable
        result["P_RAM_failure_probability"] = failure_probability

    return result


def _store_additional_objects_in_result_RAM(result, recorder, damage_calculator, component_woehler_curve_P_RAM, detector, detector_1st):
    """Store the given objects in the results dict. The ``result`` variable gets
     returned back to the user. These additional variables an be used for certain plots,
    e.g. to plot the woehler curve."""

    result["P_RAM_recorder_collective"] = recorder.collective
    result["P_RAM_collective"] = damage_calculator.collective
    result["P_RAM_woehler_curve"] = component_woehler_curve_P_RAM
    result["P_RAM_damage_calculator"] = damage_calculator
    result["P_RAM_detector"] = detector
    result["P_RAM_detector_1st"] = detector_1st
    return result


def _compute_hcm_RAJ(assessment_parameters, scaled_load_sequence):
    """Perform the HCM rainflow counting with the Seeger-Beste notch approximation.
    The HCM algorithm is executed twice, as described in the FKM nonlinear guideline."""

    # initialize notch approximation law
    E, K_prime, n_prime, K_p = assessment_parameters[["E", "K_prime", "n_prime", "K_p"]]
    seeger_beste = pylife.materiallaws.notch_approximation_law_seegerbeste.SeegerBeste(E, K_prime, n_prime, K_p)

    # create recorder object
    recorder = pylife.stress.rainflow.recorders.FKMNonlinearRecorder()

    # create detector object
    detector = pylife.stress.rainflow.fkm_nonlinear.FKMNonlinearDetector(
        recorder=recorder, notch_approximation_law=seeger_beste
    )
    detector_1st = copy.deepcopy(detector)

    # perform HCM algorithm, first run
    detector.process_hcm_first(scaled_load_sequence)

    # perform HCM algorithm, second run
    detector.process_hcm_second(scaled_load_sequence)

    return detector_1st, detector, seeger_beste, recorder


def _compute_damage_and_lifetimes_RAJ(assessment_parameters, recorder, component_woehler_curve_P_RAJ, result):
    """For P_RAJ, calculate the damage and the lifetime and store in result dict."""

    # define damage parameter
    damage_parameter = pylife.strength.damage_parameter.P_RAJ(recorder.collective, assessment_parameters,\
                                                              component_woehler_curve_P_RAJ)

    # compute the effect of the damage parameter with the woehler curve
    damage_calculator = pylife.strength.fkm_nonlinear.damage_calculator\
        .DamageCalculatorPRAJ(damage_parameter.collective, assessment_parameters, component_woehler_curve_P_RAJ)

    result["P_RAJ_damage_parameter"] = damage_parameter

    # Infinite life assessment
    result["P_RAJ_is_life_infinite"] = damage_calculator.is_life_infinite

    # finite life assessment
    result["P_RAJ_lifetime_n_cycles"] = damage_calculator.lifetime_n_cycles
    result["P_RAJ_lifetime_n_times_load_sequence"] = damage_calculator.lifetime_n_times_load_sequence

    return result, damage_calculator


def _compute_damage_and_lifetimes_RAJ_miner(assessment_parameters, recorder, component_woehler_curve_P_RAJ, result):
    """For P_RAJ, calculate the damage and the lifetime using the woehler curve directly, and store in result dict."""

    # define damage parameter
    damage_parameter = pylife.strength.damage_parameter.P_RAJ(recorder.collective, assessment_parameters,\
                                                              component_woehler_curve_P_RAJ)

    # compute the effect of the damage parameter with the woehler curve
    damage_calculator = pylife.strength.fkm_nonlinear.damage_calculator_praj_miner\
        .DamageCalculatorPRAJMinerElementary(damage_parameter.collective, component_woehler_curve_P_RAJ)

    result["P_RAJ_miner_damage_calculator"] = damage_calculator

    # Infinite life assessment
    result["P_RAJ_miner_is_life_infinite"] = damage_calculator.is_life_infinite

    # finite life assessment
    result["P_RAJ_miner_lifetime_n_cycles"] = damage_calculator.lifetime_n_cycles
    result["P_RAJ_miner_lifetime_n_times_load_sequence"] = damage_calculator.lifetime_n_times_load_sequence

    return result


def _compute_lifetimes_for_failure_probabilities_RAJ(assessment_parameters, result, damage_calculator):
    """If P_A is set to 0.5, i.e., no explicit statistical assessment is performed, do
    some statistical assessment as post-processing.

    The lifetimes for 1ppm, 10%, 50%, and 90% are calculated using the given assessment concept
    defined by the FKM nonlinear guideline. Further, two python functions for arbitrary lifetimes
    and failure probabilities are created. Everything is stored in the result dict."""

    if "P_A" in assessment_parameters and np.isclose(assessment_parameters.P_A, 0.5):

        N_max_bearable, failure_probability = damage_calculator.get_lifetime_functions()

        N_1ppm = N_max_bearable(1e-6)
        N_10 = N_max_bearable(0.1)
        N_50 = N_max_bearable(0.5)
        N_90 = N_max_bearable(0.9)

        # add lifetime and failure probability results
        result["P_RAJ_lifetime_N_1ppm"] = N_1ppm
        result["P_RAJ_lifetime_N_10"] = N_10
        result["P_RAJ_lifetime_N_50"] = N_50
        result["P_RAJ_lifetime_N_90"] = N_90
        result["P_RAJ_N_max_bearable"] = N_max_bearable
        result["P_RAJ_failure_probability"] = failure_probability

    return result


def _store_additional_objects_in_result_RAJ(result, recorder, damage_calculator, component_woehler_curve_P_RAJ, detector, detector_1st):
    """Store the given objects in the results dict. The ``result`` variable gets
     returned back to the user. These additional variables an be used for certain plots,
    e.g. to plot the woehler curve."""

    # add collectives and objects
    result["P_RAJ_recorder_collective"] = recorder.collective
    result["P_RAJ_collective"] = damage_calculator.collective
    result["P_RAJ_woehler_curve"] = component_woehler_curve_P_RAJ
    result["P_RAJ_damage_calculator"] = damage_calculator
    result["P_RAJ_detector"] = detector
    result["P_RAJ_detector_1st"] = detector_1st

    return result


def _compute_lifetimes_P_RAJ(assessment_parameters, result, scaled_load_sequence, component_woehler_curve_P_RAJ):
    """Compute the lifetimes using the given parameters and woehler curve, with P_RAJ.

    * Execute the HCM algorithm to detect closed hysteresis.
    * Use the woehler curve and the damage parameter to predict lifetimes.
    * Do statistical assessment and store all results in a dict.
    """

    detector_1st, detector, seeger_beste_binned, recorder = _compute_hcm_RAJ(assessment_parameters, scaled_load_sequence)
    result["seeger_beste_binned"] = seeger_beste_binned

    result, damage_calculator = _compute_damage_and_lifetimes_RAJ(assessment_parameters, recorder, component_woehler_curve_P_RAJ, result)

    result = _compute_damage_and_lifetimes_RAJ_miner(assessment_parameters, recorder, component_woehler_curve_P_RAJ, result)

    result = _compute_lifetimes_for_failure_probabilities_RAJ(assessment_parameters, result, damage_calculator)

    result = _store_additional_objects_in_result_RAJ(result, recorder, damage_calculator, component_woehler_curve_P_RAJ, detector, detector_1st)
    return result


def _compute_lifetimes_P_RAM(assessment_parameters, result, scaled_load_sequence, component_woehler_curve_P_RAM):
    """Compute the lifetimes using the given parameters and woehler curve, with P_RAM.

    * Execute the HCM algorithm to detect closed hysteresis.
    * Use the woehler curve and the damage parameter to predict lifetimes.
    * Do statistical assessment and store all results in a dict.
    """
    detector_1st, detector, extended_neuber_binned, recorder = _compute_hcm_RAM(assessment_parameters, scaled_load_sequence)
    result["extended_neuber_binned"] = extended_neuber_binned

    result, damage_calculator = _compute_damage_and_lifetimes_RAM(assessment_parameters, recorder, component_woehler_curve_P_RAM, result)

    result = _compute_lifetimes_for_failure_probabilities_RAM(assessment_parameters, result, damage_calculator)

    result = _store_additional_objects_in_result_RAM(result, recorder, damage_calculator, component_woehler_curve_P_RAM, detector, detector_1st)
    return result

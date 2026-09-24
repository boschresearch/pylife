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

r"""Assess FKM nonlinear lifetimes from parameters and load sequences.

This module is the top-level user entry point for the FKM nonlinear
strength assessment.  It scales the supplied load sequence, derives local
assessment parameters, builds the ``P_RAM`` and ``P_RAJ`` component Wöhler
curves, runs hysteresis counting with the FKM nonlinear rainflow detector,
and evaluates damage and lifetime results.
"""
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
    r"""Perform the FKM nonlinear lifetime assessment.

    Use this function when the material and component assessment parameters are
    already available as a :class:`pandas.Series` and the load history is given
    as a scalar load sequence or as scaled node histories from an FE model.  The
    function evaluates the ``P_RAM`` path with the extended Neuber notch
    approximation, the ``P_RAJ`` path with the Seeger-Beste notch approximation,
    or both paths.

    Parameters
    ----------
    assessment_parameters : pandas.Series
        User and derived assessment parameters.  Required user-supplied keys are
        ``MatGroupFKM`` with one of ``'Steel'``, ``'SteelCast'``, or
        ``'Al_wrought'``; ``FinishingFKM`` with currently only ``'none'``;
        ultimate tensile strength ``R_m`` in MPa; roughness factor ``K_RP`` or
        roughness ``R_z`` in µm; failure probability ``P_A`` or reliability
        index ``beta``; load occurrence probability ``P_L`` in percent;
        transfer factor ``c`` from reference load to stress in MPa per load
        unit; highly stressed surface ``A_sigma`` in mm²; reference surface
        ``A_ref`` in mm², usually ``500``; relative stress gradient ``G`` in
        1/mm as a float or node-indexed :class:`pandas.Series`; load shape
        factor ``K_p``; and optionally ``n_bins`` for ``P_RAJ`` discretization.
        Optional load scatter keys are ``s_L`` for a normal distribution in MPa
        or ``LSD_s`` for a lognormal distribution.
    load_sequence : pandas.Series or pandas.DataFrame
        Load sequence to assess.  Use a :class:`pandas.Series` for one
        assessment point.  Use a :class:`pandas.DataFrame` with a two-level
        ``('load_step', 'node_id')`` index for multiple FE nodes whose histories
        are scaled versions of the same sequence.
    calculate_P_RAM : bool, optional
        Whether to calculate the ``P_RAM`` damage-parameter path.  Default is
        ``True``.
    calculate_P_RAJ : bool, optional
        Whether to calculate the ``P_RAJ`` damage-parameter path.  Default is
        ``True``.

    Returns
    -------
    dict
        Assessment result.  For ``P_RAM`` it contains
        ``P_RAM_is_life_infinite``, ``P_RAM_lifetime_n_cycles``,
        ``P_RAM_lifetime_n_times_load_sequence``, ``P_RAM_damage_parameter``,
        ``P_RAM_collective``, ``P_RAM_recorder_collective``,
        ``P_RAM_woehler_curve``, ``P_RAM_damage_calculator``, and the two HCM
        detectors ``P_RAM_detector`` and ``P_RAM_detector_1st``.  For ``P_RAJ``
        it contains the analogous ``P_RAJ_*`` keys, plus ``P_RAJ_miner_*`` keys
        for the elementary Miner comparison.  The key ``assessment_parameters``
        stores the copied input series with all derived parameters.  If
        ``P_A = 0.5`` for a single-point assessment, additional lifetime helper
        keys such as ``P_RAM_lifetime_N_1ppm``, ``P_RAM_N_max_bearable``, and
        ``P_RAM_failure_probability`` are available, with analogous ``P_RAJ``
        keys for the ``P_RAJ`` path.

    See Also
    --------
    pylife.strength.fkm_nonlinear.parameter_calculations.calculate_cyclic_assessment_parameters : Derive cyclic material parameters.
    pylife.stress.rainflow.fkm_nonlinear.FKMNonlinearDetector : Count closed hysteresis loops for nonlinear assessment.

    Notes
    -----
    Implements the computational proof of strength according to the FKM
    nonlinear guideline 2019.  Load scatter is treated by one of three guideline
    methods: normal scatter via ``s_L``, lognormal scatter via ``LSD_s``, or the
    blanket factor from ``P_L``.  Set ``P_A = 0.5`` and ``P_L = 50`` to suppress
    statistical safety factors for experiment-like evaluations.
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
    """Assert that ``G`` is a float or node-indexed pandas Series."""

    # check that gradient G is in the correct format
    assert isinstance(assessment_parameters.G, float) \
        or (isinstance(assessment_parameters.G, pd.Series) and not isinstance(assessment_parameters.G.index, pd.MultiIndex)), \
        "stress gradient G is in a wrong format (should be either float or pd.Series indexed by node)"


def _check_K_p_is_in_range(assessment_parameters):
    """Assert that the load shape factor ``K_p`` is at least one."""

    # check that gradient G is in the correct format
    assert assessment_parameters.K_p >= 1, \
        "K_p should be at least 1"

    if assessment_parameters.K_p == 1:
        print("Note, K_p is set to 1 which means only P_RAM can be calculated. "
            f"To use P_RAJ set K_p > 1, e.g. try K_p = 1.001.")


def _scale_load_sequence_according_to_probability(assessment_parameters, load_sequence):
    r"""Scale the load sequence for the requested load occurrence probability.

    Parameters
    ----------
    assessment_parameters : pandas.Series
        Assessment parameters containing ``P_L`` and either ``s_L``, ``LSD_s``,
        or neither to select the blanket load factor.
    load_sequence : pandas.Series or pandas.DataFrame
        Load sequence before statistical scaling.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Load sequence after applying the FKM nonlinear load scatter factor.

    Notes
    -----
    Implements the load distribution treatment of the FKM nonlinear guideline:
    normal scatter, lognormal scatter, or the blanket factor
    ``gamma_L = 1.1`` for ``P_L = 2.5`` percent and ``gamma_L = 1`` for
    ``P_L = 50`` percent.
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
    r"""Scale the load sequence by the transfer factor ``c``.

    Parameters
    ----------
    assessment_parameters : pandas.Series
        Assessment parameters containing transfer factor ``c`` in MPa per load
        unit.
    scaled_load_sequence : pandas.Series or pandas.DataFrame
        Load sequence after statistical scaling.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Load sequence scaled to local equivalent stress.
    """

    # scale load sequence by reference load
    c = assessment_parameters.c
    scaled_load_sequence = scaled_load_sequence.fkm_load_sequence.scaled_by_constant(c)

    return scaled_load_sequence


def _calculate_local_parameters(assessment_parameters):
    r"""Calculate local material and component-independent parameters."""

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
    r"""Compute the ``P_RAM`` and ``P_RAJ`` component Wöhler curves."""

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
    """Run FKM nonlinear HCM counting with the extended Neuber law."""

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
    """Calculate ``P_RAM`` damage and lifetimes and store them in the result."""

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
    """Add ``P_RAM`` post-processing lifetimes for selected probabilities."""

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
    """Store ``P_RAM`` collectives, curves, calculators, and detectors."""

    result["P_RAM_recorder_collective"] = recorder.collective
    result["P_RAM_collective"] = damage_calculator.collective
    result["P_RAM_woehler_curve"] = component_woehler_curve_P_RAM
    result["P_RAM_damage_calculator"] = damage_calculator
    result["P_RAM_detector"] = detector
    result["P_RAM_detector_1st"] = detector_1st
    return result


def _compute_hcm_RAJ(assessment_parameters, scaled_load_sequence):
    """Run FKM nonlinear HCM counting with the Seeger-Beste law."""

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
    """Calculate ``P_RAJ`` damage and lifetimes and store them in the result."""

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
    """Calculate elementary Miner lifetimes for the ``P_RAJ`` collective."""

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
    """Add ``P_RAJ`` post-processing lifetimes for selected probabilities."""

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
    """Store ``P_RAJ`` collectives, curves, calculators, and detectors."""

    # add collectives and objects
    result["P_RAJ_recorder_collective"] = recorder.collective
    result["P_RAJ_collective"] = damage_calculator.collective
    result["P_RAJ_woehler_curve"] = component_woehler_curve_P_RAJ
    result["P_RAJ_damage_calculator"] = damage_calculator
    result["P_RAJ_detector"] = detector
    result["P_RAJ_detector_1st"] = detector_1st

    return result


def _compute_lifetimes_P_RAJ(assessment_parameters, result, scaled_load_sequence, component_woehler_curve_P_RAJ):
    """Compute all requested lifetime results for the ``P_RAJ`` path."""

    detector_1st, detector, seeger_beste_binned, recorder = _compute_hcm_RAJ(assessment_parameters, scaled_load_sequence)
    result["seeger_beste_binned"] = seeger_beste_binned

    result, damage_calculator = _compute_damage_and_lifetimes_RAJ(assessment_parameters, recorder, component_woehler_curve_P_RAJ, result)

    result = _compute_damage_and_lifetimes_RAJ_miner(assessment_parameters, recorder, component_woehler_curve_P_RAJ, result)

    result = _compute_lifetimes_for_failure_probabilities_RAJ(assessment_parameters, result, damage_calculator)

    result = _store_additional_objects_in_result_RAJ(result, recorder, damage_calculator, component_woehler_curve_P_RAJ, detector, detector_1st)
    return result


def _compute_lifetimes_P_RAM(assessment_parameters, result, scaled_load_sequence, component_woehler_curve_P_RAM):
    """Compute all requested lifetime results for the ``P_RAM`` path."""
    detector_1st, detector, extended_neuber_binned, recorder = _compute_hcm_RAM(assessment_parameters, scaled_load_sequence)
    result["extended_neuber_binned"] = extended_neuber_binned

    result, damage_calculator = _compute_damage_and_lifetimes_RAM(assessment_parameters, recorder, component_woehler_curve_P_RAM, result)

    result = _compute_lifetimes_for_failure_probabilities_RAM(assessment_parameters, result, damage_calculator)

    result = _store_additional_objects_in_result_RAM(result, recorder, damage_calculator, component_woehler_curve_P_RAM, detector, detector_1st)
    return result

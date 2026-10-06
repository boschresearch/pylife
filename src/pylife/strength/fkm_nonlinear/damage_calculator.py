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

r"""Calculate FKM nonlinear damage and lifetime for ``P_RAM`` and ``P_RAJ``.

The module provides the two damage calculators used after the HCM algorithm
has produced damage-parameter collectives.  ``DamageCalculatorPRAM`` evaluates
``P_RAM`` collectives by an elementary damage sum, while
``DamageCalculatorPRAJ`` implements the guideline-specific ``P_RAJ``
accumulation with logarithmic classing and crack-growth damage.

Examples
--------
>>> from pylife.strength.fkm_nonlinear.damage_calculator import DamageCalculatorPRAM
>>> DamageCalculatorPRAM.__name__
'DamageCalculatorPRAM'
"""

__author__ = "Benjamin Maier"
__maintainer__ = __author__

import pandas as pd
import numpy as np
import warnings
import scipy.optimize

import pylife.strength.woehler_fkm_nonlinear
import pylife.strength.fkm_nonlinear.parameter_calculations
from pylife.strength.fkm_nonlinear.constants import FKMNLConstants

class DamageCalculatorPRAM:
    r"""Calculate lifetime from a ``P_RAM`` damage-parameter collective.

    Use this calculator for the FKM nonlinear assessment path based on the
    ``P_RAM`` damage parameter.  It expects the first and second HCM runs in one
    collective and evaluates them against a ``WoehlerCurvePRAM`` component Wöhler
    curve.

    Parameters
    ----------
    collective : pandas.DataFrame
        Damage-parameter collective.  Each row represents one hysteresis and must
        contain ``P_RAM`` as damage parameter value, ``is_closed_hysteresis`` as
        full-cycle flag, ``run_index`` as HCM run number, and ``S_min`` for the
        hysteresis count.  A missing index is replaced by a two-level
        ``MultiIndex`` with ``hysteresis_index`` and ``assessment_point_index``.
    component_woehler_curve_P_RAM : WoehlerCurvePRAM
        Component Wöhler curve for the ``P_RAM`` damage parameter.

    Notes
    -----
    Implement the FKM nonlinear guideline assessment for ``P_RAM``, especially the
    load-sequence repetition rule of clause 2.6, equation (2.6-90).  The damage of
    hysteresis ``i`` is accumulated as

    .. math::

        D_i = \begin{cases}
            1 / N_i, & \text{closed hysteresis}, \\
            0.5 / N_i, & \text{memory-3 hysteresis}.
        \end{cases}

    The resulting damage sum is dimensionless; a value of ``1.0`` means failure.
    """

    def __init__(self, collective, component_woehler_curve_P_RAM):
        """Initialize the ``P_RAM`` damage calculation.

        Parameters
        ----------
        collective : pandas.DataFrame
            Damage-parameter collective with one row per hysteresis.  Required columns
            are ``P_RAM``, ``is_closed_hysteresis``, ``run_index``, and ``S_min``.
        component_woehler_curve_P_RAM : WoehlerCurvePRAM
            Component Wöhler curve used to convert ``P_RAM`` values to bearable cycle
            numbers.
        """

        self._collective = collective.copy()
        self._component_woehler_curve_P_RAM = component_woehler_curve_P_RAM

        self._P_RAM_Z = self._component_woehler_curve_P_RAM.P_RAM_Z

        self._initialize_collective_index()
        self._initialize_P_RAM_Z_index()

        # compute bearable number of cycles
        self._collective["N"] = np.where(self._collective["P_RAM"] >= self._P_RAM_Z,
                                         1e3 * np.power(self._collective["P_RAM"] / self._P_RAM_Z, 1/self._component_woehler_curve_P_RAM.d_1),
                                         1e3 * np.power(self._collective["P_RAM"] / self._P_RAM_Z, 1/self._component_woehler_curve_P_RAM.d_2))

        # compute individual damage per cycle
        self._collective["D"] = np.where(self._collective["is_closed_hysteresis"], 1/self._collective["N"], 0.5/self._collective["N"])

        # compute cumulative damage for every node
        self._collective["cumulative_damage"] = self._collective["D"].groupby("assessment_point_index").cumsum()

        # compute number of cycles until damage sum is 1
        self._n_cycles_until_damage = self._collective["cumulative_damage"].groupby("assessment_point_index").apply(lambda array: np.searchsorted(array, 1))

    @property
    def collective(self):
        """Return the evaluated ``P_RAM`` collective.

        Returns
        -------
        pandas.DataFrame
            Copy of the input collective enriched with bearable cycle numbers,
            damage per hysteresis, and cumulative damage per assessment point.
        """
        return self._collective

    @property
    def P_RAM_max(self):
        """Return the maximum ``P_RAM`` value of the second HCM run.

        Returns
        -------
        float or pandas.Series
            Maximum ``P_RAM`` damage parameter per assessment point.  Values below or
            equal to the fatigue strength limit indicate infinite life for this
            criterion.
        """

        # get maximum damage parameter
        P_RAM_max = self._collective.loc[self._collective["run_index"]==2, "P_RAM"].groupby("assessment_point_index").max()

        return P_RAM_max.squeeze()

    @property
    def is_life_infinite(self):
        """Return whether the ``P_RAM`` assessment predicts infinite life.

        Returns
        -------
        bool or pandas.Series
            ``True`` where ``P_RAM_max`` does not exceed the fatigue strength limit of
            the component Wöhler curve; otherwise ``False``.
        """

        # y/x = d, 1/d = x/y

        fatigue_strength_limit = self._component_woehler_curve_P_RAM.fatigue_strength_limit

        # remove any given index of fatigue_strength_limit which results from the assessment_parameters.G parameter
        if isinstance(fatigue_strength_limit, pd.Series):
            fatigue_strength_limit.reset_index(drop=True, inplace=True)

        result = self.P_RAM_max <= fatigue_strength_limit
        return result.squeeze()

    @property
    def lifetime_n_times_load_sequence(self):
        """Return load-sequence repetitions until ``P_RAM`` failure.

        Returns
        -------
        float or numpy.ndarray
            Number of complete load sequence repetitions until the damage sum reaches
            ``1.0``.  A value of ``0`` means that failure occurs before the end of the
            second HCM run.
        """

        # compute damage sums of both HCM runs
        damage_sum_first_run = self._collective.loc[self._collective["run_index"]==1, "D"].groupby("assessment_point_index").sum()
        damage_sum_second_run = self._collective.loc[self._collective["run_index"]==2, "D"].groupby("assessment_point_index").sum()

        # fill default values for all assessment point where there is no value
        damage_sum_first_run = self._fill_with_default_for_missing_assessment_points(damage_sum_first_run, 0)
        damage_sum_second_run = self._fill_with_default_for_missing_assessment_points(damage_sum_second_run, 0)

        # how often the second run can be repeated (after the first run) until damage
        # eq. (2.6-90)
        x = np.where(damage_sum_first_run == 0,
                1 / damage_sum_second_run,
                (1 - damage_sum_first_run) / damage_sum_second_run)

        # If damage sum of D=1 is reached before end of second run of HCM algorithm, set lifetime_n_times_load_sequence to 0.
        # Else the value is x + 1
        result = np.where(self._n_cycles_until_damage < self._n_hystereses,
                          0,
                          x + 1)

        # store value of x
        self._x = x

        return result.squeeze()

    @property
    def lifetime_n_cycles(self):
        """Return load cycles until ``P_RAM`` failure.

        Returns
        -------
        float or numpy.ndarray
            Number of cycles in the collective until the accumulated damage sum
            reaches ``1.0``.  The value is expressed in counted load cycles, not in
            seconds or load-sequence repetitions.
        """

        x_plus_1 = self.lifetime_n_times_load_sequence

        # if damage sum of D=1 is reached before end of second run of HCM algorithm
        lifetime = np.where(self._n_cycles_until_damage < self._n_hystereses,
                            self._n_cycles_until_damage,
                            x_plus_1 * self._n_hystereses_run_2)

        return lifetime.squeeze()

    def get_lifetime_functions(self, assessment_parameters):
        """Return probabilistic lifetime functions for the ``P_RAM`` assessment.

        The returned callables scale the deterministic lifetime with the material
        scatter and safety factor of the FKM nonlinear guideline.

        Parameters
        ----------
        assessment_parameters : pandas.Series
            Assessment parameters.  Only the material group is required here to select
            the ``f_2.5%`` scatter constant for ``P_RAM``.

        Returns
        -------
        N_max_bearable : callable
            Function ``N_max_bearable(P_A, clip_gamma=False)`` returning the maximum
            bearable number of cycles for failure probability ``P_A``.
        failure_probability : callable
            Function ``failure_probability(N)`` returning the failure probability for
            ``N`` load cycles.

        Notes
        -----
        Use FKM nonlinear guideline clause 2.5, equation (2.5-38), for the material
        safety factor.  If ``clip_gamma`` is ``True``, ``gamma_M`` is clipped to at
        least ``1.1`` for the ``P_RAM`` assessment.
        """

        constants = FKMNLConstants().for_material_group(assessment_parameters)

        f_25 = constants.f_25percent_material_woehler_RAM

        def N_max_bearable(P_A, clip_gamma=False):
            beta = pylife.strength.fkm_nonlinear.parameter_calculations.compute_beta(P_A)
            log_gamma_M = (0.8*beta - 2)*0.08

            # Note that the FKM nonlinear guideline defines a cap at 1.1 for P_RAM.
            if clip_gamma:
                log_gamma_M = max(log_gamma_M, np.log10(1.1))

            reduction_factor_P = np.log10(f_25) - log_gamma_M

            # Note: P_A shifts the woehler curve, this may switch from slope d_1 to slope d_2, slope_woehler is not constant, but depends on P_A.
            # Therefore, we do the woehler curve assessment again:

            # compute bearable number of cycles
            P_RAM_reduced = self._P_RAM_Z * 10**(reduction_factor_P)
            self._collective["N"] = np.where(self._collective["P_RAM"] >= P_RAM_reduced,
                                            1e3 * np.power(self._collective["P_RAM"] / P_RAM_reduced, 1/self._component_woehler_curve_P_RAM.d_1),
                                            1e3 * np.power(self._collective["P_RAM"] / P_RAM_reduced, 1/self._component_woehler_curve_P_RAM.d_2))

            # compute individual damage per cycle
            self._collective["D"] = np.where(self._collective["is_closed_hysteresis"], 1/self._collective["N"], 0.5/self._collective["N"])

            return self.lifetime_n_cycles

        def failure_probability(N):

            result = scipy.optimize.minimize_scalar(
                lambda x: (N_max_bearable(x) - N) ** 2,
                bounds=[1e-9, 1-1e-9], method='bounded', options={'xatol': 1e-10})

            if result.success:
                return result.x
            else:
                return 0

        return N_max_bearable, failure_probability

    def _initialize_collective_index(self):
        """Validate and, if necessary, create the collective index.

        The method requires the columns used by the ``P_RAM`` damage calculation and
        stores the number of hysteresis entries for later lifetime conversion.
        """

        # if assessment is done for multiple points at once, work with a multi-indexed data frame
        if not isinstance(self._collective.index, pd.MultiIndex):
            n_hystereses = len(self._collective)
            self._collective.index = pd.MultiIndex.from_product([range(n_hystereses), [0]], names=["hysteresis_index", "assessment_point_index"])

        # assert that the index contains the two columns "hysteresis_index" and "assessment_point_index"
        assert self._collective.index.names == ["hysteresis_index", "assessment_point_index"]

        assert "P_RAM" in self._collective
        assert "is_closed_hysteresis" in self._collective
        assert "run_index" in self._collective

        # store some statistics about the DataFrame
        self._n_hystereses = self._collective.groupby("assessment_point_index")["S_min"].count().values[0]
        self._n_hystereses_run_2 = self._collective[self._collective["run_index"]==2].groupby("assessment_point_index")["S_min"].count().values[0]

    def _initialize_P_RAM_Z_index(self):
        """Align node-dependent ``P_RAM_Z`` values with the collective index.

        If the Wöhler curve provides one ``P_RAM_Z`` value per assessment point, expand
        it to the hysteresis-level ``MultiIndex`` used by the collective.
        """
        # if P_RAM_Z is a series without multi-index
        if isinstance(self._P_RAM_Z, pd.Series):
            if not isinstance(self._P_RAM_Z.index, pd.MultiIndex):

                n_hystereses = len(self._collective.index.get_level_values("hysteresis_index").unique())
                self._P_RAM_Z = pd.Series(
                    data = (np.ones([n_hystereses,1]) * np.array([self._P_RAM_Z])).flatten(),
                    index = self._collective.index)

    def _fill_with_default_for_missing_assessment_points(self, df, default_value):
        """Return a series containing all assessment points.

        Parameters
        ----------
        df : pandas.Series
            Series indexed by ``assessment_point_index`` that may miss some assessment
            points.
        default_value : float
            Value inserted for missing assessment points.

        Returns
        -------
        pandas.Series
            Series indexed by every assessment point in the collective.
        """
        assessment_point_index = self._collective.index.get_level_values("assessment_point_index").unique()
        series_with_all_rows = pd.Series(np.nan, index=assessment_point_index, name="a")

        result = pd.concat([df, series_with_all_rows],axis=1)[df.name]
        result = result.fillna(default_value)
        return result


class DamageCalculatorPRAJ:
    r"""Calculate lifetime from a ``P_RAJ`` collective by guideline accumulation.

    Use this calculator for the official FKM nonlinear ``P_RAJ`` assessment.  It
    uses the damage values computed during damage-parameter evaluation, classes the
    second HCM run logarithmically, and accounts for the guideline's crack-growth
    based degradation of the endurance limit.

    Parameters
    ----------
    collective : pandas.DataFrame
        Damage-parameter collective with one row per hysteresis.  Required columns
        are ``P_RAJ``, ``P_RAJ_D``, ``D``, ``run_index``, and ``S_min``.  A missing
        index is replaced by a two-level ``MultiIndex`` with ``hysteresis_index``
        and ``assessment_point_index``.
    assessment_parameters : pandas.Series
        Parameters of the nonlinear ``P_RAJ`` assessment, including
        ``P_RAJ_klass_max``, ``P_RAJ_D_e``, ``d_RAJ``, ``a_0``, ``a_end``,
        ``l_star``, and optionally ``n_bins``.  Default for ``n_bins`` is ``200``.
    component_woehler_curve_P_RAJ : WoehlerCurvePRAJ
        Component Wöhler curve for the ``P_RAJ`` damage parameter.

    See Also
    --------
    DamageCalculatorPRAM : Calculate damage from ``P_RAM`` with elementary accumulation.

    Notes
    -----
    Implement the FKM nonlinear guideline assessment for ``P_RAJ`` in clause 2.9,
    including equations (2.9-126), (2.9-135), and (2.9-138).  The class evaluates

    .. math::

        \bar{N} = H_0\,(2 + \bar{x}_{-2})

    as the bearable number of cycles until crack initiation.  Choose
    ``DamageCalculatorPRAJMinerElementary`` instead only for the modified
    Miner-type ``P_RAJ`` accumulation, not for a strict guideline assessment.
    """

    def __init__(self, collective, assessment_parameters, component_woehler_curve_P_RAJ):
        """Initialize the guideline ``P_RAJ`` damage calculation.

        Parameters
        ----------
        collective : pandas.DataFrame
            Damage-parameter collective with one row per hysteresis.  Required columns
            are ``P_RAJ``, ``P_RAJ_D``, ``D``, ``run_index``, and ``S_min``.
        assessment_parameters : pandas.Series
            Assessment parameters controlling classing, crack growth, and the number
            of logarithmic bins.  Default for missing ``n_bins`` is ``200``.
        component_woehler_curve_P_RAJ : WoehlerCurvePRAJ
            Component Wöhler curve used by the ``P_RAJ`` assessment.
        """

        self._collective = collective
        self._assessment_parameters = assessment_parameters
        self._component_woehler_curve_P_RAJ = component_woehler_curve_P_RAJ
        self._P_RAJ_D_0 = self._component_woehler_curve_P_RAJ.fatigue_strength_limit

        self._initialize_collective_index()

        # get number of bins for P_RAJ
        if "n_bins" not in self._assessment_parameters:
            self._assessment_parameters.n_bins = 200

        n_bins = self._assessment_parameters.n_bins

        # setup the lookup table "self._binned_P_RAJ" and self._binned_h
        self._initialize_binning()

        # compute cumulative damage for every node
        self._collective["cumulative_damage"] = self._collective["D"].groupby("assessment_point_index").cumsum()

        # compute number of cycles until damage sum is 1, eq. (2.9-136)
        self._n_cycles_until_damage = self._collective["cumulative_damage"].groupby("assessment_point_index").apply(lambda array: np.searchsorted(array, 1))

        # calculate the value of self._xbar_minus_2
        self._compute_xbar_minus_2()

        # eq. (2.8-93)
        self._N_minus_2 = self._H_0 * self._xbar_minus_2

        # compute bearable number of cycles until crack
        self._N_bar = self._H_0 * (2 + self._xbar_minus_2)

        self._x_bar = (2 + self._xbar_minus_2)

    @property
    def collective(self):
        """Return the evaluated ``P_RAJ`` collective.

        Returns
        -------
        pandas.DataFrame
            Input collective enriched with cumulative damage per assessment point.
        """
        return self._collective

    @property
    def P_RAJ_max(self):
        """Return the maximum ``P_RAJ`` value of the second HCM run.

        Returns
        -------
        float or pandas.Series
            Maximum ``P_RAJ`` damage parameter per assessment point.  Values below or
            equal to the fatigue strength limit indicate infinite life for this
            criterion.
        """

        # get maximum damage parameter
        if isinstance(self._collective.index, pd.MultiIndex):
            P_RAJ_max = self._collective.loc[self._collective["run_index"]==2, "P_RAJ"].groupby("assessment_point_index").max()
        else:
            P_RAJ_max = self._collective.loc[self._collective["run_index"]==2, "P_RAJ"].max()

        return P_RAJ_max.squeeze()

    @property
    def is_life_infinite(self):
        """Return whether the ``P_RAJ`` assessment predicts infinite life.

        Returns
        -------
        bool or pandas.Series
            ``True`` where ``P_RAJ_max`` does not exceed the fatigue strength limit of
            the component Wöhler curve; otherwise ``False``.
        """

        result = self.P_RAJ_max <= self._component_woehler_curve_P_RAJ.fatigue_strength_limit
        return result.squeeze()

    @property
    def lifetime_n_times_load_sequence(self):
        """Return load-sequence repetitions until ``P_RAJ`` failure.

        Returns
        -------
        float or numpy.ndarray
            Number of complete load sequence repetitions until the damage sum reaches
            ``1.0``.  A value of ``0`` means that failure occurs before the end of the
            second HCM run.
        """

        # If damage sum of D=1 is reached before end of second run of HCM algorithm, set lifetime_n_times_load_sequence to 0.
        # Else the value is x + 1
        result = np.where(self._n_cycles_until_damage < self._n_hystereses,
                          0,
                          self._x_bar)
        return result.squeeze()

    @property
    def lifetime_n_cycles(self):
        """Return load cycles until ``P_RAJ`` failure.

        Returns
        -------
        float or numpy.ndarray
            Number of cycles in the collective until crack initiation according to the
            guideline ``P_RAJ`` accumulation.
        """

        # if damage sum of D=1 is reached before end of second run of HCM algorithm
        lifetime = np.where(self._n_cycles_until_damage < self._n_hystereses,
                            self._n_cycles_until_damage,
                            self._N_bar)

        return lifetime.squeeze()

    def get_lifetime_functions(self):
        """Return probabilistic lifetime functions for the ``P_RAJ`` assessment.

        Returns
        -------
        N_max_bearable : callable
            Function ``N_max_bearable(P_A, clip_gamma=False)`` returning the maximum
            bearable number of cycles for failure probability ``P_A``.
        failure_probability : callable
            Function ``failure_probability(N)`` returning the failure probability for
            ``N`` load cycles.

        Notes
        -----
        Use FKM nonlinear guideline clause 2.8, equation (2.8-38), for the material
        safety factor.  If ``clip_gamma`` is ``True``, ``gamma_M`` is clipped to at
        least ``1.2`` for the ``P_RAJ`` assessment.
        """

        constants = FKMNLConstants().for_material_group(self._assessment_parameters)
        f_25 = constants.f_25percent_material_woehler_RAJ
        slope_woehler = abs(1/self._component_woehler_curve_P_RAJ.d)
        lifetime_n_cycles = self.lifetime_n_cycles

        def N_max_bearable(P_A, clip_gamma=False):
            beta = pylife.strength.fkm_nonlinear.parameter_calculations.compute_beta(P_A)
            log_gamma_M = (0.8*beta - 2)*0.155

            # Note that the FKM nonlinear guideline defines a cap at 1.2 for P_RAJ.
            if clip_gamma:
                log_gamma_M = max(log_gamma_M, np.log10(1.2))

            reduction_factor_P = np.log10(f_25) - log_gamma_M
            reduction_factor_N = reduction_factor_P * slope_woehler

            return lifetime_n_cycles * 10**(reduction_factor_N)

        def failure_probability(N):

            result = scipy.optimize.minimize_scalar(
                lambda x: (N_max_bearable(x) - N) ** 2,
                bounds=[1e-9, 1-1e-9], method='bounded', options={'xatol': 1e-10})

            if result.success:
                return result.x
            else:
                return 0

        return N_max_bearable, failure_probability

    def _initialize_collective_index(self):
        """Validate and, if necessary, create the collective index.

        The method requires the columns used by the ``P_RAJ`` guideline damage
        calculation and stores the number of hysteresis entries for lifetime
        conversion.
        """

        # if assessment is done for multiple points at once, work with a multi-indexed data frame
        if not isinstance(self._collective.index, pd.MultiIndex):
            n_hystereses = len(self._collective)
            self._collective.index = pd.MultiIndex.from_product([range(n_hystereses), [0]], names=["hysteresis_index", "assessment_point_index"])

        # assert that the index contains the two columns "hysteresis_index" and "assessment_point_index"
        assert self._collective.index.names == ["hysteresis_index", "assessment_point_index"]

        assert "run_index" in self._collective
        assert "D" in self._collective
        assert "P_RAJ" in self._collective

        # store some statistics about the DataFrame
        self._n_hystereses = self._collective.groupby("assessment_point_index")["S_min"].count().values[0]

    def _initialize_binning(self):
        """Create logarithmic ``P_RAJ`` classes for the second HCM run.

        The class boundaries and class-middle values implement the guideline's
        ``Klassierung`` step.  The counts are stored per assessment point so that
        several nodes can be evaluated in one calculator instance.
        """

        # initialize the classes for P_RAJ
        P_RAJ_klass_max = self._assessment_parameters.P_RAJ_klass_max
        P_RAJ_D_e = self._assessment_parameters.P_RAJ_D_e
        n_bins = self._assessment_parameters.n_bins

        # eq. (2.9-126)
        delta_P = 1.0/n_bins * np.log(P_RAJ_klass_max / P_RAJ_D_e)

        # initalize binned P_RAJ values with equal logarithmic class sizes
        self._binned_P_RAJ = np.logspace(np.log10(P_RAJ_klass_max), np.log10(P_RAJ_D_e), n_bins+1)  # 201 (n_bins+1) because entry 0 is P_RAJ_class_max

        # assert that upper bin size divided by lower bin size is equal for all bins
        log_bin_sizes = [self._binned_P_RAJ[i-1] / self._binned_P_RAJ[i] for i in range(1,n_bins+1)]


        #assert np.nanstd(log_bin_sizes) < 1e-10
        if np.nanstd(log_bin_sizes) >= 1e-10:
            warnings.warn(f"std(log_bin_sizes) should be zero, but is {np.nanstd(log_bin_sizes)}.")

        # at this point, the vectorial assertions should hold, masking out nan values

        # class middle points, eq. (2.9.131)
        self._binned_P_RAJ_m = (self._binned_P_RAJ[:n_bins] + self._binned_P_RAJ[1:]) / 2

        # Now `Klassieren(P_RAJ_i, h_i, P_RAJ_m_i)`, 1 <= i <= 200   (200 is the standard value for n_bins)
        # equals (self._binned_P_RAJ[i+1], self._binned_h[i], self._binned_P_RAJ_m[i]), 0 <= i < 200.
        # The corresponding class `i` for a P_RAJ value is given such that:
        #    self._binned_P_RAJ[i] <= P_RAJ <= self._binned_P_RAJ[i+1]
        # and self._binned_P_RAJ_m[i] is the corresponding center point
        #
        # instead of self._binned_h[class_index], we use self._binned_h[assessment_point_index, class_index]

        # fill binned P_RAJ values for second run of HCM algorithm

        n_assessment_points = len(self._collective[self._collective.index.get_level_values("hysteresis_index")==0])
        self._binned_h = np.zeros((n_assessment_points,n_bins))
        self._n_not_in_bin = np.zeros(n_assessment_points)

        for index, group in self._collective[self._collective.run_index == 2].groupby("hysteresis_index"):

            # if we have a different stress gradient for every node, the bin contains different values for each node
            if len(self._binned_P_RAJ.shape) == 2:

                # this is the vectorial case where the stress gradient is different for every node,
                # every bin contains one value for every node

                def find(row):
                    return n_bins - np.searchsorted(np.flip(self._binned_P_RAJ, axis=0)[:,int(row["index"])], row.P_RAJ)

                i = group.reset_index().reset_index()[["index","P_RAJ"]].apply(find, axis=1)

            else:
                # scalar case, each bin contains one value

                # find class index of binned P_RAJ value
                i = group.P_RAJ.apply(lambda P_RAJ: n_bins - np.searchsorted(np.flip(self._binned_P_RAJ, axis=0), P_RAJ))

            # here, we have:
            #   self._binned_P_RAJ[i] <= P_RAJ <= self._binned_P_RAJ[i+1]

            # if P_RAJ > P_RAJ_D_e:
            # increment binned at the corresponding class, equivalent to self._binned_h[i] += 1
            increment = np.zeros((n_assessment_points,n_bins+1))
            increment[np.array(range(n_assessment_points)), i] = np.where(group.reset_index(drop=True).P_RAJ > P_RAJ_D_e, 1, 0)

            self._binned_h = self._binned_h + increment[:,:n_bins]

            # if P_RAJ <= P_RAJ_D_e:
            # increment n_not_in_bin
            self._n_not_in_bin += np.where(group.reset_index().P_RAJ <= P_RAJ_D_e, 1, 0)

        # eq. (2.9-135)
        self._H_0 = np.sum(self._binned_h, axis=1) + self._n_not_in_bin

    def _compute_xbar_minus_2(self):
        """Compute the additional sequence repetitions after the second run.

        The value ``xbar_minus_2`` corresponds to FKM nonlinear guideline equation
        (2.9-138) and is later converted to cycles by multiplication with ``H_0``.
        """

        n_bins = self._assessment_parameters.n_bins
        last_P_RAJ_D = self._collective["P_RAJ_D"].groupby("assessment_point_index").last()

        # find corresponding class `q` for value last_P_RAJ_D

        # if we have a different stress gradient for every node, the bin contains different values for each node
        if len(self._binned_P_RAJ.shape) == 2:

            def find(row):
                return n_bins - np.searchsorted(np.flip(self._binned_P_RAJ, axis=0)[:,int(row["index"])], row.P_RAJ_D)

            q = last_P_RAJ_D.reset_index().reset_index().apply(find, axis=1)

        else:
            # scalar case, each bin contains one value
            q = last_P_RAJ_D.apply(lambda P_RAJ: n_bins - np.searchsorted(np.flip(self._binned_P_RAJ), P_RAJ))

        # here, we have:
        #   self._binned_P_RAJ[q] <= last_P_RAJ_D <= self._binned_P_RAJ[q+1]
        # and self._binned_P_RAJ_m[q] is the corresponding center point

        # standard calculation of m according to eq. (2.8-60), (2.9-117)
        m = -1/self._assessment_parameters.d_RAJ

        # definition of the function f of eq. (2.9-139)
        def f(j):
            denominator = np.power(self._assessment_parameters.a_0, 1-m) - np.power(self._assessment_parameters.a_end, 1-m)
            bracket = self._P_RAJ_D_0 / self._binned_P_RAJ_m[j] \
                * (self._assessment_parameters.a_0 + self._assessment_parameters.l_star \
                   * (1 - self._binned_P_RAJ_m[j]/self._P_RAJ_D_0))
            nominator = np.power(self._assessment_parameters.a_0, 1-m) - np.power(bracket, 1-m)
            return nominator / denominator

        # store internal variables
        self._f = f
        self._q = q
        self._last_P_RAJ_D = last_P_RAJ_D
        self._m = m

        # eq. (2.9-138)
        self._xbar_minus_2 = np.zeros_like(q, dtype=float)

        denominator = np.zeros_like(q, dtype=float)
        previous_j = 0

        # iterate from j = q to 198 for all assessment points at once.
        # The values where j is smaller than q are masked out at the end.
        for j in range(min(q), n_bins-1):

            # compute sum in denominator, only one new summand is added in this iteration,
            # we can avoid doing the complete inner sum over i for every new j
            for i in range(previous_j, j+1):
                P_RAJ_m = self._binned_P_RAJ_m[i]    # this corresponds to the range [self._binned_P_RAJ[i], self._binned_P_RAJ[i+1]]

                # eq. (2.9-140)
                #N = (P_RAJ_m / self._component_woehler_curve_P_RAJ.P_RAJ_Z) ** (1/self._component_woehler_curve_P_RAJ.d)
                N = self._component_woehler_curve_P_RAJ.calc_N(P_RAJ_m, P_RAJ_D=last_P_RAJ_D)

                damage = np.where(P_RAJ_m > last_P_RAJ_D,
                                    self._binned_h[:,i] / N,

                                    # for N = inf (infinite life), damage is zero
                                    0)

                denominator += damage

            previous_j = j

            # silence warning "divide by zero encountered in true_divide". This happens for denominator=0, but then it will use the second branch with np.inf anyways
            with np.errstate(divide='ignore'):
                self._xbar_minus_2 += np.where(j >= q,
                                           np.where(abs(denominator) > 1e-13,
                                                    (f(j+1) - f(j)) / denominator,
                                                    np.inf),
                                           0)

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

import pandas as pd
import numpy as np
import warnings
import scipy.optimize

import pylife.strength.woehler_fkm_nonlinear
import pylife.strength.fkm_nonlinear.parameter_calculations
from pylife.strength.fkm_nonlinear.constants import FKMNLConstants

class DamageCalculatorPRAM:
    """This class performs the lifetime assessment according to the FKM nonlinear assessment.
    It holds damage values from two previous runs of the HCM algorithm and a component woehler curve of type
    `WoehlerCurvePRAM`. The outputs are lifetime numbers and detection of infinite life.

    This class implements the assessment procedure using the damage parameter P_RAM.
    """

    def __init__(self, collective, component_woehler_curve_P_RAM):
        """Initialize the computation with the Woehler curve, connect a load collective and assessment parameters to the object.

        This function should be called once after initialization and before any other method is called.

        Parameters
        ----------
        collective : pandas DataFrame
            A load collective with computed damage parameter resulting from two runs of the HCM algorithm.
            Every row corresponds to one closed hysteresis.

            More specifically, the table has to contain the following columns:

            * ``P_RAM``: the value of the P_RAM damage parameter for every hysteresis.
            * ``is_closed_hysteresis``: The hysteresis fully lies on the secondary branch and is fully counted (True),
              or it results from a "Memory 3" case and is only counted half the damage. (False)
            * ``run_index`` number of the run of the HCM algorithm, either 1 for the first hystereses or 2 for the following ones.

            The collective has to contain no index at all or a MultiIndex with two levels named
            `hysteresis_index` and `assessment_point_index`. The first index level increments for every new hysteresis.
            The second index level identifies a separate assessment point, e.g., a mesh node for which the assessment should be carried out.
            It it, thus, possible to perform the assessment against this woehler curve for multiple points at once.

        component_woehler_curve_P_RAM : object of class `pylife.strength.woehler_fkm_nonlinear.WoehlerCurvePRAM`
            The compoennt woehler curve used for the assessment.

        Returns
        -------
        None.

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
        return self._collective

    @property
    def P_RAM_max(self):
        """The maximum P_RAM damage parameter value of the second run of the HCM algorithm.
        If this value is lower than the fatigue strength limit, the component has infinite life.
        The method ``compute_damage`` needs to be called beforehand."""

        # get maximum damage parameter
        P_RAM_max = self._collective.loc[self._collective["run_index"]==2, "P_RAM"].groupby("assessment_point_index").max()

        return P_RAM_max.squeeze()

    @property
    def is_life_infinite(self):
        """Whether the component has infinite life.
        The method ``compute_damage`` needs to be called beforehand."""

        # y/x = d, 1/d = x/y

        fatigue_strength_limit = self._component_woehler_curve_P_RAM.fatigue_strength_limit

        # remove any given index of fatigue_strength_limit which results from the assessment_parameters.G parameter
        if isinstance(fatigue_strength_limit, pd.Series):
            fatigue_strength_limit.reset_index(drop=True, inplace=True)

        result = self.P_RAM_max <= fatigue_strength_limit
        return result.squeeze()

    @property
    def lifetime_n_times_load_sequence(self):
        """The number of times the whole load sequence can be traversed until failure.
        The method ``compute_damage`` needs to be called beforehand."""

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
        """The number of load cycles (as defined in the load collective) until failure.
        The method ``compute_damage`` needs to be called beforehand."""

        x_plus_1 = self.lifetime_n_times_load_sequence

        # if damage sum of D=1 is reached before end of second run of HCM algorithm
        lifetime = np.where(self._n_cycles_until_damage < self._n_hystereses,
                            self._n_cycles_until_damage,
                            x_plus_1 * self._n_hystereses_run_2)

        return lifetime.squeeze()

    def get_lifetime_functions(self, assessment_parameters):
        """Return two python functions that can be used for detailed probabilistic assessment.

        The first function, ``N_max_bearable(P_A, clip_gamma=False)``,
        calculates the maximum number of cycles
        that the component can withstand with the given failure probability.
        The parameter ``clip_gamma`` specifies whether the scaling factor gamma_M
        will be at least 1.1 (P_RAM) or 1.2 (P_RAJ), as defined
        in eq. (2.5-38) (PRAM) / eq. (2.8-38) (PRAJ)

        The second function, ``failure_probability(N)``, calculates the failure probability for a
        given number of cycles.

        Parameters
        ----------
        assessment_parameters : pd.Series
            The assessment parameters, only the material group is required to determine the respective
            f_2,5% constant.

        Returns
        -------
        N_max_bearable
            python function, ``N_max_bearable(P_A)``
        failure_probability
            python function, ``failure_probability(N)``
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
        """Assert that the variable `self._collective` contains the proper index
        and columns. If the collective does not contain any MultiIndex (or any index at all),
        create the appropriate MultiIndex"""

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
        """Properly initialize P_RAM_Z if it was computed individually for every node because of a stress gradient field G.
        In such a case, add the proper multi-index with levels "hysteresis_index", "assessment_point_index" such that
        the Series is compatible with self._collective.
        """
        # if P_RAM_Z is a series without multi-index
        if isinstance(self._P_RAM_Z, pd.Series):
            if not isinstance(self._P_RAM_Z.index, pd.MultiIndex):

                n_hystereses = len(self._collective.index.get_level_values("hysteresis_index").unique())
                self._P_RAM_Z = pd.Series(
                    data = (np.ones([n_hystereses,1]) * np.array([self._P_RAM_Z])).flatten(),
                    index = self._collective.index)

    def _fill_with_default_for_missing_assessment_points(self, df, default_value):
        """Add rows to a series df that are not yet there, such that the result has
        a row for every assessment point. Example:

        * input ``df``:

            .. code::

                assessment_point_index
                0    0.312183
                2    0.312183
                Name: stddev_log_N, dtype: float64

        * default value: 5
        * result:

            .. code::

                assessment_point_index
                0    0.312183
                1    5.000000
                2    0.312183
                Name: stddev_log_N, dtype: float64
        """
        assessment_point_index = self._collective.index.get_level_values("assessment_point_index").unique()
        series_with_all_rows = pd.Series(np.nan, index=assessment_point_index, name="a")

        result = pd.concat([df, series_with_all_rows],axis=1)[df.name]
        result = result.fillna(default_value)
        return result


class DamageCalculatorPRAJ:
    """This class performs the lifetime assessment according to the FKM nonlinear assessment.
    It holds damage values from two previous runs of the HCM algorithm and a component woehler curve of type
    `WoehlerCurvePRAJ`. The outputs are lifetime numbers and detection of infinite life.

    This class implements the assessment procedure using the damage parameter P_RAJ.
    """

    def __init__(self, collective, assessment_parameters, component_woehler_curve_P_RAJ):
        """Initialize the computation with the Woehler curve, connect a load collective and material parameters to the object.

        This function should be called once after initialization, before any other method is called.

        Parameters
        ----------
        collective : pandas DataFrame
            A load collective with computed damage parameter resulting from two runs of the HCM algorithm.
            Every row corresponds to one closed hysteresis.

            More specifically, the table has to contain the following columns:

            * ``P_RAJ``: the value of the P_RAJ damage parameter for every hysteresis.
            * ``D``: The amount of damage of the hysteresis.
            * ``run_index`` number of the run of the HCM algorithm, either 1 for the first hystereses or 2 for the following ones.

        assessment_parameters : pandas Series
            All assessment parameters collected so far. Has to contain the following fields:
            * ``P_RAJ_klass_max``
            * ``P_RAJ_D_e``
            * ``P_RAJ_D_0``
            * ``d_RAJ``
            * ``a_0``
            * ``a_end``
            * ``l_star``
            * ``P_RAJ_Z``
            * ``d_RAJ``
            * ``n_bins``: number of bins in the lookup table for speed up, "Klassierung", a larger value is more accurate but leads to longer runtimes (optional, default is 200)
            Refer to the FKM nonlinear document for a description of these parameters.

        Returns
        -------
        None.
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
        return self._collective

    @property
    def P_RAJ_max(self):
        """The maximum P_RAJ damage parameter value of the second run of the HCM algorithm.
        If this value is lower than the fatigue strength limit, the component has infinite life.
        The method ``compute_damage`` needs to be called beforehand."""

        # get maximum damage parameter
        if isinstance(self._collective.index, pd.MultiIndex):
            P_RAJ_max = self._collective.loc[self._collective["run_index"]==2, "P_RAJ"].groupby("assessment_point_index").max()
        else:
            P_RAJ_max = self._collective.loc[self._collective["run_index"]==2, "P_RAJ"].max()

        return P_RAJ_max.squeeze()

    @property
    def is_life_infinite(self):
        """Whether the component has infinite life.
        The method ``compute_damage`` needs to be called beforehand."""

        result = self.P_RAJ_max <= self._component_woehler_curve_P_RAJ.fatigue_strength_limit
        return result.squeeze()

    @property
    def lifetime_n_times_load_sequence(self):
        """The number of times the whole load sequence can be traversed until failure.
        The method ``compute_damage`` needs to be called beforehand."""

        # If damage sum of D=1 is reached before end of second run of HCM algorithm, set lifetime_n_times_load_sequence to 0.
        # Else the value is x + 1
        result = np.where(self._n_cycles_until_damage < self._n_hystereses,
                          0,
                          self._x_bar)
        return result.squeeze()

    @property
    def lifetime_n_cycles(self):
        """The number of load cycles (as defined in the load collective) until failure.
        The method ``compute_damage`` needs to be called beforehand."""

        # if damage sum of D=1 is reached before end of second run of HCM algorithm
        lifetime = np.where(self._n_cycles_until_damage < self._n_hystereses,
                            self._n_cycles_until_damage,
                            self._N_bar)

        return lifetime.squeeze()

    def get_lifetime_functions(self):
        """Return two python functions that can be used for detailed probabilistic assessment.

        The first function, ``N_max_bearable(P_A, clip_gamma=False)``,
        calculates the maximum number of cycles
        that the component can withstand with the given failure probability.
        The parameter ``clip_gamma`` specifies whether the scaling factor gamma_M
        will be at least 1.1 (P_RAM) or 1.2 (P_RAJ), as defined
        in eq. (2.5-38) (PRAM) / eq. (2.8-38) (PRAJ)        The second function, ``failure_probability(N)``, calculates the failure probability for a
        given number of cycles.

        Parameters
        ----------
        assessment_parameters : pd.Series
            The assessment parameters, only the material group is required to determine the respective
            f_2,5% constant.

        Returns
        -------
        N_max_bearable
            python function, ``N_max_bearable(P_A)``
        failure_probability
            python function, ``failure_probability(N)``
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
        """Assert that the variable `self._collective` contains the proper index
        and columns. If the collective does not contain any MultiIndex (or any index at all),
        create the appropriate MultiIndex"""

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
        """
        Create a lookup table for P_RAJ values, in ``self._binned_P_RAJ``

        The following vectorial assertions hold:

        .. code::

            if isinstance(delta_P, pd.Series):
                assert np.allclose(delta_P[~np.isnan(delta_P)], np.log(log_bin_sizes[0][~np.isnan(log_bin_sizes[0])]))

                # assert that first and last values are as desired
                assert np.allclose(self._binned_P_RAJ[~np.isnan(self._binned_P_RAJ)][-1], P_RAJ_D_e[~np.isnan(P_RAJ_D_e)])
                assert np.allclose(self._binned_P_RAJ[~np.isnan(self._binned_P_RAJ)][-1], P_RAJ_D_e[~np.isnan(P_RAJ_D_e)])
                assert np.allclose(self._binned_P_RAJ[~np.isnan(self._binned_P_RAJ)][0], P_RAJ_klass_max)

            # scalar assertions
            else:
                assert np.isclose(delta_P, np.log(log_bin_sizes[0]))

                # assert that first and last values are as desired
                assert np.isclose(self._binned_P_RAJ[-1], P_RAJ_D_e)
                assert np.isclose(self._binned_P_RAJ[0], P_RAJ_klass_max)
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
        """Compute the value of self._xbar_minus_2, described by eq. (2.9-138)
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

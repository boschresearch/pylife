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
import scipy.optimize

import pylife.strength.fkm_nonlinear
import pylife.materiallaws.rambgood
import pylife.strength.woehler_fkm_nonlinear
from pylife.strength.fkm_nonlinear.constants import FKMNLConstants

class P_RAM:
    """This class implements the damage parameter P_RAM according to guideline FKM nonlinear.
    The resulting values are added to the collective in a new column, which can be retrieved by the ``.collective`` accessor.
    """

    def __init__(self, collective, assessment_parameters):
        """Initialize the P_RAM object and compute the damage parameter P_RAM according to FKM nonlinear.
        The resulting values are added to the collective, which can be retrieved by the ``.collective`` accessor.

        Parameters
        ----------
        collective : pandas DataFrame
            A load collective with stress and strain amplitude and mean stress,
            resulting from two runs of the HCM algorithm and computed by the FKMNonlinearRecorder.
            Every row corresponds to one closed hysteresis.

            More specifically, the table has to contain the following columns:

            * ``S_a``: the stress amplitude of the hysteresis
            * ``S_m``: the mean stress of the hysteresis
            * ``epsilon_a`` the strain amplitude of the hysteresis
        assessment_parameters : pandas Series
            All material parameters collected so far. Has to contain the ``R_m`` and ``E`` entries.

        Returns
        -------
        None.

        """
        self._collective = collective.copy()
        self._assessment_parameters = assessment_parameters

        # select set of constants according to given material group
        self._constants = FKMNLConstants().for_material_group(assessment_parameters)

        # check if collective contains required columns
        assert "S_a" in self._collective.columns
        assert "S_m" in self._collective.columns
        assert "epsilon_a" in self._collective.columns

        # check if the required material parameters are present
        assert "R_m" in self._assessment_parameters
        assert "E" in self._assessment_parameters

        self._compute_values()

    @property
    def collective(self):
        return self._collective

    def _compute_values(self):
        """Compute the P_RAM damage parameter according to FKM nonlinear.
        """

        # determine R_m from HV, if not given directly
        if "R_m" in self._assessment_parameters:
            R_m = self._assessment_parameters.R_m

        # compute M_σ according to eq. (2.6-84)
        self._M_sigma = self._constants.a_M * 1e-3 * R_m + self._constants.b_M

        # compute k according to eq. (2.6-83)
        self._collective["k"] = 0.0
        self._collective.loc[self._collective["S_m"]>=0, "k"] = self._M_sigma * (self._M_sigma + 2)
        self._collective.loc[self._collective["S_m"]<0,  "k"] = self._M_sigma/3 * (self._M_sigma/3 + 2)

        # compute (σ_a + k*σ_m)
        self._collective["discriminant"] = self._collective["S_a"] +  self._collective["k"]*self._collective["S_m"]

        # compute P_RAM according to eq. (2.6-82)
        self._collective["P_RAM"] = np.where(self._collective["discriminant"] >= 0,
                                             np.sqrt(self._collective["discriminant"]
                                                     * self._collective["epsilon_a"]
                                                     * self._assessment_parameters.E), 0.0)

        # delete temporary columns
        self._collective.drop(columns = ["discriminant", "k"], inplace=True)


class P_RAJ:
    """This class implements the damage parameter P_RAJ according to guideline FKM nonlinear.
    The resulting values are added to the collective in a new column, which can be retrieved by the ``.collective`` accessor.
    """

    def __init__(self, collective, assessment_parameters, component_woehler_curve_P_RAJ):
        """Initialize the P_RAJ object and compute the damage parameter P_RAJ according to FKM nonlinear.
        The resulting values are added to the collective, which can be retrieved by the ``.collective`` accessor.

        Parameters
        ----------
        collective : pandas DataFrame
            A load collective with stress and strain amplitude and mean stress,
            resulting from two runs of the HCM algorithm and computed by the FKMNonlinearRecorder.
            Every row corresponds to one closed hysteresis.

            More specifically, the table has to contain the following columns:

            * ``S_a``: the stress amplitude of the hysteresis
            * ``S_m``: the mean stress of the hysteresis
            * ``epsilon_a``: the strain amplitude of the hysteresis
            * ``is_closed_hysteresis``:  whether the hysteresis is closed and counts as full damage
            * ``run_index``: which run of the HCM algorithm the hysteresis belows to, has to be one of {1,2}.
        assessment_parameters : pandas Series
            All material and assessment parameters collected so far. Has to contain entries
            for ``R_m``, ``E``, ``n_prime``, ``K_prime``.
        component_woehler_curve_P_RAJ : class WoehlerCurvePRAJ
            The woehler curve, which can be obtained using the ``woehler_P_RAJ`` accessor as follows:

            .. code:: python

                component_woehler_curve_parameters = assessment_parameters[["P_RAJ_Z", "P_RAJ_D_0", "d_RAJ"]]
                component_woehler_curve_P_RAJ = component_woehler_curve_parameters.woehler_P_RAJ

        Returns
        -------
        None.

        """
        self._collective = collective.copy()
        self._assessment_parameters = assessment_parameters
        self._component_woehler_curve_P_RAJ = component_woehler_curve_P_RAJ
        self._P_RAJ_Z = self._component_woehler_curve_P_RAJ.P_RAJ_Z
        self._P_RAJ_D_0 = self._component_woehler_curve_P_RAJ.fatigue_strength_limit

        # select set of constants according to given material group
        self._constants = FKMNLConstants().for_material_group(assessment_parameters)

        # check if collective contains required columns
        assert "S_a" in self._collective.columns
        assert "S_m" in self._collective.columns
        assert "epsilon_a" in self._collective.columns
        assert "is_closed_hysteresis" in self._collective.columns
        assert "run_index" in self._collective.columns

        # if assessment is done for multiple points at once, work with a multi-indexed data frame
        if not isinstance(self._collective.index, pd.MultiIndex):
            n_hystereses = len(self._collective)
            self._collective.index = pd.MultiIndex.from_product([range(n_hystereses), [0]], names=["hysteresis_index", "assessment_point_index"])

        # assert that the index contains the two columns "hysteresis_index" and "assessment_point_index"
        assert self._collective.index.names == ["hysteresis_index", "assessment_point_index"]

        assert (self._collective.run_index.unique() == np.array([1,2])).all() or (self._collective.run_index.unique() == np.array([2])).all()

        # check if the required material parameters are present
        assert "R_m" in self._assessment_parameters
        assert "E" in self._assessment_parameters
        assert "n_prime" in self._assessment_parameters
        assert "K_prime" in self._assessment_parameters


        # remove any given index of f_RAM which results from the assessment_parameters.G parameter
        if isinstance(self._P_RAJ_Z, pd.Series):
            self._P_RAJ_Z.reset_index(drop=True, inplace=True)

        if isinstance(self._P_RAJ_D_0, pd.Series):
            self._P_RAJ_D_0.reset_index(drop=True, inplace=True)

        self._compute_values()

    @property
    def collective(self):
        return self._collective

    def _compute_values(self):
        """Compute the P_RAJ damage parameter according to FKM nonlinear.
        """

        # compute the crack opening stress S_open (chapter 2.8.9.1) for every hysteresis
        self._compute_S_open()

        # compute fictitious single-step crack opening strain, eq. (2.9-99)
        self._ramberg_osgood_relation = pylife.materiallaws.rambgood.RambergOsgood(E=self._assessment_parameters.E, K=self._assessment_parameters.K_prime, n=self._assessment_parameters.n_prime)
        self._collective["epsilon_open_ein"] = self._collective.epsilon_min + self._ramberg_osgood_relation.delta_strain(self._collective.S_open - self._collective.S_min)

        #self._collective["epsilon_open_ein2"] = self._collective.epsilon_min + (self._collective.S_open - self._collective.S_min) / self._assessment_parameters.E \
        #    + 2*np.power((self._collective.S_open - self._collective.S_min) / (2*self._assessment_parameters.K_prime), (1/self._assessment_parameters.n_prime))


        # compute crack opening strain with history (chapter 2.8.9.3, chapter 2.9.8.1 is better)
        self._compute_crack_opening_loop()
        # continue on p.133 (pdf)

        # delete temporary columns
        #self._collective.drop(columns = ["A_0", "A_1", "A_2", "A_3"], inplace=True)

    def _compute_S_open(self):
        """compute the crack opening stress S_open (chapter 2.8.9.1)"""

        # determine R_m from HV, if not given directly
        if "R_m" in self._assessment_parameters:
            R_m = self._assessment_parameters.R_m

        # compute 0.2% yield strength and yield stress, eq. (2.8-64)
        self._assessment_parameters.R_p02 = 0.002**self._assessment_parameters.n_prime * self._assessment_parameters.K_prime
        self._assessment_parameters.S_F = 0.5 * (self._assessment_parameters.R_p02 + R_m)

        # compute M_σ according to eq. (2.6-84), (2.9-98)
        self._M_sigma = self._constants.a_M * 1e-3 * R_m + self._constants.b_M

        # compute mean stress parameter A_m, eq. (2.8-65)
        self._collective["A_m"] = np.where(self._collective.S_m < 0,
                                           0.4 - self._M_sigma/4,
                                           0.47 * (1 - 1.5*self._M_sigma) * (1+self._collective.R) ** (1+self._collective.R+self._M_sigma))

        # compute coefficients according to eq. (2.8-63)
        self._collective["A_0"] = 0.535 * np.cos(np.pi/2 * self._collective.S_max / self._assessment_parameters.S_F) + self._collective.A_m
        self._collective["A_1"] = 0.344 * self._collective.S_max / self._assessment_parameters.S_F + self._collective.A_m
        self._collective["A_3"] = 2*self._collective.A_0 + self._collective.A_1 - 1
        self._collective["A_2"] = 1 - self._collective.A_0 - self._collective.A_1 - self._collective.A_3

        # compute crack opening stress
        self._collective["S_open_factor"] = np.where((0 <= self._collective.R) & (self._collective.R < 1),
                                              self._collective.A_0 + self._collective.A_1*self._collective.R
                                              + self._collective.A_2*self._collective.R**2 + self._collective.A_3*self._collective.R**3,
                                              np.where(self._collective.R < 0,
                                                       self._collective.A_0 + self._collective.A_1*self._collective.R,
                                                       1)
                                              )

        self._collective["S_open"] = self._collective.S_max * self._collective.S_open_factor

        # remove temporary columns
        self._collective.drop(columns=["A_0", "A_1", "A_2", "A_3", "S_open_factor"], inplace=True)

    def _calculate_P_RAJ(self, delta_S_eff, delta_epsilon_eff):
        """compute P_RAJ according to eq. 2.9-110"""

        P_RAJ = 1.24 * np.power(delta_S_eff,2) / self._assessment_parameters.E \
            + 1.02 / np.sqrt(self._assessment_parameters.n_prime) * delta_S_eff \
                * (delta_epsilon_eff - delta_S_eff/self._assessment_parameters.E)
        return P_RAJ

    def _calculate_fatigue_limit_variables(self, D_akt):
        """Compute a_0, delta_J_eff_th, P_RAJ_D"""

        # standard calculation of m according to eq. (2.8-60), (2.9-117)
        m = -1/self._component_woehler_curve_P_RAJ.d

        # eq. (2.8-59), (2.9-121)
        C = 1e-5 * (5e5)**m * (self._assessment_parameters.E) ** (-m)

        a_end = 0.5  # [mm]
        self._assessment_parameters.a_end = a_end

        # short initial crack length a0
        # standard calculation according to eq. (2.8-58), (2.9-120)
        a_0 = (a_end**(1-m) - (1-m) * C * self._component_woehler_curve_P_RAJ.P_RAJ_Z ** m) ** (1 / (1-m))

        self._assessment_parameters.a_0 = a_0

        # eq. (2.9-118)
        delta_J_eff_th = self._assessment_parameters.E / 5e6

        # eq. (2.9-116)
        denominator = ((a_end**(1-m) - a_0**(1-m)) * D_akt + a_0**(1-m)) ** (1/(1-m)) \
                        + delta_J_eff_th/self._component_woehler_curve_P_RAJ.fatigue_strength_limit - a_0

        # update fatigue limit P_RAJ_D
        P_RAJ_D = delta_J_eff_th * 1 / denominator

        # for debugging, disable decrease of P_RAJ_D (uncomment next line)
        #P_RAJ_D = pd.Series(self._P_RAJ_D_0)

        return a_0, delta_J_eff_th, P_RAJ_D

    def _compute_crack_opening_loop(self):
        """compute crack opening strain with history (chapter 2.8.9.3, chapter 2.9.8.1 is better)"""

        # initialize new columns in collective DataFrame
        self._collective["epsilon_open_alt"] = 0.0
        self._collective["epsilon_open"] = 0.0
        self._collective["P_RAJ"] = 0.0
        self._collective["D"] = 0.0
        self._collective["S_close"] = 0.0
        self._collective["case_name"] = ""

        self._collective["epsilon_min_alt_SP"] = 0.0
        self._collective["epsilon_max_alt_SP"] = 0.0

        assessment_point_index = self._collective[self._collective.index.get_level_values("hysteresis_index")==0]\
            .index.get_level_values("assessment_point_index")

        # initialize variables
        epsilon_open_alt = -np.inf   # initialized according to 2.9.7 point 2
        epsilon_open_alt = pd.Series(0.0, index=assessment_point_index)

        epsilon_min_alt_SP = pd.Series(np.inf, index=assessment_point_index)
        epsilon_max_alt_SP = pd.Series(-np.inf, index=assessment_point_index)

        epsilon_min_alt_SP = pd.Series(0.0, index=assessment_point_index)
        epsilon_max_alt_SP = pd.Series(0.0, index=assessment_point_index)

        # cumulative damage value
        D_akt = pd.Series(0.0, index=assessment_point_index)

        # initialize current fatigue limit
        P_RAJ_D = pd.Series(self._P_RAJ_D_0, index=assessment_point_index)

        P_RAJ_D = pd.Series(self._P_RAJ_D_0, index=assessment_point_index)

        # initialize helper variables
        epsilon_open = pd.Series(0.0, index=assessment_point_index)

        # Find the last hysteresis of the first run of the HCM algorithm. After this hysteresis, we need to initialize some variables for the second HCM run.
        last_index_of_first_run = self._collective[(self._collective["run_index"]==1) & (self._collective["run_index"].shift(-1)==2)].index

        # if there is no hysteresis in the first run of the HCM algorithm, precompute some required values
        # this is described in the correction document to the FKM nonlinear
        if len(last_index_of_first_run) == 0:
            last_index_of_first_run = 0

            # compute helper variables for fatigue limit
            a_0, delta_J_eff_th, _ = self._calculate_fatigue_limit_variables(D_akt)

            # eq. (2.9-124)
            l_star = delta_J_eff_th / self._P_RAJ_D_0 - a_0
            self._assessment_parameters.l_star = l_star

            # eq. (2.9-123)
            self._assessment_parameters.P_RAJ_D_e = self._P_RAJ_D_0 * (a_0 + l_star) / (self._assessment_parameters.a_end + l_star)

            # compute maximum occuring P_RAJ
            max_abs_S_max = self._collective.S_max.abs().max()
            max_abs_S_min = self._collective.S_min.abs().max()
            maximum_abs_S = max(max_abs_S_max, max_abs_S_min)
            delta_stress = 2*maximum_abs_S
            delta_strain = self._ramberg_osgood_relation.delta_strain(delta_stress)

            # eq. (2.9-125)
            self._assessment_parameters.P_RAJ_klass_max = self._calculate_P_RAJ(delta_stress, delta_strain)

        else:
            last_index_of_first_run = last_index_of_first_run[0][0]

        # iterate over collected hystereses
        for index,group in self._collective.groupby("hysteresis_index"):

            # remove first index level, group is now indexed only by "assessment_point_index"
            group = group.droplevel("hysteresis_index")

            # create temporary series
            is_damage_in_current_hysteresis = pd.Series(True, index=assessment_point_index)
            case_name = pd.Series("", index=assessment_point_index)

            # ---------------------------------------
            # case 1: strain in entire hysteresis is lower than crack opening strain, crack does not open, no damage
            # if row.epsilon_max < epsilon_open_alt:
            condition_1 = group.epsilon_max < epsilon_open_alt

            # set case_name = "1"
            case_name.mask(condition_1, other="1", inplace=True)

            # set epsilon_open = epsilon_open_alt
            epsilon_open.mask(condition_1, other=epsilon_open_alt, inplace=True)

            # set is_damage_in_current_hysteresis = False
            is_damage_in_current_hysteresis.mask(condition_1, other=False, inplace=True)

            # ---------------------------------------
            # case 2 (a and b)
            # if row.epsilon_max < epsilon_open_alt:
            condition_2 = (~condition_1) & ((epsilon_max_alt_SP < group.epsilon_max_LF) \
                | (epsilon_min_alt_SP > group.epsilon_min_LF))

            # set case_name = "2"
            case_name.mask(condition_2, other="2", inplace=True)

            # set epsilon_open = row.epsilon_open_ein
            epsilon_open.mask(condition_2, other=group.epsilon_open_ein, inplace=True)

            # update values as described after eq. (2.9-102)
            epsilon_min_alt_SP.mask(condition_2, other=group.epsilon_min_LF, inplace=True)
            epsilon_max_alt_SP.mask(condition_2, other=group.epsilon_max_LF, inplace=True)

            # ---------------------------------------
            # case 3
            # if row.epsilon_open_ein >= epsilon_open_alt:
            condition_3 = (~condition_2) & (~condition_1) \
                & (group.epsilon_open_ein >= epsilon_open_alt)

            # set case_name = "3"
            case_name.mask(condition_3, other="3", inplace=True)

            # set epsilon_open = epsilon_open_alt
            epsilon_open.mask(condition_3, other=epsilon_open_alt, inplace=True)

            # ---------------------------------------
            # case 4
            # if row.epsilon_open_ein < epsilon_open_alt:
            condition_4 = (~condition_3) & (~condition_2) & (~condition_1) \
                & (group.epsilon_open_ein < epsilon_open_alt)

            # set case_name = "4"
            case_name.mask(condition_4, other="4", inplace=True)

            # case 4a
            # if row.S_a >= 0.4 * self._assessment_parameters.S_F:
            condition_4a = condition_4 & (group.S_a >= 0.4 * self._assessment_parameters.S_F)

            # set epsilon_open = row.epsilon_open_ein
            epsilon_open.mask(condition_4a, other=group.epsilon_open_ein, inplace=True)

            # case 4b
            condition_4b = condition_4 & (group.S_a < 0.4 * self._assessment_parameters.S_F)

            # set epsilon_open = epsilon_open_alt
            epsilon_open.mask(condition_4b, other=epsilon_open_alt, inplace=True)

            # ---------------------------------------
            # update epsilon_min_alt_SP and epsilon_max_alt_SP, as described in eq. (2.9-91), (2.9-92)
            condition = case_name.isin(["1", "3", "4"])
            epsilon_min_alt_SP.mask(condition, other=np.minimum(epsilon_min_alt_SP, group.epsilon_min), inplace=True)
            epsilon_max_alt_SP.mask(condition, other=np.maximum(epsilon_max_alt_SP, group.epsilon_max), inplace=True)

            # set value of epsilon_open
            group.epsilon_open = epsilon_open.to_numpy()

            # store the new crack opening strain in the collective DataFrame
            self._collective.loc[self._collective.index.get_level_values("hysteresis_index")==index,"epsilon_open"] = epsilon_open.to_numpy()

            # store the used values of epsilon_min_alt
            self._collective.loc[self._collective.index.get_level_values("hysteresis_index")==index,"epsilon_min_alt_SP"] = epsilon_min_alt_SP.to_numpy()
            self._collective.loc[self._collective.index.get_level_values("hysteresis_index")==index,"epsilon_max_alt_SP"] = epsilon_max_alt_SP.to_numpy()

            self._collective.loc[self._collective.index.get_level_values("hysteresis_index")==index,"case_name"] = case_name.to_numpy()

            # compute the effective stress and strain ranges, delta_S_eff and delta_epsilon_eff,
            # of the hysteresis where the crack is open

            # compute crack closing stress, S_close
            def f(S_close, S_max, epsilon_max, epsilon_open):
                return self._ramberg_osgood_relation.delta_strain(S_max - S_close) - (epsilon_max - epsilon_open)

            def fprime(S_close, S_max, epsilon_max, epsilon_open):
                """
                Note, the derivative of `ramberg_osgood_relation.delta_strain` is:
                d/dΔsigma delta_strain(Δsigma) =  d/dΔsigma 2*strain(Δsigma/2)
                  = 2*d/dΔsigma strain(Δsigma/2) = 2 * 1/2 * tangential_compliance(Δsigma/2)
                  = ramberg_osgood_relation.tangential_compliance(delta_stress/2)
                """
                delta_stress = S_max - S_close
                return -self._ramberg_osgood_relation.tangential_compliance(delta_stress/2)

            def optimize(row):

                if np.isnan(row.S_min):
                    return np.nan

                return scipy.optimize.newton(f,
                                             x0=row.S_min,
                                             args=(row.S_max, row.epsilon_max, row.epsilon_open),
                                             fprime=fprime,
                                             full_output=False)

            # --
            S_close = group.apply(optimize, axis="columns")

            # set S_close = S_min for epsilon_open < epsilon_min
            S_close.mask(epsilon_open < group.epsilon_min, other=group.S_min, inplace=True)

            # eq. (2.9-108)
            delta_S_eff = group.S_max - S_close

            # eq. (2.9-109)
            epsilon_close = epsilon_open

            # eq. (2.9-110)
            delta_epsilon_eff = np.where(epsilon_open < group.epsilon_min,
                                   group.epsilon_max - group.epsilon_min,         # eq. (2.9-108)
                                   group.epsilon_max - epsilon_close)

            # store value in collective DataFrame for current hysteresis, this is only for debugging and could be removed for performance reasons
            self._collective.loc[self._collective.index.get_level_values("hysteresis_index")==index,"delta_S_eff"] = delta_S_eff.to_numpy()
            self._collective.loc[self._collective.index.get_level_values("hysteresis_index")==index,"delta_epsilon_eff"] = delta_epsilon_eff
            self._collective.loc[self._collective.index.get_level_values("hysteresis_index")==index,"S_close"] = S_close.to_numpy()
            self._collective.loc[self._collective.index.get_level_values("hysteresis_index")==index,"is_damage_in_current_hysteresis"] = is_damage_in_current_hysteresis.to_numpy()


            # if the crack was not entirely closed in the current hysteresis (which was case 1)
            # compute value of P_RAJ, eq. (2.9-110)
            P_RAJ = np.where(is_damage_in_current_hysteresis,
                             self._calculate_P_RAJ(delta_S_eff, delta_epsilon_eff),
                             0.0)

            # store value in collective DataFrame for current hysteresis
            self._collective.loc[self._collective.index.get_level_values("hysteresis_index")==index,"P_RAJ"] = P_RAJ

            # update the infinite life limit
            self._component_woehler_curve_P_RAJ.update_P_RAJ_D(P_RAJ_D)

            # compute the lifetime using the component woehler curve
            N = self._component_woehler_curve_P_RAJ.calc_N(P_RAJ)

            # in the standard FKM nonlinear procedure, the above computation is equivalent to:
            #if P_RAJ > P_RAJ_D:
            #     N = (P_RAJ / self._P_RAJ_Z) ** (1/self._component_woehler_curve_P_RAJ.d)
            #else:
            #    N = np.inf

            # compute damage contribution
            D = np.where(group.is_closed_hysteresis,
                         1.0/N,
                         0.5/N)

            # store damage in collective
            self._collective.loc[self._collective.index.get_level_values("hysteresis_index")==index,"D"] = D

            # sum up cumulative damage
            D_akt += D

            # update the variable for the previous crack opening strain, epsilon_open_alt
            # This was already done for cases 1,2 and 4.
            # for case 3:
            condition = (case_name == "3") & (epsilon_open >= group.epsilon_min) & ~(np.isinf(N))

            # update crack opening strain according to eq. (2.8-77), (2.9-115)
            # only if the crack opening strain is larger than the minimum strain of the hysteresis
            # for N = inf, epsilon_open_alt does not change (then, exp(0)=1, epsilon_open_alt=epsilon_open_alt)

            epsilon_open_alt.mask(condition,
                                  other=group.epsilon_open_ein - (group.epsilon_open_ein - epsilon_open_alt) * np.exp(-15 / N), inplace=True)

            # for the other cases 1,2, and 4:
            epsilon_open_alt.mask(case_name.isin(["1", "2", "4"]), other=epsilon_open, inplace=True)

            # store the value of epsilon_open_alt after it has been updated
            self._collective.loc[self._collective.index.get_level_values("hysteresis_index")==index,"epsilon_open_alt"] = epsilon_open_alt.to_numpy()

            # compute helper variables for fatigue limit
            a_0, delta_J_eff_th, P_RAJ_D = self._calculate_fatigue_limit_variables(D_akt)

            # store P_RAJ_D in collective DataFrame
            self._collective.loc[self._collective.index.get_level_values("hysteresis_index")==index,"P_RAJ_D"] = P_RAJ_D.to_numpy()

            # initializations for the second run,
            # only execute once at the end of the first run
            if index == last_index_of_first_run:

                # eq. (2.9-124)
                l_star = delta_J_eff_th / self._P_RAJ_D_0 - a_0
                self._assessment_parameters.l_star = l_star

                # eq. (2.9-123)
                self._assessment_parameters.P_RAJ_D_e = self._P_RAJ_D_0 * (a_0 + l_star) / (self._assessment_parameters.a_end + l_star)

                # compute maximum occuring P_RAJ
                max_abs_S_max = self._collective.S_max.abs().max()
                max_abs_S_min = self._collective.S_min.abs().max()
                maximum_abs_S = max(max_abs_S_max, max_abs_S_min)
                delta_stress = 2*maximum_abs_S
                delta_strain = self._ramberg_osgood_relation.delta_strain(delta_stress)

                # eq. (2.9-125)
                self._assessment_parameters.P_RAJ_klass_max = self._calculate_P_RAJ(delta_stress, delta_strain)

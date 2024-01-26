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
import pylife.strength.fkm_nonlinear.damage_calculator

class DamageCalculatorPRAJMinerElementary:
    """This class performs the lifetime assessment as a modification to the FKM nonlinear guideline.
    For the P_RAJ damage parameter, the lifetime is computed directly using the woehler curve,
    analogous to the procedure with P_RAM. The whole approach with binning and infinite strength decrease is omitted.

    The class reuses the DamageCalculatorPRAM class.
    """

    class ComponentWoehlerCurvePRAMStub:
        def __init__(self, component_woehler_curve_P_RAJ):
            """component_woehler_curve_P_RAJ is of class WoehlerCurvePRAJ"""
            self._component_woehler_curve_P_RAJ = component_woehler_curve_P_RAJ

        @property
        def d_1(self):
            """The slope of the Wöhler Curve in the first section, for N < 1e3"""
            return self._component_woehler_curve_P_RAJ.d

        @property
        def d_2(self):
            """The slope of the Wöhler Curve in the second section, for N >= 1e3"""
            return self._component_woehler_curve_P_RAJ.d

        @property
        def P_RAM_Z(self):
            """The damage parameter value that separates the first and second section, corresponding to N = 1e3"""

            P_RAJ_Z = self._component_woehler_curve_P_RAJ.P_RAJ_Z
            P_RAJ_Z_1e3 = self._component_woehler_curve_P_RAJ.calc_P_RAJ(1e3)

            if isinstance(P_RAJ_Z, float):
                return P_RAJ_Z_1e3

            return pd.Series(index = P_RAJ_Z.index, data=P_RAJ_Z_1e3)

        @property
        def P_RAM_D(self):
            """The damage parameter value of the endurance limit"""
            return self._component_woehler_curve_P_RAJ.P_RAJ_D

        def calc_N(self, P_RAM):
            """Evaluate the woehler curve at the given damage paramater value, P_RAM.

            Parameters
            ----------
            P_RAM : float
                The damage parameter value where to evaluate the woehler curve.

            Returns
            -------
            N : float
                The number of cycles for the given P_RAM value.

            """

            return self._component_woehler_curve_P_RAJ.calc_N(P_RAM)

        def calc_P_RAM(self, N):
            """Evaluate the woehler curve at the specified number of cycles.

            Parameters
            ----------
            N : array-like
                Number of cycles where to evaluate the woehler curve.

            Returns
            -------
            array-like
                The P_RAM values that correspond to the given N values.

            """
            return self._component_woehler_curve_P_RAJ.calc_P_RAJ(N)

        @property
        def fatigue_strength_limit(self):
            """The fatigue strength limit of the component, i.e.,
            the P_RAM value below which we have infinite life."""

            return self._component_woehler_curve_P_RAJ.fatigue_strength_limit

        @property
        def fatigue_life_limit(self):
            """The fatigue life limit N_D of the component, i.e.,
            the number of cycles at the fatigue strength limit P_RAM_D."""

            return self._component_woehler_curve_P_RAJ.fatigue_life_limit


    def __init__(self, collective, component_woehler_curve_P_RAJ):
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


        self._collective = collective.copy()
        self._collective["P_RAM"] = self._collective["P_RAJ"]
        self._component_woehler_curve = self.ComponentWoehlerCurvePRAMStub(component_woehler_curve_P_RAJ)

        self._damage_calculator_pram = pylife.strength.fkm_nonlinear.damage_calculator\
            .DamageCalculatorPRAM(self._collective, self._component_woehler_curve)

    @property
    def collective(self):
        return self._collective

    @property
    def P_RAJ_max(self):
        """The maximum P_RAM damage parameter value of the second run of the HCM algorithm.
        If this value is lower than the fatigue strength limit, the component has infinite life."""

        return self._damage_calculator_pram.P_RAM_max

    @property
    def is_life_infinite(self):
        """Whether the component has infinite life."""

        return self._damage_calculator_pram.is_life_infinite

    @property
    def lifetime_n_times_load_sequence(self):
        """The number of times the whole load sequence can be traversed until failure."""

        return self._damage_calculator_pram.lifetime_n_times_load_sequence

    @property
    def lifetime_n_cycles(self):
        """The number of load cycles (as defined in the load collective) until failure."""

        return self._damage_calculator_pram.lifetime_n_cycles

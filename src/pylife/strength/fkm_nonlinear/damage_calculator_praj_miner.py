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

"""Calculate ``P_RAJ`` lifetime with an elementary Miner-type damage sum.

This module provides a modified alternative to the guideline ``P_RAJ`` damage
calculator.  It maps ``P_RAJ`` values to the elementary ``P_RAM`` calculator so
that users can compare guideline classing with direct Miner-type accumulation.

Examples
--------
>>> from pylife.strength.fkm_nonlinear import damage_calculator_praj_miner
>>> damage_calculator_praj_miner.DamageCalculatorPRAJMinerElementary.__name__
'DamageCalculatorPRAJMinerElementary'
"""

__author__ = "Benjamin Maier"
__maintainer__ = __author__

import pandas as pd
import pylife.strength.fkm_nonlinear.damage_calculator

class DamageCalculatorPRAJMinerElementary:
    r"""Calculate ``P_RAJ`` lifetime by elementary Miner-type accumulation.

    Use this calculator when a direct Miner-type damage sum for ``P_RAJ`` is
    wanted instead of the official guideline ``P_RAJ`` classing and crack-growth
    accumulation implemented by ``DamageCalculatorPRAJ``.  The class adapts the
    ``P_RAJ`` Wöhler curve to the ``DamageCalculatorPRAM`` interface and then
    reuses the elementary damage calculation.

    Parameters
    ----------
    collective : pandas.DataFrame
        Damage-parameter collective with one row per hysteresis.  Required columns
        are ``P_RAJ``, ``is_closed_hysteresis``, ``run_index``, and ``S_min``.
    component_woehler_curve_P_RAJ : WoehlerCurvePRAJ
        Component Wöhler curve for the ``P_RAJ`` damage parameter.

    Notes
    -----
    Implement the modified Miner-type accumulation

    .. math::

        D = \sum_i D_i

    for ``P_RAJ`` values.  This is useful for comparisons and non-guideline
    workflows; use ``DamageCalculatorPRAJ`` for the FKM nonlinear guideline's own
    ``P_RAJ`` lifetime assessment.
    """

    class ComponentWoehlerCurvePRAMStub:
        """Adapt a ``WoehlerCurvePRAJ`` to the ``P_RAM`` calculator interface.

        Parameters
        ----------
        component_woehler_curve_P_RAJ : WoehlerCurvePRAJ
            Component Wöhler curve whose ``P_RAJ`` methods and properties are exposed
            under the names expected by ``DamageCalculatorPRAM``.
        """
        def __init__(self, component_woehler_curve_P_RAJ):
            """Initialize the adapter with a ``P_RAJ`` Wöhler curve.

            Parameters
            ----------
            component_woehler_curve_P_RAJ : WoehlerCurvePRAJ
                Component Wöhler curve to adapt.
            """
            self._component_woehler_curve_P_RAJ = component_woehler_curve_P_RAJ

        @property
        def d_1(self):
            """Return the Wöhler slope below ``1e3`` cycles.

            Returns
            -------
            float
                Slope exponent of the adapted ``P_RAJ`` Wöhler curve.
            """
            return self._component_woehler_curve_P_RAJ.d

        @property
        def d_2(self):
            """Return the Wöhler slope from ``1e3`` cycles onward.

            Returns
            -------
            float
                Slope exponent of the adapted ``P_RAJ`` Wöhler curve.
            """
            return self._component_woehler_curve_P_RAJ.d

        @property
        def P_RAM_Z(self):
            """Return the adapted damage parameter at ``1e3`` cycles.

            Returns
            -------
            float or pandas.Series
                ``P_RAJ`` value at ``1e3`` cycles, exposed as ``P_RAM_Z`` for the adapted
                elementary calculator interface.
            """

            P_RAJ_Z = self._component_woehler_curve_P_RAJ.P_RAJ_Z
            P_RAJ_Z_1e3 = self._component_woehler_curve_P_RAJ.calc_P_RAJ(1e3)

            if isinstance(P_RAJ_Z, float):
                return P_RAJ_Z_1e3

            return pd.Series(index = P_RAJ_Z.index, data=P_RAJ_Z_1e3)

        @property
        def P_RAM_D(self):
            """Return the adapted endurance-limit damage parameter.

            Returns
            -------
            float or pandas.Series
                ``P_RAJ_D`` value exposed as ``P_RAM_D`` for the adapted elementary
                calculator interface.
            """
            return self._component_woehler_curve_P_RAJ.P_RAJ_D

        def calc_N(self, P_RAM):
            """Calculate cycles for an adapted damage-parameter value.

            Parameters
            ----------
            P_RAM : float or array_like
                Damage-parameter value passed through to the underlying ``P_RAJ`` Wöhler
                curve.

            Returns
            -------
            float or array_like
                Number of cycles corresponding to ``P_RAM``.
            """

            return self._component_woehler_curve_P_RAJ.calc_N(P_RAM)

        def calc_P_RAM(self, N):
            """Calculate the adapted damage parameter for cycle numbers.

            Parameters
            ----------
            N : float or array_like
                Number of cycles.

            Returns
            -------
            float or array_like
                ``P_RAJ`` values exposed as ``P_RAM`` values for the adapted elementary
                calculator interface.
            """
            return self._component_woehler_curve_P_RAJ.calc_P_RAJ(N)

        @property
        def fatigue_strength_limit(self):
            """Return the adapted fatigue strength limit.

            Returns
            -------
            float or pandas.Series
                ``P_RAJ`` value below which the adapted Wöhler curve predicts infinite
                life.
            """

            return self._component_woehler_curve_P_RAJ.fatigue_strength_limit

        @property
        def fatigue_life_limit(self):
            """Return the fatigue life limit of the adapted curve.

            Returns
            -------
            float or pandas.Series
                Number of cycles at the adapted fatigue strength limit.
            """

            return self._component_woehler_curve_P_RAJ.fatigue_life_limit


    def __init__(self, collective, component_woehler_curve_P_RAJ):
        """Initialize the Miner-type ``P_RAJ`` damage calculation.

        Parameters
        ----------
        collective : pandas.DataFrame
            Damage-parameter collective with one row per hysteresis.  Required columns
            are ``P_RAJ``, ``is_closed_hysteresis``, ``run_index``, and ``S_min``.
        component_woehler_curve_P_RAJ : WoehlerCurvePRAJ
            Component Wöhler curve used to convert ``P_RAJ`` values to bearable cycle
            numbers.
        """


        self._collective = collective.copy()
        self._collective["P_RAM"] = self._collective["P_RAJ"]
        self._component_woehler_curve = self.ComponentWoehlerCurvePRAMStub(component_woehler_curve_P_RAJ)

        self._damage_calculator_pram = pylife.strength.fkm_nonlinear.damage_calculator\
            .DamageCalculatorPRAM(self._collective, self._component_woehler_curve)

    @property
    def collective(self):
        """Return the evaluated Miner-type ``P_RAJ`` collective.

        Returns
        -------
        pandas.DataFrame
            Copy of the input collective with ``P_RAJ`` mirrored to ``P_RAM`` for the
            adapted elementary calculator.
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

        return self._damage_calculator_pram.P_RAM_max

    @property
    def is_life_infinite(self):
        """Return whether the Miner-type ``P_RAJ`` assessment predicts infinite life.

        Returns
        -------
        bool or pandas.Series
            ``True`` where ``P_RAJ_max`` does not exceed the fatigue strength limit of
            the component Wöhler curve; otherwise ``False``.
        """

        return self._damage_calculator_pram.is_life_infinite

    @property
    def lifetime_n_times_load_sequence(self):
        """Return load-sequence repetitions until Miner-type ``P_RAJ`` failure.

        Returns
        -------
        float or numpy.ndarray
            Number of complete load sequence repetitions until the elementary damage
            sum reaches ``1.0``.  A value of ``0`` means that failure occurs before the
            end of the second HCM run.
        """

        return self._damage_calculator_pram.lifetime_n_times_load_sequence

    @property
    def lifetime_n_cycles(self):
        """Return load cycles until Miner-type ``P_RAJ`` failure.

        Returns
        -------
        float or numpy.ndarray
            Number of cycles in the collective until the elementary damage sum reaches
            ``1.0``.
        """

        return self._damage_calculator_pram.lifetime_n_cycles

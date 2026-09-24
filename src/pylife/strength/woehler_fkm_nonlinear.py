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

"""Provide FKM nonlinear damage-parameter Wöhler curve accessors."""

__author__ = "Benjamin Maier"
__maintainer__ = __author__

import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings
import copy

from pylife import PylifeSignal


@pd.api.extensions.register_series_accessor('woehler_P_RAM')
@pd.api.extensions.register_dataframe_accessor('woehler_P_RAM')
class WoehlerCurvePRAM(PylifeSignal):
    r"""Represent the FKM nonlinear Wöhler curve for ``P_RAM``.

    This accessor is available as ``.woehler_P_RAM`` on
    :class:`pandas.Series` and :class:`pandas.DataFrame` objects.  It evaluates
    the component Wöhler curve used for damage parameters calculated by
    :class:`pylife.strength.damage_parameter.P_RAM`.

    Signal contract:

    * ``P_RAM_Z``: Damage-parameter value at ``N = 1e3`` cycles, in MPa.
    * ``P_RAM_D``: Endurance-limit damage-parameter value, in MPa.  It equals
      ``P_RAM_D_WS / f_RAM`` in FKM nonlinear equation 2.6-89.
    * ``d_1``: First finite-life slope for ``N < 1e3``, dimensionless and
      negative.
    * ``d_2``: Second finite-life slope for ``N >= 1e3``, dimensionless and
      negative.

    Parameters
    ----------
    pandas_obj : pandas.Series or pandas.DataFrame
        Wöhler curve parameters for one assessment point or vectorized
        parameters for several assessment points.

    See Also
    --------
    pylife.strength.damage_parameter.P_RAM : Calculate ``P_RAM`` collectives.
    WoehlerCurvePRAJ : Evaluate the FKM nonlinear Wöhler curve for ``P_RAJ``.

    Notes
    -----
    The curve has two finite-life branches with slopes :math:`d_1` and
    :math:`d_2`, followed by a horizontal endurance-limit branch at
    ``P_RAM_D``.  This follows FKM nonlinear guideline section 2.5.6.
    """

    def _validate(self):
        self.fail_if_key_missing(['P_RAM_Z', 'P_RAM_D', 'd_1', 'd_2'])

        is_not_nan = ~np.isnan(self._obj.P_RAM_Z)
        if not np.all(np.where(is_not_nan, self._obj.P_RAM_Z, 1) > np.where(is_not_nan, self._obj.P_RAM_D, 0)):
            raise ValueError(f"P_RAM_Z ({self._obj.P_RAM_Z}) has to be larger than P_RAM_D ({self._obj.P_RAM_D})!")

        if self._obj.d_1 >= 0:
            raise ValueError(f"d_1 ({self._obj.d_1}) has to be negative!")

        if self._obj.d_2 >= 0:
            raise ValueError(f"d_2 ({self._obj.d_2}) has to be negative!")

    def get_woehler_curve_minimum_lifetime(self):
        """Get the scalar curve for the assessment point with minimum lifetime.

        Use this method after a vectorized FKM nonlinear assessment, for
        example on a mesh, when the resulting Wöhler curve should be plotted
        for the most critical assessment point.  If the curve already contains
        scalar values, the copied curve is unchanged.

        Returns
        -------
        WoehlerCurvePRAM
            Deep copy of the current Wöhler curve with scalar ``P_RAM_Z`` and
            ``P_RAM_D`` values equal to the minima of the vectorized values.
        """

        # compute the minimum/maximum for the vectorized items
        woehler_curve_minimum_lifetime = copy.deepcopy(self)
        woehler_curve_minimum_lifetime._obj.P_RAM_Z = np.min(woehler_curve_minimum_lifetime._obj.P_RAM_Z)
        woehler_curve_minimum_lifetime._obj.P_RAM_D = np.min(woehler_curve_minimum_lifetime._obj.P_RAM_D)

        return woehler_curve_minimum_lifetime

    @property
    def d_1(self):
        """Return the first Wöhler curve slope for ``N < 1e3``."""
        return self._obj.d_1

    @property
    def d_2(self):
        """Return the second Wöhler curve slope for ``N >= 1e3``."""
        return self._obj.d_2

    @property
    def P_RAM_Z(self):
        """Return the ``P_RAM`` transition value at ``N = 1e3`` cycles."""
        return self._obj.P_RAM_Z

    @property
    def P_RAM_D(self):
        """Return the ``P_RAM`` endurance-limit value."""
        return self._obj.P_RAM_D

    def calc_N(self, P_RAM):
        """Evaluate the Wöhler curve at a ``P_RAM`` value.

        Parameters
        ----------
        P_RAM : float
            Damage-parameter value in MPa.

        Returns
        -------
        float
            Number of cycles to failure.  The result is ``numpy.inf`` when
            ``P_RAM`` is at or below the fatigue strength limit.
        """

        # silence warning "divide by zero in np.power. This happens for P_RAM=0, but then it will use the second branch with N=np.inf anyways
        with np.errstate(divide='ignore'):
            N = np.where(P_RAM > self.fatigue_strength_limit,
                         np.where(P_RAM >= self.P_RAM_Z,
                                  1e3 * np.power(P_RAM / self.P_RAM_Z, 1/self.d_1),
                                  1e3 * np.power(P_RAM / self.P_RAM_Z, 1/self.d_2)),
                         np.inf)

        return N

    def calc_P_RAM(self, N):
        """Evaluate the Wöhler curve at a number of cycles.

        Parameters
        ----------
        N : array_like
            Number of cycles where to evaluate the Wöhler curve.

        Returns
        -------
        numpy.ndarray
            ``P_RAM`` values in MPa that correspond to the given cycle counts.
        """
        N = np.array(N)

        # Note, this formula was derived visually from the figure 2.5 on page 43 of the FKM nonlinear document
        return np.where(N < 1e3,
                        self.P_RAM_Z * np.power(N * 1e-3, self.d_1),
                        np.where(N < self.fatigue_life_limit,
                                 self.P_RAM_Z * np.power(N * 1e-3, self.d_2),
                                 self.fatigue_strength_limit)
                        )

    @property
    def fatigue_strength_limit(self):
        """Return the ``P_RAM`` value below which lifetime is infinite."""

        return self.P_RAM_D

    @property
    def fatigue_life_limit(self):
        """Return the cycle count at the ``P_RAM`` fatigue strength limit."""

        # exp(log(P_RAM_Z) + (log(N) - log(1e3))*d_2)
        # P_RAM_Z * exp((log(N) - log(1e3))*d_2) = fatigue_strength_limit for N=fatigue_life_limit
        # =>  P_RAM_Z * exp((log(fatigue_life_limit) - log(1e3))*d_2) = fatigue_strength_limit
        # =>  fatigue_life_limit = exp(log(fatigue_strength_limit / P_RAM_Z) / d_2 + log(1e3)) = 1e3 * (fatigue_strength_limit / P_RAM_Z)^(1/d_2)

        return 1e3 * (self.fatigue_strength_limit / self.P_RAM_Z) ** (1/self._obj.d_2)


@pd.api.extensions.register_series_accessor('woehler_P_RAJ')
@pd.api.extensions.register_dataframe_accessor('woehler_P_RAJ')
class WoehlerCurvePRAJ(PylifeSignal):
    r"""Represent the FKM nonlinear Wöhler curve for ``P_RAJ``.

    This accessor is available as ``.woehler_P_RAJ`` on
    :class:`pandas.Series` and :class:`pandas.DataFrame` objects.  It evaluates
    the component Wöhler curve used for crack-mechanics damage parameters
    calculated by :class:`pylife.strength.damage_parameter.P_RAJ`.

    Signal contract:

    * ``P_RAJ_Z``: Damage-parameter value at ``N = 1`` cycle, in MPa.
    * ``P_RAJ_D_0``: Initial endurance-limit damage-parameter value, in MPa.
    * ``d_RAJ``: Finite-life slope, dimensionless and negative.

    Parameters
    ----------
    pandas_obj : pandas.Series or pandas.DataFrame
        Wöhler curve parameters for one assessment point or vectorized
        parameters for several assessment points.

    See Also
    --------
    pylife.strength.damage_parameter.P_RAJ : Calculate ``P_RAJ`` collectives.
    WoehlerCurvePRAM : Evaluate the FKM nonlinear Wöhler curve for ``P_RAM``.

    Notes
    -----
    The curve has one finite-life branch with slope :math:`d_{RAJ}` and a
    horizontal endurance-limit branch.  During the FKM nonlinear algorithm the
    active fatigue strength may be lowered from ``P_RAJ_D_0`` to
    :attr:`fatigue_strength_limit_final`.
    """

    def _validate(self):
        self.fail_if_key_missing(['P_RAJ_Z', 'P_RAJ_D_0', 'd_RAJ'])

        is_not_nan = ~np.isnan(self._obj.P_RAJ_Z)
        if not np.all(np.where(is_not_nan, self._obj.P_RAJ_Z, 1) > np.where(is_not_nan, self._obj.P_RAJ_D_0, 0)):
            raise ValueError(f"P_RAJ_Z ({self._obj.P_RAJ_Z}) has to be larger than P_RAJ_D_0 ({self._obj.P_RAJ_D_0})!")

        if self._obj.d_RAJ >= 0:
            raise ValueError(f"d_RAJ ({self._obj.d_RAJ}) has to be negative!")

        # eq. (2.9-27)
        self._P_RAJ_D = self._obj.P_RAJ_D_0


    def update_P_RAJ_D(self, P_RAJ_D):
        """Update the active ``P_RAJ`` fatigue strength limit.

        Parameters
        ----------
        P_RAJ_D : pandas.Series
            New fatigue strength limit values in MPa for one or more assessment
            points.
        """

        self._P_RAJ_D = P_RAJ_D

    def get_woehler_curve_minimum_lifetime(self):
        """Get the scalar curve for the assessment point with minimum lifetime.

        Use this method after a vectorized FKM nonlinear assessment, for
        example on a mesh, when the resulting Wöhler curve should be plotted
        for the most critical assessment point.  If the curve already contains
        scalar values, the copied curve is unchanged.

        Returns
        -------
        WoehlerCurvePRAJ
            Deep copy of the current Wöhler curve with scalar ``P_RAJ_Z`` and
            ``P_RAJ_D_0`` values equal to the minima of the vectorized values.
        """

        woehler_curve_minimum_lifetime = copy.deepcopy(self)

        # get the minimum values for all vectorized values
        woehler_curve_minimum_lifetime._obj.P_RAJ_Z = np.min(woehler_curve_minimum_lifetime._obj.P_RAJ_Z)
        woehler_curve_minimum_lifetime._obj.P_RAJ_D_0 = np.min(woehler_curve_minimum_lifetime._obj.P_RAJ_D_0)

        return woehler_curve_minimum_lifetime

    @property
    def d(self):
        """Return the finite-life Wöhler curve slope."""
        return self._obj.d_RAJ

    @property
    def P_RAJ_D(self):
        """Return the active ``P_RAJ`` fatigue strength limit."""
        return self._P_RAJ_D

    @property
    def P_RAJ_Z(self):
        """Return the ``P_RAJ`` value at ``N = 1`` cycle."""
        return self._obj.P_RAJ_Z

    def calc_P_RAJ(self, N):
        """Evaluate the Wöhler curve at a number of cycles.

        Parameters
        ----------
        N : array_like
            Number of cycles where to evaluate the Wöhler curve.

        Returns
        -------
        numpy.ndarray
            ``P_RAJ`` values in MPa that correspond to the given cycle counts.
        """
        N = np.array(N)


        # Note, this formula was derived visually from the figure 2.18 on page 93 of the FKM nonlinear document
        # N = (P_RAJ / P_RAJ_Z) ^ (1/d)
        # N^d = P_RAJ / P_RAJ_Z
        # P_RAJ = P_RAJ_Z * N^d

        return np.where(N < self.fatigue_life_limit,
                       self.P_RAJ_Z * np.power(N, self.d),
                       self.fatigue_strength_limit)

    def calc_N(self, P_RAJ, P_RAJ_D=None):
        """Evaluate the Wöhler curve at a ``P_RAJ`` value.

        Parameters
        ----------
        P_RAJ : float
            Damage-parameter value in MPa.
        P_RAJ_D : float, optional
            Alternative fatigue strength limit in MPa.  If omitted, use the
            active limit stored in the Wöhler curve.  Default is ``None``.

        Returns
        -------
        float
            Number of cycles to failure.  The result is ``numpy.inf`` when
            ``P_RAJ`` is at or below the active fatigue strength limit.
        """

        if P_RAJ_D is None:
            P_RAJ_D = self._P_RAJ_D

        # silence warning "divide by zero in np.power. This happens for P_RAJ=0, but then it will use the second branch with N=np.inf anyways
        with np.errstate(divide='ignore'):
            N = np.where(P_RAJ > P_RAJ_D,
                         np.power(P_RAJ / self._obj.P_RAJ_Z, 1/self._obj.d_RAJ),
                         np.inf)

        return N

    @property
    def fatigue_strength_limit(self):
        """Return the initial ``P_RAJ`` fatigue strength limit."""

        return self._obj.P_RAJ_D_0

    @property
    def fatigue_strength_limit_final(self):
        """Return the ``P_RAJ`` fatigue strength limit after the FKM algorithm."""

        return self._P_RAJ_D

    @property
    def fatigue_life_limit(self):
        """Return the cycle count at the initial ``P_RAJ`` fatigue limit."""
        # ND = (P_RAJ_D / P_RAJ_Z) ^ (1/d)

        return (self.fatigue_strength_limit / self.P_RAJ_Z) ** (1/self.d)

    @property
    def fatigue_life_limit_final(self):
        """Return the cycle count at the final ``P_RAJ`` fatigue limit."""

        return (self.fatigue_strength_limit_final / self.P_RAJ_Z) ** (1/self.d)

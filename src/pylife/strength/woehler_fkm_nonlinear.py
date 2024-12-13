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
import scipy.stats as stats
import warnings
import copy

from pylife import PylifeSignal


@pd.api.extensions.register_series_accessor('woehler_P_RAM')
@pd.api.extensions.register_dataframe_accessor('woehler_P_RAM')
class WoehlerCurvePRAM(PylifeSignal):
    """This class represents the type of (component) Wöhler curve that is used
    in the FKM nonlinear fatigue assessment with damage parameter P_RAM.

    The Wöhler Curve (aka SN-curve) determines after how many load cycles at a
    certain load amplitude the component is expected to fail.

    This Wöhler curve is defined piecewise with three sections: Two sections with slopes :math:`d_1`, :math:`d_2` and
    then a horizontal section at the endurance limit (cf. Sec. 2.5.6 of the FKM nonlinear document).

    The signal has the following mandatory keys:

    * ``d_1`` : The slope of the Wöhler Curve in the first section, for N < 1e3
    * ``d_2`` : The slope of the Wöhler Curve in the second section, for N >= 1e3
    * ``P_RAM_Z`` : The damage parameter value that separates the first and second section, corresponding to N = 1e3
    * ``P_RAM_D`` : The damage parameter value of the endurance limit of the component, computed as P_RAM_D_WS / f_RAM (FKM nonlinear, eq. 2.6-89)
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
        """If this woehler curve is vectorized, i.e., holds values for multiple assessment points at once,
        get a version of this woehler curve for the assessment point with the minimum lifetime.
        This function is usually needed after an FKM nonlinear assessment for multiple points (e.g, o whole mesh at once),
        if the woehler curve should be plotted afterwards. Plotting is only possible for a specific woehler curve of
        a single point.

        If the woehler curve was for a single assessment point before, nothing is changed.

        Returns
        -------
        woehler_curve_minimum_lifetime : WoehlerCurvePRAJ
            A deep copy of the current woehler curve object, but with scalar values. The
            resulting values are the minimum of the stored vectorized values.
        """

        # compute the minimum/maximum for the vectorized items
        woehler_curve_minimum_lifetime = copy.deepcopy(self)
        woehler_curve_minimum_lifetime._obj.P_RAM_Z = np.min(woehler_curve_minimum_lifetime._obj.P_RAM_Z)
        woehler_curve_minimum_lifetime._obj.P_RAM_D = np.min(woehler_curve_minimum_lifetime._obj.P_RAM_D)

        return woehler_curve_minimum_lifetime

    @property
    def d_1(self):
        """The slope of the Wöhler Curve in the first section, for N < 1e3"""
        return self._obj.d_1

    @property
    def d_2(self):
        """The slope of the Wöhler Curve in the second section, for N >= 1e3"""
        return self._obj.d_2

    @property
    def P_RAM_Z(self):
        """The damage parameter value that separates the first and second section, corresponding to N = 1e3"""
        return self._obj.P_RAM_Z

    @property
    def P_RAM_D(self):
        """The damage parameter value of the endurance limit"""
        return self._obj.P_RAM_D

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

        # silence warning "divide by zero in np.power. This happens for P_RAM=0, but then it will use the second branch with N=np.inf anyways
        with np.errstate(divide='ignore'):
            N = np.where(P_RAM > self.fatigue_strength_limit,
                         np.where(P_RAM >= self.P_RAM_Z,
                                  1e3 * np.power(P_RAM / self.P_RAM_Z, 1/self.d_1),
                                  1e3 * np.power(P_RAM / self.P_RAM_Z, 1/self.d_2)),
                         np.inf)

        return N

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
        """The fatigue strength limit of the component, i.e.,
        the P_RAM value below which we have infinite life."""

        return self.P_RAM_D

    @property
    def fatigue_life_limit(self):
        """The fatigue life limit N_D of the component, i.e.,
        the number of cycles at the fatigue strength limit P_RAM_D."""

        # exp(log(P_RAM_Z) + (log(N) - log(1e3))*d_2)
        # P_RAM_Z * exp((log(N) - log(1e3))*d_2) = fatigue_strength_limit for N=fatigue_life_limit
        # =>  P_RAM_Z * exp((log(fatigue_life_limit) - log(1e3))*d_2) = fatigue_strength_limit
        # =>  fatigue_life_limit = exp(log(fatigue_strength_limit / P_RAM_Z) / d_2 + log(1e3)) = 1e3 * (fatigue_strength_limit / P_RAM_Z)^(1/d_2)

        return 1e3 * (self.fatigue_strength_limit / self.P_RAM_Z) ** (1/self._obj.d_2)


@pd.api.extensions.register_series_accessor('woehler_P_RAJ')
@pd.api.extensions.register_dataframe_accessor('woehler_P_RAJ')
class WoehlerCurvePRAJ(PylifeSignal):
    """This class represents the type of (component) Wöhler curve that is used in the
    FKM nonlinear fatigue assessment with damage parameter P_RAJ.

    The Wöhler Curve (aka SN-curve) determines after how many load cycles at a
    certain load amplitude the component is expected to fail.

    This Wöhler curve is defined piecewise with two sections for finite and infinite life:
    The sloped section with slope :math:`d` and the horizontal section at the endurance limit
    (cf. Sec. 2.8.6 of the FKM nonlinear document).

    The signal has the following mandatory keys:

    * ``d_RAJ`` : The slope of the Wöhler Curve in the finite life section.
    * ``P_RAJ_Z`` : The load limit at N = 1
    * ``P_RAJ_D_0`` : The initial load level of the endurance limit
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
        """This method is used to update the fatigue strength P_RAJ_D, which is stored in this woehler curve.

        Parameters
        ----------
        P_RAJ_D : pandas Series
            The new fatigue strength values that will be set for the woehler curve for multiple assessment points.
        """

        self._P_RAJ_D = P_RAJ_D

    def get_woehler_curve_minimum_lifetime(self):
        """If this woehler curve is vectorized, i.e., holds values for multiple assessment points at once,
        get a version of this woehler curve for the assessment point with the minimum lifetime.
        This function is usually needed after an FKM nonlinear assessment for multiple points (e.g, o whole mesh at once),
        if the woehler curve should be plotted afterwards. Plotting is only possible for a specific woehler curve of
        a single point.

        If the woehler curve was for a single assessment point before, nothing is changed.

        Returns
        -------
        woehler_curve_minimum_lifetime : WoehlerCurvePRAJ
            A deep copy of the current woehler curve object, but with scalar values. The
            resulting values are the minimum of the stored vectorized values.
        """

        woehler_curve_minimum_lifetime = copy.deepcopy(self)

        # get the minimum values for all vectorized values
        woehler_curve_minimum_lifetime._obj.P_RAJ_Z = np.min(woehler_curve_minimum_lifetime._obj.P_RAJ_Z)
        woehler_curve_minimum_lifetime._obj.P_RAJ_D_0 = np.min(woehler_curve_minimum_lifetime._obj.P_RAJ_D_0)

        return woehler_curve_minimum_lifetime

    @property
    def d(self):
        """The slope of the Wöhler Curve in the first section"""
        return self._obj.d_RAJ

    @property
    def P_RAJ_D(self):
        """The fatigue strength for multiple assessment points."""
        return self._P_RAJ_D

    @property
    def P_RAJ_Z(self):
        """The P_RAJ value for N=1."""
        return self._obj.P_RAJ_Z

    def calc_P_RAJ(self, N):
        """Evaluate the woehler curve at the specified number of cycles.

        Parameters
        ----------
        N : array-like
            Number of cycles where to evaluate the woehler curve.

        Returns
        -------
        array-like
            The P_RAJ values that correspond to the given N values.

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
        """Evaluate the woehler curve at the given damage paramater value, P_RAJ.

        Parameters
        ----------
        P_RAJ : float
            The damage parameter value where to evaluate the woehler curve.
        P_RAJ_D : float
            (optional) A different fatigue strength limit P_RAJ_D, if not set, the normal fatigue strength limit is used.

        Returns
        -------
        N : float
            The number of cycles for the given P_RAJ value.

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
        """The fatigue strength limit of the component."""

        return self._obj.P_RAJ_D_0

    @property
    def fatigue_strength_limit_final(self):
        """The fatigue strength limit of the component, after the FKM algorithm."""

        return self._P_RAJ_D

    @property
    def fatigue_life_limit(self):
        """The fatigue strength limit N_D of the component, i.e.,
        the number of cycles at the fatigue strength limit."""
        # ND = (P_RAJ_D / P_RAJ_Z) ^ (1/d)

        return (self.fatigue_strength_limit / self.P_RAJ_Z) ** (1/self.d)

    @property
    def fatigue_life_limit_final(self):
        """The fatigue strength limit N_D of the component, i.e.,
        the number of cycles at the fatigue strength limit, after the FKM algorithm."""

        return (self.fatigue_strength_limit_final / self.P_RAJ_Z) ** (1/self.d)

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

"""Represent measured Wöhler fatigue test data as a pandas accessor.

The module registers :attr:`pandas.DataFrame.fatigue_data` and provides a
helper for deriving fracture and runout outcomes from a cycle limit.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

from pylife.utils.functions import scattering_range_to_std

from pylife import PylifeSignal
from pylife import DataValidator


@pd.api.extensions.register_dataframe_accessor('fatigue_data')
class FatigueData(PylifeSignal):
    """Validate and partition measured Wöhler fatigue test data.

    ``FatigueData`` is a :class:`~pylife.PylifeSignal` accessor registered as
    :attr:`pandas.DataFrame.fatigue_data`.  It stores individual fatigue tests
    and separates finite-life tests from endurance-limit tests for the Wöhler
    analyzers.

    The signal has the following mandatory keys:

    * ``load`` : Load level applied during the test, usually a stress or force
      amplitude in the user's units.
    * ``cycles`` : Number of cycles reached by the specimen.  For fractures it
      is the cycles to failure; for runouts it is the stopped test duration.
    * ``fracture`` : Boolean outcome flag.  ``True`` marks a fractured
      specimen, ``False`` marks a runout that survived the specified cycles.

    Parameters
    ----------
    pandas_obj : pandas.DataFrame
        Fatigue test data with the mandatory ``load``, ``cycles``, and
        ``fracture`` columns.

    Notes
    -----
    The finite zone contains fractured tests above the finite-infinite
    transition load.  The infinite zone contains all tests at or below that
    transition and is used to evaluate the endurance limit ``SD`` according to
    DIN 50100-style Wöhler testing.
    """

    def _validate(self):
        self.fail_if_key_missing(['load', 'cycles', 'fracture'])
        self._finite_infinite_transition = None

    def sanitize_check(self):
        """Perform sanitize checks on the fatigue data and raise on failure."""
        if not self._obj.fracture.any():
            raise ValueError("Need at least one fracture.")
        if self.fractures.cycles.max() == self.fractures.cycles.min():
            raise ValueError("There must be a variance in fracture cycles.")

    @property
    def num_tests(self):
        """Return the number of tests."""
        return self._obj.shape[0]

    @property
    def num_fractures(self):
        """Return the number of fracture tests."""
        return self.fractures.shape[0]

    @property
    def num_runouts(self):
        """Return the number of runout tests."""
        return self.runouts.shape[0]

    @property
    def fractures(self):
        """Return only fracture tests."""
        return self._obj[self._obj.fracture]

    @property
    def runouts(self):
        """Return only runout tests."""
        return self._obj[~self._obj.fracture]

    @property
    def load(self):
        """Return the test load levels."""
        return self._obj.load

    @property
    def cycles(self):
        """Return the reached cycle numbers."""
        return self._obj.cycles

    @property
    def fracture(self):
        """Return the fracture outcome flags."""
        return self._obj.fracture

    @property
    def finite_infinite_transition(self):
        """Return the estimated load that separates finite and infinite life.

        The transition is determined from the highest runout load and the next
        higher fracture load.  It is used as the initial endurance limit load
        ``SD`` for subsequent Wöhler analyses.
        """
        if self._finite_infinite_transition is None:
            self._calc_finite_infinite_transition()
        return self._finite_infinite_transition

    @property
    def finite_zone(self):
        """Return fracture tests above ``finite_infinite_transition``."""
        if self._finite_infinite_transition is None:
            self._calc_finite_infinite_transition()
        return self._finite_zone

    @property
    def infinite_zone(self):
        """Return tests at or below ``finite_infinite_transition``."""
        if self._finite_infinite_transition is None:
            self._calc_finite_infinite_transition()
        return self._infinite_zone

    @property
    def fractured_loads(self):
        """Return unique load levels with at least one fracture."""
        return np.unique(self.fractures.load.values)

    @property
    def runout_loads(self):
        """Return unique load levels with at least one runout."""
        return np.unique(self.runouts.load.values)

    @property
    def non_fractured_loads(self):
        """Return load levels with runouts and no fractures."""
        return np.setdiff1d(self.runout_loads, self.fractured_loads)

    @property
    def mixed_loads(self):
        """Return load levels with both fractures and runouts."""
        return np.intersect1d(self.runout_loads, self.fractured_loads)

    @property
    def pure_runout_loads(self):
        """Return load levels that contain runouts but no fractures."""
        return np.setxor1d(self.runout_loads, self.mixed_loads)

    def conservative_finite_infinite_transition(self):
        """Set a conservative finite-infinite transition load.

        The method averages all mixed load levels and the highest pure-runout
        load level [1]_.  This lowers the estimated endurance limit compared
        with the default transition search when pure runouts exist below
        mixed levels.

        Returns
        -------
        FatigueData
            The same accessor with the updated transition load.

        References
        ----------
        .. [1] Mustafa Kassem, "Open Source Software Development for
           Reliability and Lifetime Calculation", p. 34.
        """
        amps_to_consider = self.mixed_loads

        if len(self.non_fractured_loads ) > 0:
            amps_to_consider = np.concatenate((amps_to_consider, [self.non_fractured_loads.max()]))

        if len(amps_to_consider) > 0:
            self._finite_infinite_transition = amps_to_consider.mean()
            self._calc_finite_zone()

        return self

    def set_finite_infinite_transition(self, finite_infinite_transition):
        """Set the transition load between finite and infinite life manually.

        Parameters
        ----------
        finite_infinite_transition : float
            Load level used as the endurance-limit start value and as the
            boundary between finite and infinite zones.

        Returns
        -------
        FatigueData
            The same accessor with recalculated finite and infinite zones.
        """
        self._finite_infinite_transition = finite_infinite_transition
        self._calc_finite_zone_manual(finite_infinite_transition)

        return self

    def irrelevant_runouts_dropped(self):
        """Return data with pure runout levels below the relevant range dropped.

        Returns
        -------
        FatigueData
            The current accessor when no runouts are irrelevant, otherwise a
            new accessor without pure runout levels below the highest pure
            runout level.
        """
        if len(self.pure_runout_loads) <= 1:
            return self
        if self.pure_runout_loads.max() < self.fractured_loads.min():
            df = self._obj[~(self._obj.load < self.pure_runout_loads.max())]
            return FatigueData(df).set_finite_infinite_transition(self._finite_infinite_transition)
        else:
            return self

    @property
    def max_runout_load(self):
        """Return the highest load level with a runout."""
        return self.runouts.load.max()

    def _calc_finite_infinite_transition(self):
        self._calc_finite_zone()
        self._finite_infinite_transition = 0.0 if len(self.runouts) == 0 else self._half_level_above_highest_runout()

    def _half_level_above_highest_runout(self):
        if len(self._finite_zone) > 0:
            return (self._finite_zone.load.min() + self.max_runout_load) / 2.

        return self._guess_from_second_highest_runout()

    def _guess_from_second_highest_runout(self):
        max_loads = np.sort(self._obj.load.unique())[-2:]
        return max_loads[1] + (max_loads[1]-max_loads[0]) / 2.

    def _calc_finite_zone(self):
        if len(self.runouts) > 0:
            return self._calc_finite_zone_manual(self.max_runout_load)
        self._infinite_zone = self._obj[:0]
        self._finite_zone = self._obj

    def _calc_finite_zone_manual(self, limit):
        self._finite_zone = self.fractures[self.fractures.load > limit]
        self._infinite_zone = self._obj[self._obj.load <= limit]


def determine_fractures(df, load_cycle_limit=None):
    """Add a ``fracture`` column from a runout cycle limit.

    Parameters
    ----------
    df : pandas.DataFrame
        Fatigue test data with ``load`` and ``cycles`` columns but without a
        required ``fracture`` column.
    load_cycle_limit : float, optional
        Cycle count at which tests are classified as runouts.  Tests with
        ``cycles`` greater than or equal to this value become runouts, all
        others become fractures.  Default is the maximum cycle count in ``df``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the Boolean column ``fracture`` added.

    Examples
    --------
    >>> df = pd.DataFrame({"load": [300.0, 280.0], "cycles": [1000, 10000]})
    >>> determine_fractures(df, load_cycle_limit=10000)["fracture"].tolist()
    [True, False]
    """
    DataValidator().fail_if_key_missing(df, ['load', 'cycles'])
    if load_cycle_limit is None:
        load_cycle_limit = df.cycles.max()
    ret = df.copy()
    ret['fracture'] = df.cycles < load_cycle_limit
    return ret

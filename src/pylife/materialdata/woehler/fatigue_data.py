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

import pandas as pd
import numpy as np
import scipy.stats as stats

from pylife.utils.functions import scattering_range_to_std

from pylife import PylifeSignal
from pylife import DataValidator


@pd.api.extensions.register_dataframe_accessor('fatigue_data')
class FatigueData(PylifeSignal):
    """Class for fatigue data

    Mandatory keys are
        * ``load`` : float, the load level
        * ``cycles`` : float, the cycles of failure or runout
        * ``fracture``: bool, ``True`` iff the test is a fracture
    """

    def _validate(self):
        self.fail_if_key_missing(['load', 'cycles', 'fracture'])
        if not self._obj.fracture.any():
            raise ValueError("Need at least one fracture.")
        if self.fractures.cycles.max() == self.fractures.cycles.min():
            raise ValueError("There must be a variance in fracture cycles.")
        self._finite_infinite_transition = None

    @property
    def num_tests(self):
        '''The number of tests'''
        return self._obj.shape[0]

    @property
    def num_fractures(self):
        '''The number of fractures'''
        return self.fractures.shape[0]

    @property
    def num_runouts(self):
        '''The number of runouts'''
        return self.runouts.shape[0]

    @property
    def fractures(self):
        '''Only the fracture tests'''
        return self._obj[self._obj.fracture]

    @property
    def runouts(self):
        '''Only the runout tests'''
        return self._obj[~self._obj.fracture]

    @property
    def load(self):
        '''The load levels'''
        return self._obj.load

    @property
    def cycles(self):
        '''the cycle numbers'''
        return self._obj.cycles

    @property
    def finite_infinite_transition(self):
        '''The start value of the load endurance limit.

        It is determined by searching for the lowest load level before the
        appearance of a runout data point, and the first load level where a
        runout appears.  Then the median of the two load levels is the start
        value.
        '''
        if self._finite_infinite_transition is None:
            self._calc_finite_infinite_transition()
        return self._finite_infinite_transition

    @property
    def finite_zone(self):
        '''All the tests with load levels above ``finite_infinite_transition``, i.e. the finite zone'''
        if self._finite_infinite_transition is None:
            self._calc_finite_infinite_transition()
        return self._finite_zone

    @property
    def infinite_zone(self):
        '''All the tests with load levels below ``finite_infinite_transition``, i.e. the infinite zone'''
        if self._finite_infinite_transition is None:
            self._calc_finite_infinite_transition()
        return self._infinite_zone

    @property
    def fractured_loads(self):
        return np.unique(self.fractures.load.values)

    @property
    def runout_loads(self):
        return np.unique(self.runouts.load.values)

    @property
    def non_fractured_loads(self):
        return np.setdiff1d(self.runout_loads, self.fractured_loads)

    @property
    def mixed_loads(self):
        return np.intersect1d(self.runout_loads, self.fractured_loads)

    @property
    def pure_runout_loads(self):
        return np.setxor1d(self.runout_loads, self.mixed_loads)

    def conservative_finite_infinite_transition(self):
        """
        Sets a lower fatigue limit that what is expected from the algorithm given by Mustafa Kassem.
        For calculating the fatigue limit, all amplitudes where runouts and fractures are present are collected.
        To this group, the maximum amplitude with only runouts present is added.
        Then, the fatigue limit is the mean of all these amplitudes.

        Returns
        -------
        self

        See also
        --------
        Kassem, Mustafa - "Open Source Software Development for Reliability and Lifetime Calculation" pp. 34
        """
        amps_to_consider = self.mixed_loads

        if len(self.non_fractured_loads ) > 0:
            amps_to_consider = np.concatenate((amps_to_consider, [self.non_fractured_loads.max()]))

        if len(amps_to_consider) > 0:
            self._finite_infinite_transition = amps_to_consider.mean()
            self._calc_finite_zone()

        return self

    def set_finite_infinite_transition(self, finite_infinite_transition):
        """
        Allows the user to set an arbitrary fatigue limit.

        Parameters
        ----------
        finite_infinite_transition : float
            The fatigue limit for separating the finite and infinite zone is set.

        Returns
        -------
        self
        """
        self._finite_infinite_transition = finite_infinite_transition
        self._calc_finite_zone_manual(finite_infinite_transition)

        return self

    def irrelevant_runouts_dropped(self):
        '''Make a copy of the instance with irrelevant pure runout levels dropped. '''
        if len(self.pure_runout_loads) <= 1:
            return self
        if self.pure_runout_loads.max() < self.fractured_loads.min():
            df = self._obj[~(self._obj.load < self.pure_runout_loads.max())]
            return FatigueData(df)
        else:
            return self

    @property
    def max_runout_load(self):
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
    '''Adds a fracture column according to defined load cycle limit

    Parameters
    ----------
    df : DataFrame
        A ``DataFrame`` containing ``fatigue_data`` without ``fractures`` column
    load_cycle_limit : float, optional
        If given, all the tests of ``df`` with ``cycles`` equal od above
        ``load_cycle_limit`` are considered as runouts. Others as fractures.
        If not given the maximum cycle number in ``df`` is used as load cycle
        limit.

    Returns
    -------
    df : DataFrame
        A ``DataFrame`` with the column ``fracture`` added
    '''
    DataValidator().fail_if_key_missing(df, ['load', 'cycles'])
    if load_cycle_limit is None:
        load_cycle_limit = df.cycles.max()
    ret = df.copy()
    ret['fracture'] = df.cycles < load_cycle_limit
    return ret

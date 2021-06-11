# Copyright (c) 2019-2021 - for information on the respective copyright owner
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

'''
Rainflow counting
=================

A module performing rainflow counting
'''

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import numpy as np
import pandas as pd

import cython


def get_turns(samples):
    ''' Finds the turning points in a sample chunk

    Parameters
    ----------
    samples : 1D numpy.ndarray
        the sample chunk

    Returns
    -------
    positions : 1D numpy.ndarray
        the indeces where sample has a turning point
    turns : 1D numpy.ndarray
        the values of the turning points

    '''
    def plateau_turns(diffs):
        plateau_turns = np.zeros_like(diffs, dtype=np.bool_)[1:]
        duplicates = np.array(diffs == 0, dtype=np.int8)

        if duplicates.any():
            edges = np.diff(duplicates)
            dups_starts = np.where(edges > 0)[0]
            dups_ends = np.where(edges < 0)[0]
            if len(dups_starts) and len(dups_ends) and dups_ends[0] < dups_starts[0]:
                dups_ends = dups_ends[1:]
            plateau_turns[dups_starts[np.where(diffs[dups_starts] * diffs[dups_ends+1] < 0)]] = True

        return plateau_turns

    diffs = np.diff(samples)
    peak_turns = diffs[:-1] * diffs[1:] < 0.0

    positions = np.where(np.logical_or(peak_turns, plateau_turns(diffs)))[0] + 1

    return positions, samples[positions]


class AbstractRainflowCounter:
    '''The common base class for rainflow counters

    Subclasses implementing a specific rainflow counting algorithm are
    supposed to implement a method ``process()`` that takes the signal
    samples as a parameter, append all the hysteresis loop limits to
    ``self._loops_from`` and ``self.loops_to`` and return ``self``. The
    ``process()`` method is supposed to be implemented in a way, that
    the result is independent of the sample chunksize, so
    ``rfc.process(signal)`` should be equivalent to
    ``rfc.process(signal[:n]).process(signal[n:])`` for any 0 < n <
    signal length.

    Todo
    ----
    - write a 4 point rainflow counter
    - accept the histogram binning upfront so that loop information
      has not to be stored explicitly. This is important to ensure
      that the memory consumption remains O(1) rather than O(n).
      '''
    def __init__(self):
        self.loops_from = []
        self.loops_to = []
        self._sample_tail = None

    def _get_new_turns(self, samples):
        if self._sample_tail is not None:
            samples = np.concatenate((self._sample_tail, samples))
        turn_positions, turns = get_turns(samples)
        if turn_positions.size > 0:
            self._sample_tail = samples[turn_positions[-1]:]
        else:
            self._sample_tail = samples
        return turns

    def get_rainflow_matrix(self, bins):
        ''' Calculates a histogram of the recorded loops

        Parameters
        ----------
        bins : int or array_like or [int, int] or [array, array], optional
            The bin specification (see numpy.histogram2d)

        Returns
        -------
        H : ndarray, shape(nx, ny)
            The bi-dimensional histogram of samples (see numpy.histogram2d)
        xedges : ndarray, shape(nx+1,)
            The bin edges along the first dimension.
        yedges : ndarray, shape(ny+1,)
            The bin edges along the second dimension.
        '''
        return np.histogram2d(self.loops_from, self.loops_to, bins)

    def get_rainflow_matrix_frame(self, bins):
        '''Calculates a histogram of the recorded loops into a pandas.DataFrame.

        An interval index is used to index the bins.

        Parameters
        ----------
        bins : int or array_like or [int, int] or [array, array], optional
            The bin specification: see numpy.histogram2d

        Returns
        -------
        pandas.DataFrame
            A pandas.DataFrame using a multi interval index in order to
            index data point for a given from/to value pair.
        '''
        hist, fr, to = self.get_rainflow_matrix(bins)
        index_fr = pd.IntervalIndex.from_breaks(fr)
        index_to = pd.IntervalIndex.from_breaks(to)

        mult_idx = pd.MultiIndex.from_product([index_fr, index_to], names=['from', 'to'])
        return pd.DataFrame(data=hist.flatten(), index=mult_idx)

    def residuals(self):
        ''' Returns the residual turning points of the time signal so far

        The residuals are the loops not (yet) closed.
        '''
        return self._residuals


class RainflowCounterThreePoint(AbstractRainflowCounter):
    ''' Implements 3 point rainflow counting algorithm

    See the `here <subsection_TP_>`_ in the demo for an example.

    We take three turning points into account to detect closed hysteresis loops.

    * start: the point where the loop is starting from
    * front: the turning point after the start
    * back: the turning point after the front

    A loop is considered closed if following conditions are met:

    * the load difference between front and back is bigger than or
      equal the one between start and front. In other words: if the
      back goes beyond the starting point. For example (A-B-C) and
      (B-C-D) not closed, whereas (C-D-E) is.

    * the loop init has not been a loop front in a prior closed
      loop. For example F would close the loops (D-E-F) but D is
      already front of the closed loop (C-D-E).

    * the load level of the front has already been covered by a prior
      turning point. Otherwise it is considered part of the front
      residuum.

    When a loop is closed it is possible that the loop back also
    closes unclosed loops of the past by acting as loop back for an
    unclosed start/front pair. For example E closes the loop (C-D-E)
    and then also (A-B-E).

    ::

        Load -----------------------------
        |        x B               F x
        --------/-\-----------------/-----
        |      /   \   x D         /
        ------/-----\-/-\---------/-------
        |    /     C x   \       /
        --\-/-------------\-----/---------
        |  x A             \   /
        --------------------\-/-----------
        |                    x E
        ----------------------------------
        |              Time

    .. _subsection_TP: ../demos/rainflow.ipynb#Classic-Three-Point-Counting
    '''
    def __init__(self):
        super(RainflowCounterThreePoint, self).__init__()
        self._residuals = None

    @cython.locals(
        start=cython.int, front=cython.int, back=cython.int,
        highest_front=cython.int, lowest_front=cython.int,
        start_val=cython.double, front_val=cython.double, back_val=cython.double,
        turns=cython.double[:])
    def process(self, samples):
        ''' Processes a sample chunk

        Parameters
        ----------
        samples : array_like, shape (N, )
            The samples to be processed

        Returns
        -------
        self : RainflowCounterThreePoint
            The ``self`` object so that processing can be chained

        Example
        -------
        >>> rfc = RainflowCounterThreePoint().process(samples)
        >>> rfc.get_rainflow_matrix_frame(128)
        '''
        if self._residuals is None:
            residuals = samples[:1]
            residual_indeces = [0, 1]
        else:
            residuals = self._residuals[:-1]
            residual_indeces = [*range(len(residuals))]

        turns_np = np.concatenate((residuals, self._get_new_turns(samples), samples[-1:]))
        turns = turns_np

        highest_front = np.argmax(residuals)
        lowest_front = np.argmin(residuals)

        back = residual_indeces[-1] + 1
        while back < turns.shape[0]:
            if len(residual_indeces) >= 2:
                start = residual_indeces[-2]
                front = residual_indeces[-1]
                start_val, front_val, back_val = turns[start], turns[front], turns[back]

                if front_val > turns[highest_front]:
                    highest_front = front
                if front_val < turns[lowest_front]:
                    lowest_front = front

                if (start >= max(lowest_front, highest_front) and
                    np.abs(back_val - front_val) >= np.abs(front_val - start_val) and
                    front != highest_front and front != lowest_front):
                    self.loops_from.append(start_val)
                    self.loops_to.append(front_val)
                    residual_indeces.pop()
                    residual_indeces.pop()
                    continue

            residual_indeces.append(back)
            back += 1

        self._residuals = turns_np[residual_indeces]

        return self


class RainflowCounterFKM(AbstractRainflowCounter):
    '''Implements a rainflow counter as described in FKM non linear

    See the `here <subsection_FKM_>`_ in the demo for an example.

    The algorithm has been published by Clormann & Seeger 1985 and has
    been cited havily since.

    .. _subsection_FKM: ../demos/rainflow.ipynb#Algorithm-recommended-by-FKM-non-linear
    '''
    def __init__(self):
        super(RainflowCounterFKM, self).__init__()
        self._ir = 1
        self._residuals = []
        self._max_turn = 0.0

    @cython.locals(
        turns=cython.double[:],
        iz=cython.int, ir=cython.int,
        last0=cython.double, last1=cython.double,
        loop_assumed=cython.int,
        max_turn=cython.double)
    def process(self, samples):
        ''' Processes a sample chunk

        Parameters
        ----------
        samples : array_like, shape (N, )
            The samples to be processed

        Returns
        -------
        self : RainflowCounterFKM
            The ``self`` object so that processing can be chained

        Example
        -------
        >>> rfc = RainflowCounterFKM().process(samples)
        >>> rfc.get_rainflow_matrix_frame(128)
        '''
        ir = self._ir
        max_turn = self._max_turn
        turns = self._get_new_turns(samples)

        for current in turns:
            loop_assumed = True
            while loop_assumed:
                iz = len(self._residuals)
                loop_assumed = False
                if iz > ir:
                    last0 = self._residuals[-1]
                    last1 = self._residuals[-2]
                    if np.abs(current-last0) >= np.abs(last0-last1):
                        self.loops_from.append(last1)
                        self.loops_to.append(last0)
                        self._residuals.pop()
                        self._residuals.pop()
                        if np.abs(last0) < max_turn and np.abs(last1) < max_turn:
                            loop_assumed = True
                elif iz == ir:
                    if np.abs(current) > max_turn:
                        ir += 1
            max_turn = max(np.abs(current), max_turn)
            self._residuals.append(current)

            self._ir = ir
            self._max_turn = max_turn

        return self

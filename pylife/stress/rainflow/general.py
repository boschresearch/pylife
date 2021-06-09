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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import numpy as np
import pandas as pd


def find_turns(samples):
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
            if len(dups_ends) and dups_ends[0] < dups_starts[0]:
                dups_ends = dups_ends[1:]
            plateau_turns[dups_starts[np.where(diffs[dups_starts] * diffs[dups_ends+1] < 0)]] = True

        return plateau_turns

    diffs = np.diff(samples)
    peak_turns = diffs[:-1] * diffs[1:] < 0.0

    positions = np.where(np.logical_or(peak_turns, plateau_turns(diffs)))[0] + 1

    return positions, samples[positions]


class AbstractDetector:
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
    def __init__(self, recorder):
        self._sample_tail = np.array([])
        self._recorder = recorder
        self._head_index = 0
        self._residual_index = np.array([0], dtype=np.int64)
        self._residuals = np.array([])

    @property
    def residuals(self):
        '''The residual turning points of the time signal so far.

        The residuals are the loops not (yet) closed.
        '''
        return self._residuals

    @property
    def residual_index(self):
        return np.append(self._residual_index, self._head_index - 1)

    @property
    def recorder(self):
        return self._recorder

    def _new_turns(self, samples):
        sample_len = len(samples)
        samples = np.concatenate((self._sample_tail, samples))
        turn_positions, turns = find_turns(samples)
        if turn_positions.size > 0:
            old_sample_tail_length = len(self._sample_tail)
            self._sample_tail = samples[turn_positions[-1]:]
            turn_positions += self._head_index - old_sample_tail_length
        else:
            self._sample_tail = samples

        self._head_index += sample_len

        return turn_positions, turns

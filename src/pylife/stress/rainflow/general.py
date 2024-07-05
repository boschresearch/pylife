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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

from abc import ABCMeta, abstractmethod
import warnings
import numpy as np
import pandas as pd


def find_turns(samples):
    """Find the turning points in a sample chunk.

    Parameters
    ----------
    samples : 1D numpy.ndarray
        the sample chunk

    Returns
    -------
    index : 1D numpy.ndarray
        the indeces where sample has a turning point
    turns : 1D numpy.ndarray
        the values of the turning points

    Notes
    -----
    In case of plateaus i.e. multiple directly neighbored samples with exactly
    the same values, building a turning point together, the first sample of the
    plateau is indexed.


    Warnings
    --------
    Any ``NaN`` values are dropped from the input signal before processing it
    and will thus also not appear in the turns.  In those cases a warning is
    issued.  The reason for this is, that if the ``NaN`` appears next to an
    actual turning point the turning point is no longer detected which will
    lead to an underestimation of the damage sum later in the damage
    calculation.  Generally you should not have ``NaN`` values in your signal.
    If you do, it would be a good idea to clean them out before the rainflow
    detection.

    """

    def clean_nans(samples):
        nans = pd.isna(samples)
        if nans.any():
            warnings.warn(UserWarning("At least one NaN like value has been dropped from the input signal."))
            return samples[~nans], nans
        return samples, None

    def correct_turns_by_nans(index, nans):
        if nans is None:
            return
        nan_positions = np.where(nans)[0]
        for nan_pos in nan_positions:
            index[index >= nan_pos] += 1

    def plateau_turns(diffs):
        plateau_turns = np.zeros_like(diffs, dtype=np.bool_)[1:]
        duplicates = np.array(diffs == 0, dtype=np.int8)

        if duplicates.any():
            edges = np.diff(duplicates)
            dups_starts = np.where(edges > 0)[0]
            dups_ends = np.where(edges < 0)[0]
            if len(dups_starts) and len(dups_ends):
                cut_ends = dups_ends[0] < dups_starts[0]
                cut_starts = dups_starts[-1] > dups_ends[-1]
                if cut_ends:
                    dups_ends = dups_ends[1:]
                if cut_starts:
                    dups_starts = dups_starts[:-1]
                plateau_turns[dups_starts[np.where(diffs[dups_starts] * diffs[dups_ends+1] < 0)]] = True

        return plateau_turns

    samples, nans = clean_nans(samples)

    diffs = np.diff(samples)
    peak_turns = diffs[:-1] * diffs[1:] < 0.0

    index = np.where(np.logical_or(peak_turns, plateau_turns(diffs)))[0] + 1

    turns_values = samples[index]
    correct_turns_by_nans(index, nans)

    return index, turns_values


class AbstractDetector(metaclass=ABCMeta):
    """The common base class for rainflow detectors.

    Subclasses implementing a specific rainflow counting algorithm are supposed
    to implement a method ``process()`` that takes the signal samples as a
    parameter, and reports all the hysteresis loop limits to ``self._recorder``
    using its ``record_values()`` method of.

    Some detectors also report the index of the loop limiting samples to the
    recorder using its ``record_index()`` method. Those detectors should also
    report the size of each processed sample chunk to the recorder using
    ``report_chunk()``.

    The ``process()`` method is supposed return ``self`` and to be implemented
    in a way, that the result is independent of the sample chunksize, so
    ``dtor.process(signal)`` should be equivalent to
    ``dtor.process(signal[:n]).process(signal[n:])`` for any 0 < n < signal
    length.

    Should usually only be instantiated by a sublacsse's ``__init__()`` using
    ``super().__init__()``.
    """

    def __init__(self, recorder):
        """Instantiate an AbstractDetector.

        Parameters
        ----------
        recorder : subclass of :class:`.AbstractRecorder`
            The recorder that the detector will report to.
        """
        self._sample_tail = np.array([])
        self._recorder = recorder
        self._head_index = 0
        self._residual_index = np.array([0], dtype=np.uintp)
        self._residuals = np.array([])

    @property
    def residuals(self):
        """The residual turning points of the time signal so far.

        The residuals are the loops not (yet) closed.
        """
        return self._residuals

    @property
    def residual_index(self):
        """The index of the residual turning points of the time signal so far."""
        return np.append(self._residual_index, self._head_index - 1)

    @property
    def recorder(self):
        """The recorder instance the detector is reporting to."""
        return self._recorder

    @abstractmethod
    def process(self, samples):
        """Process a sample chunk.

        Parameters
        ----------
        samples : array_like, shape (N, )
            The samples to be processed

        Returns
        -------
        self : instance of the subclass
            The ``self`` object so that processing can be chained

        Notes
        -----
        Must be implemented by subclasses.
        """
        return self

    def _new_turns(self, samples):
        """Provide new turning points for the next chunk.

        Parameters
        ----------
        samples : 1-D array of float
            The samples of the chunk to be processed

        Returns
        -------
        turn_index : 1-D array of int
            The global index of the turning points of the chunk to be processed
        turn_values : 1-D array of float
            The values of the turning points

        Notes
        -----
        This method can be called by the ``process()`` implementation of
        subclasses. The sample tail i.e. the samples after the last turning
        point of the chunk are stored and prepended to the samples of the next
        call.
        """
        sample_len = len(samples)
        samples = np.concatenate((self._sample_tail, samples))
        turn_index, turn_values = find_turns(samples)
        if turn_index.size > 0:
            old_sample_tail_length = len(self._sample_tail)
            self._sample_tail = samples[turn_index[-1]:]
            turn_index += self._head_index - old_sample_tail_length
        else:
            self._sample_tail = samples

        self._head_index += sample_len

        return turn_index, turn_values


class AbstractRecorder:
    """A common base class for rainflow recorders.

    Subclasses implementing a rainflow recorder are supposed to implement the
    following methods:

    * ``record_values()``
    * ``record_index()``
    """

    def __init__(self):
        """Instantiate an AbstractRecorder."""
        self._chunks = np.array([], dtype=np.int64)

    @property
    def chunks(self):
        """The limits index of the chunks processed so far.

        Note
        ----
        The first chunk limit is the length of the first chunk, so identical to
        the index to the first sample of the second chunk, if a second chunk
        exists.
        """
        return self._chunks

    def report_chunk(self, chunk_size):
        """Report a chunk.

        Parameters
        ----------
        chunk_size : int
            The length of the chunk previously processed by the detector.

        Note
        ----
        Should be called by the detector after the end of ``process()``.
        """
        self._chunks = np.append(self._chunks, chunk_size)

    def chunk_local_index(self, global_index):
        """Transform the global index to an index valid in a certain chunk.

        Parameters
        ----------
        global_index : array-like int
            The global index to be transformed.

        Returns
        -------
        chunk_number : array of ints
            The number of the chunk the indexed sample is in.
        chunk_local_index : array of ints
            The index of the sample in its chunk.
        """
        chunk_index = np.insert(np.cumsum(self._chunks), 0, 0)
        chunk_num = np.searchsorted(chunk_index, global_index, side='right') - 1

        return chunk_num, global_index - chunk_index[chunk_num]

    def record_values(self, values_from, values_to):  # pragma: no cover
        """Report hysteresis loop values to the recorder.

        Parameters
        ----------
        values_from : list of floats
            The sample values where the hysteresis loop starts from.

        values_to : list of floats
            The sample values where the hysteresis loop goes to and turns back from.

        Note
        ----
        Default implementation does nothing. Can be implemented by recorders
        interested in the hysteresis loop values.
        """
        pass

    def record_index(self, indeces_from, indeces_to):  # pragma: no cover
        """Record hysteresis loop index to the recorder.

        Parameters
        ----------
        indeces_from : list of ints
            The sample indeces where the hysteresis loop starts from.

        indeces_to : list of ints
            The sample indeces where the hysteresis loop goes to and turns back from.

        Note
        ----
        Default implementation does nothing. Can be implemented by recorders
        interested in the hysteresis loop values.
        """
        pass

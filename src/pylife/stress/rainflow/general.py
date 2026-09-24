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

"""Provide common streaming helpers for rainflow detectors and recorders."""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

from abc import ABCMeta, abstractmethod
import warnings
import numpy as np
import pandas as pd


def find_turns(samples):
    """Find turning points in a sample chunk.

    Parameters
    ----------
    samples : numpy.ndarray
        One-dimensional sample chunk containing load values, typically in MPa.

    Returns
    -------
    index : numpy.ndarray
        Indices where ``samples`` has a turning point.
    turns : numpy.ndarray
        Load values at the turning points.

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

    Notes
    -----
    In case of plateaus, i.e. multiple directly neighboring samples with
    exactly the same value building a turning point together, the first sample
    of the plateau is indexed.
    """

    def clean_nans(samples: np.ndarray):
        nans = np.asarray(pd.isna(samples))
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

    # find indices where /\ or \/
    peak_turns = diffs[:-1] * diffs[1:] < 0.0

    index = np.where(np.logical_or(peak_turns, plateau_turns(diffs)))[0] + 1

    turns_values = samples[index]
    correct_turns_by_nans(index, nans)

    return index, turns_values


class AbstractDetector(metaclass=ABCMeta):
    """Define the common base class for streaming rainflow detectors.

    Subclasses implement a concrete rainflow counting rule, such as
    three-point or four-point counting.  A detector receives load samples and
    reports closed hysteresis loops to its recorder through
    ``record_values()``.  Detectors that know the source sample indices also
    call ``record_index()`` and ``report_chunk()``.

    Parameters
    ----------
    recorder : AbstractRecorder
        Recorder that receives the detected rainflow loops.

    Notes
    -----
    Implement ``process()`` so processing is independent of chunk size:
    ``detector.process(signal)`` should be equivalent to
    ``detector.process(signal[:n]).process(signal[n:])`` for any split point
    ``0 < n < len(signal)``.  This enables streamed rainflow counting for
    large signals.

    Rainflow counting in pyLife is based on common three-point and four-point
    algorithms, including the DIN 45667 / ASTM E1049 family of rainflow
    counting methods.
    """

    def __init__(self, recorder):
        """Instantiate a detector base.

        Parameters
        ----------
        recorder : AbstractRecorder
            Recorder that receives detected rainflow loops.
        """
        self._sample_tail = np.array([])
        self._recorder = recorder
        self._head_index = 0
        self._residual_index = np.array([0], dtype=np.uintp)
        self._residuals = np.array([])
        self._is_flushing_enabled = False

    @property
    def residuals(self):
        """Return the residual turning points of the signal processed so far.

        Returns
        -------
        numpy.ndarray
            Load values of turning points that have not yet formed closed
            hysteresis loops.
        """
        return self._residuals

    @property
    def residual_index(self):
        """Return the sample indices of residual turning points.

        Returns
        -------
        numpy.ndarray
            Global sample indices of residual turning points.
        """
        return np.append(self._residual_index, self._head_index - 1)

    @property
    def recorder(self):
        """Return the recorder that receives detected loops.

        Returns
        -------
        AbstractRecorder
            Recorder instance passed at construction time.
        """
        return self._recorder

    @abstractmethod
    def process(self, samples, flush=False):
        """Process a sample chunk.

        Parameters
        ----------
        samples : array_like
            Load samples to process, typically in MPa.
        flush : bool, optional
            Whether to flush cached values at the end.  See
            :meth:`pylife.stress.rainflow.FourPointDetector.process` for
            the user-facing behavior.  Default is ``False``.

        Returns
        -------
        AbstractDetector
            The detector itself so processing can be chained.

        See Also
        --------
        flush : Process samples and force cached values to be emitted.

        Notes
        -----
        Must be implemented by subclasses.
        """

        return self

    def flush(self, samples=[]):
        """Flush cached values after processing an optional final sample chunk.

        If ``process`` is called instead of ``flush``, the last value of a
        load sequence is cached for a subsequent call to ``process``,
        because it may or may not be a turning point of the sequence.

        Using ``flush`` forces processing of the last value. This may not be
        the desired effect as multiple increasing or decreasing values in a
        row could occur, instead of processing only turning points.

        Parameters
        ----------
        samples : array_like, optional
            Final load samples to process, typically in MPa.  Default is
            ``[]``.

        Returns
        -------
        AbstractDetector
            The detector itself so processing can be chained.

        See Also
        --------
        process : Process samples without forcing cached values to be emitted.

        Notes
        -----
        This method is equivalent to ``process(samples, flush=True)``.
        """
        return self.process(samples, flush=True)

    def _new_turns(self, samples, flush=False, preserve_start=False):
        """Provide new turning points for the next chunk.

        This method can handle samples as both one-dimensional arrays and
        multi-dimensional data frames.

        Parameters
        ----------
        samples : array_like or pandas.DataFrame
            Samples of the chunk to process.
        flush : bool, optional
            Whether to flush values at the end instead of keeping a tail.
            Default is ``False``.
        preserve_start : bool, optional
            Whether to preserve the beginning of the sequence.  If this is
            ``False``, only turning points are extracted, for example:
                _new_turns([1, 2, 1])   # -> 2
                _new_turns([0, 1])      # -> 1
            If ``preserve_start`` is True, the first point is also added, even
            though it is not a turn point:
                _new_turns([1, 2, 1], preserve_start=True)   # -> 1, 2
                _new_turns([0, 1], preserve_start=True)      # -> 0, 1

            This option has no effect if there are samples left over
            from a previous call with ``flush=False``.  Default is ``False``.

        Returns
        -------
        turn_index : numpy.ndarray
            Global indices of turning points in the processed chunk.
        turn_values : numpy.ndarray
            Load values of the turning points.

        Notes
        -----
        This method can be called by the ``process()`` implementation of
        subclasses. The sample tail i.e. the samples after the last turning
        point of the chunk are stored and prepended to the samples of the next
        call.
        """

        if len(samples) == 0:
            return np.array([]), np.array([])

        if len(self._sample_tail) > 0:
            preserve_start = False

        samples_with_last_tail = np.concatenate((self._sample_tail, samples))

        turn_index, turn_values = find_turns(samples_with_last_tail)

        sample_tail_index = turn_index[-1] if turn_index.size > 0 else 0
        turn_index += self._head_index - len(self._sample_tail)

        self._sample_tail = samples_with_last_tail[sample_tail_index:]  # FIXME: samples_with_last_tail[-1:] also possible?
        self._head_index += len(samples)

        if flush and len(self._sample_tail) > 0:
            turn_index, turn_values = self._flush_new_turns(turn_index, turn_values)

        if preserve_start:
            turn_index, turn_values = self._preserve_start(turn_index, turn_values, samples[0])

        return turn_index, turn_values

    def _flush_new_turns(self, turn_index, turn_values):
        turn_index = np.concatenate((turn_index, [self._head_index-1]))

        if isinstance(turn_values, np.ndarray):
            turn_values = np.concatenate((turn_values, [self._sample_tail[-1]]))
        else:
            turn_values.append(self._last_sample)

        self._sample_tail = self._sample_tail[-1:]
        return turn_index, turn_values

    def _preserve_start(self, turn_index, turn_values, first_sample):
        if turn_index.size > 0:
            if turn_index[0] > 0:

                # prepend first sample to results
                turn_index = np.insert(turn_index, 0, 0)

                if isinstance(turn_values, np.ndarray):
                    turn_values = np.insert(turn_values, 0, first_sample)
                else:
                    turn_values.insert(0, first_sample)
        return turn_index, turn_values



    def _new_turns_multiple_assessment_points(self, samples, flush=False, preserve_start=False):
        """Provide new turning points for multiple assessment points.

        This method is used when the assessment considers multiple points at
        once.  It is called from ``_new_turns``.

        Parameters
        ----------
        samples : pandas DataFrame
            Samples of the chunk to process.  The index must be a
            :class:`pandas.MultiIndex` with levels ``load_step`` and
            ``node_id``.
        flush : bool, optional
            Whether to flush values at the end instead of keeping a tail.
            Default is ``False``.
        preserve_start : bool, optional
            Whether to preserve the beginning of the sequence.  If this is
            ``False``, only turning points are extracted, for example:
                _new_turns([1, 2, 1])   # -> 2
                _new_turns([0, 1])      # -> 1
            If ``preserve_start`` is True, the first point is also added, even
            though it is not a turn point:
                _new_turns([1, 2, 1], preserve_start=True)   # -> 1, 2
                _new_turns([0, 1], preserve_start=True)      # -> 0, 1

        Returns
        -------
        turn_index : numpy.ndarray
            Global indices of turning points in the processed chunk.
        turn_values : list of pandas.DataFrame
            Values of the turning points for all assessment points.
        """

        assert isinstance(samples[0], pd.DataFrame)
        assert samples[0].index.names == ["load_step", "node_id"]

        # extract the representative samples for the first node
        first_node_id = samples.index.get_level_values("node_id")[0]
        samples_of_first_node = samples[samples.index.get_level_values("node_id") == first_node_id].to_numpy().flatten()

        previous_head_index = self._head_index

        turn_index, _ = self._new_turns(samples_of_first_node, flush, preserve_start)

        # the selected samples are a list of DataFrames. Each DataFrame contains the values for all nodes
        selected_samples = [samples[samples.index.get_level_values("load_step") == index-previous_head_index].reset_index(drop=True) \
                            for index in turn_index]

        return turn_index, selected_samples

class AbstractRecorder:
    """Define the common base class for rainflow recorders.

    Recorders receive loop data from an :class:`AbstractDetector`.  Subclasses
    choose which data to keep, for example only loop loads or also sample
    indices.

    Notes
    -----
    Override ``record_values()`` to store loop turning loads and
    ``record_index()`` to store the corresponding sample indices.
    """

    def __init__(self):
        """Instantiate a recorder base."""
        self._chunks = np.array([], dtype=np.int64)

    @property
    def chunks(self):
        """Return chunk boundary indices reported so far.

        Returns
        -------
        numpy.ndarray
            Sizes of the processed chunks in processing order.

        Notes
        -----
        The first chunk limit is the length of the first chunk, so identical to
        the index to the first sample of the second chunk, if a second chunk
        exists.
        """
        return self._chunks

    def report_chunk(self, chunk_size):
        """Record the size of a processed sample chunk.

        Parameters
        ----------
        chunk_size : int
            Length of the chunk previously processed by the detector.

        Notes
        -----
        Should be called by the detector after the end of ``process()``.
        """
        self._chunks = np.append(self._chunks, chunk_size)

    def chunk_local_index(self, global_index):
        """Transform global sample indices into chunk-local indices.

        Parameters
        ----------
        global_index : array_like
            Global sample indices to transform.

        Returns
        -------
        chunk_number : numpy.ndarray
            Number of the chunk containing each indexed sample.
        chunk_local_index : numpy.ndarray
            Index of each sample within its chunk.
        """
        chunk_index = np.insert(np.cumsum(self._chunks), 0, 0)
        chunk_num = np.searchsorted(chunk_index, global_index, side='right') - 1

        return chunk_num, global_index - chunk_index[chunk_num]

    def record_values(self, values_from, values_to):  # pragma: no cover
        """Report hysteresis loop values to the recorder.

        Parameters
        ----------
        values_from : array_like
            Load values where each hysteresis loop starts, typically in MPa.
        values_to : array_like
            Load values where each hysteresis loop turns back, typically in
            MPa.

        Notes
        -----
        Default implementation does nothing. Can be implemented by recorders
        interested in the hysteresis loop values.
        """
        pass

    def record_index(self, indeces_from, indeces_to):  # pragma: no cover
        """Report hysteresis loop sample indices to the recorder.

        Parameters
        ----------
        indeces_from : array_like
            Sample indices where each hysteresis loop starts.
        indeces_to : array_like
            Sample indices where each hysteresis loop turns back.

        Notes
        -----
        Default implementation does nothing. Can be implemented by recorders
        interested in the hysteresis loop indices.
        """
        pass

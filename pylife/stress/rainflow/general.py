import numpy as np
import pandas as pd


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
            if dups_ends[0] < dups_starts[0]:
                dups_ends = dups_ends[1:]
            plateau_turns[dups_starts[np.where(diffs[dups_starts] * diffs[dups_ends+1] < 0)]] = True

        return plateau_turns

    diffs = np.diff(samples)
    peak_turns = diffs[:-1] * diffs[1:] < 0.0

    positions = np.where(np.logical_or(peak_turns, plateau_turns(diffs)))[0] + 1

    return positions, samples[positions]


class AbstractRainflowDetector:
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
        self._sample_tail = None
        self._recorder = recorder
        self._residuals = []


    @property
    def residuals(self):
        '''The residual turning points of the time signal so far.

        The residuals are the loops not (yet) closed.
        '''
        return self._residuals

    @property
    def recorder(self):
        return self._recorder

    def _get_new_turns(self, samples):
        if self._sample_tail is not None:
            samples = np.concatenate((self._sample_tail, samples))
        turn_positions, turns = get_turns(samples)
        if turn_positions.size > 0:
            self._sample_tail = samples[turn_positions[-1]:]
        else:
            self._sample_tail = samples
        return turns

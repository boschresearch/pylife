# Copyright (c) 2019-2024 - for information on the respective copyright owner
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
from pylife.rainflow_ext import threepoint_loop

from .general import AbstractDetector


class ThreePointDetector(AbstractDetector):
    r"""Classic three point rainflow counting algorithm.

    .. jupyter-execute::

        from pylife.stress.timesignal import TimeSignalGenerator
        import pylife.stress.rainflow as RF

        ts = TimeSignalGenerator(10, {
            'number': 50,
            'amplitude_median': 1.0, 'amplitude_std_dev': 0.5,
            'frequency_median': 4, 'frequency_std_dev': 3,
            'offset_median': 0, 'offset_std_dev': 0.4}, None, None).query(10000)

        rfc = RF.ThreePointDetector(recorder=RF.LoopValueRecorder())
        rfc.process(ts)

        rfc.recorder.collective

    Alternatively you can ask the recorder for a histogram matrix:

    .. jupyter-execute::

        rfc.recorder.histogram(bins=16)

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
    """

    def __init__(self, recorder):
        """Instantiate a ThreePointDetector.

        Parameters
        ----------
        recorder : subclass of :class:`.AbstractRecorder`
            The recorder that the detector will report to.
        """
        super().__init__(recorder)

    def process(self, samples, flush=False):
        """Process a sample chunk.

        Parameters
        ----------
        samples : array_like, shape (N, )
            The samples to be processed

        flush : bool
            Whether to flush the cached values at the end.

            If ``flush=False``, the last value of a load sequence is
            cached for a subsequent call to ``process``, because it may or may
            not be a turning point of the sequence, which is only decided
            when the next data point arrives.

            Setting ``flush=True`` forces processing of the last value.
            When ``process`` is called again afterwards with new data,
            two increasing or decreasing values in a row might have been
            processed, as opposed to only turning points of the sequence.

            Example:
            a)
                process([1, 2], flush=False)  # processes 1
                process([3, 1], flush=True)   # processes 3, 1
                -> processed sequence is [1,3,1], only turning points

            b)
                process([1, 2], flush=True)   # processes 1, 2
                process([3, 1], flush=True)   # processes 3, 1
                -> processed sequence is [1,2,3,1], "," is not a turning point

            c)
                process([1, 2])   # processes 1
                process([3, 1])   # processes 3
                -> processed sequence is [1,3], end ("1") is missing

            d)
                process([1, 2])   # processes 1
                process([3, 1])   # processes 3
                flush()           # process 1
                -> processed sequence is [1,3,1]

        Returns
        -------
        self : ThreePointDetector
            The ``self`` object so that processing can be chained
        """
        samples = np.asarray(samples)

        if len(self._residuals) == 0:
            residuals = samples[:1]
        else:
            residuals = self._residuals[:-1]

        turns_index, turns_values = self._new_turns(samples, flush)

        turns = np.concatenate((residuals, turns_values, samples[-1:]))
        turns_index = np.concatenate((self._residual_index, turns_index.astype(np.uintp)))

        highest_front = np.argmax(residuals)
        lowest_front = np.argmin(residuals)

        (
            from_vals,
            to_vals,
            from_index,
            to_index,
            residual_index
        ) = threepoint_loop(turns, turns_index, highest_front, lowest_front, len(residuals))

        self._recorder.record_values(from_vals, to_vals)
        self._recorder.record_index(from_index, to_index)

        self._residuals = turns[residual_index]
        self._residual_index = turns_index[residual_index[:-1]]
        self._recorder.report_chunk(len(samples))

        return self

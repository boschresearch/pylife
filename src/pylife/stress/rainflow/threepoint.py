# Copyright (c) 2019-2026 - for information on the respective copyright owner
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

"""Detect rainflow cycles with the classic three-point criterion.

Use this module for rainflow counting where loops are closed by a third
turning point that passes the previous reversal and satisfies the residual
conditions of the three-point method.
"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import numpy as np
from pylife.rainflow_ext import threepoint_loop

from .general import AbstractDetector


class ThreePointDetector(AbstractDetector):
    r"""Count rainflow cycles with the classic three-point criterion.

    Use this detector for general load collectives when the three-point
    method is the desired rainflow convention. The detector reports loop
    start and end loads to :class:`pylife.stress.rainflow.LoopValueRecorder`
    or, when sample indices are needed, to
    :class:`pylife.stress.rainflow.FullRecorder`. The recorder collective can
    then be transformed to load amplitude, load range, mean load, and number
    of cycles for fatigue assessment in :mod:`pylife.strength.fatigue`.

    Parameters
    ----------
    recorder : pylife.stress.rainflow.AbstractRecorder
        Recorder receiving detected loop loads in MPa. Use
        :class:`pylife.stress.rainflow.FullRecorder` to store the sample
        indices of the two turning points in addition to the load values.

    See Also
    --------
    pylife.stress.rainflow.FourPointDetector : Count cycles with the four-point criterion.
    pylife.stress.rainflow.FKMDetector : Count cycles by the classic FKM procedure.
    pylife.stress.rainflow.FullRecorder : Store loop loads and sample indices.

    Notes
    -----
    The three-point detector evaluates a start point :math:`S`, the following
    front point :math:`F`, and the following back point :math:`B`. A closed
    loop is counted when the back point reaches beyond the start-front
    excursion,

    .. math::

        |B - F| \ge |F - S|,

    and the front is not part of an already closed loop or an uncovered front
    residual. When a loop closes, the same back point may also close older
    open loops. The recorded loop values are reversal loads in MPa; their
    peak-to-peak load range is :math:`L_R = |L_F - L_S|` and their load
    amplitude is :math:`L_a = L_R / 2`.

    The detector supports chunked processing. Repeated calls to
    :meth:`process` continue the count across chunk boundaries; set
    ``flush=True`` only for the final chunk if the last sample shall be
    considered a turning point.

    Examples
    --------
    >>> from pylife.stress.rainflow import ThreePointDetector, LoopValueRecorder
    >>> detector = ThreePointDetector(recorder=LoopValueRecorder())
    >>> detector.process([0.0, 3.0, -1.0, 2.0, -2.0, 0.0], flush=True) is detector
    True
    >>> detector.recorder.collective
       from   to
    0  -1.0  2.0
    """

    def __init__(self, recorder):
        """Instantiate a three-point detector.

        Parameters
        ----------
        recorder : pylife.stress.rainflow.AbstractRecorder
            Recorder receiving detected loop loads in MPa. The recorder must
            implement ``record_values()``; recorders that also implement
            ``record_index()`` receive sample indices.
        """
        super().__init__(recorder)

    def process(self, samples, flush=False):
        """Process a chunk of load samples.

        Parameters
        ----------
        samples : array_like
            Load samples in MPa. The detector extracts turning points and
            combines them with residual turning points from previous chunks.
        flush : bool, optional
            Force processing of the last value as a turning point. Default is
            ``False``. See
            :meth:`pylife.stress.rainflow.FourPointDetector.process` for the
            streaming consequences of flushing.

        Returns
        -------
        ThreePointDetector
            The detector itself, so that repeated ``process()`` calls can be
            chained.
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

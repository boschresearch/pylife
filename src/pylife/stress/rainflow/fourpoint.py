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

"""Detect rainflow cycles with the four-point hysteresis criterion.

Use this module for general rainflow counting where closed loops are
identified from four consecutive turning points and sample indices may be
recorded together with loop loads.
"""

__author__ = "Vishnu Pradeep"
__maintainer__ = "Johannes Mueller"


import numpy as np
from pylife.rainflow_ext import fourpoint_loop

from .general import AbstractDetector


class FourPointDetector(AbstractDetector):
    r"""Count rainflow cycles with the four-point criterion.

    Use this detector for general load collectives when the four-point
    hysteresis criterion is appropriate. The detector reports loop start and
    end loads to :class:`pylife.stress.rainflow.LoopValueRecorder` or, when
    sample indices are needed, to
    :class:`pylife.stress.rainflow.FullRecorder`. The recorder collective can
    then be transformed to load amplitude, load range, mean load, and number
    of cycles for fatigue assessment in :mod:`pylife.strength.fatigue`.

    Parameters
    ----------
    recorder : pylife.stress.rainflow.AbstractRecorder, optional
        Recorder receiving detected loop loads in MPa. If not given, a new
        :class:`pylife.stress.rainflow.LoopValueRecorder` is created. Use
        :class:`pylife.stress.rainflow.FullRecorder` to store the sample
        indices of the two turning points in addition to the load values.

    See Also
    --------
    pylife.stress.rainflow.ThreePointDetector : Count cycles with the classic three-point criterion.
    pylife.stress.rainflow.FKMDetector : Count cycles by the classic FKM procedure.
    pylife.stress.rainflow.FullRecorder : Store loop loads and sample indices.

    Notes
    -----
    The four-point detector evaluates four consecutive turning points
    :math:`A`, :math:`B`, :math:`C`, and :math:`D`. A closed loop from
    :math:`B` to :math:`C` is counted when the inner range is enclosed by
    both neighboring ranges:

    .. math::

        |D - C| \ge |C - B| \quad \text{and} \quad
        |B - A| \ge |C - B|.

    The closed loop is removed from the residual turning-point sequence and
    the algorithm continues with the joined path from :math:`A` to
    :math:`D`. The recorded loop values are reversal loads in MPa; their
    peak-to-peak load range is :math:`L_R = |L_C - L_B|` and their load
    amplitude is :math:`L_a = L_R / 2`.

    The detector supports chunked processing. Repeated calls to
    :meth:`process` continue the count across chunk boundaries; set
    ``flush=True`` only for the final chunk if the last sample shall be
    considered a turning point.

    Examples
    --------
    >>> from pylife.stress.rainflow import FourPointDetector
    >>> detector = FourPointDetector()
    >>> detector.process([0.0, 3.0, -1.0, 2.0, -2.0, 0.0], flush=True) is detector
    True
    >>> detector.recorder.collective
       from   to
    0  -1.0  2.0
    """

    def __init__(self, recorder=None):
        """Instantiate a four-point detector.

        Parameters
        ----------
        recorder : pylife.stress.rainflow.AbstractRecorder, optional
            Recorder receiving detected loop loads in MPa. The recorder must
            implement ``record_values()``; recorders that also implement
            ``record_index()`` receive sample indices. If not given, a new
            :class:`pylife.stress.rainflow.LoopValueRecorder` is created.
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
            ``False``. If ``False``, the last value is cached for a subsequent
            call because only the next data point can decide whether it is a
            turning point. If ``True``, the last value is processed now; a
            following chunk may therefore introduce two monotonic values in a
            row.

        Returns
        -------
        FourPointDetector
            The detector itself, so that repeated ``process()`` calls can be
            chained.

        Examples
        --------
        >>> from pylife.stress.rainflow import FourPointDetector, FullRecorder
        >>> detector = FourPointDetector(recorder=FullRecorder())
        >>> detector.process([1.0, 2.0], flush=False).process([3.0, 1.0]).recorder.collective
        Empty DataFrame
        Columns: [from, to, index_from, index_to]
        Index: []
        >>> detector = FourPointDetector(recorder=FullRecorder())
        >>> detector.process([1.0, 2.0], flush=True).process([3.0, 1.0]).recorder.collective
           from   to  index_from  index_to
        0   2.0  3.0           1         2
        """

        samples = np.asarray(samples)
        residuals = samples[:1] if self._residuals.size == 0 else self._residuals[:-1]

        turns_index, turns_values = self._new_turns(samples, flush)

        turns_np = np.concatenate((residuals, turns_values, samples[-1:]))
        turns_index = np.concatenate((self._residual_index, turns_index.astype(np.uintp)))

        (
            from_vals,
            to_vals,
            from_index,
            to_index,
            residual_index
        ) = fourpoint_loop(turns_np, turns_index)

        self._recorder.record_values(from_vals, to_vals)
        self._recorder.record_index(from_index, to_index)

        self._residuals = turns_np[residual_index]
        self._residual_index = turns_index[residual_index[:-1]]
        self._recorder.report_chunk(len(samples))

        return self

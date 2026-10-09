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

"""Detect rainflow cycles by the FKM recommended procedure.

The module provides the classic FKM detector for nominal load time series.
Use :class:`pylife.stress.rainflow.FKMDetector` when the assessment method
requires the FKM rainflow counting sequence, and use
:class:`pylife.stress.rainflow.fkm_nonlinear.FKMNonlinearDetector` for the nonlinear HCM
procedure with local stress-strain histories.
"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import numpy as np

from .general import AbstractDetector


class FKMDetector(AbstractDetector):
    r"""Count rainflow cycles by the FKM recommended procedure.

    Use this detector for classic FKM assessments that need closed loops
    from a nominal load signal. The detector reports the loop start and end
    load values to a recorder, usually
    :class:`pylife.stress.rainflow.LoopValueRecorder`, and the recorder
    converts them into a load collective for downstream fatigue assessment.
    For the FKM nonlinear guideline with local stress and strain histories,
    use :class:`pylife.stress.rainflow.fkm_nonlinear.FKMNonlinearDetector` instead.

    Parameters
    ----------
    recorder : pylife.stress.rainflow.AbstractRecorder, optional
        Recorder receiving the detected loop start and end load values in
        MPa. Use :class:`pylife.stress.rainflow.LoopValueRecorder` for a
        load collective and a recorder histogram, or a compatible recorder
        implementing ``record_values()``. If not given, a new
        :class:`pylife.stress.rainflow.LoopValueRecorder` is created.

    See Also
    --------
    pylife.stress.rainflow.fkm_nonlinear.FKMNonlinearDetector : Count cycles for the FKM nonlinear HCM assessment.
    pylife.stress.rainflow.FourPointDetector : Count cycles with the general four-point criterion.
    pylife.stress.rainflow.ThreePointDetector : Count cycles with the classic three-point criterion.
    pylife.stress.rainflow.LoopValueRecorder : Store loop start and end load values.

    Notes
    -----
    The detector implements the FKM recommended rainflow procedure published
    by Clormann and Seeger [FKM-Classic-Rainflow]_. It keeps residual turning
    points and closes a loop when the current load excursion covers the
    previous excursion, i.e. for three successive turning points
    :math:`x_{i-2}`, :math:`x_{i-1}`, and the current point :math:`x_i` when

    .. math::

        |x_i - x_{i-1}| \ge |x_{i-1} - x_{i-2}|.

    The resulting loop is recorded by its two reversal loads. A recorder or
    downstream collective can convert these loads to load range
    :math:`L_R = |L_\mathrm{to} - L_\mathrm{from}|`, load amplitude
    :math:`L_a = L_R / 2`, mean load
    :math:`L_m = (L_\mathrm{to} + L_\mathrm{from}) / 2`, and number of
    cycles.

    The detector supports chunked processing. Repeated calls to
    :meth:`process` continue the count across chunk boundaries and keep
    residual turning points open until later chunks close them. This detector
    does not report sample indices; use
    :class:`pylife.stress.rainflow.FourPointDetector` or
    :class:`pylife.stress.rainflow.ThreePointDetector` with
    :class:`pylife.stress.rainflow.FullRecorder` when indices are required.

    References
    ----------
    .. [FKM-Classic-Rainflow] U. Clormann and T. Seeger, "Rainflow-HCM.
       Rainflow counting for operational fatigue assessments on
       werkstoffmechanischer Grundlage", Stahlbau, 1985.

    Examples
    --------
    >>> from pylife.stress.rainflow import FKMDetector
    >>> detector = FKMDetector()
    >>> detector.process([0.0, 3.0, -1.0, 2.0, -2.0, 0.0], flush=True) is detector
    True
    >>> detector.recorder.collective
       from   to
    0  -1.0  2.0
    """

    def __init__(self, recorder=None):
        """Instantiate an FKM detector.

        Parameters
        ----------
        recorder : pylife.stress.rainflow.AbstractRecorder, optional
            Recorder receiving the detected loop start and end load values
            in MPa. The recorder must implement ``record_values()``. If not
            given, a new :class:`pylife.stress.rainflow.LoopValueRecorder`
            is created.
        """
        super().__init__(recorder)
        self._ir = 1
        self._residuals = []
        self._max_turn = 0.0

    def process(self, samples, flush=False):
        """Process a chunk of load samples.

        Parameters
        ----------
        samples : array_like
            Load samples in MPa. The values are scanned for turning points;
            residual turning points are kept for subsequent chunks.
        flush : bool, optional
            Force processing of the last value as a turning point. Default is
            ``False``. Leave this disabled for streaming data and enable it
            for the final chunk if the last value shall close possible loops.

        Returns
        -------
        FKMDetector
            The detector itself, so that repeated ``process()`` calls can be
            chained.
        """
        ir = self._ir
        max_turn = self._max_turn
        turns_index, turns = self._new_turns(samples, flush)

        from_vals = []
        to_vals = []

        for current in turns:
            loop_assumed = True
            while loop_assumed:
                iz = len(self._residuals)
                if iz < ir:
                    break
                loop_assumed = False
                if iz > ir:
                    last0 = self._residuals[-1]
                    last1 = self._residuals[-2]
                    if np.abs(current-last0) >= np.abs(last0-last1):
                        from_vals.append(last1)
                        to_vals.append(last0)
                        self._residuals.pop()
                        self._residuals.pop()
                        if np.abs(last0) < max_turn and np.abs(last1) < max_turn:
                            loop_assumed = True
                    continue
                if np.abs(current) > max_turn:
                    ir += 1
            max_turn = max(np.abs(current), max_turn)
            self._residuals.append(current)

            self._ir = ir
            self._max_turn = max_turn

        self._recorder.record_values(from_vals, to_vals)

        return self

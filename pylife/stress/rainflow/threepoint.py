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

import cython
import numpy as np

from .general import AbstractDetector


class ThreePointDetector(AbstractDetector):
    r"""Classic three point rainflow counting algorithm.

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
    """

    def __init__(self, recorder):
        """Instantiate a ThreePointDetector.

        Parameters
        ----------
        recorder : subclass of :class:`.AbstractRecorder`
            The recorder that the detector will report to.
        """
        super().__init__(recorder)

    @cython.locals(
        start=cython.int, front=cython.int, back=cython.int,
        highest_front=cython.int, lowest_front=cython.int,
        start_val=cython.double, front_val=cython.double, back_val=cython.double,
        turns=cython.double[:])
    def process(self, samples):
        """Process a sample chunk.

        Parameters
        ----------
        samples : array_like, shape (N, )
            The samples to be processed

        Returns
        -------
        self : ThreePointDetector
            The ``self`` object so that processing can be chained
        """
        if len(self._residuals) == 0:
            residuals = samples[:1]
            residual_index = [0, 1]
        else:
            residuals = self._residuals[:-1]
            residual_index = [*range(len(residuals))]

        turns_index, turns_values = self._new_turns(samples)

        turns_np = np.concatenate((residuals, turns_values, samples[-1:]))
        turns_index = np.concatenate((self._residual_index, turns_index))

        turns = turns_np

        highest_front = np.argmax(residuals)
        lowest_front = np.argmin(residuals)

        back = residual_index[-1] + 1
        while back < turns.shape[0]:
            if len(residual_index) >= 2:
                start = residual_index[-2]
                front = residual_index[-1]
                start_val, front_val, back_val = turns[start], turns[front], turns[back]

                if front_val > turns[highest_front]:
                    highest_front = front
                elif front_val < turns[lowest_front]:
                    lowest_front = front
                elif (start >= max(lowest_front, highest_front) and
                      np.abs(back_val - front_val) >= np.abs(front_val - start_val)):
                    self._recorder.record_values(start_val, front_val)
                    self._recorder.record_index(turns_index[start], turns_index[front])
                    residual_index.pop()
                    residual_index.pop()
                    continue

            residual_index.append(back)
            back += 1

        self._residuals = turns_np[residual_index]
        self._residual_index = turns_index[residual_index[:-1]]
        self._recorder.report_chunk(len(samples))

        return self

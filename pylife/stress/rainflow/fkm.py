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


class FKMDetector(AbstractDetector):
    """Rainflow detector as described in FKM non linear.

    The algorithm has been published by Clormann & Seeger 1985 and has
    been cited havily since.

    See the `here <subsection_FKM_>`_ in the demo for an example.

    Note
    ----
    This detector **does not** report the loop index.

    .. _subsection_FKM: ../demos/rainflow.ipynb#Algorithm-recommended-by-FKM-non-linear
    """

    def __init__(self, recorder):
        """Instantiate a FKMDetector.

        Parameters
        ----------
        recorder : subclass of :class:`.AbstractRecorder`
            The recorder that the detector will report to.
        """
        super().__init__(recorder)
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
        """Process a sample chunk.

        Parameters
        ----------
        samples : array_like, shape (N, )
            The samples to be processed

        Returns
        -------
        self : FKMDetector
            The ``self`` object so that processing can be chained
        """
        ir = self._ir
        max_turn = self._max_turn
        turns_index, turns = self._new_turns(samples)

        for current in turns:
            loop_assumed = True
            while loop_assumed:
                iz = len(self._residuals)
                loop_assumed = False
                if iz > ir:
                    last0 = self._residuals[-1]
                    last1 = self._residuals[-2]
                    if np.abs(current-last0) >= np.abs(last0-last1):
                        self._recorder.record_values(last1, last0)
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

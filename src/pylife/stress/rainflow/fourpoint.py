# Copyright (c) 2019-2022 - for information on the respective copyright owner
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

__author__ = "Vishnu Pradeep"
__maintainer__ = "Johannes Mueller"


import numpy as np

from .general import AbstractDetector


class FourPointDetector(AbstractDetector):
    """Implements four point rainflow counting algorithm.

    .. jupyter-execute::

        from pylife.stress.timesignal import TimeSignalGenerator
        import pylife.stress.rainflow as RF

        ts = TimeSignalGenerator(10, {
            'number': 50,
            'amplitude_median': 1.0, 'amplitude_std_dev': 0.5,
            'frequency_median': 4, 'frequency_std_dev': 3,
            'offset_median': 0, 'offset_std_dev': 0.4}, None, None).query(10000)

        rfc = RF.FourPointDetector(recorder=RF.LoopValueRecorder())
        rfc.process(ts)

        rfc.recorder.collective

    Alternatively you can ask the recorder for a histogram matrix:

    .. jupyter-execute::

        rfc.recorder.histogram(bins=16)

    We take four turning points into account to detect closed hysteresis loops.

    Consider four consecutive peak/valley points say, A, B, C, and D  If B and C are
    contained within A and B, then a cycle is counted from B to C; otherwise no cycle is
    counted.

    i.e, If ``X ≥ Y AND Z ≥ Y`` then a cycle exist ``FROM = B`` and ``TO = C``
    where, ranges ``X = |D–C|``, ``Y = |C–B|``, and ``Z = |B–A|``

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

    So, if a cycle exsist from B to C then delete these peaks from the turns array
    and perform next iteration by joining A&D else if no cylce exsists, then B would
    be the next strarting point.
    """

    def __init__(self, recorder):
        """Instantiate a FourPointDetector.

        Parameters
        ----------
        recorder : subclass of :class:`.AbstractRecorder`
            The recorder that the detector will report to.
        """
        super().__init__(recorder)

    def process(self, samples):
        """Process a sample chunk.

        Parameters
        ----------
        samples : array_like, shape (N, )
            The samples to be processed

        Returns
        -------
        self : FourPointDetector
            The ``self`` object so that processing can be chained
        """

        residuals = samples[:1] if self._residuals.size == 0 else self._residuals[:-1]

        turns_index, turns_values = self._new_turns(samples)

        turns_np = np.concatenate((residuals, turns_values, samples[-1:]))
        turns_index = np.concatenate((self._residual_index, turns_index))

        turns = turns_np

        from_vals = []
        to_vals = []
        from_index = []
        to_index = []

        residual_index = [0, 1]
        i = 2
        while i < len(turns):
            if len(residual_index) < 3:
                residual_index.append(i)
                i += 1
                continue

            a = turns_np[residual_index[-3]]
            b = turns_np[residual_index[-2]]
            c = turns_np[residual_index[-1]]
            d = turns_np[i]

            ab = np.abs(a - b)
            bc = np.abs(b - c)
            cd = np.abs(c - d)
            if bc <= ab and bc <= cd:
                from_vals.append(b)
                to_vals.append(c)

                idx_2 = turns_index[residual_index.pop()]
                idx_1 = turns_index[residual_index.pop()]
                from_index.append(idx_1)
                to_index.append(idx_2)
                continue

            residual_index.append(i)
            i += 1

        self._recorder.record_values(from_vals, to_vals)
        self._recorder.record_index(from_index, to_index)

        self._residuals = turns_np[residual_index]
        self._residual_index = turns_index[residual_index[:-1]]
        self._recorder.report_chunk(len(samples))

        return self

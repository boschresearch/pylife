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

"""Provide backward-compatible shims for the pyLife 1.x rainflow API.

The classes in this module wrap the detector/recorder API introduced in
pyLife 2.0.  They remain available so existing code keeps working, but new code
should instantiate a detector such as
:class:`pylife.stress.rainflow.ThreePointDetector` or
:class:`pylife.stress.rainflow.FKMDetector` together with a recorder such as
:class:`pylife.stress.rainflow.FullRecorder`.
"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import pandas as pd

from .threepoint import ThreePointDetector
from .fkm import FKMDetector
from .recorders import FullRecorder


class AbstractRainflowCounter:
    """Wrap a detector and recorder in the deprecated pyLife 1.x API.

    Notes
    -----
    This compatibility layer stores loops in a
    :class:`pylife.stress.rainflow.FullRecorder`.  Prefer using detectors and
    recorders directly for new code.
    """

    def __init__(self):
        """Instantiate a compatibility rainflow counter."""
        self._recorder = FullRecorder()

    @property
    def loops_from(self):
        """Return the loads where recorded loops start.

        Returns
        -------
        numpy.ndarray
            Start load values, typically in MPa.
        """
        return self._recorder.values_from

    @property
    def loops_to(self):
        """Return the loads where recorded loops turn back.

        Returns
        -------
        numpy.ndarray
            Turn-back load values, typically in MPa.
        """
        return self._recorder.values_to

    def residuals(self):
        """Return residual turning points of the wrapped detector.

        Returns
        -------
        numpy.ndarray
            Load values of turning points that have not yet formed closed
            hysteresis loops.
        """
        return self._detector.residuals

    def get_rainflow_matrix(self, bins):
        """Return a NumPy rainflow matrix of recorded loops.

        Parameters
        ----------
        bins : int, array_like or list
            Bin specification passed to :func:`numpy.histogram2d`.

        Returns
        -------
        tuple
            Histogram counts and bin edges as returned by
            :func:`numpy.histogram2d`.
        """
        return self._recorder.histogram_numpy(bins)

    def get_rainflow_matrix_frame(self, bins):
        """Return a pandas rainflow matrix of recorded loops.

        Parameters
        ----------
        bins : int, array_like or list
            Bin specification passed to :func:`numpy.histogram2d`.

        Returns
        -------
        pandas.DataFrame
            Histogram counts indexed by ``from`` and ``to`` load intervals.
        """
        return pd.DataFrame(self._recorder.histogram(bins))


class RainflowCounterThreePoint(AbstractRainflowCounter):
    """Count loops with the deprecated three-point rainflow API.

    Notes
    -----
    Prefer :class:`pylife.stress.rainflow.ThreePointDetector` with
    :class:`pylife.stress.rainflow.FullRecorder` in new code.
    """

    def __init__(self):
        """Instantiate a three-point compatibility counter."""
        super().__init__()
        self._detector = ThreePointDetector(recorder=self._recorder)

    def process(self, samples):
        """Process load samples with the wrapped three-point detector.

        Parameters
        ----------
        samples : array_like
            Load samples to process, typically in MPa.

        Returns
        -------
        RainflowCounterThreePoint
            The counter itself so processing can be chained.
        """
        self._detector.process(samples)
        return self


class RainflowCounterFKM(AbstractRainflowCounter):
    """Count loops with the deprecated FKM rainflow API.

    Notes
    -----
    Prefer :class:`pylife.stress.rainflow.FKMDetector` with
    :class:`pylife.stress.rainflow.FullRecorder` in new code.
    """

    def __init__(self):
        """Instantiate an FKM compatibility counter."""
        super().__init__()
        self._detector = FKMDetector(recorder=self._recorder)

    def process(self, samples):
        """Process load samples with the wrapped FKM detector.

        Parameters
        ----------
        samples : array_like
            Load samples to process, typically in MPa.

        Returns
        -------
        RainflowCounterFKM
            The counter itself so processing can be chained.
        """
        self._detector.process(samples)
        return self

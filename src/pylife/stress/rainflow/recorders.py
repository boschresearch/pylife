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

"""Provide recorder implementations for rainflow detectors."""

__author__ = ["Johannes Mueller", "Benjamin Maier"]
__maintainer__ = __author__


import numpy as np
import pandas as pd

from .general import AbstractRecorder


class LoopValueRecorder(AbstractRecorder):
    """Record rainflow loop turning loads.

    The recorder stores the load value where each closed hysteresis loop starts
    and the value where it turns back.  These values can be exposed as an
    explicit load collective or binned into a two-dimensional histogram.
    """

    def __init__(self):
        """Instantiate a loop-value recorder."""
        super().__init__()
        self._values_from = np.zeros((0,))
        self._values_to = np.zeros((0,))

    @property
    def values_from(self):
        """Return the loads where recorded loops start.

        Returns
        -------
        numpy.ndarray
            Start load values, typically in MPa.
        """
        return self._values_from

    @property
    def values_to(self):
        """Return the loads where recorded loops turn back.

        Returns
        -------
        numpy.ndarray
            Turn-back load values, typically in MPa.
        """
        return self._values_to

    @property
    def collective(self):
        """Return the recorded loops as an explicit load collective.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ``from`` and ``to`` containing loop turning
            loads, typically in MPa.
        """
        return pd.DataFrame({'from': self._values_from, 'to': self._values_to})

    def record_values(self, values_from, values_to):
        """Record loop turning loads.

        Parameters
        ----------
        values_from : array_like
            Load values where loops start, typically in MPa.
        values_to : array_like
            Load values where loops turn back, typically in MPa.
        """
        self._values_from = np.append(self._values_from, values_from)
        self._values_to = np.append(self._values_to, values_to)

    def histogram_numpy(self, bins=10):
        """Calculate a NumPy histogram of recorded loop loads.

        Parameters
        ----------
        bins : int, array_like or list, optional
            Bin specification passed to :func:`numpy.histogram2d`.  Default is
            ``10``.

        Returns
        -------
        H : numpy.ndarray
            Two-dimensional histogram of ``from`` and ``to`` load values.
        xedges : numpy.ndarray
            Bin edges along the ``from`` load axis.
        yedges : numpy.ndarray
            Bin edges along the ``to`` load axis.
        """
        def is_non_continous(intervals):
            lefts = intervals.left
            rights = intervals.right
            return np.any(lefts[1:] != rights[:-1])

        if isinstance(bins, pd.IntervalIndex) or isinstance(bins, pd.arrays.IntervalArray):
            if not bins.is_non_overlapping_monotonic or is_non_continous(bins):
                raise ValueError("Intervals must not overlap and must be continuous and monotonic.")
            new_bins = np.empty(len(bins) + 1)
            new_bins[:-1] = bins.left
            new_bins[-1] = bins.right[-1]
            bins = new_bins

        return np.histogram2d(self._values_from, self._values_to, bins)

    def histogram(self, bins=10):
        """Calculate a pandas histogram of recorded loop loads.

        An interval index is used to label the bins.

        Parameters
        ----------
        bins : int, array_like or list, optional
            Bin specification passed to :func:`numpy.histogram2d`.  Default is
            ``10``.

        Returns
        -------
        pandas.Series
            Histogram counts with interval index levels ``from`` and ``to``.
        """
        hist, fr, to = self.histogram_numpy(bins)
        index_fr = pd.IntervalIndex.from_breaks(fr)
        index_to = pd.IntervalIndex.from_breaks(to)

        mult_idx = pd.MultiIndex.from_product([index_fr, index_to], names=['from', 'to'])
        return pd.Series(data=hist.flatten(), index=mult_idx)


class FullRecorder(LoopValueRecorder):
    """Record rainflow loop loads and sample indices.

    This recorder extends :class:`LoopValueRecorder` with the indices of the
    samples where each loop starts and turns back.  Use it when additional
    quantities from the original time series, such as temperature or dwell
    time, must be associated with each loop.
    """

    def __init__(self):
        """Instantiate a FullRecorder."""
        super().__init__()
        self._index_from = np.array([], dtype=np.uintp)
        self._index_to = np.array([], dtype=np.uintp)

    @property
    def index_from(self):
        """Return the sample indices where recorded loops start.

        Returns
        -------
        numpy.ndarray
            Global sample indices of the loop start points.
        """
        return self._index_from

    @property
    def index_to(self):
        """Return the sample indices where recorded loops turn back.

        Returns
        -------
        numpy.ndarray
            Global sample indices of the loop turn-back points.
        """
        return self._index_to

    @property
    def collective(self):
        """Return recorded loops and indices as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ``from``, ``to``, ``index_from``, and
            ``index_to``.
        """
        return pd.DataFrame({
            'from': self._values_from,
            'to': self._values_to,
            'index_from': self._index_from,
            'index_to': self._index_to
        })

    def record_index(self, index_from, index_to):
        """Record loop sample indices.

        Parameters
        ----------
        index_from : array_like
            Sample indices where loops start.
        index_to : array_like
            Sample indices where loops turn back.
        """
        self._index_from = np.concatenate(
            (self._index_from, np.asarray(index_from, dtype=np.uintp))
        )
        self._index_to = np.concatenate(
            (self._index_to, np.asarray(index_to, dtype=np.uintp))
        )


class FKMNonlinearRecorder(AbstractRecorder):
    """Record loops for the FKM nonlinear assessment workflow.

    The recorder stores minimum and maximum load, stress, and strain values
    reported by ``FKMNonlinearDetector`` together with flags that distinguish
    closed hystereses from Memory 3 entries of the FKM nonlinear procedure.
    """

    def __init__(self):
        """Instantiate a FKMNonlinearRecorder."""
        super().__init__()
        self._results_min = pd.DataFrame(
            columns=["loads_min", "S_min", "epsilon_min", "epsilon_min_LF"],
            dtype=np.float64,
        )
        self._results_max = pd.DataFrame(
            columns=["loads_max", "S_max", "epsilon_max", "epsilon_max_LF"],
            dtype=np.float64,
        )
        self._is_closed_hysteresis = []
        self._is_zero_mean_stress_and_strain = []
        self._run_index = []

    @property
    def loads_min(self):
        """Return the minimum load values of recorded hystereses.

        Returns
        -------
        pandas.Series
            Minimum load values, typically in MPa or the unit of the input
            load history.
        """
        return self._results_min["loads_min"]

    @property
    def loads_max(self):
        """Return the maximum load values of recorded hystereses.

        Returns
        -------
        pandas.Series
            Maximum load values, typically in MPa or the unit of the input
            load history.
        """
        return self._results_max["loads_max"]

    @property
    def S_min(self):
        """Return the minimum stresses of recorded hystereses.

        Returns
        -------
        pandas.Series
            Minimum stress values in MPa.
        """
        return self._results_min["S_min"]

    @property
    def S_max(self):
        """Return the maximum stresses of recorded hystereses.

        Returns
        -------
        pandas.Series
            Maximum stress values in MPa.
        """
        return self._results_max["S_max"]

    @property
    def epsilon_min(self):
        """Return the minimum strains of recorded hystereses.

        Returns
        -------
        pandas.Series
            Minimum strain values, dimensionless.
        """
        return self._results_min["epsilon_min"]

    @property
    def epsilon_max(self):
        """Return the maximum strains of recorded hystereses.

        Returns
        -------
        pandas.Series
            Maximum strain values, dimensionless.
        """
        return self._results_max["epsilon_max"]

    @property
    def epsilon_min_LF(self):
        """Return the minimum lifetime-history strain values.

        Returns
        -------
        pandas.Series
            Minimum strain values seen in the load history up to each
            hysteresis, dimensionless.
        """
        return self._results_min["epsilon_min_LF"]

    @property
    def epsilon_max_LF(self):
        """Return the maximum lifetime-history strain values.

        Returns
        -------
        pandas.Series
            Maximum strain values seen in the load history up to each
            hysteresis, dimensionless.
        """
        return self._results_max["epsilon_max_LF"]

    @property
    def S_a(self):
        """Return stress amplitudes of recorded hystereses.

        Returns
        -------
        numpy.ndarray
            Stress amplitude in MPa.
        """
        return 0.5 * (np.array(self.S_max) - np.array(self.S_min))

    @property
    def S_m(self):
        """Return mean stresses of recorded hystereses.

        Returns
        -------
        numpy.ndarray
            Mean stress in MPa.

        Notes
        -----
        Mean stress is usually ``(S_min + S_max) / 2``.  For Memory 3
        hystereses the FKM nonlinear guideline defines ``S_m = 0``; those rows
        are indicated by ``is_zero_mean_stress_and_strain``.
        """
        median = 0.5 * (np.array(self.S_min) + np.array(self.S_max))
        return np.where(self.is_zero_mean_stress_and_strain, 0, median)

    @property
    def epsilon_a(self):
        """Return strain amplitudes of recorded hystereses.

        Returns
        -------
        numpy.ndarray
            Strain amplitude, dimensionless.
        """
        return 0.5 * (np.array(self.epsilon_max) - np.array(self.epsilon_min))

    @property
    def epsilon_m(self):
        """Return mean strains of recorded hystereses.

        Returns
        -------
        numpy.ndarray
            Mean strain, dimensionless.

        Notes
        -----
        Mean strain is usually ``(epsilon_min + epsilon_max) / 2``.  For
        Memory 3 hystereses the FKM nonlinear guideline defines
        ``epsilon_m = 0``; those rows are indicated by
        ``is_zero_mean_stress_and_strain``.
        """
        return np.where(self.is_zero_mean_stress_and_strain, \
                        0, 0.5 * (np.array(self.epsilon_min) + np.array(self.epsilon_max)))

    @property
    def is_zero_mean_stress_and_strain(self):
        """Return flags for FKM Memory 3 zero mean values.

        Returns
        -------
        list of bool or numpy.ndarray
            ``True`` for hystereses where the FKM nonlinear procedure defines
            mean stress and mean strain as zero.
        """

        # if the assessment is performed for multiple points at once
        if len(self.S_min) > 0 and len(self.S_min.index.names) > 1:
            return self._get_for_every_node(self._is_zero_mean_stress_and_strain)
        else:
            return self._is_zero_mean_stress_and_strain

    @property
    def R(self):
        """Return stress ratios ``R`` of recorded hystereses.

        Returns
        -------
        numpy.ndarray
            Stress ratio ``R = S_min / S_max``, dimensionless.

        Notes
        -----
        For Memory 3 hystereses the FKM nonlinear guideline defines
        ``R = -1``.  Those rows are indicated by
        ``is_zero_mean_stress_and_strain``.
        """
        with np.errstate(all="ignore"):
            R = np.array(self.S_min) / np.array(self.S_max)
        return np.where(self.is_zero_mean_stress_and_strain, -1, R)

    @property
    def is_closed_hysteresis(self):
        """Return whether each row is a closed hysteresis.

        Returns
        -------
        list of bool or numpy.ndarray
            ``True`` for closed hystereses and ``False`` for Memory 3 entries,
            which count only half damage in the FKM nonlinear procedure.
        """

        # if the assessment is performed for multiple points at once
        if len(self.S_min) > 0 and len(self.S_min.index.names) > 1:
            return self._get_for_every_node(self._is_closed_hysteresis)
        else:
            return self._is_closed_hysteresis

    @property
    def collective(self):
        """Return the FKM nonlinear collective as a DataFrame.

        The load values are given in the columns ``loads_min``, and ``loads_max``
        for consistency with other recoders.
        Stress and strain values for the hystereses are given in the columns
        ``S_min``, ``S_max``, and  ``epsilon_min``, ``epsilon_max``, respectively.
        The column ``is_closed_hysteresis`` indicates whether the row corresponds
        to a closed hysteresis or was recorded as a memory 3 hysteresis,
        which counts only half the damage in the FKM nonlinear procedure.
        The columns ``epsilon_min_LF`` and ``epsilon_max_LF`` describe the minimum
        and maximum seen value of epsilon in the entire load history, up to the
        current hysteresis. These values may be lower (min) or higher (max) than
        the min/max values of the previously recorded hysteresis as they also
        take into account parts of the stress-strain diagram curve that
        are not part of hystereses.

        The resulting DataFrame will have a MultiIndex with levels "hysteresis_index"
        and "assessment_point_index", both counting from 0 upwards. The nodes of a mesh
        are, thus, mapped to the index sequence 0,1,..., even if the
        node_id starts, e.g., with 1.

        Returns
        -------
        pandas.DataFrame
            Recorded FKM nonlinear collective with load, stress, strain, and
            status columns.
        """

        if len(self.S_min) > 0 and len(self.S_min.index.names) > 1:
            assessment_levels = [name for name in self.S_min.index.names if name != "load_step"]
            n_assessment_points = self.S_min.groupby(assessment_levels).first().count()
            n_hystereses = int(len(self.S_min) / n_assessment_points)

            index = pd.MultiIndex.from_product(
                [range(n_hystereses), range(n_assessment_points)],
                names=["hysteresis_index", "assessment_point_index"],
            )
        else:
            n_hystereses = len(self.S_min)
            index = pd.MultiIndex.from_product([range(n_hystereses), [0]], names=["hysteresis_index", "assessment_point_index"])

        return pd.DataFrame(
            index=index,
            data={
                "loads_min": self.loads_min.to_numpy(),
                "loads_max": self.loads_max.to_numpy(),
                "S_min": self.S_min.to_numpy(),
                "S_max": self.S_max.to_numpy(),
                "R": self.R,
                "epsilon_min": self.epsilon_min.to_numpy(),
                "epsilon_max": self.epsilon_max.to_numpy(),
                "S_a": self.S_a,
                "S_m": self.S_m,
                "epsilon_a": self.epsilon_a,
                "epsilon_m": self.epsilon_m,
                "epsilon_min_LF": self._results_min["epsilon_min_LF"].to_numpy(),
                "epsilon_max_LF": self._results_max["epsilon_max_LF"].to_numpy(),
                "is_closed_hysteresis": self.is_closed_hysteresis,
                "is_zero_mean_stress_and_strain": self.is_zero_mean_stress_and_strain,
                "run_index": np.array(self._run_index, dtype=np.int64)
            })

    def record_values_fkm_nonlinear(
        self,
        results_min,
        results_max,
        is_closed_hysteresis,
        is_zero_mean_stress_and_strain,
        run_index,
    ):
        """Record FKM nonlinear loop results.

        Parameters
        ----------
        results_min : pandas.DataFrame
            Minimum-side results with columns ``loads_min``, ``S_min``,
            ``epsilon_min``, and ``epsilon_min_LF``.
        results_max : pandas.DataFrame
            Maximum-side results with columns ``loads_max``, ``S_max``,
            ``epsilon_max``, and ``epsilon_max_LF``.
        is_closed_hysteresis : list of bool
            Flags indicating closed hystereses.
        is_zero_mean_stress_and_strain : list of bool
            Flags indicating Memory 3 hystereses with zero mean stress and
            strain according to the FKM nonlinear procedure.
        run_index : int
            Index of the detector run that produced the results.
        """

        self._results_min = results_min if len(self._results_min) == 0 else pd.concat([self._results_min, results_min])
        self._results_max = results_max if len(self._results_max) == 0 else pd.concat([self._results_max, results_max])
        self._is_closed_hysteresis += is_closed_hysteresis
        self._is_zero_mean_stress_and_strain += is_zero_mean_stress_and_strain

        self._run_index += [run_index] * len(results_min)

    def _get_for_every_node(self, boolean_array):

        # number of points, i.e., number of values for every load step
        li = self.S_min.index.to_frame()['load_step']
        m = self.S_min.groupby((li!=li.shift()).cumsum(), sort=False).count().iloc[0]
        # bring the array of boolean values to the right shape
        # numeric_array contains only 0s and 1s for False and True
        numeric_array = np.array(boolean_array).reshape(-1,1).dot(np.ones((1,m))).flatten()
        # transform the array to boolean type
        return np.where(numeric_array == 1, True, False)

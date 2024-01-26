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

__author__ = ["Johannes Mueller", "Benjamin Maier"]
__maintainer__ = __author__


import numpy as np
import pandas as pd

from .general import AbstractRecorder


class LoopValueRecorder(AbstractRecorder):
    """Rainflow recorder that collects the loop values."""

    def __init__(self):
        """Instantiate a LoopRecorder."""
        super().__init__()
        self._values_from = []
        self._values_to = []

    @property
    def values_from(self):
        """1-D float array containing the values from which the loops start."""
        return self._values_from

    @property
    def values_to(self):
        """1-D float array containing the values the loops go to before turning back."""
        return self._values_to

    @property
    def collective(self):
        """The overall collective recorded as :class:`pandas.DataFrame`.

        The columns are named ``from``, ``to``.
        """
        return pd.DataFrame({'from': self._values_from, 'to': self._values_to})

    def record_values(self, values_from, values_to):
        """Record the loop values."""
        self._values_from += values_from
        self._values_to += values_to

    def histogram_numpy(self, bins=10):
        """Calculate a histogram of the recorded values into a plain numpy.histogram2d.

        Parameters
        ----------
        bins : int or array_like or [int, int] or [array, array], optional
            The bin specification (see numpy.histogram2d)

        Returns
        -------
        H : ndarray, shape(nx, ny)
            The bi-dimensional histogram of samples (see numpy.histogram2d)
        xedges : ndarray, shape(nx+1,)
            The bin edges along the first dimension.
        yedges : ndarray, shape(ny+1,)
            The bin edges along the second dimension.
        """
        return np.histogram2d(self._values_from, self._values_to, bins)

    def histogram(self, bins=10):
        """Calculate a histogram of the recorded values into a :class:`pandas.Series`.

        An interval index is used to index the bins.

        Parameters
        ----------
        bins : int or array_like or [int, int] or [array, array], optional
            The bin specification (see numpy.histogram2d)

        Returns
        -------
        pandas.Series
            A pandas.Series using a multi interval index in order to
            index data point for a given from/to value pair.
        """
        hist, fr, to = self.histogram_numpy(bins)
        index_fr = pd.IntervalIndex.from_breaks(fr)
        index_to = pd.IntervalIndex.from_breaks(to)

        mult_idx = pd.MultiIndex.from_product([index_fr, index_to], names=['from', 'to'])
        return pd.Series(data=hist.flatten(), index=mult_idx)


class FullRecorder(LoopValueRecorder):
    """Rainflow recorder that collects the loop values and the loop index.

    Same functionality like :class:`.LoopValueRecorder` but additionally
    collects the loop index.
    """

    def __init__(self):
        """Instantiate a FullRecorder."""
        super().__init__()
        self._index_from = []
        self._index_to = []

    @property
    def index_from(self):
        """1-D int array containing the index to the samples from which the loops start."""
        return self._index_from

    @property
    def index_to(self):
        """1-D int array containing the index to the samples the loops go to before turning back."""
        return self._index_to

    @property
    def collective(self):
        """The overall collective recorded as :class:`pandas.DataFrame`.

        The columns are named ``from``, ``to``, ``index_from``, ``index_to``.
        """
        return pd.DataFrame({
            'from': self._values_from,
            'to': self._values_to,
            'index_from': self._index_from,
            'index_to': self._index_to
        })

    def record_index(self, index_from, index_to):
        """Record the index."""
        self._index_from += index_from
        self._index_to += index_to


class FKMNonlinearRecorder(AbstractRecorder):
    """Recorder that goes together with the FKMNonlinearDetector."""

    def __init__(self):
        """Instantiate a FKMNonlinearRecorder."""
        super().__init__()
        self._loads_min = []
        self._loads_max = []
        self._S_min = []
        self._S_max = []
        self._epsilon_min = []
        self._epsilon_max = []
        self._epsilon_min_LF = []
        self._epsilon_max_LF = []
        self._is_closed_hysteresis = []
        self._is_zero_mean_stress_and_strain = []
        self._run_index = []
        self._debug_output = []

    @property
    def loads_min(self):
        """1-D float array containing the start load values of the recorded hystereses."""
        return self._loads_min

    @property
    def loads_max(self):
        """1-D float array containing the end load values of the recorded hystereses,
        i.e., the values the loops go to before turning back."""
        return self._loads_max

    @property
    def S_min(self):
        """1-D float array containing the minimum stress values of the recorded hystereses."""
        return self._S_min

    @property
    def S_max(self):
        """1-D float array containing the maximum stress values of the recorded hystereses."""
        return self._S_max

    @property
    def epsilon_min(self):
        """1-D float array containing the minimum strain values of the recorded hystereses."""
        return self._epsilon_min

    @property
    def epsilon_max(self):
        """1-D float array containing the maximum strain values of the recorded hystereses."""
        return self._epsilon_max

    @property
    def S_a(self):
        """1-D numpy array containing the stress amplitudes of the recorded hystereses."""
        return 0.5 * (np.array(self._S_max) - np.array(self._S_min))

    @property
    def S_m(self):
        """1-D numpy array containing the mean stresses of the recorded hystereses,
        which are usually computed as ``(S_min + S_max) / 2``.
        Only for hystereses resulting from Memory 3, the FKM nonlinear document defines ``S_m``
        to be zero (eq. 2.9-52). This is indicated by ``_is_zero_mean_stress_and_strain=True`̀ .
        For these hystereses, this function returns 0 instead of ``(S_min + S_max) / 2``. """
        return np.where(self.is_zero_mean_stress_and_strain, \
                        0, 0.5 * (np.array(self._S_min) + np.array(self._S_max)))

    @property
    def epsilon_a(self):
        """1-D float array containing the strain amplitudes of the recorded hystereses."""
        return 0.5 * (np.array(self._epsilon_max) - np.array(self._epsilon_min))

    @property
    def epsilon_m(self):
        """1-D numpy array containing the mean strain of the recorded hystereses,
        which are usually computed as ``(epsilon_min + epsilon_max) / 2``.
        Only for hystereses resulting from Memory 3, the FKM nonlinear document defines ``epsilon_m``
        to be zero (eq. 2.9-53). This is indicated by ``_is_zero_mean_stress_and_strain=True`̀ .
        For these hystereses, this function returns 0 instead of ``(epsilon_min + epsilon_max) / 2``. """
        return np.where(self.is_zero_mean_stress_and_strain, \
                        0, 0.5 * (np.array(self._epsilon_min) + np.array(self._epsilon_max)))

    def _get_for_every_node(self, boolean_array):

        # number of points, i.e., number of values for every load step
        m = len(self._S_min[0])

        # bring the array of boolean values to the right shape
        # numeric_array contains only 0s and 1s for False and True
        numeric_array = np.array(boolean_array).reshape(-1,1).dot(np.ones((1,m)))

        # transform the array to boolean type
        return np.where(numeric_array == 1, True, False)

    @property
    def is_zero_mean_stress_and_strain(self):

        # if the assessment is performed for multiple points at once
        if len(self._S_min) > 0 and isinstance(self._S_min[0], pd.Series):
            return self._get_for_every_node(self._is_zero_mean_stress_and_strain)
        else:
            return self._is_zero_mean_stress_and_strain

    @property
    def R(self):
        """1-D numpy array containing the stress relation of the recorded hystereses,
        which are usually computed as ``S_min / S_max``.
        Only for hystereses resulting from Memory 3, the FKM nonlinear document defines ``R = -1``
        (eq. 2.9-54). This is indicated by ``_is_zero_mean_stress_and_strain=True`̀ .
        For these hystereses, this function returns -1 instead of ``S_min / S_max``, which may be different. """

        return np.where(self.is_zero_mean_stress_and_strain, \
                        -1, np.array(self._S_min) / np.array(self._S_max))

    @property
    def is_closed_hysteresis(self):
        """1-D bool array indicating whether the row corresponds to a closed hysteresis or
        was recorded as a memory 3 hysteresis, which counts only half the damage in the FKM nonlinear procedure."""

        # if the assessment is performed for multiple points at once
        if len(self._S_min) > 0 and isinstance(self._S_min[0], pd.Series):
            return self._get_for_every_node(self._is_closed_hysteresis)
        else:
            return self._is_closed_hysteresis

    @property
    def collective(self):
        """The overall collective recorded as :class:`pandas.DataFrame`.

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
        """

        # if the assessment is performed for multiple points at once
        if len(self._S_min) > 0 and isinstance(self._S_min[0], pd.Series):

            n_hystereses = self.R.shape[0]
            n_nodes = self.R.shape[1]

            index = pd.MultiIndex.from_product([range(n_hystereses), range(n_nodes)], names=["hysteresis_index", "assessment_point_index"])

            return pd.DataFrame(
                index=index,
                data={
                    "loads_min": pd.concat(self._loads_min).to_numpy().flatten(),
                    "loads_max": pd.concat(self._loads_max).to_numpy().flatten(),
                    "S_min": pd.concat(self._S_min).to_numpy().flatten(),
                    "S_max": pd.concat(self._S_max).to_numpy().flatten(),
                    "R": self.R.flatten(),
                    "epsilon_min": pd.concat(self._epsilon_min).to_numpy().flatten(),
                    "epsilon_max": pd.concat(self._epsilon_max).to_numpy().flatten(),
                    "S_a": self.S_a.flatten(),
                    "S_m": self.S_m.flatten(),
                    "epsilon_a": self.epsilon_a.flatten(),
                    "epsilon_m": self.epsilon_m.flatten(),
                    "epsilon_min_LF": pd.concat(self._epsilon_min_LF).to_numpy().flatten(),
                    "epsilon_max_LF": pd.concat(self._epsilon_max_LF).to_numpy().flatten(),
                    "is_closed_hysteresis": self.is_closed_hysteresis.flatten(),
                    "is_zero_mean_stress_and_strain": self.is_zero_mean_stress_and_strain.flatten(),
                    "run_index": np.array(self._run_index).reshape(-1,1).dot(np.ones((1,n_nodes))).flatten().astype(int),
                })

        else:
            # if the assessment is performed by a single point
            n_hystereses = len(self._S_min)
            index = pd.MultiIndex.from_product([range(n_hystereses), [0]], names=["hysteresis_index", "assessment_point_index"])

            # determine load
            loads_min = self._loads_min
            loads_max = self._loads_max

            if len(self._loads_min) == 0 or len(self._loads_max) == 0:
                loads_min = [np.nan]*n_hystereses
                loads_max = [np.nan]*n_hystereses

            return pd.DataFrame(
                index=index,
                data={
                    "loads_min": loads_min,
                    "loads_max": loads_max,
                    "S_min": self._S_min,
                    "S_max": self._S_max,
                    "R": self.R,
                    "epsilon_min": self._epsilon_min,
                    "epsilon_max": self._epsilon_max,
                    "S_a": self.S_a,
                    "S_m": self.S_m,
                    "epsilon_a": self.epsilon_a,
                    "epsilon_m": self.epsilon_m,
                    "epsilon_min_LF": self._epsilon_min_LF,
                    "epsilon_max_LF": self._epsilon_max_LF,
                    "is_closed_hysteresis": self._is_closed_hysteresis,
                    "is_zero_mean_stress_and_strain": self._is_zero_mean_stress_and_strain,
                    "run_index": self._run_index,
                    "debug_output": self._debug_output
            })

    def record_values_fkm_nonlinear(self, loads_min, loads_max, S_min, S_max, epsilon_min, epsilon_max,
                      epsilon_min_LF, epsilon_max_LF,
                      is_closed_hysteresis, is_zero_mean_stress_and_strain, run_index, debug_output):
        """Record the loop values."""

        if loads_min:
            self._loads_min += loads_min
        if loads_max:
            self._loads_max += loads_max
        self._S_min += S_min
        self._S_max += S_max
        self._epsilon_min += epsilon_min
        self._epsilon_max += epsilon_max
        self._epsilon_min_LF += epsilon_min_LF
        self._epsilon_max_LF += epsilon_max_LF
        self._is_closed_hysteresis += is_closed_hysteresis
        self._is_zero_mean_stress_and_strain += is_zero_mean_stress_and_strain

        if len(debug_output) == 0:
            debug_output = [""] * len(S_min)
        self._debug_output += debug_output

        self._run_index += [run_index] * len(S_min)

    def _get_for_every_node(self, boolean_array):

        # number of points, i.e., number of values for every load step
        m = len(self._S_min[0])

        # bring the array of boolean values to the right shape
        # numeric_array contains only 0s and 1s for False and True
        numeric_array = np.array(boolean_array).reshape(-1,1).dot(np.ones((1,m)))

        # transform the array to boolean type
        return np.where(numeric_array == 1, True, False)

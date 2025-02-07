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

__author__ = "Benjamin Maier"
__maintainer__ = __author__

import numpy as np
import pandas as pd

import pylife.stress.rainflow.general as RFG
import pylife.materiallaws.notch_approximation_law as NAL

INDEX = 0
LOAD_TYPE = 1

IS_CLOSED = 0
FROM = 1
TO = 2
CLOSE = 3

LOAD = 0
STRESS = 1
STRAIN = 2
EPS_MIN_LF = 3
EPS_MAX_LF = 4
STRESS_AND_STRAIN = slice(STRESS, STRAIN+1)

PRIMARY = 0
SECONDARY = 1

MEMORY_1_2 = 1
MEMORY_3 = 0

HISTORY_COLUMNS = ["load", "stress", "strain", "secondary_branch"]
HISTORY_INDEX_LEVELS = [
    "load_segment", "load_step", "run_index", "turning_point", "hyst_from", "hyst_to", "hyst_close"
]


class _ResidualsRecord:

    def __init__(self):
        self._index = []
        self._values = []

    def append(self, idx, val):
        self._index.append(idx)
        self._values.append(val)

    def pop(self):
        return self._index.pop(), self._values.pop()

    @property
    def index(self):
        return np.array(self._index, dtype=np.int64)

    @property
    def current_index(self):
        return self._index[-1]

    def reindex(self):
        self._index = list(range(-len(self._values), 0))

    def will_remain_open_by(self, load):
        current_load_extent = np.abs(load - self._values[-1])
        previous_load_extent = np.abs(self._values[-1] - self._values[-2])
        return current_load_extent < previous_load_extent

    def __len__(self):
        return len(self._index)


class FKMNonlinearDetector(RFG.AbstractDetector):
    """HCM-Algorithm detector as described in FKM nonlinear.

    """

    def __init__(self, recorder, notch_approximation_law, binner=NAL.NotchApproxBinner):
        super().__init__(recorder)

        if binner is not None:
            self._binner = binner(notch_approximation_law)
            self._notch_approximation_law = self._binner
        else:
            self._binner = None
            self._notch_approximation_law = notch_approximation_law

        if notch_approximation_law is not None:
            self._ramberg_osgood_relation = self._notch_approximation_law.ramberg_osgood_relation

        # state of the hcm algorithm
        self._iz = 0                # indicator how many open hystereses there are
        self._ir = 1                # indicator how many of the open hystereses start at the primary branch (and, thus, cannot be closed)
        self._load_max_seen = 0.0    # maximum seen load value
        self._run_index = 0     # which run through the load sequence is currently performed

        self._last_deformation_record = None
        self._residuals_record = _ResidualsRecord()
        self._residuals = np.array([])
        self._record_vals_residuals = pd.DataFrame()

        self._history_record = []

        self._num_turning_points = 0

    def process_hcm_first(self, samples):
        """Perform the HCM algorithm for the first time.
        This processes the given samples accordingly, only considering
        "turning points", neglecting consecutive duplicate values,
        making sure that the beginning starts with 0.

        Parameters
        ----------
        samples : list of floats or list of pd.DataFrame`s
            The samples to be processed by the HCM algorithm.
        """
        assert len(samples) >= 2
        samples, flush = self._adjust_samples_and_flush_for_hcm_first_run(samples)

        return self.process(samples, flush=flush)

    def process_hcm_second(self, samples):
        """Perform the HCM algorithm for the second time,
        after it has been executed with ``process_hcm_first``.
        This processes the given samples accordingly, only considering
        "turning points", neglecting consecutive duplicate values,
        making sure that the beginning of the sequence is properly fitted to
        the samples of the first run, such that no samples are lost
        and no non-turning points are introducted between the two runs.

        Parameters
        ----------
        samples : list of floats or list of pd.DataFrame`s
            The samples to be processed by the HCM algorithm.
        """
        assert len(samples) >= 2
        return self.process(samples, flush=True)

    def process(self, samples, flush=False):
        """Process a sample chunk. This method implements the actual HCM algorithm.

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
                -> processed sequence is [1,2,3,1], "2" is not a turning point

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
        self : FKMNonlinearDetector
            The ``self`` object so that processing can be chained
        """

        # collected values, which will be passed to the recorder at the end of `process()`
        assert not isinstance(samples, pd.DataFrame)
        self._run_index += 1

        load_turning_points = self._determine_load_turning_points(samples, flush)

        self._current_load_index = load_turning_points.index

        li = load_turning_points.index.to_frame()['load_step']
        turning_point_idx = pd.Index((li != li.shift()).cumsum() - 1, name="turning_point")

        load_turning_points_rep = np.asarray(
            load_turning_points.groupby(turning_point_idx, sort=False).first()
        )

        deform_type_record, hysts = self._perform_hcm_algorithm(load_turning_points_rep)

        if self._last_deformation_record is None:
            self._last_deformation_record = np.zeros((5, self._group_size))

        num_turning_points = len(load_turning_points_rep)
        record_vals = self._process_deformation(load_turning_points, num_turning_points, deform_type_record)

        self._store_recordings_for_history(deform_type_record, record_vals, turning_point_idx, hysts)

        results = self._process_hysteresis(record_vals, hysts)
        results_min, results_max = results

        self._update_residuals(record_vals, turning_point_idx, load_turning_points_rep)

        self._num_turning_points += num_turning_points

        # TODO: check if these are really that redundant
        is_closed_hysteresis = (hysts[:, 0] != MEMORY_3).tolist()
        is_zero_mean_stress_and_strain = (hysts[:, 0] == MEMORY_3).tolist()

        self._recorder.record_values_fkm_nonlinear(
            results_min, results_max,
            is_closed_hysteresis=is_closed_hysteresis,
            is_zero_mean_stress_and_strain=is_zero_mean_stress_and_strain,
            run_index=self._run_index
        )

        return self

    def _determine_load_turning_points(self, samples, flush):
        old_head_index = self._head_index

        have_multi_index = isinstance(samples, pd.Series) and len(samples.index.names) > 1

        if have_multi_index:
            rep_samples = samples.groupby('load_step', sort=False).first().to_numpy()
        else:
            rep_samples = np.asarray(samples)

        if self._binner is not None:
            if have_multi_index:
                load_max_idx = samples.groupby("load_step").first().abs().idxmax()
                load_max = samples.xs(load_max_idx, level="load_step").abs()
            else:
                load_max = np.abs(samples).max()
            self._binner.initialize(load_max)

        loads_indices, load_turning_points = self._new_turns(rep_samples, flush)

        self._group_size = len(samples) // len(rep_samples)

        if have_multi_index:
            load_steps = samples.index.get_level_values('load_step').unique()
            if len(loads_indices) > 0:
                turns_idx = loads_indices - old_head_index
                idx = load_steps[turns_idx]
                load_turning_points = samples.loc[idx]
                if turns_idx[0] < 0:
                    load_turning_points.iloc[:self._group_size] = self._last_sample
            else:
                load_turning_points = pd.Series(
                    [], index=pd.MultiIndex.from_tuples([], names=samples.index.names)
                )
            idx = load_steps[-1]
            self._last_sample = samples.loc[idx]

        if isinstance(load_turning_points, pd.Series):
            return load_turning_points

        return pd.Series(
            load_turning_points, index=pd.Index(loads_indices, name="load_step")
        )

    def _store_recordings_for_history(self, record, record_vals, turning_point, hysts):
        record_repr = (
            record_vals.reset_index(["load_step", "turning_point"])
            .groupby(turning_point)
            .first()
            .drop(["epsilon_min_LF", "epsilon_max_LF"], axis=1)
        )

        record_repr["run_index"] = self._run_index
        record_repr["secondary_branch"] = record[:, SECONDARY] != 0

        rec_hysts = hysts.copy()
        rec_hysts[:, 1:] += self._num_turning_points

        self._history_record.append((record_repr, rec_hysts))

    def _update_residuals(self, record_vals, turning_point, load_turning_points_rep):
        residuals_index = self._residuals_record.index
        old_residuals_index = residuals_index[residuals_index < 0]
        new_residuals_index = residuals_index[residuals_index >= 0] + self._num_turning_points

        remaining_vals_residuals = self._record_vals_residuals.loc[
            self._record_vals_residuals.index[old_residuals_index]
        ]

        new_vals_residuals = record_vals.loc[
                record_vals.index.isin(new_residuals_index, level="turning_point")
            ]

        self._record_vals_residuals = pd.concat([remaining_vals_residuals, new_vals_residuals])
        self._record_vals_residuals.index.names = record_vals.index.names

        self._residuals = (
            load_turning_points_rep[residuals_index] if len(residuals_index) else np.array([])
        )

        self._residuals_record.reindex()

    def _process_deformation(self, load_turning_points, num_turning_points, deform_type_record):
        """Calculate the local stress and strain of all turning points

        In ._perform_hcm_algorithm we recorded which turning point is in
        PRIMARY deformation regime and which in SECONDARY

        Now we use this information to calculate the local stress and strain
        according to the notch approximation law for each turining point for
        each point in the mesh.

        Parameters
        ----------
        load_turning_points : pd.Series (N * self._group_size) float
            The load distribution of all the turning points. It carries all th
            index levels of the initial load signal.

        num_turning_points : int
            The number of tunring_points

        deform_type_record : np.ndarray(N, 2) int
            The inndex and deformation type of each turning point
            (see _perform_hcm_algorithm)

        Returns pd.DataFrame
            columns: ["load", "stress", "strain", "epsilon_min_LF", "epsilon_max_LF"]
            index: the same like load_turning_points
        """
        def primary(_prev, load):
            return self._notch_approximation_law.primary(load)

        def secondary(prev, load):
            prev_load = prev[LOAD]

            delta_L = load - prev_load
            delta_stress_strain = self._notch_approximation_law.secondary(delta_L)

            return prev[STRESS_AND_STRAIN].T + delta_stress_strain

        def prev_record_from_residuals(prev_idx):
            idx = len(self._record_vals_residuals) + prev_idx*self._group_size
            return self._record_vals_residuals.iloc[idx:idx+self._group_size].to_numpy().T

        def prev_record_from_this_run(prev_idx):
            idx = prev_idx * self._group_size
            return record_vals[:, idx:idx+self._group_size]

        def determine_prev_record(prev_idx):
            if prev_idx < 0:
                return prev_record_from_residuals(prev_idx)
            if prev_idx == curr_idx:
                return self._last_deformation_record
            return prev_record_from_this_run(prev_idx)

        record_vals = np.empty((5, num_turning_points*self._group_size))

        turning_points = load_turning_points.to_numpy()

        for curr_idx in range(num_turning_points):
            prev_record = determine_prev_record(deform_type_record[curr_idx, INDEX])

            idx = curr_idx * self._group_size
            load = turning_points[idx:idx+self._group_size]

            deformation_function = secondary if deform_type_record[curr_idx, SECONDARY] else primary

            result_buf = record_vals[:, idx:idx+self._group_size]

            result_buf[LOAD] = load
            result_buf[STRESS_AND_STRAIN] = deformation_function(prev_record, load).T

            self._calculate_epsilon_LF(result_buf)
            self._last_deformation_record = result_buf

        record_vals = pd.DataFrame(
            record_vals.T,
            columns=["load", "stress", "strain", "epsilon_min_LF", "epsilon_max_LF"],
            index=load_turning_points.index,
        )

        new_sum_tp = self._num_turning_points + num_turning_points
        tp_index = [np.arange(self._num_turning_points, new_sum_tp)] * self._group_size

        record_vals["turning_point"] = np.stack(tp_index).T.flatten()
        return record_vals.set_index("turning_point", drop=True, append=True)

    def _calculate_epsilon_LF(self, deformation_record):
        """Calculate epsilon_LF values for the current deformation record

        Parameters
        ----------
        deformation_record : np.ndarray (5)
            [:3] load, stress, strain
            [3:] reserved for epsilon_min_LF and epsilon_max_LF
        """

        old_load = self._last_deformation_record[LOAD, 0]
        current_load = deformation_record[LOAD, 0]

        if old_load < current_load:
            deformation_record[EPS_MAX_LF] = (
                self._last_deformation_record[EPS_MAX_LF]
                if self._last_deformation_record[EPS_MAX_LF, 0] > deformation_record[STRAIN, 0]
                else deformation_record[STRAIN, :]
            )
            deformation_record[EPS_MIN_LF] = self._last_deformation_record[EPS_MIN_LF]
        else:
            deformation_record[EPS_MIN_LF] = (
                self._last_deformation_record[EPS_MIN_LF]
                if self._last_deformation_record[EPS_MIN_LF, 0] < deformation_record[STRAIN, 0]
                else deformation_record[STRAIN, :]
            )
            deformation_record[EPS_MAX_LF] = self._last_deformation_record[EPS_MAX_LF]


    def _process_hysteresis(self, record_vals, hysts):
        """Calcuclate all the recorded hysteresis values

        For each hysteresis we calculate the two records consisting of
        load, stress, strain, epsilon_min_LF, epsilon_max_LF

        Parameters
        ----------
        record_vals: pd.DataFrame
            colimns: load, stress, strain, epsilon_min_LF, epsilon_max_LF
            index of load_turning_points

        hysts: np.ndarray(N, 2)
            the recorded hysteresis information
            (see ._perform_hcm_algorithm)

        Returns
        -------
        result_min, result_max: pd.DataFrame
        """
        def turn_memory_1_2(values, index):
            if values[0][0, 0] < values[1][0, 0]:
                return (values[0], values[1], index[0], index[1])
            return (values[1], values[0], index[1], index[0])

        def turn_memory_3(values, index):
            abs_point = np.abs(values[0])
            point_min = -abs_point
            point_max = abs_point
            point_min[:, EPS_MIN_LF:] = values[1][:, EPS_MIN_LF:]
            point_max[:, EPS_MIN_LF:] = values[1][:, EPS_MIN_LF:]
            return (point_min, point_max, index[0], index[0])

        memory_functions = [turn_memory_3, turn_memory_1_2]

        start = len(self._residuals)

        record_vals_with_residuals = pd.concat([self._record_vals_residuals, record_vals])

        value_array = record_vals_with_residuals.to_numpy()

        index_array = np.asarray(
            record_vals_with_residuals.index.droplevel("turning_point").to_frame()
        )

        signal_index_names = self._current_load_index.names
        signal_index_num = len(signal_index_names)

        result_len = len(hysts) * self._group_size

        results_min = np.zeros((result_len, 4))
        results_min_idx = np.zeros((result_len, signal_index_num), dtype=np.int64)

        results_max = np.zeros((result_len, 4))
        results_max_idx = np.zeros((result_len, signal_index_num), dtype=np.int64)

        for i, hyst in enumerate(hysts):
            idx = (hyst[FROM:CLOSE] + start) * self._group_size

            hyst_type = hyst[IS_CLOSED]

            beg0, beg1 = idx[0], idx[1]
            vbeg1 = beg1 - self._group_size if hyst_type == MEMORY_3 else beg1

            end0, end1 = beg0 + self._group_size, beg1 + self._group_size
            vend1 = vbeg1 + self._group_size

            values = value_array[beg0:end0], value_array[vbeg1:vend1]
            index = index_array[beg0:end0], index_array[beg1:end1]

            min_val, max_val, min_idx, max_idx = memory_functions[hyst_type](values, index)

            beg = i * self._group_size
            end = beg + self._group_size

            results_min[beg:end, :3] = min_val[:, :3]
            results_max[beg:end, :3] = max_val[:, :3]

            results_min_idx[beg:end] = min_idx
            results_max_idx[beg:end] = max_idx

            results_min[beg:end, -1] = min_val[:, EPS_MIN_LF]
            results_max[beg:end, -1] = max_val[:, EPS_MAX_LF]

        results_min = pd.DataFrame(
            results_min,
            columns=["loads_min", "S_min", "epsilon_min", "epsilon_min_LF"],
            index=pd.MultiIndex.from_arrays(results_min_idx.T, names=signal_index_names)
        )
        results_max = pd.DataFrame(
            results_max,
            columns=["loads_max", "S_max", "epsilon_max", "epsilon_max_LF"],
            index=pd.MultiIndex.from_arrays(results_max_idx.T, names=signal_index_names)
        )

        return results_min, results_max

    def _adjust_samples_and_flush_for_hcm_first_run(self, samples):

        is_multi_index = isinstance(samples, pd.Series) and len(samples.index.names) > 1

        if not is_multi_index:
            samples = np.concatenate([[0], np.asarray(samples)])
        else:
            assessment_levels = [name for name in samples.index.names if name != "load_step"]
            assessment_idx = samples.groupby(assessment_levels).first().index

            # create a new sample with 0 load for all nodes
            multi_index = pd.MultiIndex.from_product([[0], assessment_idx], names=samples.index.names)
            first_sample = pd.Series(0, index=multi_index)

            # increase the load_step index value by one for all samples
            samples_without_index = samples.reset_index()
            samples_without_index.load_step += 1
            samples = samples_without_index.set_index(samples.index.names)[0]

            # prepend the new zero load sample
            samples = pd.concat([first_sample, samples], axis=0)

        # determine first, second and last samples
        scalar_samples = samples

        if is_multi_index:
            # convert to list
            scalar_samples = samples.groupby("load_step", sort=False).first()

        scalar_samples_twice = np.concatenate([scalar_samples, scalar_samples])
        turn_indices, _ = RFG.find_turns(scalar_samples_twice)

        flush = len(scalar_samples)-1 in turn_indices

        return samples, flush

    def _perform_hcm_algorithm(self, load_turning_points):
        """Perform the entire HCM algorithm for all load samples

        Parameters
        ----------
            load_turning_points : np.ndarray
                The representative tunring points of the load signal

        Returns
        -------
        deform_type_record : np.ndarray (N,2) integer
            first column: index in the turning point array
            second column: indicate whether the deformation is in PRIMARY or SECONDARY regime

        hysts : np.ndarray (N,4) integer
            first column: Type of hysteresis memort (MEMORY_1_2 or MEMORY_3)
            second column: index in turning point array of the hysteresis origin
            third column: index in turning point array of the hysteresis front
            fourth column: index in turning point array of the hysteresis
               closing point (-1 if hysteresis not closed)
        """

        hysts = np.zeros((len(load_turning_points), 4), dtype=np.int64)
        hyst_ptr = 0

        deform_type_record = -np.ones((len(load_turning_points), 2), dtype=np.int64)
        deform_ptr = 0

        for index, current_load in enumerate(load_turning_points):
            hyst_ptr = self._hcm_process_sample(
                current_load, index, hysts, hyst_ptr, deform_type_record[deform_ptr, :]
            )

            if np.abs(current_load) > self._load_max_seen:
                self._load_max_seen = np.abs(current_load)

            self._iz += 1

            self._residuals_record.append(deform_ptr, current_load)

            deform_ptr += 1

        hysts = hysts[:hyst_ptr, :]
        return deform_type_record, hysts

    def _hcm_process_sample(
        self,
        current_load,
        current_idx,
        hysts,
        hyst_ptr,
        deform_type_record,
    ):
        """Process one sample in the HCM algorithm, i.e., one load value

        Parameters
        ----------
        current_load: float
            The current representative load

        current_index: int
            The index of the current load in the turning point list

        hysts: np.ndarray (N,4) integer
            The hysteresis record (see ._perform_hcm_algorithm)

        hyst_ptr: int
            The pointer to the next hysteresis record

        deform_type_record : np.ndarray (N,2) integer
            The deformation type record (see ._perform_hcm_algorithm)
        """

        record_idx = current_idx

        while True:
            if self._iz == self._ir:

                if np.abs(current_load) > self._load_max_seen:  # case a) i, "Memory 3"
                    deform_type_record[:] = [record_idx, PRIMARY]

                    residuals_idx = self._residuals_record.current_index
                    hysts[hyst_ptr, :] = [MEMORY_3, residuals_idx, current_idx, -1]
                    hyst_ptr += 1

                    self._ir += 1

                else:
                    deform_type_record[:] = [record_idx, SECONDARY]
                break

            if self._iz < self._ir:
                deform_type_record[:] = [record_idx, PRIMARY]
                break

            # here we have iz > ir:

            if self._residuals_record.will_remain_open_by(current_load):
                deform_type_record[:] = [record_idx, SECONDARY]
                break

            # no -> we have a new hysteresis

            prev_idx_1, prev_load_1 = self._residuals_record.pop()
            prev_idx_0, prev_load_0 = self._residuals_record.pop()

            if len(self._residuals_record):
                record_idx = self._residuals_record.current_index

            self._iz -= 2

            # if the points of the hysteresis lie fully inside the seen range of loads, i.e.,
            # the turn points are smaller than the maximum turn point so far
            # (this happens usually in the second run of the HCM algorithm)
            if np.abs(prev_load_0) < self._load_max_seen and np.abs(prev_load_1) < self._load_max_seen:
                # case "Memory 2", "c) ii B"
                # the primary branch is not yet reached, continue processing residual loads, potentially
                # closing even more hysteresis

                hysts[hyst_ptr, :] = [MEMORY_1_2, prev_idx_0, prev_idx_1, current_idx]
                hyst_ptr += 1

                continue

            # case "Memory 1", "c) ii A"
            # The last hysteresis saw load values equal to the previous maximum.
            # (Higher values cannot happen here.)
            # The load curve follows again the primary path.
            # No further hystereses can be closed, continue with the next load value.
            # Previously, iz was decreased by 2 and will be increased by 1 at the end of this loop ,
            # effectively `iz = iz - 1` as described on p.70

            # Proceed on primary path for the rest, which was not part of the closed hysteresis

            deform_type_record[:] = [record_idx, PRIMARY]
            hysts[hyst_ptr, :] = [MEMORY_1_2, prev_idx_0, prev_idx_1, current_idx]
            hyst_ptr += 1

        return hyst_ptr

    @property
    def strain_values(self):
        """
        Get the strain values of the turning points in the stress-strain diagram.
        They are needed in the FKM nonlinear roughness & surface layer algorithm, which adds residual stresses in another pass of the HCM algorithm.

        Returns
        -------
        list of float
           The strain values of the turning points that are visited during the HCM algorithm.
        """
        return self.history().query("load_step >= 0").strain.to_numpy()

    @property
    def strain_values_first_run(self):
        """
        Get the strain values of the turning points in the stress-strain diagram, for the first run of the HCM algorithm.
        They are needed in the FKM nonlinear roughness & surface layer algorithm, which adds residual stresses in another pass of the HCM algorithm.

        Returns
        -------
        list of float
           The strain values of the turning points that are visited during the first run of the HCM algorithm.
        """

        return self.history().query("load_step >= 0 and run_index == 1").strain.to_numpy()

    @property
    def strain_values_second_run(self):
        """
        Get the strain values of the turning points in the stress-strain diagram, for the second and any further run of the HCM algorithm.
        They are needed in the FKM nonlinear roughness & surface layer algorithm, which adds residual stresses in another pass of the HCM algorithm.

        Returns
        -------
        list of float
           The strain values of the turning points that are visited during the second run of the HCM algorithm.
        """

        return self.history().query("load_step >= 0 and run_index == 2").strain.to_numpy()

    def history(self):
        """Compile the history of noteworthy points.

        Returns
        -------

        history : pd.DataFrame
            The history containing of
            ``load``, ``stress``, ``strain`` and ``secondary_branch``.
            The ``secondary_branch`` column is ``bool`` and indicates if the point
            is on secondary load branch.

            The index consists of the following levels:
                * ``load_segment``: the number of the point
                * ``load_step``: the index of the point in the actual samples
                * ``run_index``: the index of the run (usually 1 or 2)
                * ``turning_point``: the number of the turning point (-1 if it is not a turning point)
                * ``hyst_from``: the number of the hysteresis starting at the point (-1 if there isn't one)
                * ``hyst_to``: the number of the hysteresis opened at the point (-1 if there isn't one)
                * ``hyst_close``: the number hof the hysteresis closed at the point (-1 if there isn't one)

        Notes
        -----

        The history contains all the turning points with two other kinds of points injected:
          * The primary hysteresis opening (Memory 3 of the guidline)
          * The closing points of a hysteresis

        Note that the ``load_step`` index of the injected points is always `-1`, so you
        can't use it to determine the index of a hysteresis closing in the original
        signal.

        """
        history = pd.concat([rr for rr, _ in self._history_record]).reset_index(
            drop=True
        )
        history["load_segment"] = np.arange(1, len(history) + 1)

        hysts = np.concatenate([hs for _, hs in self._history_record])
        hyst_index = np.concatenate(
            [[np.arange(len(hysts))], hysts[:, FROM:CLOSE].T, [hysts[:, IS_CLOSED].T]]
        ).T

        hyst_from_marker = pd.Series(-1, index=history.index)
        hyst_to_marker = pd.Series(-1, index=history.index)

        if len(hysts):
            hyst_from_marker.iloc[hyst_index[:, FROM]] = hyst_index[:, IS_CLOSED]
            hyst_to_marker.iloc[hyst_index[:, TO]] = hyst_index[:, IS_CLOSED]

        history["hyst_from"] = hyst_from_marker
        history["hyst_to"] = hyst_to_marker
        history["hyst_close"] = pd.Series(-1, index=history.index)

        to_insert = []
        negate = []
        turning_point_drop_idx = []
        hyst_close_index = []

        for hyst_index, hyst in enumerate(hysts):
            if hyst[IS_CLOSED] == MEMORY_1_2:
                hyst_close = int(hyst[CLOSE]) + len(to_insert)
                hyst_from = int(hyst[FROM])
                turning_point_drop_idx.append(hyst_close)
                hyst_close_index.append([hyst_close, hyst_index])
                to_insert.append((hyst_close, hyst_from))
            else:
                hyst_from = int(hyst[FROM])
                hyst_to = int(hyst[TO]) + len(to_insert)
                negate.append(hyst_to)
                turning_point_drop_idx.append(hyst_to)
                to_insert.append((hyst_to, hyst_from))

        hyst_close_index = np.array(hyst_close_index, dtype=np.int64)

        negate = np.array(negate, dtype=np.int64)

        index = list(np.arange(len(history)))

        for target, idx in to_insert:
            index.insert(target, int(idx))

        history = history.iloc[index].reset_index(drop=True)

        history.loc[
            turning_point_drop_idx,
            ["turning_point", "load_step", "hyst_from", "hyst_to"],
        ] = -1
        history.loc[turning_point_drop_idx, "secondary_branch"] = True
        history.loc[negate, HISTORY_COLUMNS] = -history.loc[
            negate, HISTORY_COLUMNS
        ]
        history.loc[negate, "hyst_to"] = history.loc[negate + 1, "hyst_to"].to_numpy()
        history.loc[negate + 1, "hyst_to"] = -1
        history.loc[negate, "secondary_branch"] = True

        if len(hyst_close_index):
            history.loc[hyst_close_index[:, 0], "hyst_close"] = hyst_close_index[:, 1]

        history["load_segment"] = np.arange(len(history), dtype=np.int64)

        history.set_index(HISTORY_INDEX_LEVELS, inplace=True)

        return history

    def interpolated_stress_strain_data(
            self,
            *,
            load_segment=None,
            hysteresis_index=None,
            n_points_per_branch=100
    ):
        """Caclulate interpolated stress and strain data.

        Parameters
        ----------
        load_segment : int, Optional
            The number of the load segment for which the stress strain data is to be
            interpolated.
        hysteresis_index : int, Optional
            The number of the hysteresis for which the stress strain data is to be
            interpolated.
        n_points_per_branch : int, Optional
            The number of points to be interpolated to of each load segment

        Returns
        -------
        stress_strain_data : pd.DataFrame
            The resulting ``DataFrame`` will contain the following columns:

              * ``stress``, ``strain`` – the stress strain data
              * ``secondary_branch``– a ``bool`` column indicating if the point is
                on a secondary load branch
              * ``hyst_index`` – the number of the hysteresis the load segment is part of (-1 if  there isn't one)
              * ``load_segment`` the number of the load segment
              * ``run_index`` the number of the run

        """
        history = self.history()

        if hysteresis_index is not None:

            hyst_to = history.query(f"hyst_to == {hysteresis_index}")
            if hysteresis_index in history.index.get_level_values("hyst_close"):
                hyst_close = history.query(f"hyst_close == {hysteresis_index}")
                load_segment_close = hyst_close.index.get_level_values("load_segment")[0]
            else:
                load_segment_close = None

            load_segment_to = hyst_to.index.get_level_values("load_segment")[0]

            segments = [
                self._interpolate_deformation(load_segment_to, n_points_per_branch)
            ]
            if load_segment_close is not None:
                segments.append(
                    self._interpolate_deformation(
                        load_segment_close, n_points_per_branch
                    )
                )

            result = pd.concat(segments).reset_index(drop=True)
            result["hyst_index"] = hysteresis_index

            return result

        if load_segment is not None:
            return self._interpolate_deformation(load_segment, n_points_per_branch)

        return (
            pd.concat(
                [
                    self._interpolate_deformation(
                        row.load_segment, n_points_per_branch
                    )
                    for _, row in history.reset_index().iterrows()
                ]
            )
            .reset_index(drop=True)
        )


    def _interpolate_deformation(self, load_segment, n_points):
        history = self.history()
        idx = history.index.get_level_values("load_segment").get_loc(load_segment)

        to_value = history.iloc[idx]

        run_index = history.index.get_level_values("run_index")[idx]

        hyst_open_idx = history.index.get_level_values("hyst_to")[idx]
        hyst_close_idx = history.index.get_level_values("hyst_close")[idx]

        hyst_index = hyst_open_idx if hyst_open_idx >= 0 else hyst_close_idx

        if idx == 0:
            from_value = pd.Series({"stress": 0.0})
        elif hyst_close_idx >= 0:
            from_value = history.query(f"hyst_to == {hyst_close_idx}").iloc[0]
        elif hyst_open_idx >= 0:
            from_value = history.query(f"hyst_from == {hyst_open_idx}").iloc[0]
        else:
            from_value = history.iloc[idx-1]

        stress = np.linspace(from_value.stress, to_value.stress, n_points)

        if to_value.secondary_branch:
            delta_stress = from_value.stress - stress
            strain = from_value.strain - self._ramberg_osgood_relation.delta_strain(delta_stress)
        else:
            strain = self._ramberg_osgood_relation.strain(stress)

        return pd.DataFrame(
            {
                "stress": stress,
                "strain": strain,
                "secondary_branch": to_value.secondary_branch,
                "hyst_index": hyst_index,
                "load_segment": load_segment,
                "run_index": run_index,
            }
        )

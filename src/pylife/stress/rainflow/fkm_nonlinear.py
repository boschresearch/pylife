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

IS_CLOSED = 0
FROM = 1
TO = 2
CLOSE = 3

LOAD = 0
STRESS = 1
STRAIN = 2
EPS_MIN_LF = 3
EPS_MAX_LF = 4

PRIMARY = 0
SECONDARY = 1

MEMORY_1_2 = 1
MEMORY_3 = 0

PHYSICAL_HIST_COLUMNS = ["load", "stress", "strain", "secondary_branch"]
INDEX_HIST_LEVELS = [
    "load_segment", "load_step", "run_index", "hyst_from", "hyst_to", "hyst_close"
]


class FKMNonlinearDetector(RFG.AbstractDetector):
    """HCM-Algorithm detector as described in FKM nonlinear.

    """

    def __init__(self, recorder, notch_approximation_law):
        super().__init__(recorder)
        self._notch_approximation_law = notch_approximation_law

        if notch_approximation_law is not None:
            self._ramberg_osgood_relation = self._notch_approximation_law.ramberg_osgood_relation

        # state of the hcm algorithm
        self._iz = 0                # indicator how many open hystereses there are
        self._ir = 1                # indicator how many of the open hystereses start at the primary branch (and, thus, cannot be closed)
        self._load_max_seen = 0.0    # maximum seen load value
        self._run_index = 0     # which run through the load sequence is currently performed

        # whether the load sequence starts and the first value should be considered (normally, only "turns" in the sequence get extracted, which would omit the first value)
        self._is_load_sequence_start = True

        self._last_record = None
        self._residuals_array = []
        self._residuals_ndarray = np.array([])
        self._record_vals_residuals = pd.DataFrame()

        self._history_record = []

        self._num_turning_points = 0

        self._last_sample = pd.DataFrame()

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
        multi_index = isinstance(samples, pd.Series) and len(samples.index.names) > 1

        self._run_index += 1

        old_head_index = self._head_index

        if multi_index:
            _samples = samples.groupby('load_step', sort=False).first().to_numpy().flatten()
        else:
            _samples = np.asarray(samples)

        # get the turning points
        loads_indices, load_turning_points = self._new_turns(_samples, flush)

        self._group_size = 1

        if multi_index:
            load_steps = samples.index.get_level_values('load_step').unique().to_series()
            self._group_size = len(samples) // len(load_steps)
            if len(loads_indices) > 0:
                turns_idx = loads_indices - old_head_index
                idx = load_steps.iloc[turns_idx]
                vals = samples.loc[idx]
                if turns_idx[0] < 0:
                    vals.iloc[:len(self._last_sample)] = self._last_sample

                load_turning_points = vals

            else:
                load_turning_points = []
            idx = load_steps.iloc[-1]
            self._last_sample = samples.loc[idx]

        if not isinstance(load_turning_points, pd.Series):
            load_turning_points = pd.Series(load_turning_points)
            load_turning_points.index.name = 'load_step'

        self._current_load_index = load_turning_points.index

        li = load_turning_points.index.to_frame()['load_step']
        load_step = (li != li.shift()).cumsum() - 1

        turning_groups = load_turning_points.groupby(load_step, sort=False)

        load_turning_points_rep = np.array([
            self._get_scalar_current_load(current_load)
            for _, current_load in turning_groups
        ])

        record, hysts = self._perform_hcm_algorithm(load_turning_points_rep)

        if self._last_record is None:
            self._last_record = np.zeros((5, self._group_size))

        record_vals = self._collect_record(load_turning_points, turning_groups, load_step, record)

        self._store_recordings_for_history(record, record_vals, load_step, hysts)

        self._num_turning_points += (len(load_turning_points))

        results = self._process_recording(load_turning_points_rep, record_vals, hysts)
        results_min, results_max, epsilon_min_LF, epsilon_max_LF = results

        residuals_index = np.array([i for i, _ in self._residuals_array])

        remaining_vals_residuals = (
            self._record_vals_residuals.loc[
                self._record_vals_residuals.index[residuals_index[residuals_index < 0]]
            ]
            if len(residuals_index) > 0
            else pd.DataFrame()
        )

        new_residuals_index = residuals_index[residuals_index >= 0]

        # TODO: check if this can be simplified once `load_step` will be internalized

        index_series = record_vals.index.get_level_values("load_step").to_series()
        indexer = pd.DataFrame(
            {
                "load_step": index_series.values,
                "load_num": load_step,
            },
            index=load_step.index
        ).groupby("load_num").first()["load_step"].values

        new_vals_residuals = (
            record_vals.loc[
                record_vals.index.isin(
                    indexer[new_residuals_index],
                    level="load_step",
                )
            ]
            if len(new_residuals_index) > 0
            else pd.DataFrame()
        )

        self._record_vals_residuals = pd.concat([remaining_vals_residuals, new_vals_residuals])

        self._residuals_ndarray = (
            load_turning_points_rep[residuals_index] if len(residuals_index) else np.array([])
        )

        res_len = len(self._residuals_array)
        self._residuals_array = [(i-res_len, val) for i, (_, val) in enumerate(self._residuals_array)]

        # TODO: check if these are really that redundant
        is_closed_hysteresis = (hysts[:, 0] != MEMORY_3).tolist()
        is_zero_mean_stress_and_strain = (hysts[:, 0] == MEMORY_3).tolist()

        self._recorder.record_values_fkm_nonlinear(
            loads_min=results_min["loads_min"],
            loads_max=results_max["loads_max"],
            S_min=results_min["S_min"],
            S_max=results_max["S_max"],
            epsilon_min=results_min["epsilon_min"],
            epsilon_max=results_max["epsilon_max"],
            epsilon_min_LF=epsilon_min_LF,
            epsilon_max_LF=epsilon_max_LF,
            is_closed_hysteresis=is_closed_hysteresis,
            is_zero_mean_stress_and_strain=is_zero_mean_stress_and_strain,
            run_index=self._run_index
        )

        return self

    def _store_recordings_for_history(self, record, record_vals, load_step, hysts):
        record_repr = (
            record_vals.groupby(load_step)
            .first()
            .reset_index(drop=False)
            .drop(["epsilon_min_LF", "epsilon_max_LF"], axis=1)
        )
        record_repr["run_index"] = self._run_index
        record_repr["secondary_branch"] = record[:, SECONDARY] != 0

        rec_hysts = hysts.copy()
        rec_hysts[:, 1:] += self._num_turning_points

        self._history_record.append((record_repr, rec_hysts))


    def _collect_record(self, load_turning_points, turning_groups, load_step, record):
        record_vals = np.empty((5, len(turning_groups)*self._group_size))

        turning_points = load_turning_points.to_numpy()
        for i in range(len(turning_groups)):
            prev_idx = int(record[i, IS_CLOSED])

            if prev_idx < 0:
                idx = len(self._record_vals_residuals) + prev_idx*self._group_size
                prev_record = self._record_vals_residuals.iloc[idx:idx+self._group_size].to_numpy().T
            elif prev_idx < i:
                idx = prev_idx * self._group_size
                prev_record = record_vals[:, idx:idx+self._group_size]
            else:
                prev_record = self._last_record
            idx = i * self._group_size

            load_turning_point = turning_points[idx:idx+self._group_size]
            self._process_deformation(record[i, :], record_vals, idx, load_turning_point, prev_record)

        return pd.DataFrame(
            record_vals.T,
            columns=["load", "stress", "strain", "epsilon_min_LF", "epsilon_max_LF"],
            index=load_turning_points.index,
        )

    def _process_deformation(self, record, record_vals, idx, load, prev_record):
        function_map = [self._primary, self._secondary]

        function = function_map[record[SECONDARY]]

        result = record_vals[:, idx:idx+self._group_size]
        result[:3, :] = function(prev_record, load)

        old_load = self._last_record[LOAD]

        if old_load[0] < load[0]:
            result[EPS_MAX_LF] = (
                self._last_record[EPS_MAX_LF]
                if self._last_record[EPS_MAX_LF, 0] > result[STRAIN, 0]
                else result[STRAIN, :]
            )
            result[EPS_MIN_LF] = self._last_record[EPS_MIN_LF]
        else:
            result[EPS_MIN_LF] = (
                self._last_record[EPS_MIN_LF]
                if self._last_record[EPS_MIN_LF, 0] < result[STRAIN, 0]
                else result[STRAIN, :]
            )
            result[EPS_MAX_LF] = self._last_record[EPS_MAX_LF]

        self._last_record = record_vals[:, idx:idx+self._group_size]

    def _primary(self, _prev, load):
        sigma = self._notch_approximation_law.stress(load)
        epsilon = self._notch_approximation_law.strain(sigma, load)
        return np.array([load, sigma, epsilon])

    def _secondary(self, prev, load):
        prev_load = prev[LOAD]

        delta_L = load - prev_load
        delta_sigma = self._notch_approximation_law.stress_secondary_branch(delta_L)
        delta_epsilon = self._notch_approximation_law.strain_secondary_branch(delta_sigma, delta_L)

        sigma = prev[STRESS] + delta_sigma
        epsilon = prev[STRAIN] + delta_epsilon

        return np.array([load, sigma, epsilon])


    def _process_recording(self, turning_points, record_vals, hysts):
        def turn_memory_1_2(values, index):
            if values[0][0, 0] < values[1][0, 0]:
                return (values[0], values[1], index[0], index[1])
            return (values[1], values[0], index[1], index[0])

        def turn_memory_3(values, index):
            abs_point = np.abs(values[0])
            return (-abs_point, abs_point, index[0], index[0])

        memory_functions = [turn_memory_3, turn_memory_1_2]

        start = len(self._residuals_ndarray)
        if start:
            turning_points = np.concatenate((self._residuals_ndarray, turning_points))
        record_vals_with_residuals = pd.concat([self._record_vals_residuals, record_vals])

        value_array = record_vals_with_residuals.to_numpy()
        index_array = record_vals_with_residuals.index.to_frame().to_numpy()

        result_len = len(hysts) * self._group_size

        signal_index_names = self._current_load_index.names
        signal_index_num = len(signal_index_names)

        results_min = np.zeros((result_len, 3))
        results_min_idx = np.zeros((result_len, signal_index_num), dtype=np.int64)

        results_max = np.zeros((result_len, 3))
        results_max_idx = np.zeros((result_len, signal_index_num), dtype=np.int64)

        epsilon_min_LF = np.zeros(result_len)
        epsilon_max_LF = np.zeros(result_len)

        for i, hyst in enumerate(hysts):
            idx = (hyst[FROM:CLOSE] + start) * self._group_size

            beg0, beg1 = idx[0], idx[1]
            end0, end1 = beg0 + self._group_size, beg1 + self._group_size

            values = value_array[beg0:end0], value_array[beg1:end1]
            index = index_array[beg0:end0], index_array[beg1:end1]

            hyst_type = hyst[IS_CLOSED]
            min_val, max_val, min_idx, max_idx = memory_functions[hyst_type](values, index)

            beg = i * self._group_size
            end = beg + self._group_size

            results_min[beg:end] = min_val[:, :3]
            results_max[beg:end] = max_val[:, :3]

            results_min_idx[beg:end] = min_idx
            results_max_idx[beg:end] = max_idx

            epsilon_min_LF[beg:end] = min_val[:, EPS_MIN_LF]
            epsilon_max_LF[beg:end] = max_val[:, EPS_MAX_LF]

        results_min = pd.DataFrame(
            results_min,
            columns=["loads_min", "S_min", "epsilon_min"],
            index=pd.MultiIndex.from_arrays(results_min_idx.T, names=signal_index_names)
        )
        results_max = pd.DataFrame(
            results_max,
            columns=["loads_max", "S_max", "epsilon_max"],
            index=pd.MultiIndex.from_arrays(results_max_idx.T, names=signal_index_names)
        )

        return results_min, results_max, pd.Series(epsilon_min_LF), pd.Series(epsilon_max_LF)

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


    def _adjust_samples_and_flush_for_hcm_first_run(self, samples):

        is_multi_index = isinstance(samples, pd.Series) and len(samples.index.names) > 1

        if not is_multi_index:
            samples = np.concatenate([[0], np.asarray(samples)])
        else:
            # get the index with all node_id`s
            node_id_index = samples.groupby("node_id").first().index

            # create a new sample with 0 load for all nodes
            multi_index = pd.MultiIndex.from_product([[0], node_id_index], names=["load_step","node_id"])
            first_sample = pd.Series(0, index=multi_index)

            # increase the load_step index value by one for all samples
            samples_without_index = samples.reset_index()
            samples_without_index.load_step += 1
            samples = samples_without_index.set_index(["load_step", "node_id"])[0]

            # prepend the new zero load sample
            samples = pd.concat([first_sample, samples], axis=0)

        # determine first, second and last samples
        scalar_samples = samples

        if is_multi_index:
            # convert to list
            scalar_samples = samples.groupby("load_step", sort=False).first()

        scalar_samples_twice = np.concatenate([scalar_samples, scalar_samples])
        turn_indices, _ = RFG.find_turns(scalar_samples_twice)

        flush = True
        if len(scalar_samples)-1 not in turn_indices:
            flush = False

        return samples, flush

    def _perform_hcm_algorithm(self, load_turning_points):
        """Perform the entire HCM algorithm for all load samples"""

        # iz: number of not yet closed branches
        # ir: number of residual loads corresponding to hystereses that cannot be closed,
        #     because they contain parts of the primary branch

        # iterate over loads from the given list of samples

        hysts = np.zeros((len(load_turning_points), 4), dtype=np.int64)
        hyst_index = 0

        record = -np.ones((len(load_turning_points), 2), dtype=np.int64)
        rec_index = 0

        for index, current_load in enumerate(load_turning_points):
            hyst_index = self._hcm_process_sample(current_load, index, hysts, hyst_index, record, rec_index)

            if np.abs(current_load) > self._load_max_seen:
                self._load_max_seen = np.abs(current_load)

            self._iz += 1

            self._residuals_array.append((rec_index, current_load))

            rec_index += 1

        hysts = hysts[:hyst_index, :]
        return record, hysts


    def _hcm_process_sample(self, current_load, current_index, hysts, hyst_index, record, rec_index):
        """ Process one sample in the HCM algorithm, i.e., one load value """

        record_index = current_index

        while True:
            if self._iz == self._ir:

                if np.abs(current_load) > self._load_max_seen:  # case a) i, "Memory 3"
                    record[rec_index, :] = [record_index, PRIMARY]

                    residuals_idx = self._residuals_array[-1][0]
                    hysts[hyst_index, :] = [MEMORY_3, residuals_idx, current_index, -1]
                    hyst_index += 1

                    self._ir += 1

                else:
                    record[rec_index, :] = [record_index, SECONDARY]
                break

            if self._iz < self._ir:
                record[rec_index, :] = [record_index, PRIMARY]
                break

            # here we have iz > ir:

            prev_idx_0, prev_load_0 = self._residuals_array[-2]
            prev_idx_1, prev_load_1 = self._residuals_array[-1]

            current_load_extent = np.abs(current_load - prev_load_1)
            previous_load_extent = np.abs(prev_load_1 - prev_load_0)

            if current_load_extent < previous_load_extent:
                record[rec_index, :] = [record_index, SECONDARY]
                break

            # no -> we have a new hysteresis

            self._residuals_array.pop()
            self._residuals_array.pop()

            if len(self._residuals_array):
                record_index = self._residuals_array[-1][0]

            self._iz -= 2

            # if the points of the hysteresis lie fully inside the seen range of loads, i.e.,
            # the turn points are smaller than the maximum turn point so far
            # (this happens usually in the second run of the HCM algorithm)
            if np.abs(prev_load_0) < self._load_max_seen and np.abs(prev_load_1) < self._load_max_seen:
                # case "Memory 2", "c) ii B"
                # the primary branch is not yet reached, continue processing residual loads, potentially
                # closing even more hysteresis

                hysts[hyst_index, :] = [MEMORY_1_2, prev_idx_0, prev_idx_1, current_index]
                hyst_index += 1

                continue

            # case "Memory 1", "c) ii A"
            # The last hysteresis saw load values equal to the previous maximum.
            # (Higher values cannot happen here.)
            # The load curve follows again the primary path.
            # No further hystereses can be closed, continue with the next load value.
            # Previously, iz was decreased by 2 and will be increased by 1 at the end of this loop ,
            # effectively `iz = iz - 1` as described on p.70

            # Proceed on primary path for the rest, which was not part of the closed hysteresis

            record[rec_index, :] = [record_index, PRIMARY]
            hysts[hyst_index, :] = [MEMORY_1_2, prev_idx_0, prev_idx_1, current_index]
            hyst_index += 1

        return hyst_index


    def _get_scalar_current_load(self, current_load):
        """Get a scalar value that represents the current load.
        This is either the load itself if it is already scaler,
        or the node from the first assessment point if multiple points are
        considered at once."""

        if isinstance(current_load, pd.Series):
            current_load_representative = current_load.iloc[0]
        else:
            current_load_representative = current_load
        return current_load_representative



    def interpolated_stress_strain_data(
            self,
            *,
            load_segment=None,
            hysteresis_index=None,
            n_points_per_branch=100
    ):
        history = self.history()

        if hysteresis_index is not None:

            hyst_to = history.xs(hysteresis_index, level="hyst_to")
            if hysteresis_index in history.index.get_level_values("hyst_close"):
                hyst_close = history.xs(hysteresis_index, level="hyst_close")
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
            stress = np.linspace(0.0, to_value.stress, n_points)
            strain = self._ramberg_osgood_relation.strain(stress)

            return pd.DataFrame(
                {
                    "stress": stress,
                    "strain": strain,
                    "secondary_branch": False,
                    "hyst_index": hyst_index,
                    "load_segment": load_segment,
                    "run_index": run_index,
                }
            )

        if hyst_close_idx >= 0:
            from_value = history.xs(hyst_close_idx, level="hyst_to").iloc[0]
        elif hyst_open_idx >= 0:
            from_value = history.xs(hyst_open_idx, level="hyst_from").iloc[0]
        elif idx == 0:
            from_value = history.xs(run_index-1, level="run_index").iloc[-1]
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

    def history(self):
        history = pd.concat([rr for rr, _ in self._history_record]).reset_index(drop=True)
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
        load_step_drop_idx = []
        hyst_close_index = []

        for hyst_index, hyst in enumerate(hysts):
            if hyst[IS_CLOSED] == MEMORY_1_2:
                hyst_close = int(hyst[CLOSE]) + len(to_insert)
                hyst_from = int(hyst[FROM])
                load_step_drop_idx.append(hyst_close)
                hyst_close_index.append([hyst_close, hyst_index])
                to_insert.append((hyst_close, hyst_from))
            else:
                hyst_from = int(hyst[FROM])
                hyst_to = int(hyst[TO]) + len(to_insert)
                negate.append(hyst_to)
                load_step_drop_idx.append(hyst_to)
                to_insert.append((hyst_to, hyst_from))

        hyst_close_index = np.array(hyst_close_index, dtype=np.int64)

        negate = np.array(negate, dtype=np.int64)

        index = list(np.arange(len(history)))

        for target, idx in to_insert:
            index.insert(target, int(idx))

        history = history.iloc[index].reset_index(drop=True)

        history.loc[load_step_drop_idx, ["load_step", "hyst_from", "hyst_to"]] = -1
        history.loc[load_step_drop_idx, "secondary_branch"] = True
        history.loc[negate, PHYSICAL_HIST_COLUMNS] = -history.loc[negate, PHYSICAL_HIST_COLUMNS]
        history.loc[negate, "hyst_to"] = history.loc[negate+1, "hyst_to"].to_numpy()
        history.loc[negate+1, "hyst_to"] = -1
        history.loc[negate, "secondary_branch"] = True

        if len(hyst_close_index):
            history.loc[hyst_close_index[:, 0], "hyst_close"] = hyst_close_index[:, 1]

        history["load_segment"] = np.arange(len(history), dtype=np.int64)

        history.set_index(INDEX_HIST_LEVELS, inplace=True)

        return history

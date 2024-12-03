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
import itertools

import pylife.stress.rainflow.general as RFG

INDEX=0
CASE=1

LOAD=0
STRESS=1
STRAIN=2
EPS_MIN_LF=3
EPS_MAX_LF=4

PRIMARY=0
SECONDARY=1
DISCONTINUITY=2
DISCONTINUITY_MASK=0b01
NO_DISCONTINUITY=0

MEMORY_1_2 = 1
MEMORY_3 = 0

PHYSICAL_HIST_COLUMNS = ["load", "stress", "strain", "secondary_branch"]
INDEX_HIST_LEVELS = ["load_segment", "load_step", "run_index", "hyst_from", "hyst_to", "hyst_close"]


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

        self._epsilon_min_LF = None
        self._epsilon_max_LF = None

        # whether the load sequence starts and the first value should be considered (normally, only "turns" in the sequence get extracted, which would omit the first value)
        self._is_load_sequence_start = True

        self._last_record = None
        self._residuals_array = []
        self._residuals_ndarray = np.array([])
        self._record_vals_residuals = pd.DataFrame()

        self._history_record = []

        self._recorded_deformation = pd.DataFrame()
        self._chunk_sizes = []

        self._hcm_message = ""


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

        self._hcm_message += f"HCM first run starts\n"
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

        self._hcm_message += f"\nHCM second run starts\n"
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
        self._chunk_sizes.append(len(load_turning_points))

        li = load_turning_points.index.to_frame()['load_step']
        load_step = (li != li.shift()).cumsum() - 1

        load_turning_points_rep = np.array([
            self._get_scalar_current_load(current_load)
            for _, current_load in load_turning_points.groupby(load_step, sort=False)
        ])

        record, hysts = self._perform_hcm_algorithm(load_turning_points_rep)

        if self._last_record is None:
            self._last_record = np.zeros((5, self._group_size))

        record_vals = self._collect_record(load_turning_points, load_step, record)

        self._store_recordings_for_history(record, record_vals, load_step, hysts)

        self._recorded_deformation = pd.concat([self._recorded_deformation, record_vals])

        _results_min, _results_max, _epsilon_min_LF_new, _epsilon_max_LF_new = self._process_recording(load_turning_points_rep, record_vals, hysts)

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
            loads_min=_results_min["loads_min"],
            loads_max=_results_max["loads_max"],
            S_min=_results_min["S_min"],
            S_max=_results_max["S_max"],
            epsilon_min=_results_min["epsilon_min"],
            epsilon_max=_results_max["epsilon_max"],
            epsilon_min_LF=_epsilon_min_LF_new,
            epsilon_max_LF=_epsilon_max_LF_new,
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
        record_repr["secondary_branch"] = record[:, CASE] != 0

        rec_hysts = hysts.copy()
        rec_hysts[:, 1:] += sum(self._chunk_sizes[:-1])

        self._history_record.append((record_repr, rec_hysts))


    def _collect_record(self, load_turning_points, load_step, record):
        record_vals = np.empty((len(load_turning_points), 5))

        for i, (_, load_turning_point) in enumerate(load_turning_points.groupby(load_step, sort=False)):
            prev_idx = int(record[i, INDEX])
            prev_record = self._last_record
            if prev_idx < 0:
                idx = len(self._record_vals_residuals) + prev_idx*self._group_size
                prev_record = self._record_vals_residuals.iloc[idx:idx+self._group_size].to_numpy().T
            elif prev_idx < i:
                prev_record = record_vals[prev_idx*self._group_size:(prev_idx+1)*self._group_size].T
            idx = i * self._group_size
            record_vals[idx:idx+self._group_size] = self._process_deformation(record[i, :], load_turning_point, prev_record).T

        return pd.DataFrame(
            record_vals,
            columns=["load", "stress", "strain", "epsilon_min_LF", "epsilon_max_LF"],
            index=load_turning_points.index,
        )


    def _process_deformation(self, record, load, prev_record):
        function_map = [self._primary, self._secondary]

        function = function_map[record[CASE] & DISCONTINUITY_MASK]

        result = np.empty((5, self._group_size))
        result[:3, :] = function(prev_record, load)

        old_load = self._last_record[LOAD]

        if old_load[0] < load.iloc[0]:
            result[EPS_MAX_LF] = self._last_record[EPS_MAX_LF] if self._last_record[EPS_MAX_LF, 0] > result[STRAIN, 0] else result[STRAIN, :]
            result[EPS_MIN_LF] = self._last_record[EPS_MIN_LF]
        else:
            result[EPS_MIN_LF] = self._last_record[EPS_MIN_LF] if self._last_record[EPS_MIN_LF, 0] < result[STRAIN, 0] else result[STRAIN, :]
            result[EPS_MAX_LF] = self._last_record[EPS_MAX_LF]

        self._last_record = result.copy()

        return result

    def load_memory_1_2(self, point_0, point_1):
        return (point_0, point_1, point_0, point_1) if point_0.values[0, 0] < point_1.values[0, 0] else (point_1, point_0, point_1, point_0)

    def load_memory_3(self, point_0, point_1):
        abs_point = point_0.abs()
        return (-abs_point, abs_point, point_0, point_0)

    def _process_recording(self, turning_points, record_vals, hysts):
        start = len(self._residuals_ndarray)
        if start:
            turning_points = np.concatenate((self._residuals_ndarray, turning_points))
        record_vals_with_residuals = pd.concat([self._record_vals_residuals, record_vals])
        num = len(hysts)

        inames = self._current_load_index.names

        results_min = pd.DataFrame(
            np.zeros((num * self._group_size, len(inames) + 3)),
            columns=["loads_min", "S_min", "epsilon_min"] + inames,
        )

        index_min = []

        results_max = pd.DataFrame(
            np.zeros((num * self._group_size, len(inames) + 3)),
            columns=["loads_max", "S_max", "epsilon_max"] + inames
        )

        epsilon_min_LF = pd.Series(np.zeros((num * self._group_size)))
        epsilon_max_LF = pd.Series(np.zeros((num * self._group_size)))

        index_max = []

        memory_functions = [self.load_memory_3, self.load_memory_1_2]

        for i, hyst in enumerate(hysts):
            idx = (hyst[1:3] + start) * self._group_size
            point_0 = record_vals_with_residuals[idx[0]:idx[0]+self._group_size]
            point_1 = record_vals_with_residuals[idx[1]:idx[1]+self._group_size]
            hyst_type = hyst[0]

            _min, _max, _lf_min, _lf_max = memory_functions[hyst_type](point_0, point_1)
            rmin = _min.reset_index(drop=False)[["load", "stress", "strain"] + inames]
            results_min.iloc[i*self._group_size:(i+1)*self._group_size] = rmin
            results_max.iloc[i*self._group_size:(i+1)*self._group_size] = _max.reset_index(drop=False)[["load", "stress", "strain"] + inames]
            index_min.append(_min.index)
            index_max.append(_max.index)

            epsilon_min_LF.iloc[i*self._group_size:(i+1)*self._group_size] = _lf_min['epsilon_min_LF']
            epsilon_max_LF.iloc[i*self._group_size:(i+1)*self._group_size] = _lf_max['epsilon_max_LF']

        for iname in inames:
            results_min[iname] = results_min[iname].astype(np.int64)
            results_max[iname] = results_max[iname].astype(np.int64)
        return results_min.set_index(inames, drop=True), results_max.set_index(inames), epsilon_min_LF, epsilon_max_LF

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
        return self._recorded_deformation.strain.to_numpy()

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

        return self.strain_values[:self._chunk_sizes[0]]

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

        return self.strain_values[self._chunk_sizes[0]:]



    def _primary(self, _prev, load):
        sigma = self._notch_approximation_law.stress(load)
        epsilon = self._notch_approximation_law.strain(sigma, load)
        return np.array([load, sigma, epsilon])

    def _secondary(self, prev, load):
        prev_load = prev[LOAD]
        secondary_load = np.sign(load) * np.minimum(np.abs(load), np.abs(prev_load))

        delta_L = load - prev_load
        delta_sigma = self._notch_approximation_law.stress_secondary_branch(delta_L)
        delta_epsilon = self._notch_approximation_law.strain_secondary_branch(delta_sigma, delta_L)

        sigma = prev[STRESS] + delta_sigma
        epsilon = prev[STRAIN] + delta_epsilon

        return np.array([load, sigma, epsilon])

        delta_L = load - secondary_load
        delta_sigma = self._notch_approximation_law.stress_secondary_branch(delta_L)
        delta_epsilon = self._notch_approximation_law.strain_secondary_branch(delta_sigma, delta_L)

        return np.array([load, sigma, epsilon])

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
        history = pd.DataFrame(
            {c: pd.Series(dtype=np.float64) for c in PHYSICAL_HIST_COLUMNS}
            | {l: pd.Series(dtype=np.int64) for l in INDEX_HIST_LEVELS}
            | {"secondary_branch": pd.Series(dtype=np.bool_)},
        ).set_index(INDEX_HIST_LEVELS, drop=True)

        record_repr = pd.concat([rr for rr, _ in self._history_record]).reset_index(drop=True)
        record_repr["load_segment"] = np.arange(1, len(record_repr) + 1)

        hysts = np.concatenate([hs for _, hs in self._history_record])

        hyst_index = np.concatenate(
            [[np.arange(len(hysts))], hysts[:, 1:4].T + len(history), hysts[:, 0:1].T]
        ).T

        history.reset_index(inplace=True, drop=False)

        history = pd.concat([history, record_repr]).reset_index(drop=True)

        hyst_from_marker = pd.Series(-1, index=history.index)
        hyst_to_marker = pd.Series(-1, index=history.index)
        hyst_close_marker = pd.Series(-1, index=history.index)

        if len(hysts):
            hyst_from_marker.iloc[hyst_index[:, 1]] = hyst_index[:, 0]
            hyst_to_marker.iloc[hyst_index[:, 2]] = hyst_index[:, 0]

        history["hyst_from"] = hyst_from_marker
        history["hyst_to"] = hyst_to_marker
        history["hyst_close"] = hyst_close_marker

        index = list(np.arange(len(record_repr)))

        to_insert = []
        negate = []
        load_step_drop_idx = []
        hyst_close_index = []

        for hyst_index, hyst in enumerate(hysts):
            if hyst[0] == MEMORY_1_2:
                hyst_close = int(hyst[3])
                hyst_from = int(hyst[1])
                load_step_drop_idx.append(hyst_close+len(to_insert))
                hyst_close_index.append([hyst_close+len(to_insert), hyst_index])
                to_insert.append((hyst_close, hyst_from))
            else:
                hyst_from = int(hyst[1])
                hyst_to = int(hyst[2])
                negate.append(hyst_to+len(to_insert))
                load_step_drop_idx.append(hyst_to+len(to_insert))
                to_insert.append((hyst_to, hyst_from))

        hyst_close_index = np.array(hyst_close_index, dtype=np.int64)

        negate = np.array(negate, dtype=np.int64)

        for target, idx in reversed(to_insert):
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

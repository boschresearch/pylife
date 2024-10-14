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

import pylife.stress.rainflow.general

INDEX=0
CASE=1

LOAD=0
STRESS=1
STRAIN=2
EPS_MIN_LF=3
EPS_MAX_LF=4

PRIMARY=0
SECONDARY=1


class FKMNonlinearDetector(pylife.stress.rainflow.general.AbstractDetector):
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

        empty_index = [] if not isinstance(samples, pd.Series) else pd.DataFrame(columns=samples.index.names, dtype=np.int64).set_index(samples.index.names, drop=True).index
        self._is_zero_mean_stress_and_strain = []

        # initialization of _epsilon_min_LF see FKM nonlinear p.122
        if self._epsilon_min_LF is None:
            self._epsilon_min_LF = pd.Series(0.0)

        if self._epsilon_max_LF is None:
            self._epsilon_max_LF = pd.Series(0.0)

        # store all lists together

        previous_load = 0

        self._run_index += 1

        # convert from Series to np.array

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
                tindex = loads_indices - old_head_index
                idx = load_steps.iloc[tindex]
                vals = samples.loc[idx].reset_index()
                vals = vals.set_index(['load_step', 'node_id'])
                if tindex[0] < 0:
                    vals.iloc[:len(self._last_sample), 0] = self._last_sample

                load_turning_points = vals.iloc[:, 0]

            else:
                load_turning_points = []
            idx = load_steps.iloc[-1]
            self._last_sample = samples.loc[idx]

        self._initialize_epsilon_min_for_hcm_run(samples, load_turning_points)

        if not isinstance(load_turning_points, pd.Series):
            load_turning_points = pd.Series(load_turning_points)
            load_turning_points.index.name = 'load_step'

        self._current_load_index = load_turning_points.index
        self._chunk_sizes.append(len(load_turning_points))

        li = load_turning_points.index.to_frame()['load_step']
        load_step = (li != li.shift()).cumsum()

        self._current_load_indexer = load_turning_points.index.get_level_values("load_step").unique()

        load_turning_points_rep = np.array([
            self._get_scalar_current_load(current_load)
            for _, current_load in load_turning_points.groupby(load_step, sort=False)
        ])

        self._hysts = []
        self._record = -np.ones((len(load_turning_points_rep), 2), dtype=np.int64)

        self._index = 0
        self._hyst_index = 0

        self._load_max_seen, self._iz, self._ir = self._perform_hcm_algorithm(
            previous_load=previous_load, iz=self._iz, ir=self._ir,
            load_max_seen=self._load_max_seen, load_turning_points=load_turning_points)

        residuals_index = np.array([i for i, _, _ in self._residuals_array])

        self._hysts = np.array(self._hysts)

        if self._last_record is None:
            self._last_record = np.zeros((5, self._group_size))

        record_vals = np.empty((len(load_turning_points), 5))
        self._epsilon_LF = np.zeros((len(load_turning_points), 2))

        sli = 0
        for i, (_, load_turning_point) in enumerate(load_turning_points.groupby(load_step, sort=False)):
            prev_index = int(self._record[i, INDEX])
            prev_record = self._last_record
            if prev_index < 0:
                rl = len(self._record_vals_residuals)
                l = rl + prev_index*self._group_size
                u = l+self._group_size
                prev_record = self._record_vals_residuals.iloc[l:u].to_numpy().T
            elif prev_index < i:
                prev_record = record_vals[prev_index*self._group_size:(prev_index+1)*self._group_size].T
            record_vals[sli:sli+len(load_turning_point)] = self.process_deformation(self._record[i, :], load_turning_point, prev_record).T
            sli += len(load_turning_point)

        self._record_vals = pd.DataFrame(record_vals, columns=["load", "stress", "strain", "epsilon_min_LF", "epsilon_max_LF"], index=self._current_load_index)


        self._recorded_deformation = pd.concat([self._recorded_deformation, self._record_vals])

        _results_min, _results_max, _epsilon_min_LF_new, _epsilon_max_LF_new = self._process_recording(load_turning_points_rep)

        remaining_vals_residuals = (
            self._record_vals_residuals.loc[
                self._record_vals_residuals.index[residuals_index[residuals_index < 0]]
            ]
            if len(residuals_index) > 0
            else pd.DataFrame()
        )

        new_residuals_index = residuals_index[residuals_index >= 0]
        li = self._record_vals.index.to_frame()['load_step']
        load_step = (li != li.shift()).cumsum()

        index_series = self._record_vals.index.get_level_values("load_step").to_series()
        indexer = pd.DataFrame(
            {
                "load_step": index_series.values,
                "load_num": load_step,
            },
            index=load_step.index
        ).groupby("load_num").first()["load_step"].values

        new_vals_residuals = (
            self._record_vals.loc[
                self._record_vals.index.isin(
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
        self._residuals_array = [(i-res_len, val, arr) for i, (_, val, arr) in enumerate(self._residuals_array)]


        if len(self._hysts):
            _is_closed_hysteresis = self._hysts[:, 0] > 0
            _is_closed_hysteresis = _is_closed_hysteresis.tolist()
        else:
            _is_closed_hysteresis = []

        self._recorder.record_values_fkm_nonlinear(
            loads_min=_results_min["loads_min"],
            loads_max=_results_max["loads_max"],
            S_min=_results_min["S_min"],
            S_max=_results_max["S_max"],
            epsilon_min=_results_min["epsilon_min"],
            epsilon_max=_results_max["epsilon_max"],
            epsilon_min_LF=_epsilon_min_LF_new,
            epsilon_max_LF=_epsilon_max_LF_new,
            is_closed_hysteresis=_is_closed_hysteresis,
            is_zero_mean_stress_and_strain=self._is_zero_mean_stress_and_strain,
            run_index=self._run_index
        )

        return self

    def process_deformation(self, record, load, prev_record):
        function_map = [self._primary, self._secondary]

        function = function_map[record[CASE]]

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

    def _process_recording(self, turning_points):
        start = len(self._residuals_ndarray)
        if start:
            turning_points = np.concatenate((self._residuals_ndarray, turning_points))
        record_vals_with_residuals = pd.concat([self._record_vals_residuals, self._record_vals])
        num = len(self._hysts)

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

        memory_functions = [self.load_memory_3, self.load_memory_1_2, self.load_memory_1_2, self.load_memory_3]

        for i, hyst in enumerate(self._hysts):
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

        delta_L = load - prev_load #secondary_load
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
        turn_indices, _ = pylife.stress.rainflow.general.find_turns(scalar_samples_twice)

        flush = True
        if len(scalar_samples)-1 not in turn_indices:
            flush = False

        return samples, flush

    def _perform_hcm_algorithm(self, *, previous_load, iz, ir, load_max_seen, load_turning_points):
        """Perform the entire HCM algorithm for all load samples"""

        # iz: number of not yet closed branches
        # ir: number of residual loads corresponding to hystereses that cannot be closed,
        #     because they contain parts of the primary branch

        self._hcm_message += f"turning points: {load_turning_points}\n"

        # iterate over loads from the given list of samples
        li = load_turning_points.index.to_frame()['load_step']
        load_step = (li != li.shift()).cumsum()

        for index, current_load in load_turning_points.groupby(load_step, sort=False):
            current_load_representative = self._get_scalar_current_load(current_load)

            self._hcm_message += f"* load {current_load}:"

            iz, ir = self._hcm_process_sample(
                iz=iz, ir=ir,
                load_max_seen=load_max_seen, current_load_representative=current_load_representative,
                current_index=index-1
            )

            if np.abs(current_load_representative) > load_max_seen:
                load_max_seen = np.abs(current_load_representative)

            iz += 1

            self._residuals_array.append((self._index, current_load_representative, self._record[self._index, :]))

            self._index += 1

        return load_max_seen, iz, ir


    def _hcm_process_sample(self, *, iz, ir, load_max_seen, current_load_representative, current_index):
        """ Process one sample in the HCM algorithm, i.e., one load value """

        record_index = current_index

        while True:
            if iz == ir:

                # if the current load is a new maximum
                if np.abs(current_load_representative) > load_max_seen:
                    # case a) i., "Memory 3"
                    self._record[self._index, :] = [record_index, PRIMARY]
                    self._hysts.append([0, current_index-1, current_index+1])
                    self._hyst_index += 1
                    self._is_zero_mean_stress_and_strain.append(True)

                    ir += 1

                else:
                    self._record[self._index, :] = [record_index, SECONDARY]
                    self._hyst_index += 1
                # end the inner loop and fetch the next load from the load sequence
                break

            if iz < ir:
                self._record[self._index, :] = [record_index, PRIMARY]
                self._hyst_index += 1
                # branch is fully part of the initial curve, case "Memory 1"
                break

            # here we have iz > ir:

            prev_idx_0, prev_load_0, _ = self._residuals_array[-2]
            prev_idx_1, prev_load_1, _ = self._residuals_array[-1]

            # is the current load extent smaller than the last one?
            current_load_extent = np.abs(current_load_representative - prev_load_1)
            previous_load_extent = np.abs(prev_load_1 - prev_load_0)
            # yes
            if current_load_extent < previous_load_extent:
                self._record[self._index, :] = [record_index, SECONDARY]
                self._hyst_index += 1

                # continue with the next load value
                break

            # no -> we have a new hysteresis

            self._is_zero_mean_stress_and_strain.append(False)

            self._residuals_array.pop()
            self._residuals_array.pop()

            if len(self._residuals_array):
                record_index = self._residuals_array[-1][0]

            iz -= 2

            # if the points of the hysteresis lie fully inside the seen range of loads, i.e.,
            # the turn points are smaller than the maximum turn point so far
            # (this happens usually in the second run of the HCM algorithm)
            if np.abs(prev_load_0) < load_max_seen and np.abs(prev_load_1) < load_max_seen:
                # case "Memory 2", "c) ii B"
                # the primary branch is not yet reached, continue processing residual loads, potentially
                # closing even more hysteresis

                self._hcm_message += ","
                self._hysts.append([2, prev_idx_0, prev_idx_1])
                self._hyst_index += 1
                continue

            # case "Memory 1", "c) ii A"
            # The last hysteresis saw load values equal to the previous maximum.
            # (Higher values cannot happen here.)
            # The load curve follows again the primary path.
            # No further hystereses can be closed, continue with the next load value.
            # Previously, iz was decreased by 2 and will be increased by 1 at the end of this loop ,
            # effectively `iz = iz - 1` as described on p.70

            # Proceed on primary path for the rest, which was not part of the closed hysteresis

            self._record[self._index, :] = [record_index, PRIMARY]
            self._hysts.append([1, prev_idx_0, prev_idx_1])

            # store strain values, this is for the FKM nonlinear roughness & surface layer algorithm, which adds residual stresses in another pass of the HCM algorithm

            # count number of strain values in the first run of the HCM algorithm
            break

        return iz, ir

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

    def _initialize_epsilon_min_for_hcm_run(self, samples, load_turning_points):
        """initializes the values of epsilon_min_LF and epsilon_max_LF to
        have the proper dimensions."""

        if self._is_load_sequence_start:
            self._is_load_sequence_start = False

            if not isinstance(load_turning_points, np.ndarray):
                # properly initialize self._epsilon_min_LF and self._epsilon_max_LF
                first_sample = samples[samples.index.get_level_values("load_step") == 0].reset_index(drop=True)

                n_nodes = len(first_sample)
                self._epsilon_min_LF = pd.Series([0.0]*n_nodes, index=pd.Index(np.arange(n_nodes), name='node_id'))
                self._epsilon_max_LF = pd.Series([0.0]*n_nodes, index=pd.Index(np.arange(n_nodes), name='node_id'))


    def interpolated_deformation(self, *, load_step, run_index, n_points):
        run_history = self._history.xs(run_index, level="run_index")
        idx = run_history.index.droplevel(["hyst_from", "hyst_to"]).get_loc(load_step)

        to_value = run_history.iloc[idx]

        if idx == 0:
            if run_index == self._history.index.get_level_values("run_index").min():
                stress = np.linspace(0.0, to_value.stress, n_points)
                strain = self._ramberg_osgood_relation.strain(stress)

                return pd.DataFrame({"stress": stress, "strain": strain, "secondary_branch": False})

            from_value = self._history.xs(run_index-1, level="run_index").iloc[-1]
        else:
            from_value = run_history.iloc[idx-1]

        stress = np.linspace(from_value.stress, to_value.stress, n_points)

        if to_value.secondary_branch:
            secondary = np.ones(n_points, dtype=np.bool_)
        else:
            secondary = np.abs(stress) <= np.abs(from_value.stress)

        delta_stress = from_value.stress - stress[secondary]
        strain = np.empty(n_points)
        strain[secondary] = from_value.strain - self._ramberg_osgood_relation.delta_strain(delta_stress)
        strain[~secondary] = self._ramberg_osgood_relation.strain(stress[~secondary])

        return pd.DataFrame({"stress": stress, "strain": strain, "secondary_branch": secondary})


    def interpolated_deformation_history(self, n_points):
        result = self._history.drop(["load", "epsilon_min_LF", "epsilon_max_LF"], axis=1).droplevel("hyst_to")
        result.index = result.index.rename({"hyst_from": "hyst_index"})
        return result

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


class FKMNonlinearDetector(pylife.stress.rainflow.general.AbstractDetector):
    """HCM-Algorithm detector as described in FKM nonlinear.

    """

    class _HCM_Point:
        """A point in the stress-strain diagram on which the HCM algorithm operates on.

        .. note::
            For an assessment for multiple points (FEM mesh nodes) at once,
            we assume that the load time series for the different points are
            multiples of each other. In consequence, the hysteresis graph in
            the stress-strain diagram follows the same sequence of primary and
            secondary paths for every assessment point.
            It suffices to consider a single point to find out when a hysteresis
            gets closed and when to reach the primary path etc. However, the actual
            stress/strain values will be computed individually for every point.
        """

        def __init__(self, load=None, strain=None, stress=None):
            self._load = load
            self._stress = stress
            self._strain = strain

        @property
        def load(self):
            return self._load

        @property
        def load_representative(self):
            if isinstance(self._load, pd.Series):
                return self._load.iloc[0]
            else:
                return self._load

        @property
        def strain_representative(self):
            if isinstance(self._strain, pd.Series):
                return self._strain.iloc[0]
            else:
                return self._strain

        @property
        def stress(self):
            return self._stress

        @property
        def strain(self):
            return self._strain

        def __str__(self):
            assert not isinstance(self._load, pd.DataFrame)
            if isinstance(self._load, pd.Series):
                if self._stress is None:
                    if self._load is None:
                        return "()"
                    return f"(load:{self._load.values[0]:.1f})"

                if self._load is None:
                    return f"(sigma:{self._stress.values[0]:.1f}, eps:{self._strain.values[0]:.1e})"
                return f"(load:{self._load.values[0]:.1f}, sigma:{self._stress.values[0]:.1f}, eps:{self._strain.values[0]:.1e})"

            if self._stress is None:
                if self._load is None:
                    return "()"
                return f"(load:{self._load:.1f})"
            if self._load is None:
                return f"sigma:{self._stress:.1f}, eps:{self._strain:.1e})"
            return f"(load:{self._load:.1f}, sigma:{self._stress:.1f}, eps:{self._strain:.1e})"

        def __repr__(self):
            return self.__str__()

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

        self._epsilon_min_LF = np.inf         # the current value for _epsilon_min_LF, initialization see FKM nonlinear p.122
        self._epsilon_max_LF = -np.inf        # the current value for _epsilon_max_LF, initialization see FKM nonlinear p.122

        # deviation from FKM nonlinear algorithm to match given example in FKM nonlinear
        self._epsilon_min_LF = 0       # the current value for _epsilon_min_LF, initialization see FKM nonlinear p.122
        self._epsilon_max_LF = 0       # the current value for _epsilon_max_LF, initialization see FKM nonlinear p.122

        self._epsilon_min_LF = None
        self._epsilon_max_LF = None

        # whether the load sequence starts and the first value should be considered (normally, only "turns" in the sequence get extracted, which would omit the first value)
        self._is_load_sequence_start = True

        self._residuals = []        # unclosed hysteresis points

        self._hcm_point_history = []   # all traversed points, for plotting and debugging
        # list of tuples (type, hcm_point, index), e.g., [ ("primary", hcm_point, 0), ("secondary", hcm_point, 1), ...]
        # where the type is one of {"primary", "secondary"} and indicates the hysteresis branch up to the current point
        # and the index is the hysteresis number to which the points belong
        self._hcm_message = ""

        # the current index of the row in the recorded `collective` DataFrame, used only for debugging,
        # i.e., the `interpolated_stress_strain_data` method
        self._hysteresis_index = 0

        self._current_debug_output = ""
        self._strain_values = []
        self._n_strain_values_first_run = 0

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
        _loads_min = pd.Series(index=empty_index, dtype=np.float64)
        _loads_max = pd.Series(index=empty_index, dtype=np.float64)
        _S_min = pd.Series(index=empty_index, dtype=np.float64)
        _S_max = pd.Series(index=empty_index, dtype=np.float64)
        _epsilon_min = pd.Series(index=empty_index, dtype=np.float64)
        _epsilon_max = pd.Series(index=empty_index, dtype=np.float64)
        _epsilon_min_LF = pd.Series(index=empty_index, dtype=np.float64)      # minimum strain of the load history up to (including) the current hysteresis (LF=Lastfolge), mentioned on p.127 of FKM nonlinear
        _epsilon_max_LF = pd.Series(index=empty_index, dtype=np.float64)      # maximum strain of the load history up to (including) the current hysteresis (LF=Lastfolge), mentioned on p.127 of FKM nonlinear
        _is_closed_hysteresis = []            # whether the hysteresis is fully closed and counts as a normal damage hysteresis
        _is_zero_mean_stress_and_strain = []  # whether the mean stress and strain are forced to be zero (occurs in eq. 2.9-52)
        _debug_output = []

        # initialization of _epsilon_min_LF see FKM nonlinear p.122
        if self._epsilon_min_LF is None:
            self._epsilon_min_LF = pd.Series(0.0)

        if self._epsilon_max_LF is None:
            self._epsilon_max_LF = pd.Series(0.0)

        # store all lists together
        recording_lists = [_loads_min, _loads_max, _S_min, _S_max, _epsilon_min,
            _epsilon_max, _epsilon_min_LF, _epsilon_max_LF, _is_closed_hysteresis,
            _is_zero_mean_stress_and_strain, _debug_output]

        largest_point = self._HCM_Point(load=0)
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

        if multi_index:
            load_steps = samples.index.get_level_values('load_step').unique().to_series()
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

        self._load_max_seen, self._iz, self._ir, recording_lists = self._perform_hcm_algorithm(
            samples=samples, recording_lists=recording_lists, largest_point=largest_point,
            previous_load=previous_load, iz=self._iz, ir=self._ir,
            load_max_seen=self._load_max_seen, load_turning_points=load_turning_points)

        # transfer the detected hystereses to the recorder
        [_loads_min, _loads_max, _S_min, _S_max, _epsilon_min, _epsilon_max, _epsilon_min_LF,
         _epsilon_max_LF, _is_closed_hysteresis, _is_zero_mean_stress_and_strain, _debug_output] = recording_lists

        self._recorder.record_values_fkm_nonlinear(
            loads_min=_loads_min, loads_max=_loads_max,
            S_min=_S_min, S_max=_S_max,
            epsilon_min=_epsilon_min, epsilon_max=_epsilon_max,
            epsilon_min_LF=_epsilon_min_LF, epsilon_max_LF=_epsilon_max_LF,
            is_closed_hysteresis=_is_closed_hysteresis, is_zero_mean_stress_and_strain=_is_zero_mean_stress_and_strain,
            run_index=self._run_index, debug_output=_debug_output)


        return self

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
        return np.array(self._strain_values)

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

        return np.array(self._strain_values[:self._n_strain_values_first_run])

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

        return np.array(self._strain_values[self._n_strain_values_first_run:])

    def interpolated_stress_strain_data(self, *, n_points_per_branch=100, only_hystereses=False):
        """Return points on the traversed hysteresis curve, mainly intended for plotting.
        The curve including all hystereses, primary and secondary branches is sampled
        at a fixed number of points within each hysteresis branch.
        These points can be used for plotting.

        The intended use is to generate plots as follows:

        .. code:: python

            fkm_nonlinear_detector.process_hcm_first(...)
            sampling_parameter = 100    # choose larger for smoother plot or smaller for lower runtime
            plotting_data = detector.interpolated_stress_strain_data(n_points_per_branch=sampling_parameter)

            strain_values_primary = plotting_data["strain_values_primary"]
            stress_values_primary = plotting_data["stress_values_primary"]
            hysteresis_index_primary = plotting_data["hysteresis_index_primary"]
            strain_values_secondary = plotting_data["strain_values_secondary"]
            stress_values_secondary = plotting_data["stress_values_secondary"]
            hysteresis_index_secondary = plotting_data["hysteresis_index_secondary"]

            plt.plot(strain_values_primary, stress_values_primary, "g-", lw=3)
            plt.plot(strain_values_secondary, stress_values_secondary, "b-.", lw=1)

        Parameters
        ----------
        n_points_per_branch : int, optional
            How many sampling points to use per hysteresis branch, default 100.
            A larger value values means smoother curves but longer runtime.

        only_hystereses : bool, optional
            Default ``False``. If only graphs of the closed hystereses should be output.
            Note that this does not work for hysteresis that have multiple smaller hysterseses included.

        Returns
        -------
        plotting_data : dict
            A dict with the following keys:

            * "strain_values_primary"
            * "stress_values_primary"
            * "hysteresis_index_primary"
            * "strain_values_secondary"
            * "stress_values_secondary"
            * "hysteresis_index_secondary"

            The values are lists of strains and stresses of the points on the stress-strain curve,
            separately for primary and secondary branches. The lists contain nan values whenever the
            curve of the *same* branch is discontinuous. This allows to plot the entire curve
            with different colors for primary and secondary branches.

            The entries for hysteresis_index_primary and hysteresis_index_secondary are the row
            indices into the collective DataFrame returned by the recorder. This allows, e.g.,
            to separate the output of multiple runs of the HCM algorithm or to plot the traversed
            paths on the stress-strain diagram for individual steps of the algorithm.

        """

        assert n_points_per_branch >= 2

        plotter = FKMNonlinearHysteresisPlotter(self._hcm_point_history, self._ramberg_osgood_relation)
        return plotter.interpolated_stress_strain_data(n_points_per_branch=n_points_per_branch, only_hystereses=only_hystereses)

    def _proceed_on_primary_branch(self, current_point):
        """Follow the primary branch (de: Erstbelastungskurve) of a notch approximation material curve.

        Parameters
        ----------
        previous_point : _HCM_Point
            The starting point in the stress-strain diagram where to begin to follow the primary branch.
        current_point : _HCM_Point
            The end point until where to follow the primary branch. This variable only needs to have the load value.

        Returns
        -------
        current_point : _HCM_Point
            The initially given current point, but with  updated values of stress and strain.

        """
        sigma = self._notch_approximation_law.stress(current_point.load)
        epsilon = self._notch_approximation_law.strain(sigma, current_point.load)

        current_point._stress = pd.Series(sigma.values, index=current_point.load.index)
        current_point._strain = pd.Series(epsilon.values, index=current_point.load.index)

        # log point for later plotting
        self._hcm_point_history.append(("primary", current_point, self._hysteresis_index))

        return current_point

    def _proceed_on_secondary_branch(self, previous_point, current_point):
        """Follow the secondary branch of a notch approximation material curve.

        Parameters
        ----------
        previous_point : _HCM_Point
            The starting point in the stress-strain diagram where to begin to follow the primary branch.
        current_point : _HCM_Point
            The end point until where to follow the primary branch. This variable only needs to have the load value.

        Returns
        -------
        current_point : _HCM_Point
            The initially given current point, but with  updated values of stress and strain.

        """
        delta_L = current_point.load.values - previous_point.load.values   # as described in FKM nonlinear
        index = current_point.load.index
        obsolete_index_levels = [n for n in index.names if n != 'load_step']
        delta_L = pd.Series(
            delta_L, index=current_point.load.index.droplevel(obsolete_index_levels)
        )

        delta_sigma = self._notch_approximation_law.stress_secondary_branch(delta_L)
        delta_epsilon = self._notch_approximation_law.strain_secondary_branch(delta_sigma, delta_L)

        current_point._stress = pd.Series(previous_point._stress.values + delta_sigma.values, index=current_point.load.index)
        current_point._strain = pd.Series(previous_point._strain.values + delta_epsilon.values, index=current_point.load.index)

        # log point for later plotting
        self._hcm_point_history.append(("secondary", current_point, self._hysteresis_index))
        return current_point

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

    def _perform_hcm_algorithm(self, *, samples, recording_lists, largest_point, previous_load, iz, ir, load_max_seen, load_turning_points):
        """Perform the entire HCM algorithm for all load samples,
        record the found hysteresis parameters in the recording_lists."""

        # iz: number of not yet closed branches
        # ir: number of residual loads corresponding to hystereses that cannot be closed,
        #     because they contain parts of the primary branch

        self._hcm_message += f"turning points: {samples}\n"

        # iterate over loads from the given list of samples
        li = load_turning_points.index.to_frame()['load_step']
        load_step = (li != li.shift()).cumsum()

        for _, current_load in load_turning_points.groupby(load_step, sort=False):
            current_load_representative = self._get_scalar_current_load(current_load)

            self._hcm_message += f"* load {current_load}:"

            # initialize the point in the stress-strain diagram corresponding to the current load.
            # The stress and strain values will be computed during the current iteration of the present loop.
            current_point = self._HCM_Point(load=current_load)

            current_point, iz, ir, recording_lists = self._hcm_process_sample(
                current_point=current_point,
                recording_lists=recording_lists,
                largest_point=largest_point, iz=iz, ir=ir,
                load_max_seen=load_max_seen, current_load_representative=current_load_representative
            )

            # update the maximum seen absolute load
            if np.abs(current_load_representative) > load_max_seen+1e-12:
                load_max_seen = np.abs(current_load_representative)
                largest_point = current_point

            # increment the indicator how many open hystereses there are
            iz += 1

            # store the previously processed point to the list of residuals to be processed in the next iterations
            self._residuals.append(current_point)

            self._hcm_update_min_max_strain_values(
                previous_load=previous_load,
                current_load_representative=current_load_representative,
                current_point=current_point
            )
            self._hcm_message += f"\n"

            previous_load = current_load_representative

        return load_max_seen, iz, ir, recording_lists

    def _hcm_update_min_max_strain_values(self, *, previous_load, current_load_representative, current_point):
        """Update the minimum and maximum yet seen strain values
        This corresponds to the "steigend=1 or 2" assignment at chapter 2.9.7 point 5 and
        the rules under point 7.

        5->6, VZ = 5-6=-1 < 0, steigend = 1
        7->4, VZ = 7-4=3 >= 0, steigend = 2, L(0)=0
        """

        if previous_load < current_load_representative-1e-12:
            # case "steigend=1", i.e., load increases
            new_val = self._epsilon_max_LF if self._epsilon_max_LF.values[0] > current_point.strain.values[0] else current_point.strain
            self._epsilon_max_LF = new_val #pd.Series(new_val.values, index=current_point.strain.index)


        else:
            # case "steigend=2", i.e., load decreases
            new_val = self._epsilon_min_LF if self._epsilon_min_LF.values[0] < current_point.strain.values[0] else current_point.strain
            self._epsilon_min_LF = pd.Series(new_val.values, index=current_point.strain.index)

    def _hcm_process_sample(self, *, current_point, recording_lists, largest_point, iz, ir, load_max_seen, current_load_representative):
        """ Process one sample in the HCM algorithm, i.e., one load value """

        while True:
            # iz = len(self._residuals)
            if iz == ir:
                previous_point = self._residuals[-1]

                # if the current load is a new maximum
                if np.abs(current_load_representative) > load_max_seen+1e-12:
                    # case a) i., "Memory 3"
                    current_point, recording_lists = self._handle_case_a_i(
                        current_point=current_point, previous_point=previous_point,
                        recording_lists=recording_lists
                    )
                    ir += 1

                else:
                    current_point = self._handle_case_a_ii(current_point=current_point, previous_point=previous_point)


                # end the inner loop and fetch the next load from the load sequence
                break

            if iz < ir:
                # branch is fully part of the initial curve, case "Memory 1"
                current_point = self._handle_case_b(current_point)
                # do not further process this load
                break

            # here we have iz > ir:
            previous_point_0 = self._residuals[-2]
            previous_point_1 = self._residuals[-1]

            # is the current load extent smaller than the last one?
            current_load_extent = np.abs(current_load_representative-previous_point_1.load_representative)
            previous_load_extent = np.abs(previous_point_1.load_representative-previous_point_0.load_representative)
            # yes
            if current_load_extent < previous_load_extent-1e-12:
                current_point = self._handle_case_c_i(current_point=current_point, previous_point_1=previous_point_1)

                # continue with the next load value
                break

            # no -> we have a new hysteresis
            recording_lists = self._handle_case_c_ii(
                recording_lists=recording_lists, previous_point_0=previous_point_0, previous_point_1=previous_point_1
            )

            iz -= 2

            # if the points of the hysteresis lie fully inside the seen range of loads, i.e.,
            # the turn points are smaller than the maximum turn point so far
            # (this happens usually in the second run of the HCM algorithm)
            if np.abs(previous_point_0.load_representative) < load_max_seen-1e-12 and np.abs(previous_point_1.load_representative) < load_max_seen-1e-12:
                # case "Memory 2", "c) ii B"
                # the primary branch is not yet reached, continue processing residual loads, potentially
                # closing even more hysteresis

                self._hcm_message += ","

                # add a discontinuity marker
                self._hcm_point_history.append(("discontinuity", None, self._hysteresis_index))
                continue

            # case "Memory 1", "c) ii A"
            # The last hysteresis saw load values equal to the previous maximum.
            # (Higher values cannot happen here.)
            # The load curve follows again the primary path.
            # No further hystereses can be closed, continue with the next load value.
            # Previously, iz was decreased by 2 and will be increased by 1 at the end of this loop ,
            # effectively `iz = iz - 1` as described on p.70

            # Proceed on primary path for the rest, which was not part of the closed hysteresis
            current_point = self._proceed_on_primary_branch(current_point)

            # store strain values, this is for the FKM nonlinear roughness & surface layer algorithm, which adds residual stresses in another pass of the HCM algorithm
            self._strain_values.append(current_point.strain.values[0])

            # count number of strain values in the first run of the HCM algorithm
            if self._run_index == 1:
                self._n_strain_values_first_run += 1
            break

        return current_point, iz, ir, recording_lists

    def _handle_case_c_ii(self, *, recording_lists, previous_point_0, previous_point_1):
        """ Handle case c) ii. in the HCM algorithm, which detects a new hysteresis."""

        self._hcm_message += f" case c) ii., detected full hysteresis"

        epsilon_min = previous_point_0.strain if previous_point_0.strain.values[0] < previous_point_1.strain.values[0] else previous_point_1.strain
        epsilon_max = previous_point_0.strain if previous_point_0.strain.values[0] > previous_point_1.strain.values[0] else previous_point_1.strain

        [_loads_min, _loads_max, _S_min, _S_max, _epsilon_min, _epsilon_max, _epsilon_min_LF,
         _epsilon_max_LF, _is_closed_hysteresis, _is_zero_mean_stress_and_strain, _debug_output] = recording_lists

        # consume the last two loads, process this hysteresis
        current_load_min = previous_point_0.load if previous_point_0.load.values[0] < previous_point_1.load.values[0] else previous_point_1.load
        _loads_min = pd.concat([_loads_min, current_load_min])
        current_load_max = previous_point_0.load if previous_point_0.load.values[0] > previous_point_1.load.values[0] else previous_point_1.load
        _loads_max = pd.concat([_loads_max, current_load_max])

        current_S_min = previous_point_0.stress if previous_point_0.stress.values[0] < previous_point_1.stress.values[0] else previous_point_1.stress
        _S_min = pd.concat([_S_min, current_S_min])
        current_S_max = previous_point_0.stress if previous_point_0.stress.values[0] > previous_point_1.stress.values[0] else previous_point_1.stress
        _S_max = pd.concat([_S_max, current_S_max])

        _epsilon_min = pd.concat([_epsilon_min, epsilon_min])
        _epsilon_max = pd.concat([_epsilon_max, epsilon_max])
        _epsilon_min_LF = pd.concat([_epsilon_min_LF, self._epsilon_min_LF])
        _epsilon_max_LF = pd.concat([_epsilon_max_LF, self._epsilon_max_LF])
        _is_closed_hysteresis.append(True)
        _is_zero_mean_stress_and_strain.append(False)       # do not force the mean stress and strain to be zero
        # save point for the plotting utility / `interpolated_stress_strain_data` method
        # The hysteresis goes: previous_point_0 -> previous_point_1 -> previous_point_0.
        # previous_point_0,previous_point_1 are already logged, now store only previous_point_0 again to visualize the closed hysteresis
        self._hcm_point_history.append(("secondary", previous_point_0, self._hysteresis_index))

        self._hysteresis_index += 1         # increment the hysteresis counter, only needed for the `interpolated_stress_strain` method which helps in plotting the hystereses

        # remove the last two loads from the list of residual loads
        self._residuals.pop()
        self._residuals.pop()

        return [_loads_min, _loads_max, _S_min, _S_max, _epsilon_min, _epsilon_max, _epsilon_min_LF,
                _epsilon_max_LF, _is_closed_hysteresis, _is_zero_mean_stress_and_strain, _debug_output]

    def _handle_case_c_i(self, *, current_point, previous_point_1):
        """Handle case c) i. of the HCM algorithm."""

        self._hcm_message += f" case c) i."

        # yes -> we are on a new secondary branch, there is no new hysteresis to be closed with this
        current_point = self._proceed_on_secondary_branch(previous_point_1, current_point)

        # store strain values, this is for the FKM nonlinear roughness & surface layer algorithm, which adds residual stresses in another pass of the HCM algorithm
        self._strain_values.append(current_point.strain.values[0])

        # count number of strain values in the first run of the HCM algorithm
        if self._run_index == 1:
            self._n_strain_values_first_run += 1
        return current_point

    def _handle_case_b(self, current_point):
        """ Handle case b) of the HCM algorithm.
        The branch is fully part of the initial curve, case "Memory 1"
        """

        self._hcm_message += f" case b)"

        # compute stress and strain of the current point
        current_point = self._proceed_on_primary_branch(current_point)

        # store strain values, this is for the FKM nonlinear roughness & surface layer algorithm, which adds residual stresses in another pass of the HCM algorithm
        self._strain_values.append(current_point.strain.values[0])

        # count number of strain values in the first run of the HCM algorithm
        if self._run_index == 1:
            self._n_strain_values_first_run += 1

        return current_point

    def _handle_case_a_ii(self, *, current_point, previous_point):
        """Handle the case a) ii. in the HCM algorithm."""

        self._hcm_message += f" case a) ii."

        # secondary branch
        current_point = self._proceed_on_secondary_branch(previous_point, current_point)

        # store strain values, this is for the FKM nonlinear roughness & surface layer algorithm, which adds residual stresses in another pass of the HCM algorithm
        self._strain_values.append(current_point.strain.values[0])

        # count number of strain values in the first run of the HCM algorithm
        if self._run_index == 1:
            self._n_strain_values_first_run += 1

        return current_point

    def _handle_case_a_i(self, *, current_point, previous_point, recording_lists):
        """Handle the case a) i. in the HCM algorithm where
        the memory 3 effect is considered."""

        self._hcm_message += f" case a) i., detected half counted hysteresis"

        # case "Memory 3"
        # the first part is still on the secondary branch, the second part is on the primary branch
        # split these two parts

        # the secondary branch corresponds to the load range [L, -L], where L is the previous load,
        # which is named L_{q-1} in the FKM document
        flipped_previous_point = self._HCM_Point(load=-previous_point.load)
        flipped_previous_point = self._proceed_on_secondary_branch(previous_point, flipped_previous_point)

        # the primary branch is the rest
        current_point = self._proceed_on_primary_branch(current_point)

        [_loads_min, _loads_max, _S_min, _S_max, _epsilon_min, _epsilon_max, _epsilon_min_LF,
         _epsilon_max_LF, _is_closed_hysteresis, _is_zero_mean_stress_and_strain, _debug_output] = recording_lists
        _loads_min = pd.concat([_loads_min, -abs(previous_point.load)])
        _loads_max = pd.concat([_loads_max, abs(previous_point.load)])
        _S_min = pd.concat([_S_min, -abs(previous_point.stress)])
        _S_max = pd.concat([_S_max, abs(previous_point.stress)])
        _epsilon_min = pd.concat([_epsilon_min, -abs(previous_point.strain)])
        _epsilon_max = pd.concat([_epsilon_max, abs(previous_point.strain)])
        _epsilon_min_LF = pd.concat([_epsilon_min_LF, self._epsilon_min_LF])
        _epsilon_max_LF = pd.concat([_epsilon_max_LF, self._epsilon_max_LF])
        _is_closed_hysteresis.append(False)             # the hysteresis is not fully closed and will be considered half damage
        _is_zero_mean_stress_and_strain.append(True)    # force the mean stress and strain to be zero

        # store strain values, this is for the FKM nonlinear roughness & surface layer algorithm, which adds residual stresses in another pass of the HCM algorithm
        self._strain_values.append(current_point.strain.values[0])

        # count number of strain values in the first run of the HCM algorithm
        if self._run_index == 1:
            self._n_strain_values_first_run += 1

        # A note on _is_zero_mean_stress_and_strain: the FKM document specifies zero mean stress and strain in the current case,
        # sigma_m=0, and epsilon_m=0 (eq. (2.9-52, 2.9-53)).
        # Due to rounding errors as a result of the binning (de: Klassierung), the sigma_m and epsilon_m values are
        # normally not zero. The FKM

        self._hysteresis_index += 1         # increment the hysteresis counter, only needed for the `interpolated_stress_strain` method which helps in plotting the hystereses

        return (
            current_point,
            [
                _loads_min, _loads_max, _S_min, _S_max, _epsilon_min, _epsilon_max, _epsilon_min_LF,
                _epsilon_max_LF, _is_closed_hysteresis, _is_zero_mean_stress_and_strain, _debug_output
            ]
        )

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


class FKMNonlinearHysteresisPlotter:

    def __init__(self, hcm_point_history, ramberg_osgood_relation):
        self._hcm_point_history = hcm_point_history
        self._ramberg_osgood_relation = ramberg_osgood_relation

    def interpolated_stress_strain_data(self, *, n_points_per_branch=100, only_hystereses=False):
        """Return points on the traversed hysteresis curve, mainly intended for plotting.
        The curve including all hystereses, primary and secondary branches is sampled
        at a fixed number of points within each hysteresis branch.
        These points can be used for plotting.

        The intended use is to generate plots as follows:

        .. code:: python

            fkm_nonlinear_detector.process_hcm_first(...)
            sampling_parameter = 100    # choose larger for smoother plot or smaller for lower runtime
            plotting_data = detector.interpolated_stress_strain_data(n_points_per_branch=sampling_parameter)

            strain_values_primary = plotting_data["strain_values_primary"]
            stress_values_primary = plotting_data["stress_values_primary"]
            hysteresis_index_primary = plotting_data["hysteresis_index_primary"]
            strain_values_secondary = plotting_data["strain_values_secondary"]
            stress_values_secondary = plotting_data["stress_values_secondary"]
            hysteresis_index_secondary = plotting_data["hysteresis_index_secondary"]

            plt.plot(strain_values_primary, stress_values_primary, "g-", lw=3)
            plt.plot(strain_values_secondary, stress_values_secondary, "b-.", lw=1)

        Parameters
        ----------
        n_points_per_branch : int, optional
            How many sampling points to use per hysteresis branch, default 100.
            A larger value values means smoother curves but longer runtime.

        only_hystereses : bool, optional
            Default ``False``. If only graphs of the closed hystereses should be output.
            Note that this does not work for hysteresis that have multiple smaller hysterseses included.

        Returns
        -------
        plotting_data : dict
            A dict with the following keys:

            * "strain_values_primary"
            * "stress_values_primary"
            * "hysteresis_index_primary"
            * "strain_values_secondary"
            * "stress_values_secondary"
            * "hysteresis_index_secondary"

            The values are lists of strains and stresses of the points on the stress-strain curve,
            separately for primary and secondary branches. The lists contain nan values whenever the
            curve of the *same* branch is discontinuous. This allows to plot the entire curve
            with different colors for primary and secondary branches.

            The entries for hysteresis_index_primary and hysteresis_index_secondary are the row
            indices into the collective DataFrame returned by the recorder. This allows, e.g.,
            to separate the output of multiple runs of the HCM algorithm or to plot the traversed
            paths on the stress-strain diagram for individual steps of the algorithm.

        """

        """self._hcm_point_history contains all traversed points:
        It is a list of tuples (type, hcm_point, hysteresis_index), e.g., [ ("primary", hcm_point, 0), ("secondary", hcm_point, 1), ...]
        where the type is one of {"primary", "secondary"} and indicates the hysteresis branch up to the current point
        and the index is the hysteresis number to which the points belong. """

        strain_values_primary = []
        stress_values_primary = []
        hysteresis_index_primary = []
        strain_values_secondary = []
        stress_values_secondary = []
        hysteresis_index_secondary = []

        previous_point = FKMNonlinearDetector._HCM_Point(stress=0, strain=0, load=0)
        previous_point._stress = pd.Series(0)
        previous_point._strain = pd.Series(0)
        previous_type = "primary"
        previous_is_direction_up = None
        last_secondary_start_point = None
        is_direction_up = None

        # split primary parts if necessary
        self._split_primary_parts(previous_point)

        # determine which points are part of closed hysteresis and which are only part
        # of other parts in the stress-strain diagram
        point_is_part_of_closed_hysteresis = self._determine_point_is_part_of_closed_hysteresis()

        previous_point = FKMNonlinearDetector._HCM_Point(strain=0)
        previous_point._stress = pd.Series(0)
        previous_point._strain = pd.Series(0)

        # iterate over all previously stored points of the curve
        for (type, hcm_point, hysteresis_index), is_part_of_closed_hysteresis in zip(self._hcm_point_history, point_is_part_of_closed_hysteresis):

            # determine current direction ("upwards"/"downwards" in stress direction) of the hysteresis branch to be plotted
            if hcm_point is not None and previous_point is not None:
                is_direction_up = hcm_point.stress.iloc[0] - previous_point.stress.iloc[0] > 0

            # depending on branch type, compute interpolated points on the branch
            if type == "primary":
                self._handle_primary_branch(n_points_per_branch=n_points_per_branch, only_hystereses=only_hystereses,
                    strain_values_primary=strain_values_primary, stress_values_primary=stress_values_primary,
                    hysteresis_index_primary=hysteresis_index_primary, strain_values_secondary=strain_values_secondary,
                    stress_values_secondary=stress_values_secondary, previous_point=previous_point, previous_type=previous_type,
                    type=type, hcm_point=hcm_point, hysteresis_index=hysteresis_index,
                    is_part_of_closed_hysteresis=is_part_of_closed_hysteresis)

                last_secondary_start_point = None

            elif type == "secondary":
                hcm_point, secondary_start_point = self._handle_secondary_branch(
                    n_points_per_branch=n_points_per_branch, only_hystereses=only_hystereses,
                    strain_values_primary=strain_values_primary, stress_values_primary=stress_values_primary,
                    strain_values_secondary=strain_values_secondary, stress_values_secondary=stress_values_secondary,
                    hysteresis_index_secondary=hysteresis_index_secondary, previous_point=previous_point,
                    previous_type=previous_type, previous_is_direction_up=previous_is_direction_up,
                    last_secondary_start_point=last_secondary_start_point, hcm_point=hcm_point, hysteresis_index=hysteresis_index,
                    is_part_of_closed_hysteresis=is_part_of_closed_hysteresis, is_direction_up=is_direction_up)

                if previous_type == 'primary':
                    last_secondary_start_point = secondary_start_point
                elif previous_type == 'discontinuity':
                    last_secondary_start_point = hcm_point

            elif type == "discontinuity":

                # if the option "only_hystereses" is set, only output point if it is part of a closed hysteresis
                if is_part_of_closed_hysteresis or not only_hystereses:

                    stress_values_secondary.append(np.nan)
                    strain_values_secondary.append(np.nan)
                    hysteresis_index_secondary.append(hysteresis_index)

                previous_type = type
                continue

            previous_point = hcm_point
            previous_type = type
            previous_is_direction_up = is_direction_up

        result = {
            "strain_values_primary": np.array(strain_values_primary),
            "stress_values_primary": np.array(stress_values_primary),
            "hysteresis_index_primary": np.array(hysteresis_index_primary),
            "strain_values_secondary": np.array(strain_values_secondary),
            "stress_values_secondary": np.array(stress_values_secondary),
            "hysteresis_index_secondary": np.array(hysteresis_index_secondary)
        }
        return result

    def _handle_secondary_branch(self, *, n_points_per_branch, only_hystereses, strain_values_primary, stress_values_primary,
        strain_values_secondary, stress_values_secondary, hysteresis_index_secondary, previous_point,
        previous_type, previous_is_direction_up, last_secondary_start_point, hcm_point, hysteresis_index,
        is_part_of_closed_hysteresis, is_direction_up):

        secondary_start_point = None

        # if the option "only_hystereses" is set, only output point if it is part of a closed hysteresis
        if is_part_of_closed_hysteresis or not only_hystereses:
            # whenever a new segment of the secondary branch starts,
            # add the previous point as starting point
            if previous_type == "primary":
                stress_values_secondary.append(np.nan)
                strain_values_secondary.append(np.nan)
                hysteresis_index_secondary.append(hysteresis_index)

                if stress_values_primary:
                    stress_values_secondary.append(stress_values_primary[-1])
                    strain_values_secondary.append(strain_values_primary[-1])
                    hysteresis_index_secondary.append(hysteresis_index)

            # determine starting point of the current secondary branch
            # After hanging hystereses that consist entirely of secondary branches, the line continues on a previous secondary branch
            # Such case is detected if the previous direction up or downwards (from the hanging hystereses) is the same as the current direction (continuing after hanging hysteresis)
            if previous_is_direction_up == is_direction_up and last_secondary_start_point is not None:
                secondary_start_point = last_secondary_start_point
            else:
                secondary_start_point = previous_point

            new_points_stress = []
            new_points_strain = []

            # iterate over sampling point within current curve segment
            for stress in np.linspace(previous_point.stress, hcm_point.stress, n_points_per_branch):
                # compute point on secondary branch
                delta_stress = stress - secondary_start_point.stress
                delta_strain = self._ramberg_osgood_relation.delta_strain(delta_stress)

                stress = secondary_start_point._stress + delta_stress
                strain = secondary_start_point._strain + delta_strain

                new_points_stress.append(stress.iloc[0])
                new_points_strain.append(strain.iloc[0])

            # if the hysteresis ends on the primary path, then the current assumption that we only need to plot the secondary branch is incorrect.
            # In that case, the end points are not equal, do not output any curve then.

            # If the end points are equal
            if np.isclose(stress, hcm_point.stress):
                stress_values_secondary += new_points_stress
                strain_values_secondary += new_points_strain
                hysteresis_index_secondary += [hysteresis_index] * len(new_points_strain)

            # if the end points are not equal (see explanation above)
            else:
                # reuse the previous point for the next part of the graph
                hcm_point = previous_point

        return hcm_point, secondary_start_point

    def _handle_primary_branch(self, *, n_points_per_branch, only_hystereses, strain_values_primary, stress_values_primary,
        hysteresis_index_primary, strain_values_secondary, stress_values_secondary, previous_point, previous_type, type,
        hcm_point, hysteresis_index, is_part_of_closed_hysteresis):

        # if the option "only_hystereses" is set, only output point if it is part of a closed hysteresis
        if is_part_of_closed_hysteresis or not only_hystereses:
            # whenever a new segment of the primary branch starts,
            # add the previous point as starting point
            if previous_type != type:
                stress_values_primary.append(np.nan)
                strain_values_primary.append(np.nan)
                hysteresis_index_primary.append(hysteresis_index)
                stress_values_primary.append(stress_values_secondary[-1])
                strain_values_primary.append(strain_values_secondary[-1])
                hysteresis_index_primary.append(hysteresis_index)

            # iterate over sampling point within current curve segment
            for stress in np.linspace(previous_point.stress, hcm_point.stress, n_points_per_branch):
                # compute point on primary branch
                strain = self._ramberg_osgood_relation.strain(stress)
                stress_values_primary.append(stress[0])
                strain_values_primary.append(strain[0])
                hysteresis_index_primary.append(hysteresis_index)

    def _determine_point_is_part_of_closed_hysteresis(self):
        """Determine which points are part of a closed hysteresis"""
        point_is_part_of_closed_hysteresis = []
        previous_index = -1
        first_point_set = False

        for (_, _, index) in reversed(self._hcm_point_history):
            if index != previous_index:
                point_is_part_of_closed_hysteresis.insert(0, True)
                first_point_set = True

            elif first_point_set:
                first_point_set = False
                point_is_part_of_closed_hysteresis.insert(0, True)
            else:
                point_is_part_of_closed_hysteresis.insert(0, False)

            previous_index = index

        return point_is_part_of_closed_hysteresis

    def _split_primary_parts(self, previous_point):
        """Adjust the _hcm_point_history, split parts on the primary branch
        that appear for positive residual stresses."""
        old_hcm_point_history = self._hcm_point_history.copy()
        largest_abs_stress_seen = pd.Series(0)
        largest_abs_strain = 0
        self._hcm_point_history = []

        for (type, hcm_point, hysteresis_index) in old_hcm_point_history:
            if type != "primary":
                self._hcm_point_history.append((type, hcm_point, hysteresis_index))
                previous_point = hcm_point
                continue

            if all(previous_point.stress * hcm_point.stress < 0):
                sign = np.sign(hcm_point.stress)
                intermediate_point = FKMNonlinearDetector._HCM_Point(
                    strain=sign*largest_abs_strain, stress=sign*largest_abs_stress_seen
                )
                self._hcm_point_history.append(("secondary", intermediate_point, hysteresis_index))
                self._hcm_point_history.append(("primary", hcm_point, hysteresis_index))
            else:
                self._hcm_point_history.append((type, hcm_point, hysteresis_index))

            if all(abs(hcm_point.stress.values) > largest_abs_stress_seen.values):
                largest_abs_stress_seen = abs(hcm_point.stress)
                largest_abs_strain = abs(hcm_point.strain)

            previous_point = hcm_point

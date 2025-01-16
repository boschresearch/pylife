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

'''
Meanstress routines
===================

Mean stress transformation methods
----------------------------------

* FKM Goodman
* Five Segment Correction

'''

__author__ = "Johannes Mueller, Lena Rapp"
__maintainer__ = "Johannes Mueller"

from collections.abc import Iterable
import operator as op

import numpy as np
import pandas as pd

from pylife import PylifeSignal, Broadcaster
import pylife.stress.collective as CL


@pd.api.extensions.register_series_accessor('haigh_diagram')
class HaighDiagram(PylifeSignal):
    """Model for a Haigh diagram in order to perform meanstress transformations.

    A Haigh diagram a set of meanstress sensitivity slopes $M$ that is changing
    with the R-values.  The values of the ```pd.Series`` represents that slopes
    $M$ and the `pd.IntervalIndex` represents the R-ranges.
    """

    @classmethod
    def from_dict(cls, segments_dict):
        """Create a Haigh diagram from a dict.

        Parameters
        ----------
        segments_dict : dict
            dict resolving the R-value intervals to the meanstress slope

        Example
        -------
        >>> hd = HaighDiagram.from_dict({
        ...    (1.0, np.inf): 0.0,
        ...    (-np.inf, 0.0): 0.5,
        ...    (0.0, 1.0): 0.167
        ... })

        sets up a FKM Goodman like Haigh diagram.
        """
        vals = np.array(list(segments_dict.values()))
        idx = pd.IntervalIndex.from_tuples(list(segments_dict.keys()), name='R')
        return HaighDiagram(pd.Series(vals, index=idx))

    @classmethod
    def fkm_goodman(cls, haigh_fkm_goodman):
        """Create a Haigh diagram according to FKM Goodman.

        Parameters
        ----------
        haigh_fkm_goodman : pd.Series or pd.DataFrame
            a series containing one or a dataframe containing multiple values for
            `M` and optionally `M2`.

        The Haigh diagram according to FKM Goodman comes with the slope ``M``
        which is valid between ``R==-inf`` and ``R==0``.  Beyond ``R==0`` the slope
        is ``M2` if ``M2`` is given or ``M/3`` if not.
        """
        if 'M2' not in haigh_fkm_goodman:
            haigh_fkm_goodman['M2'] = haigh_fkm_goodman['M'] / 3.0

        M = haigh_fkm_goodman.M
        M2 = haigh_fkm_goodman.M2
        interval_index = pd.IntervalIndex.from_tuples([(1.0, np.inf), (-np.inf, 0.0), (0.0, 1.)], name='R')

        if isinstance(haigh_fkm_goodman, pd.Series):
            haigh_index = interval_index
            dummy_index = pd.Index([0, 1, 2], name='R')
        else:
            haigh_frame, _ = Broadcaster(haigh_fkm_goodman.index.to_frame()).broadcast(interval_index.to_frame())
            haigh_index = haigh_frame.index
            dummy_index = pd.Index([0, 1, 2] * len(haigh_fkm_goodman), name='R')

        haigh = pd.Series(0.0, index=dummy_index)

        R_index = haigh.index.get_level_values('R')

        haigh.iloc[R_index.get_indexer_for([1])] = M
        haigh.iloc[R_index.get_indexer_for([2])] = M2

        haigh.index = haigh_index
        return cls(haigh)

    @classmethod
    def five_segment(cls, five_segment_haigh_diagram):
        """Create a five segment slope Haigh diagram.

        Parameters
        ----------
        five_segment_haigh_diagram : :class:`pandas.Series` or :class:`pandas.DataFrame`
            The five segment meanstress slope data.

        Notes
        -----
        ``five_segment_hagih_diagram`` has to provide the following keys:
            * ``M0``: the mean stress sensitivity between ``R==-inf`` and ``R==0``
            * ``M1``: the mean stress sensitivity between ``R==0`` and ``R==R12``
            * ``M2``: the mean stress sensitivity betwenn ``R==R12`` and ``R==R23``
            * ``M3``: the mean stress sensitivity between ``R==R23`` and ``R==1``
            * ``M4``: the mean stress sensitivity beyond ``R==1``
            * ``R12``: R-value between ``M1`` and ``M2``
            * ``R23``: R-value between ``M2`` and ``M3``
        """
        was_series = isinstance(five_segment_haigh_diagram, pd.Series)

        if was_series:
            five_segment_haigh_diagram = pd.DataFrame(five_segment_haigh_diagram).T

        index_names = five_segment_haigh_diagram.index.names + ['R']

        def make_index(h):
            orig_index = [h.name] if not isinstance(h.name, Iterable) else list(h.name)

            return pd.MultiIndex.from_tuples([
                tuple(orig_index + [pd.Interval(1.0, np.inf)]),
                tuple(orig_index + [pd.Interval(-np.inf, 0.)]),
                tuple(orig_index + [pd.Interval(0., h.R12)]),
                tuple(orig_index + [pd.Interval(h.R12, h.R23)]),
                tuple(orig_index + [pd.Interval(h.R23, 1.)]),
            ], names=index_names).to_frame()

        haigh_index = pd.concat(list(five_segment_haigh_diagram.apply(make_index, axis=1))).index

        haigh = pd.Series(0.0, index=haigh_index)

        R_index = haigh.index.get_level_values('R')
        h, _ = Broadcaster(haigh).broadcast(five_segment_haigh_diagram)

        M4_locs = R_index.get_indexer_for([pd.Interval(1.0, np.inf)])
        M0_locs = R_index.get_indexer_for([pd.Interval(-np.inf, 0.)])
        M1_locs = R_index.get_indexer_for([pd.Interval(0., R12) for R12 in h.R12])
        M2_locs = R_index.get_indexer_for([pd.Interval(R12, R23) for R12, R23 in zip(h.R12, h.R23)])
        M3_locs = R_index.get_indexer_for([pd.Interval(R23, 1.) for R23 in h.R23])

        haigh.iloc[M4_locs] = h.M4.iloc[M4_locs]
        haigh.iloc[M0_locs] = h.M0.iloc[M0_locs]
        haigh.iloc[M1_locs] = h.M1.iloc[M1_locs]
        haigh.iloc[M2_locs] = h.M2.iloc[M2_locs]
        haigh.iloc[M3_locs] = h.M3.iloc[M3_locs]

        if was_series:
            haigh = haigh.xs(0)

        return cls(haigh)

    def transform(self, cycles, R_goal):
        """Transform a load collective to defined R-value.

        Parameters
        ----------
        cycles : :class:`pd.Series` accepted by class:``LoadCollective` or class:`LoadHistogram``
            The load collective

        Returns
        -------
        transformed_cycles : :class:`pd.Series`
            The transformed cycles
        """
        cycles, obj = self.broadcast(cycles, droplevel=['R'])

        transformer = _SegmentTransformer(cycles, obj, self._R_index, R_goal)

        for interval in transformer.segments_left_from_R_goal():
            interval_boundary = interval.right if interval.right < 1.0 else interval.left
            transformer.transform_cycles_in_interval(interval, interval_boundary)

        for interval in transformer.segments_right_from_R_goal():
            transformer.transform_cycles_in_interval(interval, interval.left)

        for interval in transformer.segments_containing_R_goal():
            transformer.transform_cycles_in_interval(interval, R_goal)

        transfomed_cycles = transformer.transformed_cycles
        res = pd.DataFrame({
            'range': 2. * transfomed_cycles.amplitude,
            'mean': transfomed_cycles.amplitude * ((1.+transfomed_cycles.R)/(1.-transfomed_cycles.R)).fillna(-1.0)
        }, index=cycles.index)

        return res

    def _validate(self):

        def has_gaps(idx):
            if len(idx) <= 1:
                return False
            return (pd.DataFrame({'l': idx.left[1:], 'r': idx.right[:-1]})
                    .apply(lambda r: r.l != r.r and not (r.l == -np.inf and r.r == np.inf), axis=1)
                    .any())

        self._R_index = self._find_R_index()

        if self._check_if_R_index(lambda idx: idx.is_overlapping):
            raise AttributeError("The intervals of the 'R' IntervalIndex must not overlap.")

        if self._check_if_R_index(has_gaps):
            raise AttributeError("The intervals of the 'R' IntervalIndex must not have gaps.")

    def _find_R_index(self):
        if 'R' not in self._obj.index.names:
            raise AttributeError("A Haigh Diagram needs an index level 'R'.")
        if isinstance(self._obj.index, pd.MultiIndex):
            R_index = self._obj.index.unique('R')
        else:
            R_index = self._obj.index
        if not isinstance(R_index, pd.IntervalIndex):
            raise AttributeError("The 'R' index must be an IntervalIndex.")
        return R_index

    def _check_if_R_index(self, check_func):
        if isinstance(self._obj.index, pd.IntervalIndex):
            return check_func(self._obj.index)

        all_but_R = [n or 0 for n in self._obj.index.names if n != 'R']

        return (
            self
            ._obj.index.to_frame(index=False)
            .groupby(all_but_R)
            .apply(lambda g: check_func(g.set_index('R').index), include_groups=False)
            .any()
        )


class _SegmentTransformer:

    def __init__(self, cycles, haigh, R_segments, R_goal):
        rf = cycles.load_collective
        self.transformed_cycles = pd.DataFrame({
            'amplitude': rf.amplitude,
            'R': rf.R
        }, index=cycles.index)
        self._haigh = haigh
        self._R_index = R_segments
        self._R_goal = R_goal

        self._distances = self._distance_from_R_goal()

    def segments_left_from_R_goal(self):
        return self._distances[self._distances < 0.].sort_values(ascending=True).index

    def segments_right_from_R_goal(self):
        return self._distances[self._distances > 0.].sort_values(ascending=False).index

    def segments_containing_R_goal(self):
        goal_segments = self._R_index.contains(self._R_goal)
        if not goal_segments.any():
            goal_segments = self._R_index.set_closed('left').contains(self._R_goal)

        return self._R_index[goal_segments]

    def _distance_from_R_goal(self):
        def fake_meanstress(R):
            return (1.+R)/(1.-R)

        meanstress = fake_meanstress(self._R_index.mid).fillna(-1.0)
        meanstress_goal = -1.0 if self._R_goal == -np.inf else fake_meanstress(self._R_goal)

        return pd.Series(meanstress.values - meanstress_goal, index=self._R_index)

    def transform_cycles_in_interval(self, interval, R_goal):
        def push_over_flipping_point(R):
            if R == -np.inf and R_goal > 1.0:
                return np.inf
            if R == np.inf and R_goal < 1.0:
                return -np.inf
            return R

        def cycles_in_current_interval():
            R = self.transformed_cycles.R.apply(push_over_flipping_point)
            test_interval = pd.Interval(interval.left, interval.right, closed='both')
            return R.apply(lambda R: R in test_interval)

        def cycles_in_current_segments(in_test_interval, segments_index):
            in_segments_index = pd.Series(False, index=self.transformed_cycles.index)
            in_segments_index[segments_index] = True

            return in_test_interval & in_segments_index

        def meanstress_sensitivity_segments_of_current_interval():
            return self._haigh.xs(interval, level='R')

        def transformed_amplitude():
            rf = self.transformed_cycles.loc[to_shift]
            amp = rf.amplitude
            mean = amp * (1.+rf.R)/(1.-rf.R)
            mean[rf.R == -np.inf] = -amp[rf.R == -np.inf]
            mean[rf.R == 1.0] = -amp[rf.R == 1.0]

            if R_goal == -np.inf:
                trans_amp = (amp + M * mean) / (1. - M)
            else:
                trans_amp = (1. - R_goal) * (amp + M*mean) / (1. - R_goal + M*(1.+R_goal))

            return trans_amp.fillna(0.0)

        to_shift = cycles_in_current_interval()
        if not to_shift.any():
            return

        M = meanstress_sensitivity_segments_of_current_interval()

        to_shift = cycles_in_current_segments(to_shift, M.index)
        if not to_shift.any():
            return

        if R_goal == 1.0:
            R_goal = -np.inf

        self.transformed_cycles.loc[to_shift, 'amplitude'] = transformed_amplitude()
        self.transformed_cycles.loc[to_shift, 'R'] = R_goal


def experimental_mean_stress_sensitivity(sn_curve_R0, sn_curve_Rn1, N_c=np.inf):
    r"""Estimate the mean stress sensitivity from two `FiniteLifeCurve` objects for the same amount of cycles `N_c`.

    The formula for calculation is taken from: "Betriebsfestigkeit", Haibach, 3. Auflage 2006

    Formula (2.1-24):

    .. math::
        M_{\sigma} = {S_a}^{R=-1}(N_c) / {S_a}^{R=0}(N_c) - 1

    Alternatively the mean stress sensitivity is calculated based on both SD values
    (if N_c is not given).

    Parameters
    ----------
    sn_curve_R0: pylife.strength.sn_curve.FiniteLifeCurve
        Instance of FiniteLifeCurve for R == 0
    sn_curve_Rn1: pylife.strength.sn_curve.FiniteLifeCurve
        Instance of FiniteLifeCurve for R == -1
    N_c: float, (default=np.inf)
        Amount of cycles where the amplitudes should be compared.
        If N_c is higher than a fatigue transition point (ND) for the SN-Curves, SD is taken.
        If N_c is None, SD values are taken as stress amplitudes instead.

    Returns
    -------
    float
        Mean stress sensitivity M_sigma

    Raises
    ------
    ValueError
        if the resulting M_sigma doesn't lie in the range from 0 to 1 a ValueError is raised, as this value would
        suggest higher strength with additional loads.
    """
    S_a_R0 = sn_curve_R0.woehler.basquin_load(N_c) if N_c < sn_curve_R0.ND else sn_curve_R0.SD
    S_a_Rn1 = sn_curve_Rn1.woehler.basquin_load(N_c) if N_c < sn_curve_Rn1.ND else sn_curve_Rn1.SD
    M_sigma = S_a_Rn1 / S_a_R0 - 1
    if not 0 <= M_sigma <= 1:
        raise ValueError("M_sigma: %.2f exceeds the interval [0, 1] which is not plausible." % M_sigma)
    return M_sigma


@pd.api.extensions.register_dataframe_accessor('meanstress_transform')
class MeanstressTransformCollective(CL.LoadCollective):

    def fkm_goodman(self, ms_sens, R_goal):
        hd = HaighDiagram.fkm_goodman(ms_sens)
        res = hd.transform(self._obj, R_goal)
        return res.load_collective

    def five_segment(self, haigh, R_goal):
        hd = HaighDiagram.five_segment(haigh)
        res = hd.transform(self._obj, R_goal)
        return res.load_collective


@pd.api.extensions.register_series_accessor('meanstress_transform')
class MeanstressTransformMatrix(CL.LoadHistogram):

    def _validate(self):
        super()._validate()

        if set(self._obj.index.names).issuperset({'from', 'to'}):
            f = self._obj.index.get_level_values('from').mid
            t = self._obj.index.get_level_values('to').mid
            self._Sa = np.abs(f-t)/2.
            self._Sm = (f+t)/2.
            self._binsize_x = self._obj.index.get_level_values('from').length.min()
            self._binsize_y = self._obj.index.get_level_values('to').length.min()
            self._remaining_names = list(filter(lambda n: n not in ['from', 'to'], self._obj.index.names))
        else:
            self._Sa = self._obj.index.get_level_values('range').mid / 2.
            self._Sm = self._obj.index.get_level_values('mean').mid
            self._binsize_x = self._obj.index.get_level_values('range').length.min()
            self._binsize_y = self._obj.index.get_level_values('mean').length.min()
            self._remaining_names = list(filter(lambda n: n not in ['range', 'mean'], self._obj.index.names))

    def fkm_goodman(self, haigh, R_goal):
        ranges = HaighDiagram.fkm_goodman(haigh).transform(self._obj, R_goal)['range']
        return self._rebin_results(ranges, R_goal).load_collective

    def _rebin_results(self, ranges, R_goal):

        def resulting_intervals():
            ranges_max = ranges.max()
            binsize = np.hypot(self._binsize_x, self._binsize_y) / np.sqrt(2.)
            bincount = int(np.ceil(ranges_max / binsize))
            range_bins = np.linspace(0, ranges_max, bincount+1)
            means_bins = range_bins * (1. + R_goal) / (2.0 * (1. - R_goal))
            range_itv = pd.IntervalIndex.from_breaks(range_bins, name="range")
            means_itv = pd.IntervalIndex.from_breaks(means_bins, name="mean")
            return range_itv, means_itv

        def aggregate_on_projection(projection, itvs, ranges, obj):
            level_values = tuple(projection)
            level_names = list(projection.index)
            ranges = ranges.xs(level_values, level=level_names)
            obj = obj.xs(level_values, level=level_names)
            sums = (
                itvs
                .to_series()
                .apply(lambda iv: sum_intervals(iv, ranges, obj))
            )
            return sums

        def sum_intervals(iv, ranges, obj):
            op_left = op.ge if iv.left == 0.0 else op.gt
            return obj.iloc[op_left(ranges.values, iv.left) & op.le(ranges.values, iv.right)].sum()

        if ranges.shape[0] == 0:
            new_idx = pd.IntervalIndex(pd.interval_range(0.,  0., 0), name='range')
            return pd.Series([], index=new_idx, name='cycles', dtype=np.float64)

        range_itv_idx, means_itg_idx = resulting_intervals()

        if len(self._remaining_names) > 0:
            remaining_idx = self._obj.index.to_frame(index=False).groupby(self._remaining_names).first().index
            result = (
                remaining_idx.to_frame(index=False)
                .apply(lambda p: aggregate_on_projection(p, range_itv_idx, ranges, self._obj), axis=1)
            )
            result.index = remaining_idx
            result.columns = pd.MultiIndex.from_arrays([result.columns, means_itg_idx])
            result = result.stack(['range', 'mean'], future_stack=True).reorder_levels(
                ['range', 'mean'] + self._remaining_names
            )
        else:
            result = range_itv_idx.to_series().apply(lambda iv: sum_intervals(iv, ranges, self._obj))
            result.index = pd.MultiIndex.from_arrays([result.index, means_itg_idx])

        return result


def fkm_goodman(amplitude, meanstress, M, M2, R_goal):
    cycles = pd.DataFrame({
        'range': 2.*amplitude,
        'mean': meanstress
    })

    haigh_fkm_goodman = pd.Series({
        'M': M,
        'M2': M2
    })
    hd = HaighDiagram.fkm_goodman(haigh_fkm_goodman)

    res = hd.transform(cycles, R_goal)
    return res.load_collective.amplitude.to_numpy()


def five_segment_correction(amplitude, meanstress, M0, M1, M2, M3, M4, R12, R23, R_goal):
    ''' Performs a mean stress transformation to R_goal according to the
        Five Segment Mean Stress Correction

    :param Sa: the stress amplitude
    :param Sm: the mean stress
    :param Rgoal: the R-value to transform to
    :param M: the mean stress sensitivity between R=-inf and R=0
    :param M1: the mean stress sensitivity between R=0 and R=R12
    :param M2: the mean stress sensitivity betwenn R=R12 and R=R23
    :param M3: the mean stress sensitivity between R=R23 and R=1
    :param M4: the mean stress sensitivity beyond R=1
    :param R12: R-value between M1 and M2
    :param R23: R-value between M2 and M3

    :returns: the transformed stress range
    '''

    cycles = pd.DataFrame({
        'range': 2.*amplitude,
        'mean': meanstress
    })

    haigh_five_segment = pd.Series({
        'M0': M0,
        'M1': M1,
        'M2': M2,
        'M3': M3,
        'M4': M4,
        'R12': R12,
        'R23': R23
    })

    hd = HaighDiagram.five_segment(haigh_five_segment)
    res = hd.transform(cycles, R_goal)
    return res.load_collective.amplitude.to_numpy()

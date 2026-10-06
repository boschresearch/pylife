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

"""Provide mean stress transformations for load collectives and histograms.

Mean stress transformations convert a cyclic load with stress amplitude
``S_a`` and mean stress ``S_m`` to an equivalent amplitude at a target stress
ratio ``R_goal``. The equivalent cycle is intended to produce the same fatigue
assessment result in an S-N curve calculation.

The module represents mean stress sensitivity by a Haigh diagram: each stress
ratio interval ``R = S_min / S_max`` is assigned a slope ``M``. Convenience
constructors implement the FKM-Goodman and five-segment diagrams, while
pandas accessors apply the transformation to load collectives and rainflow
histograms.

See Also
--------
pylife.strength.meanstress.HaighDiagram : Store piecewise mean stress sensitivities.
pylife.strength.meanstress.MeanstressTransformCollective : Transform load collectives.
pylife.strength.meanstress.MeanstressTransformMatrix : Transform load histograms.
"""

__author__ = "Johannes Mueller, Lena Rapp"
__maintainer__ = "Johannes Mueller"

from collections.abc import Iterable
import operator as op

import numpy as np
import pandas as pd

from pylife import PylifeSignal, Broadcaster
import pylife.stress.collective as CL


@pd.api.extensions.register_series_accessor("haigh_diagram")
class HaighDiagram(PylifeSignal):
    r"""Represent a piecewise Haigh diagram for mean stress correction.

    A Haigh diagram assigns a mean stress sensitivity ``M`` to intervals of
    the stress ratio ``R``. In pyLife it is stored as a
    :class:`pandas.Series` whose values are dimensionless sensitivities and
    whose ``R`` index level is a :class:`pandas.IntervalIndex`. The diagram is
    used to transform stress cycles from their actual ``R`` value to a target
    ``R_goal``.

    Parameters
    ----------
    pandas_obj : pandas.Series
        Series containing mean stress sensitivities ``M`` indexed by the
        supplied ``R`` intervals. Additional index levels may identify several
        diagrams.

    See Also
    --------
    pylife.strength.meanstress.HaighDiagram.fkm_goodman : Create an FKM-Goodman diagram.
    pylife.strength.meanstress.HaighDiagram.five_segment : Create a five-segment diagram.
    pylife.strength.meanstress.HaighDiagram.transform : Transform a load collective.

    Notes
    -----
    For one segment with sensitivity ``M`` the transformation keeps the damage
    equivalent quantity ``S_a + M S_m`` constant. With

    .. math::

        S_m = S_a\,\frac{1 + R}{1 - R},

    an amplitude transformed to ``R_goal`` is obtained from

    .. math::

        S_{a,goal} =
        \frac{(1 - R_{goal})(S_a + M S_m)}
             {1 - R_{goal} + M(1 + R_{goal})}.

    For ``R_goal = -\infty`` the implemented limiting form is

    .. math::

        S_{a,goal} = \frac{S_a + M S_m}{1 - M}.
    """

    @classmethod
    def from_dict(cls, segments_dict):
        """Create a Haigh diagram from interval boundaries and sensitivities.

        Parameters
        ----------
        segments_dict : dict
            Mapping from ``(left, right)`` stress-ratio interval tuples to
            dimensionless mean stress sensitivities ``M``.

        Returns
        -------
        pylife.strength.meanstress.HaighDiagram
            Haigh diagram accessor wrapping a :class:`pandas.Series` indexed by
            the supplied ``R`` intervals.

        Examples
        --------
        >>> from pylife.strength.meanstress import HaighDiagram
        >>> HaighDiagram.from_dict({
        ...    (1.0, np.inf): 0.0,
        ...    (-np.inf, 0.0): 0.5,
        ...    (0.0, 1.0): 0.167
        ... }).to_pandas()
        R
        (1.0, inf]     0.000
        (-inf, 0.0]    0.500
        (0.0, 1.0]     0.167
        dtype: float64
        """
        vals = np.array(list(segments_dict.values()))
        idx = pd.IntervalIndex.from_tuples(list(segments_dict.keys()), name="R")
        return HaighDiagram(pd.Series(vals, index=idx))

    @classmethod
    def fkm_goodman(cls, haigh_fkm_goodman):
        r"""Create an FKM-Goodman Haigh diagram.

        Parameters
        ----------
        haigh_fkm_goodman : pandas.Series or pandas.DataFrame
            Mean stress sensitivity data. It must contain ``M`` for
            ``-inf < R <= 0`` and may contain ``M2`` for ``0 < R <= 1``. If
            ``M2`` is missing, ``M / 3`` is used.

        Returns
        -------
        pylife.strength.meanstress.HaighDiagram
            Haigh diagram with segments ``(1, inf]``, ``(-inf, 0]`` and
            ``(0, 1]``.

        Limitations
        -----------
        The FKM-Goodman correction implemented here assumes the FKM linear
        guideline piecewise slopes: ``M`` for alternating to pulsating
        compression/tension cycles, ``M2`` for tensile mean stresses up to
        ``R = 1``, and ``0`` beyond ``R = 1``. Use :meth:`five_segment` when
        material-specific transition ratios ``R12`` and ``R23`` are available.

        Notes
        -----
        The implemented slopes are

        .. math::

            M(R) =
            \begin{cases}
            0, & 1 < R \\
            M, & -\infty < R \le 0 \\
            M_2, & 0 < R \le 1.
            \end{cases}

        Examples
        --------
        Create a diagram with default ``M2``.

        >>> from pylife.strength.meanstress import HaighDiagram
        >>> HaighDiagram.fkm_goodman(pd.Series({"M": 0.5})).to_pandas()
        R
        (1.0, inf]     0.000000
        (-inf, 0.0]    0.500000
        (0.0, 1.0]     0.166667
        dtype: float64

        Create a diagram with a manual ``M2``.

        >>> from pylife.strength.meanstress import HaighDiagram
        >>> HaighDiagram.fkm_goodman(pd.Series({"M": 0.5, "M2": 0.2})).to_pandas()
        R
        (1.0, inf]     0.0
        (-inf, 0.0]    0.5
        (0.0, 1.0]     0.2
        dtype: float64

        >>> from pylife.strength.meanstress import HaighDiagram
        >>> collective = pd.DataFrame(
        ...     {
        ...         "range": [600.0, 300.0, 500.0],
        ...         "mean": [100.0, 50.0, 80.0],
        ...         "cycles": [1.0, 10.0, 100.0],
        ...     }
        ... )
        >>> HaighDiagram.fkm_goodman(pd.Series({"M": 0.5})).transform(collective, -1.0)
           range  mean  cycles
        0  700.0   0.0     1.0
        1  350.0   0.0    10.0
        2  580.0   0.0   100.0

        """
        if "M2" not in haigh_fkm_goodman:
            haigh_fkm_goodman["M2"] = haigh_fkm_goodman["M"] / 3.0

        M = haigh_fkm_goodman.M
        M2 = haigh_fkm_goodman.M2
        interval_index = pd.IntervalIndex.from_tuples(
            [(1.0, np.inf), (-np.inf, 0.0), (0.0, 1.0)], name="R"
        )

        if isinstance(haigh_fkm_goodman, pd.Series):
            haigh_index = interval_index
            dummy_index = pd.Index([0, 1, 2], name="R")
        else:
            haigh_frame, _ = Broadcaster(haigh_fkm_goodman.index.to_frame()).broadcast(
                interval_index.to_frame()
            )
            haigh_index = haigh_frame.index
            dummy_index = pd.Index([0, 1, 2] * len(haigh_fkm_goodman), name="R")

        haigh = pd.Series(0.0, index=dummy_index)

        R_index = haigh.index.get_level_values("R")

        haigh.iloc[R_index.get_indexer_for([1])] = M
        haigh.iloc[R_index.get_indexer_for([2])] = M2

        haigh.index = haigh_index
        return cls(haigh)

    @classmethod
    def five_segment(cls, five_segment_haigh_diagram):
        r"""Create a five-segment Haigh diagram.

        Parameters
        ----------
        five_segment_haigh_diagram : pandas.Series or pandas.DataFrame
            Five-segment mean stress data containing dimensionless
            sensitivities ``M0`` through ``M4`` and transition stress ratios
            ``R12`` and ``R23``.

        Returns
        -------
        pylife.strength.meanstress.HaighDiagram
            Haigh diagram with five stress-ratio segments.

        Notes
        -----
        The five-segment diagram defines

        .. math::

            M(R) =
            \begin{cases}
            M_4, & 1 < R \\
            M_0, & -\infty < R \le 0 \\
            M_1, & 0 < R \le R_{12} \\
            M_2, & R_{12} < R \le R_{23} \\
            M_3, & R_{23} < R \le 1.
            \end{cases}

        ``R12`` and ``R23`` are dimensionless transition ratios and must
        satisfy the physical ordering used by the selected material model.

        Examples
        --------
        >>> from pylife.strength.meanstress import HaighDiagram
        >>> haigh = HaighDiagram.five_segment(
        ...    pd.Series(
        ...        {"M0": 0.5, "M1": 0.25, "M2": 0.125, "M3": 1.0, "M4": -2.0, "R12": 0.2, "R23": 0.8}
        ...    )
        ... )
        >>> haigh.to_pandas()
        R
        (1.0, inf]    -2.000
        (-inf, 0.0]    0.500
        (0.0, 0.2]     0.250
        (0.2, 0.8]     0.125
        (0.8, 1.0]     1.000
        dtype: float64

        >>> from pylife.strength.meanstress import HaighDiagram
        >>> collective = pd.DataFrame(
        ...     {
        ...         "range": [600.0, 300.0, 500.0],
        ...         "mean": [400.0, -150.0, 0.0],
        ...         "cycles": [1.0, 10.0, 100.0],
        ...     }
        ... )
        >>> haigh = HaighDiagram.five_segment(
        ...    pd.Series(
        ...        {"M0": 0.5, "M1": 0.25, "M2": 0.125, "M3": 1.0, "M4": -2.0, "R12": 0.2, "R23": 0.8}
        ...    )
        ... )
        >>> haigh.transform(collective, 0.0)
                range        mean  cycles
        0  640.000000  320.000000     1.0
        1  100.000000   50.000000    10.0
        2  333.333333  166.666667   100.0
        """
        was_series = isinstance(five_segment_haigh_diagram, pd.Series)

        if was_series:
            five_segment_haigh_diagram = pd.DataFrame(five_segment_haigh_diagram).T

        index_names = five_segment_haigh_diagram.index.names + ["R"]

        def make_index(h):
            orig_index = [h.name] if not isinstance(h.name, Iterable) else list(h.name)

            return pd.MultiIndex.from_tuples(
                [
                    tuple(orig_index + [pd.Interval(1.0, np.inf)]),
                    tuple(orig_index + [pd.Interval(-np.inf, 0.0)]),
                    tuple(orig_index + [pd.Interval(0.0, h.R12)]),
                    tuple(orig_index + [pd.Interval(h.R12, h.R23)]),
                    tuple(orig_index + [pd.Interval(h.R23, 1.0)]),
                ],
                names=index_names,
            ).to_frame()

        haigh_index = pd.concat(
            list(five_segment_haigh_diagram.apply(make_index, axis=1))
        ).index

        haigh = pd.Series(0.0, index=haigh_index)

        R_index = haigh.index.get_level_values("R")
        h, _ = Broadcaster(haigh).broadcast(five_segment_haigh_diagram)

        M4_locs = R_index.get_indexer_for([pd.Interval(1.0, np.inf)])
        M0_locs = R_index.get_indexer_for([pd.Interval(-np.inf, 0.0)])
        M1_locs = R_index.get_indexer_for([pd.Interval(0.0, R12) for R12 in h.R12])
        M2_locs = R_index.get_indexer_for(
            [pd.Interval(R12, R23) for R12, R23 in zip(h.R12, h.R23)]
        )
        M3_locs = R_index.get_indexer_for([pd.Interval(R23, 1.0) for R23 in h.R23])

        haigh.iloc[M4_locs] = h.M4.iloc[M4_locs]
        haigh.iloc[M0_locs] = h.M0.iloc[M0_locs]
        haigh.iloc[M1_locs] = h.M1.iloc[M1_locs]
        haigh.iloc[M2_locs] = h.M2.iloc[M2_locs]
        haigh.iloc[M3_locs] = h.M3.iloc[M3_locs]

        if was_series:
            haigh = haigh.xs(0)

        return cls(haigh)

    def transform(self, collective, R_goal):
        """Transform a load collective to a target stress ratio.

        Parameters
        ----------
        collective : pandas.DataFrame
            Load collective to transform. It must contain either ``range`` and
            ``mean`` columns or ``from`` and ``to`` columns. Ranges are stress
            ranges in MPa or another consistent stress unit; means are mean
            stresses in the same unit. Additional columns, for example
            ``cycles``, are copied.
        R_goal : float
            Target stress ratio ``R = S_min / S_max``, dimensionless.

        Returns
        -------
        pandas.DataFrame
            Transformed collective with columns ``range`` for stress range,
            ``mean`` for mean stress, and all non-load columns copied from the
            input.

        Notes
        -----
        Each cycle is moved segment by segment across the Haigh diagram until
        it reaches ``R_goal``. The result uses stress ranges, so the
        transformed range is ``2 * S_a``.

        Examples
        --------
        >>> from pylife.strength.meanstress import HaighDiagram
        >>> collective = pd.DataFrame(
        ...     {
        ...         "from": [300.0, -150.0, -250.0],
        ...         "to": [-300.0, 150.0, 250.0],
        ...         "cycles": [1.0, 10.0, 100.0],
        ...     }
        ... )
        >>> HaighDiagram.from_dict({(-np.inf, np.inf): 0.5}).transform(collective, 0.0)
                range        mean  cycles
        0  400.000000  200.000000     1.0
        1  200.000000  100.000000    10.0
        2  333.333333  166.666667   100.0

        >>> from pylife.strength.meanstress import HaighDiagram
        >>> collective = pd.DataFrame(
        ...     {
        ...         "range": [600.0, 300.0, 500.0],
        ...         "mean": [100.0, 50.0, 80.0],
        ...         "cycles": [1.0, 10.0, 100.0],
        ...     }
        ... )
        >>> HaighDiagram.from_dict({(-np.inf, np.inf): 0.5}).transform(collective, -1.0)
           range  mean  cycles
        0  700.0   0.0     1.0
        1  350.0   0.0    10.0
        2  580.0   0.0   100.0
        """

        broadcasted_coll, haigh = self.broadcast(collective, droplevel=["R"])

        coll = CL.LoadCollective(broadcasted_coll)

        transformer = _SegmentTransformer(coll, haigh, self._R_index, R_goal)

        for interval in transformer.segments_left_from_R_goal():
            interval_boundary = (
                interval.right if interval.right < 1.0 else interval.left
            )
            transformer.transform_cycles_in_interval(interval, interval_boundary)

        for interval in transformer.segments_right_from_R_goal():
            transformer.transform_cycles_in_interval(interval, interval.left)

        for interval in transformer.segments_containing_R_goal():
            transformer.transform_cycles_in_interval(interval, R_goal)

        transformed_cycles = transformer.transformed_cycles
        res = pd.DataFrame(
            {
                "range": 2.0 * transformed_cycles.amplitude,
                "mean": transformed_cycles.amplitude
                * ((1.0 + transformed_cycles.R) / (1.0 - transformed_cycles.R)).fillna(
                    -1.0
                ),
            },
            index=broadcasted_coll.index,
        )

        for col in collective:
            if col not in coll.columns:
                res[col] = collective[col]

        return res

    def _validate(self):

        def has_gaps(idx):
            if len(idx) <= 1:
                return False
            return (
                pd.DataFrame({"l": idx.left[1:], "r": idx.right[:-1]})
                .apply(
                    lambda r: r.l != r.r and not (r.l == -np.inf and r.r == np.inf),
                    axis=1,
                )
                .any()
            )

        self._R_index = self._find_R_index()

        if self._check_if_R_index(lambda idx: idx.is_overlapping):
            raise AttributeError(
                "The intervals of the 'R' IntervalIndex must not overlap."
            )

        if self._check_if_R_index(has_gaps):
            raise AttributeError(
                "The intervals of the 'R' IntervalIndex must not have gaps."
            )

    def _find_R_index(self):
        if "R" not in self._obj.index.names:
            raise AttributeError("A Haigh Diagram needs an index level 'R'.")
        if isinstance(self._obj.index, pd.MultiIndex):
            R_index = self._obj.index.unique("R")
        else:
            R_index = self._obj.index
        if not isinstance(R_index, pd.IntervalIndex):
            raise AttributeError("The 'R' index must be an IntervalIndex.")
        return R_index

    def _check_if_R_index(self, check_func):
        if isinstance(self._obj.index, pd.IntervalIndex):
            return check_func(self._obj.index)

        all_but_R = [n or 0 for n in self._obj.index.names if n != "R"]

        return (
            self._obj.index.to_frame(index=False)
            .groupby(all_but_R)
            .apply(lambda g: check_func(g.set_index("R").index), include_groups=False)
            .any()
        )


class _SegmentTransformer:

    def __init__(self, collective, haigh, R_segments, R_goal):
        self.transformed_cycles = pd.DataFrame(
            {"amplitude": collective.amplitude, "R": collective.R},
            index=collective.to_pandas().index,
        )
        self._haigh = haigh
        self._R_index = R_segments
        self._R_goal = R_goal

        self._distances = self._distance_from_R_goal()

    def segments_left_from_R_goal(self):
        return self._distances[self._distances < 0.0].sort_values(ascending=True).index

    def segments_right_from_R_goal(self):
        return self._distances[self._distances > 0.0].sort_values(ascending=False).index

    def segments_containing_R_goal(self):
        goal_segments = self._R_index.contains(self._R_goal)
        if not goal_segments.any():
            goal_segments = self._R_index.set_closed("left").contains(self._R_goal)

        return self._R_index[goal_segments]

    def _distance_from_R_goal(self):
        def fake_meanstress(R):
            return (1.0 + R) / (1.0 - R)

        meanstress = fake_meanstress(self._R_index.mid).fillna(-1.0)
        meanstress_goal = (
            -1.0 if self._R_goal == -np.inf else fake_meanstress(self._R_goal)
        )

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
            test_interval = pd.Interval(interval.left, interval.right, closed="both")
            return R.apply(lambda R: R in test_interval)

        def cycles_in_current_segments(in_test_interval, segments_index):
            in_segments_index = pd.Series(False, index=self.transformed_cycles.index)
            in_segments_index[segments_index] = True

            return in_test_interval & in_segments_index

        def meanstress_sensitivity_segments_of_current_interval():
            return self._haigh.xs(interval, level="R")

        def transformed_amplitude():
            rf = self.transformed_cycles.loc[to_shift]
            amp = rf.amplitude
            mean = amp * (1.0 + rf.R) / (1.0 - rf.R)
            mean[rf.R == -np.inf] = -amp[rf.R == -np.inf]
            mean[rf.R == 1.0] = -amp[rf.R == 1.0]

            if R_goal == -np.inf:
                trans_amp = (amp + M * mean) / (1.0 - M)
            else:
                trans_amp = (
                    (1.0 - R_goal)
                    * (amp + M * mean)
                    / (1.0 - R_goal + M * (1.0 + R_goal))
                )

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

        self.transformed_cycles.loc[to_shift, "amplitude"] = transformed_amplitude()
        self.transformed_cycles.loc[to_shift, "R"] = R_goal


def experimental_mean_stress_sensitivity(sn_curve_R0, sn_curve_Rn1, N_c=np.inf):
    r"""Estimate mean stress sensitivity from two S-N curves.

    Parameters
    ----------
    sn_curve_R0 : pylife.materiallaws.WoehlerCurve
        Wöhler curve accessor for stress ratio ``R = 0``.
    sn_curve_Rn1 : pylife.materiallaws.WoehlerCurve
        Wöhler curve accessor for stress ratio ``R = -1``.
    N_c : float, optional
        Number of cycles at which the amplitudes are compared. If ``N_c`` is
        greater than or equal to the knee point ``ND`` of a curve, its ``SD``
        value is used. Default is ``numpy.inf``.

    Returns
    -------
    float
        Mean stress sensitivity ``M_sigma``, dimensionless.

    Raises
    ------
    ValueError
        Raised if the resulting sensitivity is outside the physically
        plausible interval ``[0, 1]``.

    Notes
    -----
    Following Haibach [Haibach-Meanstress]_, the sensitivity is estimated as

    .. math::

        M_{\sigma} = \frac{S_a^{R=-1}(N_c)}{S_a^{R=0}(N_c)} - 1.

    References
    ----------
    .. [Haibach-Meanstress] E. Haibach, "Betriebsfestigkeit", Springer-Verlag,
       2006, p. 21.
    """
    S_a_R0 = (
        sn_curve_R0.woehler.basquin_load(N_c)
        if N_c < sn_curve_R0.ND
        else sn_curve_R0.SD
    )
    S_a_Rn1 = (
        sn_curve_Rn1.woehler.basquin_load(N_c)
        if N_c < sn_curve_Rn1.ND
        else sn_curve_Rn1.SD
    )
    M_sigma = S_a_Rn1 / S_a_R0 - 1
    if not 0 <= M_sigma <= 1:
        raise ValueError(
            "M_sigma: %.2f exceeds the interval [0, 1] which is not plausible."
            % M_sigma
        )
    return M_sigma


@pd.api.extensions.register_dataframe_accessor("meanstress_transform")
class MeanstressTransformCollective(CL.LoadCollective):
    """Transform counted load collectives to a target stress ratio.

    The accessor is registered as ``.meanstress_transform`` on
    :class:`pandas.DataFrame` load collectives. The input data must satisfy the
    :mod:`pylife.stress.collective` load collective contract and provide
    either ``range`` and ``mean`` columns or ``from`` and ``to`` columns.

    Parameters
    ----------
    pandas_obj : pandas.DataFrame
        Load collective data passed by the pandas accessor machinery.

    See Also
    --------
    pylife.strength.meanstress.HaighDiagram : Represent the correction diagram.
    pylife.strength.meanstress.MeanstressTransformMatrix : Transform rainflow histograms.
    """

    def fkm_goodman(self, goodman, R_goal):
        """Apply the FKM-Goodman transformation to a load collective.

        Parameters
        ----------
        goodman : pandas.Series or pandas.DataFrame
            Mean stress sensitivity data containing ``M`` and optionally
            ``M2``.
        R_goal : float
            Target stress ratio ``R = S_min / S_max``, dimensionless.

        Returns
        -------
        pylife.stress.collective.LoadCollective
            Transformed load collective accessor. Stress amplitudes and mean
            stresses use the same unit as the input.

        See Also
        --------
        pylife.strength.meanstress.HaighDiagram.fkm_goodman : Create the underlying diagram.

        Examples
        --------
        >>> collective = pd.DataFrame(
        ...     {
        ...         "from": [300.0, -150.0, -250.0],
        ...         "to": [-300.0, 150.0, 250.0],
        ...         "cycles": [1.0, 10.0, 100.0],
        ...     }
        ... )
        >>> collective.meanstress_transform.fkm_goodman(pd.Series({"M": 0.5}), 0.0).amplitude
        0    200.000000
        1    100.000000
        2    166.666667
        Name: amplitude, dtype: float64
        """
        res = HaighDiagram.fkm_goodman(goodman).transform(self._obj, R_goal)
        return res.load_collective

    def five_segment(self, five_segment, R_goal):
        """Apply the five-segment transformation to a load collective.

        Parameters
        ----------
        five_segment : pandas.Series or pandas.DataFrame
            Mean stress sensitivities ``M0`` through ``M4`` and transition
            ratios ``R12`` and ``R23``.
        R_goal : float
            Target stress ratio ``R = S_min / S_max``, dimensionless.

        Returns
        -------
        pylife.stress.collective.LoadCollective
            The transformed load collective. After the meanstress transformation,
            the resulting ``(range, mean)`` interval bins may no longer be
            continuous.

        See Also
        --------
        pylife.strength.meanstress.HaighDiagram.five_segment : Create the underlying diagram.

        Examples
        --------
        >>> collective = pd.DataFrame(
        ...     {
        ...         "range": [600.0, 300.0, 500.0],
        ...         "mean": [400.0, -150.0, 0.0],
        ...         "cycles": [1.0, 10.0, 100.0],
        ...     }
        ... )
        >>> haigh = pd.Series(
        ...    {"M0": 0.5, "M1": 0.25, "M2": 0.125, "M3": 1.0, "M4": -2.0, "R12": 0.2, "R23": 0.8}
        ... )
        >>> transformed = collective.meanstress_transform.five_segment(haigh, 0.0)
        >>> transformed.amplitude
        0    320.000000
        1     50.000000
        2    166.666667
        Name: amplitude, dtype: float64

        >>> transformed.meanstress
        0    320.000000
        1     50.000000
        2    166.666667
        Name: meanstress, dtype: float64
        """
        hd = HaighDiagram.five_segment(five_segment)
        res = hd.transform(self._obj, R_goal)
        return res.load_collective


@pd.api.extensions.register_series_accessor("meanstress_transform")
class MeanstressTransformMatrix(CL.LoadHistogram):
    """Transform rainflow histograms to a target stress ratio.

    The accessor is registered as ``.meanstress_transform`` on
    :class:`pandas.Series` load histograms. Histograms may be indexed by
    ``from`` and ``to`` interval bins or by ``range`` and ``mean`` interval
    bins.

    Parameters
    ----------
    pandas_obj : pandas.Series
        Load histogram data passed by the pandas accessor machinery.

    See Also
    --------
    pylife.strength.meanstress.HaighDiagram : Represent the correction diagram.
    pylife.strength.meanstress.MeanstressTransformCollective : Transform load collectives.
    """

    def _validate(self):
        super()._validate()

        if set(self._obj.index.names).issuperset({"from", "to"}):
            f = self._obj.index.get_level_values("from").mid
            t = self._obj.index.get_level_values("to").mid
            self._Sa = np.abs(f - t) / 2.0
            self._Sm = (f + t) / 2.0
            self._binsize_x = self._obj.index.get_level_values("from").length.min()
            self._binsize_y = self._obj.index.get_level_values("to").length.min()
            self._remaining_names = list(
                filter(lambda n: n not in ["from", "to"], self._obj.index.names)
            )
        else:
            self._Sa = self._obj.index.get_level_values("range").mid / 2.0
            self._Sm = self._obj.index.get_level_values("mean").mid
            self._binsize_x = self._obj.index.get_level_values("range").length.min()
            self._binsize_y = self._obj.index.get_level_values("mean").length.min()
            self._remaining_names = list(
                filter(lambda n: n not in ["range", "mean"], self._obj.index.names)
            )

    def fkm_goodman(self, goodman, R_goal):
        """Apply the FKM-Goodman transformation to a load histogram.

        Parameters
        ----------
        goodman : pandas.Series or pandas.DataFrame
            Mean stress sensitivity data containing ``M`` and optionally
            ``M2``.
        R_goal : float
            Target stress ratio ``R = S_min / S_max``, dimensionless.

        Returns
        -------
        pylife.stress.collective.LoadHistogram
            Transformed load histogram accessor. The resulting ``range`` and
            ``mean`` interval bins may no longer be continuous.

        Warnings
        --------
        The transformed interval bins are geometrically correct for the
        transformed corner points. Rebin only as a later visualization step
        because rebinning may change the numerical damage result.

        See Also
        --------
        pylife.strength.meanstress.HaighDiagram.fkm_goodman : Create the underlying diagram.

        Examples
        --------
        >>> histogram = pd.Series(
        ...     [10, 90, 900, 9000, 90000, 900000],
        ...     index=pd.MultiIndex.from_arrays(
        ...         [
        ...             pd.IntervalIndex.from_arrays(
        ...                 [650, 550, 450, 350, 250, 150], [750, 650, 550, 450, 350, 250]
        ...             ),
        ...             pd.IntervalIndex.from_arrays([0, 0, 0, 0 ,0, 0], [0, 0, 0, 0, 0, 0]),
        ...         ],
        ...         names=["range", "mean"],
        ...     ),
        ...     name="cycles"
        ... )
        >>> transformed = histogram.meanstress_transform.fkm_goodman(pd.Series({"M": 0.0}), 0)
        >>> transformed.amplitude
        range           mean
        (650.0, 750.0]  (325.0, 375.0]    350.0
        (550.0, 650.0]  (275.0, 325.0]    300.0
        (450.0, 550.0]  (225.0, 275.0]    250.0
        (350.0, 450.0]  (175.0, 225.0]    200.0
        (250.0, 350.0]  (125.0, 175.0]    150.0
        (150.0, 250.0]  (75.0, 125.0]     100.0
        Name: amplitude, dtype: float64
        """
        transformer = HaighDiagram.fkm_goodman(goodman)
        return self._perform_transformation(transformer, R_goal)

    def five_segment(self, five_segment, R_goal):
        """Apply the five-segment transformation to a load histogram.

        Parameters
        ----------
        five_segment : pandas.Series or pandas.DataFrame
            Mean stress sensitivities ``M0`` through ``M4`` and transition
            ratios ``R12`` and ``R23``.
        R_goal : float
            Target stress ratio ``R = S_min / S_max``, dimensionless.

        Returns
        -------
        pylife.stress.collective.LoadHistogram
            Transformed load histogram accessor. The resulting ``range`` and
            ``mean`` interval bins may no longer be continuous.

        Warnings
        --------
        The transformed interval bins are geometrically correct for the
        transformed corner points. Rebin only as a later visualization step
        because rebinning may change the numerical damage result.

        See Also
        --------
        pylife.strength.meanstress.HaighDiagram.five_segment : Create the underlying diagram.

        Examples
        --------
        >>> histogram = pd.Series(
        ...     [10, 90, 900, 9000, 90000, 900000],
        ...     index=pd.MultiIndex.from_arrays(
        ...         [
        ...             pd.IntervalIndex.from_arrays(
        ...                 [650, 550, 450, 350, 250, 150], [750, 650, 550, 450, 350, 250]
        ...             ),
        ...             pd.IntervalIndex.from_arrays([0, 0, 0, 0 ,0, 0], [0, 0, 0, 0, 0, 0]),
        ...         ],
        ...         names=["range", "mean"],
        ...     ),
        ...     name="cycles"
        ... )
        >>> five_segments = pd.Series(
        ...     {
        ...         "M0": 0.5,
        ...         "M1": 0.2,
        ...         "M2": 0.1,
        ...         "M3": 1.0,
        ...         "M4": 2.0,
        ...         "R12": 0.2,
        ...         "R23": 0.8,
        ...     }
        ... )
        >>> transformed = histogram.meanstress_transform.five_segment(five_segments, 0)
        >>> transformed.amplitude
        range                                     mean
        (433.3333333333333, 500.0]                (216.66666666666666, 250.0]                 233.333333
        (366.6666666666667, 433.3333333333333]    (183.33333333333334, 216.66666666666666]    200.000000
        (300.0, 366.6666666666667]                (150.0, 183.33333333333334]                 166.666667
        (233.33333333333334, 300.0]               (116.66666666666667, 150.0]                 133.333333
        (166.66666666666669, 233.33333333333334]  (83.33333333333334, 116.66666666666667]     100.000000
        (100.0, 166.66666666666669]               (50.0, 83.33333333333334]                    66.666667
        Name: amplitude, dtype: float64
        """
        transformer = HaighDiagram.five_segment(five_segment)
        return self._perform_transformation(transformer, R_goal)

    def _perform_transformation(self, transformer, R_goal):
        mean_left = self.use_class_left().meanstress.reset_index(drop=True)
        mean_right = self.use_class_right().meanstress.reset_index(drop=True)

        range_index = self.amplitude_histogram.index
        range_left = 2.0 * range_index.left.to_series().reset_index(drop=True)
        range_right = 2.0 * range_index.right.to_series().reset_index(drop=True)

        orig_lele = pd.DataFrame({"range": range_left, "mean": mean_left})
        orig_lere = pd.DataFrame({"range": range_left, "mean": mean_right})
        orig_rele = pd.DataFrame({"range": range_right, "mean": mean_left})
        orig_rere = pd.DataFrame({"range": range_right, "mean": mean_right})

        transformed_lele = transformer.transform(orig_lele, R_goal)
        transformed_lere = transformer.transform(orig_lere, R_goal)
        transformed_rele = transformer.transform(orig_rele, R_goal)
        transformed_rere = transformer.transform(orig_rere, R_goal)

        transformed = self._obj.to_frame()

        have_additional_indeces = len(transformed.index.names) > 2
        if have_additional_indeces:
            transformed.index = transformed.index.droplevel(self.index_levels)

        range = pd.DataFrame(
            {
                0: transformed_lele["range"],
                1: transformed_lere["range"],
                2: transformed_rele["range"],
                3: transformed_rere["range"],
            }
        )
        transformed["range"] = pd.IntervalIndex.from_arrays(
            range.min(axis=1), range.max(axis=1), name="range"
        )
        mean = pd.DataFrame(
            {
                0: transformed_lele["mean"],
                1: transformed_lere["mean"],
                2: transformed_rele["mean"],
                3: transformed_rere["mean"],
            }
        )
        transformed["mean"] = pd.IntervalIndex.from_arrays(
            mean.min(axis=1), mean.max(axis=1), name="mean"
        )

        result = transformed.set_index(["range", "mean"], append=have_additional_indeces, drop=True).iloc[:, 0]
        new_names = self._obj.index.to_frame().rename(columns={"from": "range", "to": "mean"}).columns
        result.index = result.index.reorder_levels(new_names)
        result.name = self._obj.name

        return CL.LoadHistogram(result)


def fkm_goodman(amplitude, meanstress, M, M2, R_goal):
    """Transform amplitudes with the FKM-Goodman mean stress correction.

    Parameters
    ----------
    amplitude : array_like
        Stress amplitudes in MPa or another consistent stress unit.
    meanstress : array_like
        Mean stresses in the same unit as ``amplitude``.
    M : float
        Mean stress sensitivity for ``-inf < R <= 0``, dimensionless.
    M2 : float
        Mean stress sensitivity for ``0 < R <= 1``, dimensionless.
    R_goal : float
        Target stress ratio ``R = S_min / S_max``, dimensionless.

    Returns
    -------
    numpy.ndarray
        Transformed stress amplitudes in the same unit as ``amplitude``.

    See Also
    --------
    pylife.strength.meanstress.HaighDiagram.fkm_goodman : Create an FKM-Goodman diagram.
    """
    cycles = pd.DataFrame({"range": 2.0 * amplitude, "mean": meanstress})

    haigh_fkm_goodman = pd.Series({"M": M, "M2": M2})
    hd = HaighDiagram.fkm_goodman(haigh_fkm_goodman)

    res = hd.transform(cycles, R_goal)
    return res.load_collective.amplitude.to_numpy()


def five_segment_correction(
    amplitude, meanstress, M0, M1, M2, M3, M4, R12, R23, R_goal
):
    """Transform amplitudes with the five-segment mean stress correction.

    Parameters
    ----------
    amplitude : array_like
        Stress amplitudes in MPa or another consistent stress unit.
    meanstress : array_like
        Mean stresses in the same unit as ``amplitude``.
    M0 : float
        Mean stress sensitivity for ``-inf < R <= 0``, dimensionless.
    M1 : float
        Mean stress sensitivity for ``0 < R <= R12``, dimensionless.
    M2 : float
        Mean stress sensitivity for ``R12 < R <= R23``, dimensionless.
    M3 : float
        Mean stress sensitivity for ``R23 < R <= 1``, dimensionless.
    M4 : float
        Mean stress sensitivity for ``1 < R``, dimensionless.
    R12 : float
        Transition stress ratio between ``M1`` and ``M2``, dimensionless.
    R23 : float
        Transition stress ratio between ``M2`` and ``M3``, dimensionless.
    R_goal : float
        Target stress ratio ``R = S_min / S_max``, dimensionless.

    Returns
    -------
    numpy.ndarray
        Transformed stress amplitudes in the same unit as ``amplitude``.

    See Also
    --------
    pylife.strength.meanstress.HaighDiagram.five_segment : Create a five-segment diagram.
    """

    cycles = pd.DataFrame({"range": 2.0 * amplitude, "mean": meanstress})

    haigh_five_segment = pd.Series(
        {"M0": M0, "M1": M1, "M2": M2, "M3": M3, "M4": M4, "R12": R12, "R23": R23}
    )

    hd = HaighDiagram.five_segment(haigh_five_segment)
    res = hd.transform(cycles, R_goal)
    return res.load_collective.amplitude.to_numpy()

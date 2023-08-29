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

import sys, os, copy
import warnings
import pytest

import numpy as np
import pandas as pd
import numpy.testing as testing

import pylife.strength.meanstress as MST


def test_haigh_diagram_fail_no_meanstress_sens_index():
    hd = pd.Series(dtype=np.float64)
    with pytest.raises(AttributeError, match="A Haigh Diagram needs an index level 'R'."):
        hd.haigh_diagram


def test_haigh_diagram_fail_no_interval_index():
    hd = pd.Series([], index=pd.Index([], name='R'), dtype=np.float64)
    with pytest.raises(AttributeError, match="The 'R' index must be an IntervalIndex."):
        hd.haigh_diagram


def test_haigh_diagram_multi_index_fail_no_interval_index():
    hd = pd.Series([], index=pd.MultiIndex.from_frame(pd.DataFrame([], columns=['foo', 'R'])), dtype=np.float64)
    with pytest.raises(AttributeError, match="The 'R' index must be an IntervalIndex."):
        hd.haigh_diagram


def test_haigh_diagram_from_dict_empty():
    hd = MST.HaighDiagram.from_dict({})
    expected = pd.Series([], index=pd.IntervalIndex([], name='R'), dtype=np.float64)
    pd.testing.assert_series_equal(hd.to_pandas(), expected)


def test_haigh_diagram_from_dict_filled():
    hd = MST.HaighDiagram.from_dict({
        (1.0, np.inf): 0.0,
        (-np.inf, 0.0): 0.5,
        (0.0, 1.0): 0.167
    })

    expected_index = pd.IntervalIndex.from_tuples([(1.0, np.inf), (-np.inf, 0.0), (0.0, 1.0)], name='R')
    expected = pd.Series([0.0, 0.5, 0.167], index=expected_index)
    pd.testing.assert_series_equal(hd.to_pandas(), expected)


def test_haigh_diagram_fkm_goodman_single_M():
    hd = MST.HaighDiagram.fkm_goodman(pd.Series({'M': 0.5}))

    expected_index = pd.IntervalIndex.from_tuples([(1.0, np.inf), (-np.inf, 0.0), (0.0, 1.0)], name='R')
    expected = pd.Series([0.0, 0.5, 0.166667], index=expected_index)
    pd.testing.assert_series_equal(hd.to_pandas(), expected, rtol=1e-4)


def test_haigh_diagram_fkm_goodman_single_M_M2():
    hd = MST.HaighDiagram.fkm_goodman(pd.Series({'M': 0.4, 'M2': 0.15}))

    expected_index = pd.IntervalIndex.from_tuples([(1.0, np.inf), (-np.inf, 0.0), (0.0, 1.0)], name='R')
    expected = pd.Series([0.0, 0.4, 0.15], index=expected_index)
    pd.testing.assert_series_equal(hd.to_pandas(), expected, rtol=1e-4)


def test_haigh_diagram_fkm_goodman_multiple_M():
    hd = MST.HaighDiagram.fkm_goodman(pd.DataFrame({
        'M': [0.5, 0.4]
    }, index=pd.Index([1, 2], name='element_id')))

    expected_index = pd.MultiIndex.from_tuples([
        (1, pd.Interval(1.0, np.inf)),
        (1, pd.Interval(-np.inf, 0.0)),
        (1, pd.Interval(0.0, 1.0)),
        (2, pd.Interval(1.0, np.inf)),
        (2, pd.Interval(-np.inf, 0.0)),
        (2, pd.Interval(0.0, 1.0)),
    ], names=['element_id', 'R'])
    expected = pd.Series([0.0, 0.5, 0.166667, 0.0, 0.4, 0.13333], index=expected_index)
    pd.testing.assert_series_equal(hd.to_pandas(), expected, rtol=1e-4)


def test_haigh_diagram_fkm_goodman_sized_M():
    MST.HaighDiagram.fkm_goodman(pd.DataFrame({
        'M': 0.5
    }, index=pd.Index(np.arange(101), name='element_id')))


def test_haigh_diagram_fkm_goodman_multiple_M_M2():
    hd = MST.HaighDiagram.fkm_goodman(pd.DataFrame({
        'M': [0.5, 0.4],
        'M2': [0.2, 0.15]
    }, index=pd.Index([1, 2], name='element_id')))

    expected_index = pd.MultiIndex.from_tuples([
        (1, pd.Interval(1.0, np.inf)),
        (1, pd.Interval(-np.inf, 0.0)),
        (1, pd.Interval(0.0, 1.0)),
        (2, pd.Interval(1.0, np.inf)),
        (2, pd.Interval(-np.inf, 0.0)),
        (2, pd.Interval(0.0, 1.0)),
    ], names=['element_id', 'R'])
    expected = pd.Series([0.0, 0.5, 0.2, 0.0, 0.4, 0.15], index=expected_index)
    pd.testing.assert_series_equal(hd.to_pandas(), expected, rtol=1e-4)


## TODO
##
## * left closed intervals?

# @pytest.mark.parametrize('left, right', [(-np.inf, 0.1), (0.0, 1.0), (-np.inf, np.inf)])
# def test_haigh_diagram_fail_R_range_not_covered(left, right):
#     hd = pd.Series([0.5], index=pd.IntervalIndex.from_tuples([(left, right)], name='R'), dtype=np.float64)
#     with pytest.raises(AttributeError, match=r"The 'R' IntervalIndex must cover the exact range \(-inf, 1.0\]."):
#         hd.haigh_diagram


# def test_haigh_diagram_multi_index_interval_index_fail_R_range_not_covered():
#     foo_level = pd.Index([], name='foo')
#     R_level = pd.IntervalIndex([], name='R')
#     hd = pd.Series([], index=pd.MultiIndex.from_product([foo_level, R_level]), dtype=np.float64)
#     with pytest.raises(AttributeError, match=r"The 'R' IntervalIndex must cover the exact range \(-inf, 1.0\]."):
#         hd.haigh_diagram


def test_haigh_diagram_fail_R_is_overlapping_multi():
    idx = pd.MultiIndex.from_tuples([
        (1, pd.Interval(1.0, np.inf)),
        (1, pd.Interval(-np.inf, -1.0)),
        (1, pd.Interval(-1.0, 0.0)),
        (2, pd.Interval(1.0, np.inf)),
        (2, pd.Interval(-np.inf, -0.5)),
        (2, pd.Interval(-0.6, 0.0)),
    ], names=['element_id', 'R'])
    hd = pd.Series(1.0, index=idx)
    with pytest.raises(AttributeError, match=r"The intervals of the 'R' IntervalIndex must not overlap."):
        hd.haigh_diagram


def test_haigh_diagram_fail_R_is_overlapping():
    idx = pd.IntervalIndex.from_tuples([(-np.inf, 0.1), (0.0, 1.0)], name='R')
    hd = pd.Series([0.5, 0.5], index=idx)
    with pytest.raises(AttributeError, match=r"The intervals of the 'R' IntervalIndex must not overlap."):
        hd.haigh_diagram


def test_haigh_diagram_fail_if_R_has_gaps():
    idx = pd.IntervalIndex.from_tuples([(-np.inf, 0.0), (0.1, 1.0)], name='R')
    hd = pd.Series([0.5, 0.5], index=idx)
    with pytest.raises(AttributeError, match=r"The intervals of the 'R' IntervalIndex must not have gaps."):
        hd.haigh_diagram


def test_haigh_diagram_fail_if_R_has_gaps_multi():
    idx = pd.MultiIndex.from_tuples([
        (1, pd.Interval(1.0, np.inf)),
        (1, pd.Interval(-np.inf, -1.0)),
        (1, pd.Interval(-1.0, 0.0)),
        (2, pd.Interval(1.0, np.inf)),
        (2, pd.Interval(-np.inf, -0.5)),
        (2, pd.Interval(-0.4, 0.0)),
    ], names=['element_id', 'R'])
    hd = pd.Series(1.0, index=idx)
    with pytest.raises(AttributeError, match=r"The intervals of the 'R' IntervalIndex must not have gaps."):
        hd.haigh_diagram


def test_haigh_diagram_multiindex_unique():
    foo_level = pd.Index([1, 2, 3], name='foo')
    R_level = pd.IntervalIndex.from_tuples([(1., np.inf), (-np.inf, 1.)], name='R')
    hd = pd.Series([1.]*6, index=pd.MultiIndex.from_product([foo_level, R_level]), dtype=np.float64)
    pd.testing.assert_index_equal(hd.haigh_diagram._R_index, R_level)


def test_haigh_diagram():
    hd = pd.Series([0.5], index=pd.IntervalIndex.from_tuples([(-np.inf, 1.0)], name='R'))
    hd.haigh_diagram


@pytest.mark.parametrize('M, from_val, to_val, R_goal, expected_range, expected_mean', [
    (0.5, -1., 1., -1.0, 2.0, 0.0),
    (0.5, -0.4, 1.2, -1.0, 2.0, 0.0),
    (0.5, 0., 4./3., -1.0, 2.0, 0.0),
    (0.5, -1., 1., 0.0, 4./3., 2./3.),
    (0.5, 1., -1., -1./3., 1.6, 0.4),
    (0.25, -0.4, 1.2, -1.0, 1.8, 0.0),
    (0.25, -0.9, 0.9, -1./3., 1.6, 0.4),
    (0.5/3., 7./12., 21./12., 0.0, 4./3., 2./3.)
])
def test_haigh_diagram_one_segment(M, from_val, to_val, R_goal, expected_range, expected_mean):
    hd = pd.Series([M], index=pd.IntervalIndex.from_tuples([(-np.inf, 1.0)], name='R'))
    cycle = pd.DataFrame({
        'from': [from_val],
        'to': [to_val]
    })

    res = hd.haigh_diagram.transform(cycle, R_goal)
    expected = pd.DataFrame({'range': [expected_range], 'mean': [expected_mean]})
    pd.testing.assert_frame_equal(res, expected)


@pytest.mark.parametrize('M, from_val, to_val, R_goal, expected_range, expected_mean', [
    (0.5, -1., 1., -1.0, 2.0, 0.0),
    (0.5, -0.4, 1.2, -1.0, 2.0, 0.0),
    (0.5, 0., 4./3., -1.0, 2.0, 0.0),
    (0.5, -1., 1., 0.0, 4./3., 2./3.),
    (0.5, 1., -1., -1./3., 1.6, 0.4),
    (0.25, -0.4, 1.2, -1.0, 1.8, 0.0),
    (0.25, -0.9, 0.9, -1./3., 1.6, 0.4),
    (0.5, 7./12., 21./12., -1.0, 2.0, 0.0),
    (0.5, -1., 1., 1./3., 7./6., 7./6.)
])
def test_haigh_diagram_two_segment(M, from_val, to_val, R_goal, expected_range, expected_mean):
    idx = pd.IntervalIndex.from_tuples([
        (-np.inf, 0.0),
        (0.0, 1.0)
    ], name='R')
    hd = pd.Series([M, M/3.], index=idx)

    cycle = pd.DataFrame({
        'from': [from_val],
        'to': [to_val]
    })

    res = hd.haigh_diagram.transform(cycle, R_goal)
    expected = pd.DataFrame({'range': [expected_range], 'mean': [expected_mean]})
    pd.testing.assert_frame_equal(res, expected)


def test_haigh_diagram_transform_to_inf():
    idx = pd.IntervalIndex.from_tuples([
        (-np.inf, 0.0),
        (0.0, 1.0)
    ], name='R')
    hd = pd.Series([0., 0.], index=idx)

    cycle = pd.DataFrame({
        'from': [-1.],
        'to': [1.]
    })

    res = hd.haigh_diagram.transform(cycle, R_goal=-np.inf)
    expected = pd.DataFrame({'range': [2.], 'mean': [-1.]})
    pd.testing.assert_frame_equal(res, expected)


def test_haigh_diagram_transform_to_R_gt_1():
    idx = pd.IntervalIndex.from_tuples([
        (1.0, np.inf),
        (-np.inf, 0.0),
    ], name='R')

    hd = pd.Series([0.5, 0.5], index=idx)

    cycle = pd.DataFrame({
        'from': [-1.],
        'to': [1.]
    })

    res = hd.haigh_diagram.transform(cycle, R_goal=5.0)

    expected = pd.DataFrame({'range': [8.], 'mean': [-6.]})
    pd.testing.assert_frame_equal(res, expected)


def test_haigh_diagram_broadcast_constant_intervals():
    R_intervals = pd.IntervalIndex.from_tuples([
        (1.0, np.inf),
        (-np.inf, 0.0),
    ], name='R')

    elements = pd.Index([1, 2], name='element_id')

    idx = pd.MultiIndex.from_product([elements, R_intervals])

    hd = pd.Series([0.0, 0.5, 0.0, 1./3.], index=idx)

    cycles = pd.DataFrame({
        'from': [-1., -2.],
        'to': [1., 2.]
    }, index=pd.Index([10, 11], name='cycle_number'))

    expected = pd.DataFrame({
        'range': [4.0, 8.0, 3.0, 6.0],
        'mean': [-2.0, -4.0, -1.5, -3.0]
    }, index=pd.MultiIndex.from_tuples([
        (1, 10), (1, 11),
        (2, 10), (2, 11)
    ], names=['element_id', 'cycle_number']))

    res = hd.haigh_diagram.transform(cycles, R_goal=-np.inf)

    pd.testing.assert_frame_equal(res, expected)


def test_haigh_diagram_broadcast_variable_intervals():
    R_intervals = pd.IntervalIndex.from_tuples([
        (1.0, np.inf),
        (-np.inf, 0.0),
    ], name='R')

    elements = pd.Index([1, 2], name='element_id')

    idx = pd.MultiIndex.from_product([elements, R_intervals])

    idx = pd.MultiIndex.from_tuples([
        (1, pd.Interval(1.0, np.inf)),
        (1, pd.Interval(-np.inf, -1.0)),
        (1, pd.Interval(-1.0, 0.0)),
        (2, pd.Interval(1.0, np.inf)),
        (2, pd.Interval(-np.inf, -0.5)),
        (2, pd.Interval(-0.5, 0.0)),
    ], names=['element_id', 'R'])

    hd = pd.Series([0.0, 0.5, 0.5, 0.0, 1./3., 1./3.], index=idx)

    cycles = pd.DataFrame({
        'from': [-1., -2.],
        'to': [1., 2.]
    }, index=pd.Index([10, 11], name='cycle_number'))

    expected = pd.DataFrame({
        'range': [4.0, 8.0, 3.0, 6.0],
        'mean': [-2.0, -4.0, -1.5, -3.0]
    }, index=pd.MultiIndex.from_tuples([
        (1, 10), (1, 11),
        (2, 10), (2, 11)
    ], names=['element_id', 'cycle_number']))

    res = hd.haigh_diagram.transform(cycles, R_goal=-np.inf)
    pd.testing.assert_frame_equal(res, expected)

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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import pytest
import numpy as np
import pandas as pd

from pylife.core.broadcaster import Broadcaster


foo_bar_series = pd.Series({'foo': 1.0, 'bar': 2.0})
foo_bar_series_twice_in_frame = pd.DataFrame([foo_bar_series, foo_bar_series])

series_named_index = foo_bar_series.copy()
series_named_index.index.name = 'idx1'

foo_bar_frame = pd.DataFrame({'foo': [1.0, 1.5], 'bar': [2.0, 1.5]})


def test_broadcast_series_to_array():
    param, obj = Broadcaster(foo_bar_series).broadcast([1.0, 2.0])

    pd.testing.assert_series_equal(param, pd.Series([1.0, 2.0]))
    pd.testing.assert_frame_equal(foo_bar_series_twice_in_frame, obj)


def test_broadcast_frame_to_array_match():
    param, obj = Broadcaster(foo_bar_frame).broadcast([1.0, 2.0])

    np.testing.assert_array_equal(param, [1.0, 2.0])
    pd.testing.assert_frame_equal(foo_bar_frame, obj)


def test_broadcast_frame_to_array_mismatch():
    with pytest.raises(ValueError, match=r"Dimension mismatch. "
                       "Cannot map 3 value array-like to a 2 element DataFrame signal."):
        Broadcaster(foo_bar_frame).broadcast([1.0, 2.0, 3.0])


def test_broadcast_series_to_scalar():
    param, obj = Broadcaster(foo_bar_series).broadcast(1.0)

    assert param == 1.0
    pd.testing.assert_series_equal(foo_bar_series, obj)


def test_broadcast_frame_to_scalar():
    param, obj = Broadcaster(foo_bar_frame).broadcast(1.0)

    expected_param = pd.Series([1.0, 1.0], index=foo_bar_frame.index)
    pd.testing.assert_series_equal(expected_param, param)
    pd.testing.assert_frame_equal(foo_bar_frame, obj)


def test_broadcast_series_index_named_to_series_index_named():
    series = pd.Series([5.0, 6.0], index=pd.Index(['x', 'y'], name='idx2'))
    param, obj = Broadcaster(series_named_index).broadcast(series)

    expected_param = pd.Series({
        ('foo', 'x'): 5.0,
        ('foo', 'y'): 6.0,
        ('bar', 'x'): 5.0,
        ('bar', 'y'): 6.0
    })
    expected_obj = pd.Series({
        ('foo', 'x'): 1.0,
        ('foo', 'y'): 1.0,
        ('bar', 'x'): 2.0,
        ('bar', 'y'): 2.0
    })
    expected_obj.index.names = ['idx1', 'idx2']
    expected_param.index.names = ['idx1', 'idx2']

    pd.testing.assert_series_equal(expected_param, param)
    pd.testing.assert_series_equal(expected_obj, obj)


def test_broadcast_series_index_named_to_series_index_none():
    series = pd.Series([5.0, 6.0], index=pd.Index([3, 4]))
    param, obj = Broadcaster(series_named_index).broadcast(series)

    expected_param = pd.Series({
        ('foo', 3): 5.0,
        ('foo', 4): 6.0,
        ('bar', 3): 5.0,
        ('bar', 4): 6.0
    })
    expected_obj = pd.Series({
        ('foo', 3): 1.0,
        ('foo', 4): 1.0,
        ('bar', 3): 2.0,
        ('bar', 4): 2.0
    })
    expected_obj.index.names = ['idx1', None]
    expected_param.index.names = ['idx1', None]

    pd.testing.assert_series_equal(expected_param, param)
    pd.testing.assert_series_equal(expected_obj, obj)


def test_broadcast_series_index_none_to_series_index_none():
    series = pd.Series([1.0, 2.0], index=pd.Index([3, 4]))
    param, obj = Broadcaster(foo_bar_series).broadcast(series)

    expected = pd.DataFrame([foo_bar_series, foo_bar_series], index=series.index)
    pd.testing.assert_series_equal(series, param)
    pd.testing.assert_frame_equal(expected, obj)


def test_broadcast_series_index_none_to_series_index_none_no_string_index():
    series = pd.Series([1.0, 2.0], index=pd.Index([3, 4]))
    obj = foo_bar_series.copy()
    obj.index = pd.Index([1, 2])
    param, obj = Broadcaster(obj).broadcast(series)

    expected = pd.DataFrame([foo_bar_series, foo_bar_series],
                            index=series.index)
    expected.columns = [1, 2]
    pd.testing.assert_series_equal(series, param)
    pd.testing.assert_frame_equal(expected, obj)


def test_broadcast_series_index_none_to_series_index_named():
    series = pd.Series([1.0, 2.0], index=pd.Index([3, 4], name='idx2'))
    foo_bar = foo_bar_series.copy()
    foo_bar.index.name = None
    param, obj = Broadcaster(foo_bar).broadcast(series)

    expected = pd.DataFrame([foo_bar_series, foo_bar_series], index=series.index)
    pd.testing.assert_series_equal(series, param)
    pd.testing.assert_frame_equal(expected, obj)


def test_broadcast_series_to_frame_2_elements_index_none():
    df = pd.DataFrame({
        'a': [1, 3],
        'b': [2, 4]
    }, index=['x', 'y'])

    param, obj = Broadcaster(foo_bar_series).broadcast(df)

    expected_obj = pd.DataFrame({
        'foo': [1.0, 1.0], 'bar': [2.0, 2.0]
    }, index=['x', 'y'])

    pd.testing.assert_frame_equal(param, df)
    pd.testing.assert_frame_equal(obj, expected_obj)


def test_broadcast_series_to_frame_3_elements_index_none():
    df = pd.DataFrame({
        'a': [1, 3, 5],
        'b': [2, 4, 6]
    }, index=['x', 'y', 'z'])

    param, obj = Broadcaster(foo_bar_series).broadcast(df)

    expected_obj = pd.DataFrame({
        'foo': [1.0, 1.0, 1.0], 'bar': [2.0, 2.0, 2.0],
    }, index=['x', 'y', 'z'])

    pd.testing.assert_frame_equal(param, df)
    pd.testing.assert_frame_equal(obj, expected_obj)


def test_broadcast_series_to_series_same_single_index():
    series = pd.Series([1, 3], index=pd.Index(['x', 'y'], name='iname1'), name='src')

    foo_bar = pd.Series([1, 2], index=pd.Index(['x', 'y'], name='iname1'), name='dst')

    param, obj = Broadcaster(foo_bar).broadcast(series)
    pd.testing.assert_series_equal(param, series)
    pd.testing.assert_series_equal(obj, foo_bar)


def test_broadcast_series_to_series_different_single_index_name():
    series = pd.Series([1, 3], index=pd.Index(['x', 'y'], name='iname1'), name='dest')

    foo_bar = pd.Series([1, 2], index=pd.Index([1, 2], name='srcname'), name='src')

    expected_index = pd.MultiIndex.from_tuples([(1, 'x'), (1, 'y'), (2, 'x'), (2, 'y')], names=['srcname', 'iname1'])
    expected_obj = pd.Series([1, 1, 2, 2], name='src', index=expected_index)
    expected_param = pd.Series([1, 3, 1, 3], name='dest', index=expected_index)

    param, obj = Broadcaster(foo_bar).broadcast(series)

    pd.testing.assert_series_equal(param, expected_param)
    pd.testing.assert_series_equal(obj, expected_obj)


def test_broadcast_frame_to_series_one_common():
    series = pd.Series(3.0, index=pd.Index([1,2], name='bar'))
    df = pd.DataFrame(
        {'col': np.arange(6)},
        index=pd.MultiIndex.from_product([pd.Index([1, 2, 3]), series.index])
    )

    param, obj = Broadcaster(df).broadcast(series)
    Broadcaster(df.col).broadcast(series)

    pd.testing.assert_series_equal(param, pd.Series(3.0, index=df.index))


def test_broadcast_frame_to_frame_same_single_index():
    df = pd.DataFrame({
        'a': [1, 3],
        'b': [2, 4]
    }, index=pd.Index(['x', 'y'], name='iname1'))

    foo_bar = pd.DataFrame({'foo': [1, 2], 'bar': [3, 4]}, index=pd.Index(['x', 'y'], name='iname1'))

    param, obj = Broadcaster(foo_bar).broadcast(df)
    pd.testing.assert_frame_equal(param, df)
    pd.testing.assert_frame_equal(obj, foo_bar)


def test_broadcast_frame_to_frame_same_multi_index():
    index = pd.MultiIndex.from_tuples([('x', 1), ('x', 2), ('y', 1), ('y', 2)], names=['iname1', 'iname2'])
    df = pd.DataFrame({
        'a': [1, 3, 5, 7],
        'b': [2, 4, 6, 8]
    }, index=index)

    foo_bar = pd.DataFrame({'foo': [1, 2, 3, 4], 'bar': [3, 4, 5, 6]}, index=index)

    param, obj = Broadcaster(foo_bar).broadcast(df)
    pd.testing.assert_frame_equal(param, df)
    pd.testing.assert_frame_equal(obj, foo_bar)


def test_broadcast_frame_to_frame_same_single_index_name_different_elements():
    df = pd.DataFrame({
        'a': [1, 3, 5],
        'b': [2, 4, 6]
    }, index=pd.Index(['x', 'y', 'z'], name='iname1'))

    foo_bar = pd.DataFrame({
        'foo': [1, 2, 3],
        'bar': [3, 4, 5]
    }, index=pd.Index(['y', 'a', 'x'], name='iname1'))

    param, obj = Broadcaster(foo_bar).broadcast(df)

    expected_param = pd.DataFrame({
        'a': [1, 3, 5, np.nan],
        'b': [2, 4, 6, np.nan]
    }, index=pd.Index(['x', 'y', 'z', 'a'], name='iname1')).sort_index()

    expected_obj = pd.DataFrame({
        'foo': [1, 2, 3, np.nan],
        'bar': [3, 4, 5, np.nan]
    }, index=pd.Index(['y', 'a', 'x', 'z'], name='iname1')).sort_index()

    pd.testing.assert_frame_equal(param.sort_index(), expected_param)
    pd.testing.assert_frame_equal(obj.sort_index(), expected_obj)


def test_broadcast_frame_to_frame_same_multi_index_name_different_elements():
    df = pd.DataFrame({
        'a': [1, 3, 5, 7, 9, 11, 13, 15, 17],
        'b': [2, 4, 6, 8, 10, 12, 14, 16, 18]
    }, index=pd.MultiIndex.from_tuples([('x', 1), ('x', 2), ('x', 3),
                                        ('y', 1), ('y', 2), ('y', 3),
                                        ('z', 1), ('z', 2), ('z', 3)], names=['iname1', 'iname2']))

    foo_bar = pd.DataFrame({
        'foo': [1, 2, 3, 4, 5, 6, 7, 8],
        'bar': [3, 4, 5, 6, 7, 8, 9, 0]
    }, index=pd.MultiIndex.from_tuples([('x', 1), ('x', 2), ('x', 3), ('x', 4),
                                        ('y', 1), ('y', 2), ('y', 3), ('y', 4)], names=['iname1', 'iname2']))

    param, obj = Broadcaster(foo_bar).broadcast(df)

    expected_param = pd.DataFrame({
        'a': [1, 3, 5, np.nan, 7, 9, 11, np.nan, 13, 15, 17],
        'b': [2, 4, 6, np.nan, 8, 10, 12, np.nan, 14, 16, 18]
    }, index=pd.MultiIndex.from_tuples([('x', 1), ('x', 2), ('x', 3), ('x', 4),
                                        ('y', 1), ('y', 2), ('y', 3), ('y', 4),
                                        ('z', 1), ('z', 2), ('z', 3)], names=['iname1', 'iname2']))

    expected_obj = pd.DataFrame({
        'foo': [1, 2, 3, 4, 5, 6, 7, 8, np.nan, np.nan, np.nan],
        'bar': [3, 4, 5, 6, 7, 8, 9, 0, np.nan, np.nan, np.nan]
    }, index=pd.MultiIndex.from_tuples([('x', 1), ('x', 2), ('x', 3), ('x', 4),
                                        ('y', 1), ('y', 2), ('y', 3), ('y', 4),
                                        ('z', 1), ('z', 2), ('z', 3)], names=['iname1', 'iname2']))

    pd.testing.assert_frame_equal(param.sort_index(), expected_param.sort_index())
    pd.testing.assert_frame_equal(obj.sort_index(), expected_obj.sort_index())


def test_broadcast_frame_to_frame_different_single_index_name():
    df = pd.DataFrame({
        'a': [1, 3],
        'b': [2, 4]
    }, index=pd.Index(['x', 'y'], name='iname1'))

    foo_bar = pd.DataFrame({'foo': [1, 2], 'bar': [3, 4]}, index=pd.Index([1, 2], name='srcname'))

    expected_obj = pd.DataFrame({
        'foo': [1, 1, 2, 2],
        'bar': [3, 3, 4, 4],
    }, index=pd.MultiIndex.from_tuples([(1, 'x'), (1, 'y'), (2, 'x'), (2, 'y')], names=['srcname', 'iname1']))

    expected_param = pd.DataFrame({
        'a': [1, 3, 1, 3],
        'b': [2, 4, 2, 4]
    }, index=pd.MultiIndex.from_tuples([(1, 'x'), (1, 'y'), (2, 'x'), (2, 'y')], names=['srcname', 'iname1']))

    param, obj = Broadcaster(foo_bar).broadcast(df)

    pd.testing.assert_frame_equal(param, expected_param)
    pd.testing.assert_frame_equal(obj, expected_obj)


def test_broadcast_frame_to_frame_different_multi_index_name():
    df = pd.DataFrame({
        'a': [1, 3, 5, 7],
        'b': [2, 4, 6, 8]
    }, index=pd.MultiIndex.from_tuples([('x', 1), ('x', 2), ('y', 1), ('y', 2)], names=['iname1', 'iname2']))

    foo_bar = pd.DataFrame({
        'foo': [1, 2, 3, 4],
        'bar': [3, 4, 5, 6]
    }, index=pd.MultiIndex.from_tuples([('a', 10), ('a', 20), ('b', 10), ('b', 20)], names=['srcname1', 'srcname2']))

    param, obj = Broadcaster(foo_bar).broadcast(df)

    expected_obj = pd.DataFrame({
        'foo': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
        'bar': [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6]
    }, index=pd.MultiIndex.from_tuples([
        ('a', 10, 'x', 1), ('a', 10, 'x', 2), ('a', 10, 'y', 1), ('a', 10, 'y', 2),
        ('a', 20, 'x', 1), ('a', 20, 'x', 2), ('a', 20, 'y', 1), ('a', 20, 'y', 2),
        ('b', 10, 'x', 1), ('b', 10, 'x', 2), ('b', 10, 'y', 1), ('b', 10, 'y', 2),
        ('b', 20, 'x', 1), ('b', 20, 'x', 2), ('b', 20, 'y', 1), ('b', 20, 'y', 2)
    ], names=['srcname1', 'srcname2', 'iname1', 'iname2']))

    expected_param = pd.DataFrame({
        'a': [1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7],
        'b': [2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8]
    }, index=pd.MultiIndex.from_tuples([
        ('a', 10, 'x', 1), ('a', 10, 'x', 2), ('a', 10, 'y', 1), ('a', 10, 'y', 2),
        ('a', 20, 'x', 1), ('a', 20, 'x', 2), ('a', 20, 'y', 1), ('a', 20, 'y', 2),
        ('b', 10, 'x', 1), ('b', 10, 'x', 2), ('b', 10, 'y', 1), ('b', 10, 'y', 2),
        ('b', 20, 'x', 1), ('b', 20, 'x', 2), ('b', 20, 'y', 1), ('b', 20, 'y', 2)
    ], names=['srcname1', 'srcname2', 'iname1', 'iname2']))

    pd.testing.assert_frame_equal(param, expected_param)
    pd.testing.assert_frame_equal(obj, expected_obj)


def test_broadcast_frame_to_frame_different_multi_index_name_drop_level():
    df = pd.DataFrame({
        'a': [1, 3, 5, 7],
        'b': [2, 4, 6, 8]
    }, index=pd.MultiIndex.from_tuples([('x', 1), ('x', 2), ('y', 1), ('y', 2)], names=['iname1', 'iname2']))

    foo_bar = pd.DataFrame({
        'foo': [1, 2, 3, 4],
        'bar': [3, 4, 5, 6]
    }, index=pd.MultiIndex.from_tuples([('a', 10), ('a', 20), ('b', 10), ('b', 20)], names=['srcname1', 'srcname2']))

    param, obj = Broadcaster(foo_bar).broadcast(df, droplevel=['srcname2'])

    expected_obj = pd.DataFrame({
        'foo': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
        'bar': [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6]
    }, index=pd.MultiIndex.from_tuples([
        ('a', 10, 'x', 1), ('a', 10, 'x', 2), ('a', 10, 'y', 1), ('a', 10, 'y', 2),
        ('a', 20, 'x', 1), ('a', 20, 'x', 2), ('a', 20, 'y', 1), ('a', 20, 'y', 2),
        ('b', 10, 'x', 1), ('b', 10, 'x', 2), ('b', 10, 'y', 1), ('b', 10, 'y', 2),
        ('b', 20, 'x', 1), ('b', 20, 'x', 2), ('b', 20, 'y', 1), ('b', 20, 'y', 2)
    ], names=['srcname1', 'srcname2', 'iname1', 'iname2']))

    expected_param = pd.DataFrame({
        'a': [1, 3, 5, 7, 1, 3, 5, 7],
        'b': [2, 4, 6, 8, 2, 4, 6, 8]
    }, index=pd.MultiIndex.from_tuples([
        ('a', 'x', 1), ('a', 'x', 2), ('a', 'y', 1), ('a', 'y', 2),
        ('b', 'x', 1), ('b', 'x', 2), ('b', 'y', 1), ('b', 'y', 2)
    ], names=['srcname1', 'iname1', 'iname2']))

    pd.testing.assert_frame_equal(param, expected_param)
    pd.testing.assert_frame_equal(obj, expected_obj)


def test_broadcast_frame_to_frame_mixed_multi_index_name():
    df = pd.DataFrame({
        'a': [1, 3, 5, 7],
        'b': [2, 4, 6, 8]
    }, index=pd.MultiIndex.from_tuples([('x', 1), ('x', 2), ('y', 1), ('y', 2)], names=['iname1', 'srcname2']))

    foo_bar = pd.DataFrame({
        'foo': [1, 2, 3, 4],
        'bar': [3, 4, 5, 6]
    }, index=pd.MultiIndex.from_tuples([('a', 1), ('b', 1), ('a', 2), ('b', 2)], names=['srcname1', 'srcname2']))

    param, obj = Broadcaster(foo_bar).broadcast(df)

    expected_obj = pd.DataFrame({
        'foo': [1, 1, 2, 2, 3, 3, 4, 4],
        'bar': [3, 3, 4, 4, 5, 5, 6, 6]
    }, index=pd.MultiIndex.from_tuples([
        ('a', 1, 'x'), ('a', 1, 'y'),
        ('b', 1, 'x'), ('b', 1, 'y'),
        ('a', 2, 'x'), ('a', 2, 'y'),
        ('b', 2, 'x'), ('b', 2, 'y')
    ], names=['srcname1', 'srcname2', 'iname1']))

    expected_prm = pd.DataFrame({
        'a': [1, 5, 1, 5, 3, 7, 3, 7],
        'b': [2, 6, 2, 6, 4, 8, 4, 8]
    }, index=pd.MultiIndex.from_tuples([
        ('a', 1, 'x'), ('a', 1, 'y'),
        ('b', 1, 'x'), ('b', 1, 'y'),
        ('a', 2, 'x'), ('a', 2, 'y'),
        ('b', 2, 'x'), ('b', 2, 'y')
    ], names=['srcname1', 'srcname2', 'iname1']))

    pd.testing.assert_frame_equal(param, expected_prm)
    pd.testing.assert_frame_equal(obj, expected_obj)


def test_broadcast_frame_to_frame_mixed_multi_index_name_drop_level():
    df = pd.DataFrame({
        'a': [1, 3, 5, 7],
        'b': [2, 4, 6, 8]
    }, index=pd.MultiIndex.from_tuples([('x', 1), ('x', 2), ('y', 1), ('y', 2)], names=['iname1', 'srcname2']))

    foo_bar = pd.DataFrame({
        'foo': [1, 2, 3, 4],
        'bar': [3, 4, 5, 6]
    }, index=pd.MultiIndex.from_tuples([('a', 1), ('b', 1), ('a', 2), ('b', 2)], names=['srcname1', 'srcname2']))

    param, obj = Broadcaster(foo_bar).broadcast(df, droplevel=['srcname1'])

    expected_obj = pd.DataFrame({
        'foo': [1, 1, 2, 2, 3, 3, 4, 4],
        'bar': [3, 3, 4, 4, 5, 5, 6, 6]
    }, index=pd.MultiIndex.from_tuples([
        ('a', 1, 'x'), ('a', 1, 'y'),
        ('b', 1, 'x'), ('b', 1, 'y'),
        ('a', 2, 'x'), ('a', 2, 'y'),
        ('b', 2, 'x'), ('b', 2, 'y')
    ], names=['srcname1', 'srcname2', 'iname1']))

    expected_prm = pd.DataFrame({
        'a': [1, 5, 3, 7],
        'b': [2, 6, 4, 8]
    }, index=pd.MultiIndex.from_tuples([
        (1, 'x'), (1, 'y'),
        (2, 'x'), (2, 'y')
    ], names=['srcname2', 'iname1']))

    pd.testing.assert_frame_equal(param, expected_prm)
    pd.testing.assert_frame_equal(obj, expected_obj)


def test_broadcast_series_to_series_overlapping_interval_index():
    interval_index = pd.IntervalIndex.from_tuples([
        (0.0, 1.0), (1.0, 2.0), (1.5, 2.5)
    ], name='interval')

    obj = pd.Series([1, 2, 3], index=interval_index, name='series')

    operand = pd.Series([4, 5, 6], index=pd.RangeIndex(3, name='operand'), name='foo')

    prm, obj = Broadcaster(obj).broadcast(operand)

    expected_index = pd.MultiIndex.from_tuples([
        (pd.Interval(0.0, 1.0), 0),
        (pd.Interval(0.0, 1.0), 1),
        (pd.Interval(0.0, 1.0), 2),
        (pd.Interval(1.0, 2.0), 0),
        (pd.Interval(1.0, 2.0), 1),
        (pd.Interval(1.0, 2.0), 2),
        (pd.Interval(1.5, 2.5), 0),
        (pd.Interval(1.5, 2.5), 1),
        (pd.Interval(1.5, 2.5), 2),
    ], names=['interval', 'operand'])

    expected_obj = pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3], index=expected_index, name='series')
    expected_prm = pd.Series([4, 5, 6, 4, 5, 6, 4, 5, 6], index=expected_index, name='foo')

    pd.testing.assert_series_equal(obj, expected_obj)
    pd.testing.assert_series_equal(prm, expected_prm)

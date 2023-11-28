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

from pylife.core import *

foo_bar_baz = pd.DataFrame({'foo': [1.0, 1.0], 'bar': [1.0, 1.0], 'baz': [1.0, 1.0]})

def test_keys_dataframe():
    pd.testing.assert_index_equal(foo_bar_baz.test_accessor_none.keys(), pd.Index(['foo', 'bar', 'baz']))


def test_keys_series():
    pd.testing.assert_index_equal(foo_bar_baz.iloc[0].test_accessor_none.keys(), pd.Index(['foo', 'bar', 'baz']))


def test_missing_keys_none():
    assert foo_bar_baz.test_accessor_none.get_missing_keys(['foo', 'bar']) == []


def test_missing_keys_one():
    assert foo_bar_baz.test_accessor_none.get_missing_keys(['foo', 'foobar']) == ['foobar']


def test_missing_keys_two():
    assert set(foo_bar_baz.test_accessor_none.get_missing_keys(['foo', 'foobar', 'barfoo'])) == set(['foobar', 'barfoo'])


def test_from_parameters_frame():
    foo = [1.0, 2.0, 3.0]
    bar = [10.0, 20.0, 30.0]
    baz = [11.0, 12.0, 13.0]
    accessor = AccessorNone.from_parameters(foo=foo, bar=bar, baz=baz)
    pd.testing.assert_index_equal(accessor.keys(), pd.Index(['foo', 'bar', 'baz']))
    expected_obj = pd.DataFrame({'foo': foo, 'bar': bar, 'baz': baz})
    pd.testing.assert_frame_equal(accessor._obj, expected_obj)
    assert accessor.some_property == 42


def test_from_parameters_series_columns():
    foo = 1.0
    bar = 10.0
    baz = 11.0
    accessor = AccessorNone.from_parameters(foo=foo, bar=bar, baz=baz)
    pd.testing.assert_index_equal(accessor.keys(), pd.Index(['foo', 'bar', 'baz']))
    expected_obj = pd.Series({'foo': foo, 'bar': bar, 'baz': baz})
    pd.testing.assert_series_equal(accessor._obj, expected_obj)
    assert accessor.some_property == 42


def test_from_parameters_series_index():
    foo = [1.0, 2.0, 3.0]
    accessor = AccessorOneDim.from_parameters(foo=foo)
    pd.testing.assert_index_equal(accessor.keys(), pd.Index(['foo']))
    expected_obj = pd.Series({'foo': foo})
    pd.testing.assert_series_equal(accessor._obj, expected_obj)
    assert accessor.some_property == 42


def test_from_parameters_missing_keys():
    foo = 1.0
    baz = 10.0
    with pytest.raises(AttributeError, match=r'^AccessorNone.*bar'):
        AccessorNone.from_parameters(foo=foo, baz=baz)


@pd.api.extensions.register_series_accessor('test_accessor_one_dim')
class AccessorOneDim(PylifeSignal):
    def _validate(self):
        if not isinstance(self._obj, pd.Series):
            raise TypeError("This accessor takes only pd.Series")

    @property
    def some_property(self):
        return 42

@pd.api.extensions.register_series_accessor('test_accessor_none')
@pd.api.extensions.register_dataframe_accessor('test_accessor_none')
class AccessorNone(PylifeSignal):
    def _validate(self):
        self.fail_if_key_missing(['foo', 'bar'])

    def already_here(self):
        return 23

    @property
    def some_property(self):
        return 42

    @property
    def not_working_property(self):
        self._missing_attribute


@pd.api.extensions.register_series_accessor('test_accessor_one')
@pd.api.extensions.register_dataframe_accessor('test_accessor_one')
class AccessorOne(PylifeSignal):
    def _validate(self):
        self.fail_if_key_missing(['foo', 'foobar'])


@pd.api.extensions.register_dataframe_accessor('test_accessor_two')
class AccessorTwo(PylifeSignal):
    def _validate(self):
        self.fail_if_key_missing(['foo', 'foobar', 'barfoo'])


def test_signal_broadcast_inheritance_series():
    assert isinstance(foo_bar_baz.loc[0].test_accessor_none, Broadcaster)


def test_signal_broadcast_inheritance_frame():
    assert isinstance(foo_bar_baz.test_accessor_none, Broadcaster)


def test_fail_if_missing_keys_none():
    foo_bar_baz.test_accessor_none


def test_fail_if_missing_keys_one_dataframe():
    with pytest.raises(AttributeError, match=r'^AccessorOne.*foobar'):
        foo_bar_baz.test_accessor_one


def test_fail_if_missing_keys_one_series():
    with pytest.raises(AttributeError, match=r'^AccessorOne.*foobar'):
        foo_bar_baz.loc[0].test_accessor_one


def test_fail_if_missing_keys_two():
    with pytest.raises(AttributeError, match=r'^AccessorTwo.*(foobar|barfoo).*(barfoo|foobar)'):
        foo_bar_baz.test_accessor_two


def test_register_method():
    @register_method(AccessorNone, 'foo_method')
    def foo(df):
        return pd.DataFrame({'baz': df['foo'] + df['bar']})

    accessor = foo_bar_baz.test_accessor_none
    pd.testing.assert_frame_equal(accessor.foo_method(), pd.DataFrame({'baz': [2.0, 2.0]}))


def test_getattr_no_method():
    accessor = foo_bar_baz.test_accessor_none
    assert accessor.already_here() == 23
    assert accessor.some_property == 42


def test_register_method_missing_attribute():
    @register_method(AccessorNone, 'another_method')
    def foo(df):
        return pd.DataFrame({'baz': df['foo'] + df['bar']})

    accessor = foo_bar_baz.test_accessor_none
    with pytest.raises(AttributeError, match=r'^\'AccessorNone\' object has no attribute \'_missing_attribute\''):
        accessor.not_working_property


def test_register_method_fail_duplicate():
    with pytest.raises(ValueError, match=r'^Method \'bar_method\' already registered in AccessorNone'):
        @register_method(AccessorNone, 'bar_method')
        def bar1(df):
            return pd.DataFrame({'baz': df['foo'] + df['bar']})

        @register_method(AccessorNone, 'bar_method')
        def bar2(df):
            return pd.DataFrame({'baz': df['foo'] - df['bar']})


def test_register_method_fail_already():
    with pytest.raises(ValueError, match=r'^AccessorNone already has an attribute \'already_here\''):
        @register_method(AccessorNone, 'already_here')
        def already_here_method(df):
            return pd.DataFrame({'baz': df['foo'] + df['bar']})


def test_pandas_series():
    series = foo_bar_baz.iloc[0]
    pd.testing.assert_series_equal(series.test_accessor_none.to_pandas(), series)


def test_pandas_frame():
    pd.testing.assert_frame_equal(foo_bar_baz.test_accessor_none.to_pandas(), foo_bar_baz)

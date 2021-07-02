# Copyright (c) 2019-2021 - for information on the respective copyright owner
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
import pandas as pd

import pylife.core.signal as signal
from pylife.core.data_validator import DataValidator

foo_bar_baz = pd.DataFrame({'foo': [1.0], 'bar': [1.0], 'baz': [1.0]})
val = DataValidator()


def test_keys_dataframe():
    pd.testing.assert_index_equal(val.keys(foo_bar_baz), pd.Index(['foo', 'bar', 'baz']))


def test_keys_series():
    pd.testing.assert_index_equal(val.keys(foo_bar_baz.iloc[0]), pd.Index(['foo', 'bar', 'baz']))


def test_keys_invalid_type():
    with pytest.raises(AttributeError, match="An accessor object needs to be either a pandas.Series or a pandas.DataFrame"):
        val.keys('lllll')


def test_missing_keys_none():
    assert val.get_missing_keys(foo_bar_baz, ['foo', 'bar']) == []


def test_missing_keys_one():
    assert val.get_missing_keys(foo_bar_baz, ['foo', 'foobar']) == ['foobar']


def test_missing_keys_two():
    assert set(val.get_missing_keys(foo_bar_baz, ['foo', 'foobar', 'barfoo'])) == set(['foobar', 'barfoo'])


@pd.api.extensions.register_dataframe_accessor('test_accessor_none')
class AccessorNone(signal.PylifeSignal):
    def _validate(self, obj, validator):
        validator.fail_if_key_missing(obj, ['foo', 'bar'])

    def already_here(self):
        return 23

    @property
    def some_property(self):
        return 42

    @property
    def not_working_property(self):
        self._missing_attribute


@pd.api.extensions.register_dataframe_accessor('test_accessor_one')
class AccessorOne(signal.PylifeSignal):
    def _validate(self, obj, validator):
        validator.fail_if_key_missing(obj, ['foo', 'foobar'])


@pd.api.extensions.register_dataframe_accessor('test_accessor_two')
class AccessorTwo(signal.PylifeSignal):
    def _validate(self, obj, validator):
        validator.fail_if_key_missing(obj, ['foo', 'foobar', 'barfoo'])


def test_fail_if_missing_keys_none():
    foo_bar_baz.test_accessor_none


def test_fail_if_missing_keys_one():
    with pytest.raises(AttributeError, match=r'^AccessorOne.*foobar'):
        foo_bar_baz.test_accessor_one


def test_fail_if_missing_keys_two():
    with pytest.raises(AttributeError, match=r'^AccessorTwo.*(foobar|barfoo).*(barfoo|foobar)'):
        foo_bar_baz.test_accessor_two


def test_register_method():
    @signal.register_method(AccessorNone, 'foo_method')
    def foo(df):
        return pd.DataFrame({'baz': df['foo'] + df['bar']})

    accessor = foo_bar_baz.test_accessor_none
    pd.testing.assert_frame_equal(accessor.foo_method(), pd.DataFrame({'baz': [2.0]}))


def test_getattr_no_method():
    accessor = foo_bar_baz.test_accessor_none
    assert accessor.already_here() == 23
    assert accessor.some_property == 42


def test_register_method_missing_attribute():
    @signal.register_method(AccessorNone, 'another_method')
    def foo(df):
        return pd.DataFrame({'baz': df['foo'] + df['bar']})

    accessor = foo_bar_baz.test_accessor_none
    with pytest.raises(AttributeError, match=r'^\'AccessorNone\' object has no attribute \'_missing_attribute\''):
        accessor.not_working_property


def test_register_method_fail_duplicate():
    with pytest.raises(ValueError, match=r'^Method \'bar_method\' already registered in AccessorNone'):
        @signal.register_method(AccessorNone, 'bar_method')
        def bar1(df):
            return pd.DataFrame({'baz': df['foo'] + df['bar']})

        @signal.register_method(AccessorNone, 'bar_method')
        def bar2(df):
            return pd.DataFrame({'baz': df['foo'] - df['bar']})


def test_register_method_fail_already():
    with pytest.raises(ValueError, match=r'^AccessorNone already has an attribute \'already_here\''):
        @signal.register_method(AccessorNone, 'already_here')
        def already_here_method(df):
            return pd.DataFrame({'baz': df['foo'] + df['bar']})

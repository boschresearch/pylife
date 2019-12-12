
import pytest
import pandas as pd

from pylife import signal


foo_bar_baz = pd.DataFrame({'foo': [1.0], 'bar': [1.0], 'baz': [1.0]})


def test_missing_keys_none():
    assert signal.get_missing_keys(foo_bar_baz, ['foo', 'bar']) == []


def test_missing_keys_one():
    assert signal.get_missing_keys(foo_bar_baz, ['foo', 'foobar']) == ['foobar']


def test_missing_keys_two():
    assert set(signal.get_missing_keys(foo_bar_baz, ['foo', 'foobar', 'barfoo'])) == set(['foobar', 'barfoo'])


@pd.api.extensions.register_dataframe_accessor('test_accessor_none')
class AccessorNone(signal.PylifeSignal):
    def _validate(self, obj):
        signal.fail_if_key_missing(obj, ['foo', 'bar'])

    def already_here(self):
        pass

@pd.api.extensions.register_dataframe_accessor('test_accessor_one')
class AccessorOne:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    def _validate(self, obj):
        signal.fail_if_key_missing(obj, ['foo', 'foobar'])


@pd.api.extensions.register_dataframe_accessor('test_accessor_two')
class AccessorTwo:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    def _validate(self, obj):
        signal.fail_if_key_missing(obj, ['foo', 'foobar', 'barfoo'])


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

    pd.testing.assert_frame_equal(foo_bar_baz.test_accessor_none.foo_method(), pd.DataFrame({'baz': [2.0]}))


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

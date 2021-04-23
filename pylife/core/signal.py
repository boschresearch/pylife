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

from pylife.core.data_validator import DataValidator


class PylifeSignal:
    '''Base class for signal accessor classes

    Parameters
    ----------
    pandas_obj : pandas.DataFrame or pandas.Series

    Notes
    -----
    Derived classes need to implement the method `_validate(self, obj)`
    that gets `pandas_obj` as `obj` parameter. This `validate()` method
    must raise an Exception (e.g. AttributeError or ValueError) in case
    `obj` is not a valid DataFrame for the kind of signal.

    For these validation :func:`fail_if_key_missing()` and
    :func:`get_missing_keys()` might be helpful.

    For a derived class you can register methods without modifying the
    class' code itself. This can be useful if you want to make signal
    accessor classes extendable.

    See also
    --------
    :func:`fail_if_key_missing()`
    :func:`get_missing_keys()`
    :func:`register_method()`
    '''
    _method_dict = {}

    def __init__(self, pandas_obj):
        self._validator = DataValidator()
        self._validate(pandas_obj, self._validator)
        self._obj = pandas_obj

    class _MethodCaller:
        def __init__(self, method, obj):
            self._method = method
            self._obj = obj

        def __call__(self, *args, **kwargs):
            return self._method(self._obj, *args, **kwargs)

    def __getattr__(self, itemname):
        method = self._method_dict.get(itemname)

        if method is None:
            return super(PylifeSignal, self).__getattribute__(itemname)

        return self._MethodCaller(method, self._obj)

    @classmethod
    def _register_method(cls, method_name):
        def method_decorator(method):
            if method_name in cls._method_dict.keys():
                raise ValueError("Method '%s' already registered in %s" % (method_name, cls.__name__))
            if hasattr(cls, method_name):
                raise ValueError("%s already has an attribute '%s'" % (cls.__name__, method_name))
            cls._method_dict[method_name] = method
        return method_decorator


def register_method(cls, method_name):
    '''Registers a method to a class derived from :class:`PyifeSignal`

    Parameters
    ----------
    cls : class
        The class the method is registered to.
    method_name : str
        The name of the method

    Raises
    ------
    ValueError
        if `method_name` is already registered for the class
    ValueError
        if `method_name` the class has already an attribute `method_name`

    Notes
    -----
    The function is meant to be used as a decorator for a function
    that is to be installed as a method for a class. The class is
    assumed to contain a pandas object in `self._obj`.

    Examples
    --------
    .. code-block:: python

        import pandas as pd
        import pylife as pl


        @pd.api.extensions.register_dataframe_accessor('foo')
        class FooAccessor(pl.signal.PyifeSignal):
            def __init__(self, obj):
                # self._validate(obj) could come here
                self._obj = obj


        @pl.signal.register_method(FooAccessor, 'bar')
        def bar(df):
            return pd.DataFrame({'baz': df['foo'] + df['bar']})

    >>> df = pd.DataFrame({'foo': [1.0, 2.0], 'bar': [-1.0, -2.0]})
    >>> df.foo.bar()
       baz
    0  0.0
    1  0.0
    '''
    return cls._register_method(method_name)

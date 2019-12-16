# Copyright (c) 2019 - for information on the respective copyright owner
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

import inspect
import pandas as pd

class SignalValidator:
    def _get_missing_keys(self, signal, keys_to_check):
        '''Gets a list of missing keys that are needed for a signal object

        Parameters
        ----------
        signal : pandas.DataFrame or pandas.Series
            The object to be checked

        keys_to_check : list
            A list of keys that need to be available in `signal`

        Returns
        -------
        missing_keys : list
            a list of missing keys

        Raises
        ------
        AttributeError
            If `signal` is neither a `pandas.DataFrame` nor a `pandas.Series`

        Notes
        -----
        If `signal` is a `pandas.DataFrame`, all keys of
        `keys_to_check` not found in the `signal.columns` are
        returned.

        If `signal` is a `pandas.Series`, all keys of
        `keys_to_check` not found in the `signal.index` are
        returned.
        '''
        if isinstance(signal, pd.Series):
            keys_avail = signal.index
        elif isinstance(signal, pd.DataFrame):
            keys_avail = signal.columns
        else:
            raise AttributeError("An accessor object needs to be either a pandas.Series or a pandas.DataFrame")

        missing_keys = []
        for k in keys_to_check:
            if k not in keys_avail:
                missing_keys.append(k)
        return missing_keys


    def fail_if_key_missing(self, signal, keys_to_check, msg=None):
        '''Raises an exception if any key is missing in a signal object

        Parameters
        ----------
        signal : pandas.DataFrame or pandas.Series
            The object to be checked

        keys_to_check : list
            A list of keys that need to be available in `signal`

        Raises
        ------
        AttributeError
            if `signal` is neither a `pandas.DataFrame` nor a `pandas.Series`
        AttributeError
            if any of the keys is not found in the signal's keys.

        Notes
        -----
        If `signal` is a `pandas.DataFrame`, all keys of
        `keys_to_check` meed to be found in the `signal.columns`.

        If `signal` is a `pandas.Series`, all keys of
        `keys_to_check` meed to be found in the `signal.index`.

        See also
        --------
        :func:`signal.get_missing_keys`
        :class:`stresssignal.StressTensorVoigtAccessor`
        '''
        missing_keys = self._get_missing_keys(signal, keys_to_check)
        if not missing_keys:
            return
        if msg is None:
            stack = inspect.stack()
            the_class = stack[2][0].f_locals['self'].__class__
            msg = the_class.__name__ + ' must have the items %s. Missing %s.'
        raise AttributeError(msg % (', '.join(keys_to_check), ', '.join(missing_keys)))
        
class PylifeSignal:
    '''Base class for signal accessor classes

    Parameters
    ----------
    pandas_obj : pandas.DataFame or pandas.Series

    Notes
    -----

    Derived classes need to implement the method `_validate(self, obj)`
    that gets `pandas_obj` as `obj` parameter. This `validate()` method
    must rais an Exception (e.g. AttributeError or ValueError) in case
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
        self._validator = SignalValidator()
        self._validate(pandas_obj, self._validator)
        self._obj = pandas_obj

    class _MethodCaller:
        def __init__(self, method, obj):
            self._method = method
            self._obj = obj

        def __call__(self, *args, **kwargs):
            return self._method(self._obj, *args, **kwargs)

    def __getattr__(self, itemname):
        if itemname not in self._method_dict.keys():
            raise AttributeError("Method '%s' not registered", itemname)
        method = self._method_dict[itemname]
        return self._MethodCaller(method, self._obj)

    @classmethod
    def _register_method(cls, method_name):
        def equistress_decorator(method):
            def method_wrapper(df):
                df.equistress
                return method(df)
            if method_name in cls._method_dict.keys():
                raise ValueError("Method '%s' already registered in %s" % (method_name, cls.__name__))
            if hasattr(cls, method_name):
                raise ValueError("%s already has an attribute '%s'" % (cls.__name__, method_name))
            cls._method_dict[method_name] = method
            return method_wrapper
        return equistress_decorator


def register_method(cls, method_name):
    '''Registeres a method to a class derived from :class:`PyifeSignal`

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

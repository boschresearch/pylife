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


import pandas as pd

from .broadcaster import Broadcaster
from .data_validator import DataValidator


class PylifeSignal(Broadcaster):
    """Base class for signal accessor classes.

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
    """

    _method_dict = {}

    def __init__(self, pandas_obj):
        """Instantiate a :class:`signal.PyLifeSignal`.

        Parameters
        ----------
        pandas_obj : pandas.DataFrame or pandas.Series

        """
        self._obj = pandas_obj
        self._validate()

    @classmethod
    def from_parameters(cls, **kwargs):
        """Make a signal instance from a parameter set.

        This is a convenience function to instantiate a signal from individual
        parameters rather than pandas objects.

        A signal class like

        .. code-block:: python

            @pd.api.extensions.register_dataframe_accessor('foo_signal')
            class FooSignal(PylifeSignal):
                pass

        The following two blocks are equivalent:

        .. code-block:: python

            pd.Series({'foo': 1.0, 'bar': 2.0}).foo_signal

        .. code-block:: python

            FooSignal.from_parameters(foo=1.0, bar=1.0)

        """
        # TODO: better error handling
        if len(kwargs) > 1 and hasattr(next(iter(kwargs.values())), '__iter__'):
            obj = pd.DataFrame(kwargs)
        else:
            obj = pd.Series(kwargs)

        return cls(obj)

    def keys(self):
        """Get a list of missing keys that are needed for a signal object.

        Returns
        -------
        keys : pd.Index
            a pandas index of keys

        Raises
        ------
        AttributeError
            if `self._obj` is neither a `pandas.DataFrame` nor a `pandas.Series`

        Notes
        -----
        If `self._obj` is a `pandas.DataFrame`, the `self._obj.columns` are returned.

        If `self._obj` is a `pandas.Series`, the `self._obj.index` are returned.
        """
        if isinstance(self._obj, pd.Series):
            return self._obj.index
        elif isinstance(self._obj, pd.DataFrame):
            return self._obj.columns
        raise AttributeError("An accessor object needs to be either a pandas.Series or a pandas.DataFrame")

    def get_missing_keys(self, keys_to_check):
        """Get a list of missing keys that are needed for a self._obj object.

        Parameters
        ----------
        keys_to_check : list
            A list of keys that need to be available in `self._obj`

        Returns
        -------
        missing_keys : list
            a list of missing keys

        Raises
        ------
        AttributeError
            if `self._obj` is neither a `pandas.DataFrame` nor a `pandas.Series`

        Notes
        -----
        If `self._obj` is a `pandas.DataFrame`, all keys of
        `keys_to_check` not found in the `self._obj.columns` are
        returned.

        If `self._obj` is a `pandas.Series`, all keys of
        `keys_to_check` not found in the `self._obj.index` are
        returned.
        """
        return DataValidator().get_missing_keys(self._obj, keys_to_check)

    def fail_if_key_missing(self, keys_to_check, msg=None):
        """Raise an exception if any key is missing in a self._obj object.

        Parameters
        ----------
        self._obj : pandas.DataFrame or pandas.Series
            The object to be checked

        keys_to_check : list
            A list of keys that need to be available in `self._obj`

        Raises
        ------
        AttributeError
            if `self._obj` is neither a `pandas.DataFrame` nor a `pandas.Series`
        AttributeError
            if any of the keys is not found in the self._obj's keys.

        Notes
        -----
        If `self._obj` is a `pandas.DataFrame`, all keys of
        `keys_to_check` meed to be found in the `self._obj.columns`.

        If `self._obj` is a `pandas.Series`, all keys of
        `keys_to_check` meed to be found in the `self._obj.index`.

        See also
        --------
        :func:`get_missing_keys`
        :class:`stresssignal.StressTensorVoigt`
        """
        DataValidator().fail_if_key_missing(self._obj, keys_to_check)

    class _MethodCaller:
        def __init__(self, method, obj):
            self._method = method
            self._obj = obj

        def __call__(self, *args, **kwargs):
            return self._method(self._obj, *args, **kwargs)

    def __getattr__(self, itemname):
        method = self._method_dict.get(itemname)

        if method is None:
            return super().__getattribute__(itemname)

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

    def to_pandas(self):
        """Expose the pandas object of the signal.

        Returns
        -------
        pandas_object : pd.DataFrame or pd.Series
            The pandas object representing the signal


        Notes
        -----

        The default implementation just returns the object given when
        instantiating the signal class. Derived classes may return a modified
        object or augmented, if they store some extra information.

        By default the object is **not** copied. So make a copy yourself, if
        you intent to modify it.
        """
        return self._obj


def register_method(cls, method_name):
    """Registers a method to a class derived from :class:`PyifeSignal`

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
        class Foo(pl.signal.PyifeSignal):
            def __init__(self, obj):
                # self._validate(obj) could come here
                self._obj = obj


        @pl.signal.register_method(Foo, 'bar')
        def bar(df):
            return pd.DataFrame({'baz': df['foo'] + df['bar']})

        df = pd.DataFrame({'foo': [1.0, 2.0], 'bar': [-1.0, -2.0]})
        df.foo.bar()

           baz
        0  0.0
        1  0.0
    """
    return cls._register_method(method_name)

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

"""Provide the base class for all pyLife signal accessors.

A *signal* in pyLife is an ordinary :class:`pandas.Series` or
:class:`pandas.DataFrame` whose keys follow a documented contract, for
example a Wöhler curve or a load collective.  :class:`PylifeSignal` turns
such an object into a domain object with methods, using the pandas accessor
mechanism, and adds index broadcasting via
:class:`~pylife.Broadcaster`.
"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__


import pandas as pd

from .broadcaster import Broadcaster
from .data_validator import DataValidator


class PylifeSignal(Broadcaster):
    """Base class for signal accessor classes.

    A signal accessor wraps a :class:`pandas.Series` or
    :class:`pandas.DataFrame` and gives it domain specific methods while
    leaving the data itself untouched.  Derived classes register themselves
    with :func:`pandas.api.extensions.register_dataframe_accessor` or
    :func:`pandas.api.extensions.register_series_accessor`, which makes them
    available as an attribute of any pandas object, e.g.
    ``df.woehler.basquin_cycles(...)``.

    Parameters
    ----------
    pandas_obj : pandas.DataFrame or pandas.Series
        The object the accessor is instantiated with.

    Raises
    ------
    AttributeError
        If ``pandas_obj`` does not fulfill the signal contract of the
        derived class.

    See Also
    --------
    fail_if_key_missing : Assert mandatory keys inside ``_validate()``.
    get_missing_keys : Query mandatory keys inside ``_validate()``.
    register_method : Add a method to a signal class from outside.

    Notes
    -----
    Derived classes need to implement the method ``_validate(self)`` which
    checks ``self._obj`` and raises an exception (e.g. ``AttributeError`` or
    ``ValueError``) if it is not a valid object for the kind of signal.  The
    methods :meth:`fail_if_key_missing` and :meth:`get_missing_keys` are
    meant to be used for that.

    For a derived class you can register methods without modifying the
    class' code itself.  This can be useful if you want to make signal
    accessor classes extendable.
    """

    _method_dict = {}

    def __init__(self, pandas_obj):
        """Instantiate a signal accessor.

        Parameters
        ----------
        pandas_obj : pandas.DataFrame or pandas.Series
            The object the accessor is instantiated with.  It is validated
            by the ``_validate()`` implementation of the derived class.
        """
        self._obj = pandas_obj
        self._validate()

    @classmethod
    def from_parameters(cls, **kwargs):
        """Make a signal instance from a parameter set.

        This is a convenience function to instantiate a signal from individual
        parameters rather than from a pandas object.

        Parameters
        ----------
        **kwargs : scalar or array_like
            The signal keys and their values.  If more than one value is
            given and the values are iterable, a
            :class:`pandas.DataFrame` backed signal is created, otherwise a
            :class:`pandas.Series` backed one.

        Returns
        -------
        PylifeSignal
            An instance of the signal class the method is called on.

        Examples
        --------
        For a signal class like

        .. code-block:: python

            @pd.api.extensions.register_dataframe_accessor('foo_signal')
            class FooSignal(PylifeSignal):
                pass

        the following two expressions are equivalent

        .. code-block:: python

            pd.Series({'foo': 1.0, 'bar': 2.0}).foo_signal
            FooSignal.from_parameters(foo=1.0, bar=2.0)
        """
        # TODO: better error handling
        if len(kwargs) > 1 and hasattr(next(iter(kwargs.values())), '__iter__'):
            obj = pd.DataFrame(kwargs)
        else:
            obj = pd.Series(kwargs)

        return cls(obj)

    def keys(self):
        """Get the keys the signal provides.

        Returns
        -------
        pandas.Index
            The keys of the underlying pandas object.

        Raises
        ------
        AttributeError
            If the underlying object is neither a
            :class:`pandas.DataFrame` nor a :class:`pandas.Series`.

        Notes
        -----
        For a :class:`pandas.DataFrame` backed signal the columns are
        returned, for a :class:`pandas.Series` backed one the index.
        """
        if isinstance(self._obj, pd.Series):
            return self._obj.index
        elif isinstance(self._obj, pd.DataFrame):
            return self._obj.columns
        raise AttributeError("An accessor object needs to be either a pandas.Series or a pandas.DataFrame")

    def get_missing_keys(self, keys_to_check):
        """Get a list of keys that are missing in the signal.

        Parameters
        ----------
        keys_to_check : list of str
            The keys that need to be available in the signal.

        Returns
        -------
        list of str
            The keys of ``keys_to_check`` that the signal does not provide,
            in the order in which they were given.  An empty list means that
            the signal is complete.

        Raises
        ------
        AttributeError
            If the underlying object is neither a
            :class:`pandas.DataFrame` nor a :class:`pandas.Series`.

        See Also
        --------
        fail_if_key_missing : Raise instead of returning the missing keys.

        Notes
        -----
        This method is meant to be used in the ``_validate()`` implementation
        of a derived class when a missing key is not necessarily an error,
        e.g. when a default value can be derived.
        """
        return DataValidator().get_missing_keys(self._obj, keys_to_check)

    def fail_if_key_missing(self, keys_to_check, msg=None):
        """Raise an exception if any key is missing in the signal.

        Parameters
        ----------
        keys_to_check : list of str
            The keys that need to be available in the signal.
        msg : str, optional
            A printf style format string for the error message taking two
            ``%s`` arguments: the expected keys and the missing keys.  By
            default a message naming the signal class is generated.

        Raises
        ------
        AttributeError
            If the underlying object is neither a
            :class:`pandas.DataFrame` nor a :class:`pandas.Series`, or if any
            of ``keys_to_check`` is not provided by the signal.

        See Also
        --------
        get_missing_keys : Return the missing keys instead of raising.

        Notes
        -----
        This is the canonical way to assert a signal contract in the
        ``_validate()`` implementation of a derived class.
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
        pandas.DataFrame or pandas.Series
            The pandas object representing the signal.

        Notes
        -----
        The default implementation just returns the object given when
        instantiating the signal class. Derived classes may return a modified
        or augmented object, if they store some extra information.

        By default the object is **not** copied. So make a copy yourself, if
        you intend to modify it.
        """
        return self._obj


def register_method(cls, method_name):
    """Register a method to a class derived from :class:`PylifeSignal`.

    Parameters
    ----------
    cls : class
        The class the method is registered to.
    method_name : str
        The name under which the method becomes available.

    Returns
    -------
    callable
        A decorator that registers the decorated function as
        ``method_name`` of ``cls`` and returns it unchanged.

    Raises
    ------
    ValueError
        If ``method_name`` is already registered for the class, or if the
        class already has an attribute ``method_name``.

    Notes
    -----
    The function is meant to be used as a decorator for a function
    that is to be installed as a method for a class. The class is
    assumed to contain a pandas object in ``self._obj``, which is passed to
    the decorated function as its first argument.

    Examples
    --------
    .. code-block:: python

        import pandas as pd
        from pylife import PylifeSignal
        from pylife.core import register_method

        @pd.api.extensions.register_dataframe_accessor('foo')
        class Foo(PylifeSignal):
            def _validate(self):
                self.fail_if_key_missing(['foo', 'bar'])

        @register_method(Foo, 'baz')
        def baz(df):
            return pd.DataFrame({'baz': df['foo'] + df['bar']})

        df = pd.DataFrame({'foo': [1.0, 2.0], 'bar': [-1.0, -2.0]})
        df.foo.baz()

           baz
        0  0.0
        1  0.0
    """
    return cls._register_method(method_name)

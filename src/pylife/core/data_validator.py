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

"""Check that a pandas object carries the keys a pyLife signal requires.

Every pyLife signal accessor declares a set of mandatory keys -- columns of
a :class:`pandas.DataFrame` or index entries of a :class:`pandas.Series`.
:class:`DataValidator` implements the lookup and the error reporting for
those checks.
"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import inspect
import pandas as pd


class DataValidator:
    """Check pandas objects for the keys a pyLife signal requires.

    The validator works on both :class:`pandas.Series` and
    :class:`pandas.DataFrame` objects and hides the difference between the
    two: for a ``DataFrame`` the *keys* are the columns, for a ``Series``
    they are the index entries.

    Usually you do not instantiate a ``DataValidator`` yourself.  Signal
    accessors derived from :class:`~pylife.PylifeSignal` expose the same
    checks as :meth:`~pylife.PylifeSignal.fail_if_key_missing` and
    :meth:`~pylife.PylifeSignal.get_missing_keys`.

    See Also
    --------
    pylife.PylifeSignal : Signal base class using these checks in ``_validate()``.

    Examples
    --------
    >>> import pandas as pd
    >>> from pylife import DataValidator
    >>> DataValidator().get_missing_keys(pd.Series({'sigma_a': 100.0}),
    ...                                  ['sigma_a', 'sigma_m'])
    ['sigma_m']
    """

    def keys(self, signal):
        """Get the keys a signal object provides.

        Parameters
        ----------
        signal : pandas.DataFrame or pandas.Series
            The object to be checked.

        Returns
        -------
        pandas.Index
            The keys of ``signal``.

        Raises
        ------
        AttributeError
            If ``signal`` is neither a :class:`pandas.DataFrame` nor a
            :class:`pandas.Series`.

        Notes
        -----
        If ``signal`` is a :class:`pandas.DataFrame`, ``signal.columns`` is
        returned.  If ``signal`` is a :class:`pandas.Series`, ``signal.index``
        is returned.

        Examples
        --------
        >>> import pandas as pd
        >>> from pylife import DataValidator
        >>> list(DataValidator().keys(pd.Series({'sigma_a': 100.0})))
        ['sigma_a']
        """
        if isinstance(signal, pd.Series):
            return signal.index
        elif isinstance(signal, pd.DataFrame):
            return signal.columns
        raise AttributeError("An accessor object needs to be either a pandas.Series or a pandas.DataFrame")

    def get_missing_keys(self, signal, keys_to_check):
        """Get a list of keys that are missing in a signal object.

        Parameters
        ----------
        signal : pandas.DataFrame or pandas.Series
            The object to be checked.
        keys_to_check : list of str
            The keys that need to be available in ``signal``.

        Returns
        -------
        list of str
            The keys of ``keys_to_check`` that are not present in ``signal``,
            in the order in which they were given.  An empty list means that
            ``signal`` is complete.

        Raises
        ------
        AttributeError
            If ``signal`` is neither a :class:`pandas.DataFrame` nor a
            :class:`pandas.Series`.

        See Also
        --------
        fail_if_key_missing : Raise instead of returning the missing keys.

        Notes
        -----
        If ``signal`` is a :class:`pandas.DataFrame`, the keys are looked up
        in ``signal.columns``, if it is a :class:`pandas.Series`, in
        ``signal.index``.
        """
        missing_keys = []
        for k in keys_to_check:
            if k not in self.keys(signal):
                missing_keys.append(k)
        return missing_keys

    def fail_if_key_missing(self, signal, keys_to_check, msg=None):
        """Raise an exception if any key is missing in a signal object.

        Parameters
        ----------
        signal : pandas.DataFrame or pandas.Series
            The object to be checked.
        keys_to_check : list of str
            The keys that need to be available in ``signal``.
        msg : str, optional
            A printf style format string for the error message taking two
            ``%s`` arguments: the expected keys and the missing keys.  By
            default a message naming the calling signal class is generated.

        Raises
        ------
        AttributeError
            If ``signal`` is neither a :class:`pandas.DataFrame` nor a
            :class:`pandas.Series`, or if any of ``keys_to_check`` is not
            found in the keys of ``signal``.

        See Also
        --------
        get_missing_keys : Return the missing keys instead of raising.

        Notes
        -----
        If ``signal`` is a :class:`pandas.DataFrame`, all keys of
        ``keys_to_check`` need to be found in ``signal.columns``.  If
        ``signal`` is a :class:`pandas.Series`, they need to be found in
        ``signal.index``.
        """
        missing_keys = self.get_missing_keys(signal, keys_to_check)
        if not missing_keys:
            return
        if msg is None:
            stack = inspect.stack()
            the_class = stack[2][0].f_locals['self'].__class__
            msg = the_class.__name__ + ' must have the items %s. Missing %s.'
        raise AttributeError(msg % (', '.join(keys_to_check), ', '.join(missing_keys)))

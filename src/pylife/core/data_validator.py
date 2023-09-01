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

import inspect
import pandas as pd


class DataValidator:

    def keys(self, signal):
        """Get a list of missing keys that are needed for a signal object.

        Parameters
        ----------
        signal : pandas.DataFrame or pandas.Series
            The object to be checked

        Returns
        -------
        keys : pd.Index
            a pandas index of keys

        Raises
        ------
        AttributeError
            if `signal` is neither a `pandas.DataFrame` nor a `pandas.Series`

        Notes
        -----
        If `signal` is a `pandas.DataFrame`, the `signal.columns` are returned.

        If `signal` is a `pandas.Series`, the `signal.index` are returned.
        """
        if isinstance(signal, pd.Series):
            return signal.index
        elif isinstance(signal, pd.DataFrame):
            return signal.columns
        raise AttributeError("An accessor object needs to be either a pandas.Series or a pandas.DataFrame")

    def get_missing_keys(self, signal, keys_to_check):
        """Get a list of missing keys that are needed for a signal object.

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
            if `signal` is neither a `pandas.DataFrame` nor a `pandas.Series`

        Notes
        -----
        If `signal` is a `pandas.DataFrame`, all keys of
        `keys_to_check` not found in the `signal.columns` are
        returned.

        If `signal` is a `pandas.Series`, all keys of
        `keys_to_check` not found in the `signal.index` are
        returned.
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
        :class:`stresssignal.StressTensorVoigt`
        """
        missing_keys = self.get_missing_keys(signal, keys_to_check)
        if not missing_keys:
            return
        if msg is None:
            stack = inspect.stack()
            the_class = stack[2][0].f_locals['self'].__class__
            msg = the_class.__name__ + ' must have the items %s. Missing %s.'
        raise AttributeError(msg % (', '.join(keys_to_check), ', '.join(missing_keys)))

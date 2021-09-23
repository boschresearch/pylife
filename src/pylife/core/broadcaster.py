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


import numpy as np
import pandas as pd


class Broadcaster:
    """The Broadcaster to align pyLife signals to operands.

    Parameters
    ----------
    pandas_obj : :class:`pandas.Series` or :class:`pandas.DataFrame`
       the object of the ``Broadcaster``

    In most cases the ``Broadcaster`` class is not used directly.  The
    functionality is in most cases used by the derived class
    :class:`pylife.PylifeSignal`.

    The purpose of the ``Broadcaster`` is to take two numerical objects and
    return two objects of the same numerical data with an aligned index.  That
    means that mathematical operations using the two objects as operands can be
    implemented using numpy's broadcasting functionality.

    See method :method:`pylife.Broadcaster.broadcast` documentation for details.

    The broadcasting is done in the following ways:


    ::

        object                 parameter              returned object         returned parameter

        Series                 Scalar                 Series                  Scalar
        |------|-----|                                |------|-----|
        | idx  |     |                                | idx  |     |
        |------|-----|         5.0               ->   |------|-----|          5.0
        | foo  | 1.0 |                                | foo  | 1.0 |
        | bar  | 2.0 |                                | bar  | 2.0 |
        |------|-----|                                |------|-----|


        DataFrame              Scalar                 DataFrame               Series
        |------|-----|-----|                          |------|-----|-----|    |------|-----|
        | idx  | foo | bar |                          | idx  | foo | bar |    | idx  |     |
        |------|-----|-----|                          |------|-----|-----|    |------|-----|
        | 0    | 1.0 | 2.0 |   5.0               ->   | 0    | 1.0 | 2.0 |    | 0    | 5.0 |
        | 1    | 1.0 | 2.0 |                          | 1    | 1.0 | 2.0 |    | 1    | 5.0 |
        | ...  | ... | ... |                          | ...  | ... | ... |    | ...  | ... |
        |------|-----|-----|                          |------|-----|-----|    |------|-----|


        Series                 Series/DataFrame       DataFrame               Series/DataFrame
        |------|-----|         |------|-----|         |------|-----|-----|    |------|-----|
        | None |     |         | idx  |     |         | idx  | foo | bar |    | idx  |     |
        |------|-----|         |------|-----|    ->   |------|-----|-----|    |------|-----|
        | foo  | 1.0 |         | 0    | 5.0 |         | 0    | 1.0 | 2.0 |    | 0    | 5.0 |
        | bar  | 2.0 |         | 1    | 6.0 |         | 1    | 1.0 | 2.0 |    | 1    | 6.0 |
        |------|-----|         | ...  | ... |         | ...  | ... | ... |    | ...  | ... |
                               |------|-----|         |------|-----|-----|    |------|-----|


        Series/DataFrame       Series/DataFrame       Series/DataFrame        Series/DataFrame
        |------|-----|         |------|-----|         |------|-----|          |------|-----|
        | xidx |     |         | xidx |     |         | xidx |     |          | xidx |     |
        |------|-----|         |------|-----|    ->   |------|-----|          |------|-----|
        | foo  | 1.0 |         | tau  | 5.0 |         | foo  | 1.0 |          | foo  | nan |
        | bar  | 2.0 |         | bar  | 6.0 |         | bar  | 2.0 |          | bar  | 6.0 |
        |------|-----|         |------|-----|         | tau  | nan |          | tau  | 5.0 |
                                                      |------|-----|          |------|-----|


        Series/DataFrame       Series/DataFrame       Series/DataFrame        Series/DataFrame
        |------|-----|         |------|-----|         |------|------|-----|   |------|------|-----|
        | xidx |     |         | yidx |     |         | xidx | yidx |     |   | xidx | yidx |     |
        |------|-----|         |------|-----|   ->    |------|------|-----|   |------|------|-----|
        | foo  | 1.0 |         | tau  | 5.0 |         | foo  | tau  | 1.0 |   | foo  | tau  | 5.0 |
        | bar  | 2.0 |         | chi  | 6.0 |         |      | chi  | 1.0 |   |      | chi  | 6.0 |
        |------|-----|         |------|-----|         | bar  | tau  | 2.0 |   | bar  | tau  | 5.0 |
                                                      |      | chi  | 2.0 |   |      | chi  | 6.0 |
                                                      |------|------|-----|   |------|------|-----|

    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def broadcast(self, parameter):
        """Broadcast the parameter to the object of ``self``.

        Parameters
        ----------

        parameters : scalar, numpy array or pandas object
            The parameter to broadcast to

        Returns
        -------
        parameter, object : index aligned numerical objects


        The


        Examples
        --------

        The behavior of the Broadcaster is best illustrated by examples:

        .. jupyter-execute::
           :hide-code:

           import pandas as pd
           from pylife import Broadcaster

        * Broadcasting :class:`pandas.Series` to a scalar results in a scalar
          and a :class:`pandas.Series`.

          .. jupyter-execute::

              obj = pd.Series([1.0, 2.0], index=pd.Index(['foo', 'bar'], name='idx'))
              obj

          .. jupyter-execute::

              parameter, obj = Broadcaster(obj).broadcast(5.0)

              parameter

          .. jupyter-execute::

              obj


        * Broadcasting :class:`pandas.DataFrame` to a scalar results in a
          :class:`pandas.DataFrame` and a :class:`pandas.Series`.

          .. jupyter-execute::

              obj = pd.DataFrame({
                  'foo': [1.0, 2.0],
                  'bar': [3.0, 4.0]
              }, index=pd.Index([1, 2], name='idx'))
              obj

          .. jupyter-execute::

              parameter, obj = Broadcaster(obj).broadcast(5.0)

              parameter

          .. jupyter-execute::

              obj


        * Broadcasting :class:`pandas.DataFrame` to a a :class:`pandas.Series`
          results in a :class:`pandas.DataFrame` and a :class:`pandas.Series`,
          **if and only if** the index name of the object is ``None``.

          .. jupyter-execute::

              obj = pd.Series([1.0, 2.0], index=pd.Index(['tau', 'chi']))
              obj

          .. jupyter-execute::

              parameter = pd.Series([3.0, 4.0], index=pd.Index(['foo', 'bar'], name='idx'))
              parameter

          .. jupyter-execute::

              parameter, obj = Broadcaster(obj).broadcast(parameter)

              parameter

          .. jupyter-execute::

              obj

        """
        if not isinstance(parameter, pd.Series) and not isinstance(parameter, pd.DataFrame):
            if isinstance(self._obj, pd.Series):
                return self._broadcast_series(parameter)
            return self._broadcast_frame(parameter)

        if self._obj.index.names == [None] and isinstance(self._obj, pd.Series):
            df = pd.DataFrame(index=parameter.index, columns=self._obj.index)
            for c in self._obj.index:
                df.loc[:, c] = self._obj[c]
            return parameter, df

        return self._broadcast_frame_to_frame(parameter)

    def _broadcast_series(self, parameter):
        prm = np.asarray(parameter)
        if prm.shape == ():
            return prm, self._obj

        df = self._broadcasted_dataframe(parameter)
        if isinstance(parameter, pd.Series):
            return parameter, df.set_index(parameter.index, inplace=True)

        return pd.Series(prm), df

    def _broadcast_series_to_frame(self, parameter):
        return parameter, self._broadcasted_dataframe(parameter).set_index(parameter.index)

    def _broadcast_frame_to_frame(self, parameter):
        def align_and_reorder():
            obj, prm = self._obj.align(parameter, axis=0)
            if obj.index.nlevels > 2:
                prm = prm.reorder_levels(total_columns)
                obj = obj.reorder_levels(total_columns)
            return prm, obj

        def cross_join_and_align_obj_and_parameter():
            prm_index = parameter.index.to_frame().reset_index(drop=True)
            obj_index = self._obj.index.to_frame()[obj_index_names]
            new_index = (obj_index
                         .join(prm_index, how='cross')
                         .set_index(total_columns)
                         .reorder_levels(total_columns).index)

            obj = _broadcast_to(self._obj, new_index)
            prm = _broadcast_to(parameter, new_index)

            return obj.align(prm, axis=0)

        uuids = _replace_none_index_names_with_unique_string([parameter, self._obj])

        prm_index_names = list(parameter.index.names)
        obj_index_names = list(self._obj.index.names)

        total_columns = obj_index_names + [lv for lv in prm_index_names if lv not in obj_index_names]
        have_commons = len(total_columns) < len(prm_index_names) + len(obj_index_names)

        if have_commons:
            prm, obj = align_and_reorder()
        else:
            obj, prm = cross_join_and_align_obj_and_parameter()

        _replace_unique_string_with_none_name([obj, prm, self._obj, parameter], uuids)

        return prm, obj

    def _broadcast_frame(self, parameter):
        try:
            parameter = np.broadcast_to(parameter, len(self._obj))
        except ValueError:
            raise ValueError("Dimension mismatch. "
                             "Cannot map %d value array-like to a %d element DataFrame signal."
                             %(len(parameter), len(self._obj)))
        return pd.Series(parameter, index=self._obj.index), self._obj

    def _broadcasted_dataframe(self, parameter):
        data = np.empty((len(parameter), len(self._obj)))
        df = pd.DataFrame(data, columns=self._obj.index).assign(**self._obj)
        return df


def _broadcast_to(obj, new_index):
    if isinstance(obj, pd.DataFrame):
        new = obj
    else:
        new = pd.DataFrame(obj)

    new = pd.DataFrame(index=new_index).join(new, how='left')
    if isinstance(obj, pd.Series):
        new = new.iloc[:, 0]
        new.name = obj.name
    return new


def _replace_none_index_names_with_unique_string(objs):
    import uuid

    def make_uuid():
        this_uuid = uuid.uuid4().hex
        uuids.append(this_uuid)
        return this_uuid

    uuids = []

    for obj in objs:
        obj.index.names = [name if name is not None else make_uuid() for name in obj.index.names]

    return uuids


def _replace_unique_string_with_none_name(objs, uuids):
    for obj in objs:
        obj.index.names = [None if name in uuids else name for name in obj.index.names]

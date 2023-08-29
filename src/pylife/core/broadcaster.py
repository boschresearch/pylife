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
    :class:`~pylife.PylifeSignal`.

    The purpose of the ``Broadcaster`` is to take two numerical objects and
    return two objects of the same numerical data with an aligned index.  That
    means that mathematical operations using the two objects as operands can be
    implemented using numpy's broadcasting functionality.

    See method :meth:`~pylife.Broadcaster.broadcast` documentation for details.

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

    def broadcast(self, parameter, droplevel=None):
        """Broadcast the parameter to the object of ``self``.

        Parameters
        ----------

        parameters : scalar, numpy array or pandas object
            The parameter to broadcast to


        Returns
        -------
        parameter, object : index aligned numerical objects


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
        droplevel = droplevel or []

        if not isinstance(parameter, pd.Series) and not isinstance(parameter, pd.DataFrame):
            if isinstance(self._obj, pd.Series):
                return self._broadcast_series(parameter)
            return self._broadcast_frame(parameter)

        if self._obj.index.names == [None] and isinstance(self._obj, pd.Series):
            df = pd.DataFrame(index=parameter.index, columns=self._obj.index)
            for c in self._obj.index:
                df[c] = self._obj[c]
            return parameter, df

        return self._broadcast_frame_to_frame(parameter, droplevel)

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

    def _broadcast_frame_to_frame(self, parameter, droplevel):
        def align_and_reorder():
            if isinstance(self._obj, pd.DataFrame) and isinstance(parameter, pd.Series):
                obj, prm = self._obj.align(pd.DataFrame({0: parameter}), axis=0)
                prm = prm.iloc[:, 0]
                prm.name = parameter.name
            else:
                obj, prm = self._obj.align(parameter, axis=0)

            if len(droplevel) > 0:
                prm_columns = list(filter(lambda level: level not in droplevel, total_columns))
                prm = prm.groupby(prm_columns).first()
            else:
                prm_columns = total_columns

            if obj.index.nlevels > 2:
                prm = prm.reorder_levels(prm_columns)
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

            obj, prm = obj.align(prm, axis=0)

            if len(droplevel) > 0:
                prm_columns = list(filter(lambda level: level not in droplevel, total_columns))
                prm = prm.groupby(prm_columns).first()

            return obj, prm

        uuids = _replace_none_index_names_with_unique_string([parameter, self._obj])

        index_level_cache = _IndexLevelCache(self._obj, parameter)

        prm_index_names = list(parameter.index.names)
        obj_index_names = list(self._obj.index.names)

        total_columns = obj_index_names + [lv for lv in prm_index_names if lv not in obj_index_names]
        have_commons = len(total_columns) < len(prm_index_names) + len(obj_index_names)

        if have_commons:
            prm, obj = align_and_reorder()
        else:
            obj, prm = cross_join_and_align_obj_and_parameter()

        obj.index = index_level_cache.restore_real_index(obj.index)
        prm.index = index_level_cache.restore_real_index(prm.index)

        index_level_cache.restore_original_indeces()
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
    if isinstance(new_index, pd.MultiIndex):
        new = new.reorder_levels(new_index.names)
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


class _IndexLevelCache:

    def __init__(self, obj, operand):

        self._obj_index = obj.index
        self._operand_index = operand.index

        self._obj = obj
        self._operand = operand

        common = set(obj.index.names).intersection(operand.index.names)
        only_obj = set(obj.index.names).difference(common)
        only_operand = set(operand.index.names).difference(common)

        self.index_levels = {}

        for name in common:
            obj_level = obj.index.get_level_values(name)
            operand_level = operand.index.get_level_values(name)
            self.index_levels[name] = obj_level.append(operand_level).unique()

        for name in only_obj:
            self.index_levels[name] = obj.index.get_level_values(name).unique()

        for name in only_operand:
            self.index_levels[name] = operand.index.get_level_values(name).unique()

        self.new_index_obj = self._make_new_index(obj.index)
        self.new_index_operand = self._make_new_index(operand.index)

        obj.index = self.new_index_obj
        operand.index = self.new_index_operand

    def restore_original_indeces(self):
        self._obj.index = self._obj_index
        self._operand.index =self._operand_index

    def restore_real_index(self, new_index):

        if len(new_index.names) > 1:
            real_index = pd.MultiIndex.from_arrays(
                [
                    self.index_levels[name][new_index.get_level_values(name)]
                    for name in new_index.names
                ],
                names=new_index.names
            )
        else:
            real_index = pd.Index(self.index_levels[new_index.name][new_index], name=new_index.name)

        return real_index

    def _make_new_index(self, index):
        if len(index.names) == 1:
            return pd.Index(self.index_levels[index.name].get_indexer_for(index), name=index.name)

        return pd.MultiIndex.from_arrays(
            [
                self.index_levels[name].get_indexer_for(index.get_level_values(name))
                for name in index.names
            ],
            names=index.names
        )

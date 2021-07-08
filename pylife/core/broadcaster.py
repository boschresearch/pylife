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
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def broadcast(self, parameter):
        if isinstance(self._obj, pd.Series):
            if isinstance(parameter, pd.DataFrame):
                return self._broadcast_series_to_frame(parameter)
            return self._broadcast_series(parameter)
        if isinstance(parameter, pd.DataFrame):
            return self._broadcast_frame_to_frame(parameter)
        return self._broadcast_frame(parameter)

    def _broadcast_series(self, parameter):
        prm = np.asarray(parameter)
        if prm.shape == ():
            return prm, self._obj

        df = self._broadcasted_dataframe(parameter)
        if isinstance(parameter, pd.Series):
            df.set_index(parameter.index, inplace=True)

        return prm, df

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
            new_index = (prm_index
                         .join(obj_index, how='cross')
                         .set_index(total_columns)
                         .reorder_levels(total_columns).index)

            obj = pd.DataFrame(index=new_index).join(self._obj, how='left')
            prm = parameter.join(obj, how='right')[parameter.columns]

            return obj.align(prm, axis=0)

        prm_index_names = list(parameter.index.names)
        obj_index_names = list(self._obj.index.names)
        total_columns = obj_index_names + [lv for lv in prm_index_names if lv not in obj_index_names]
        have_commons = len(total_columns) < len(prm_index_names) + len(obj_index_names)

        if have_commons:
            return align_and_reorder()

        obj, prm = cross_join_and_align_obj_and_parameter()
        return prm, obj

    def _broadcast_frame(self, parameter):
        try:
            parameter = np.broadcast_to(parameter, len(self._obj))
        except ValueError:
            raise ValueError("Dimension mismatch. "
                             "Cannot map %d value array-like to a %d element DataFrame signal."
                             %(len(parameter), len(self._obj)))
        return parameter, self._obj

    def _broadcasted_dataframe(self, parameter):
        data = np.empty((len(parameter), len(self._obj)))
        df = pd.DataFrame(data, columns=self._obj.index).assign(**self._obj)
        return df

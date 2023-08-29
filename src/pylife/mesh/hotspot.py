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

__author__ = "Daniel Christopher Kreuter"
__maintainer__ = "Johannes Mueller"

import pandas as pd
import pylife.mesh.meshsignal as meshsignal


@pd.api.extensions.register_dataframe_accessor('hotspot')
class HotSpot(meshsignal.Mesh):

    def calc(self, value_key, limit_frac=0.9, artefact_threshold=None):
        '''Calculates hotspots on a FE mesh

        Parameters
        ----------
        value_key : string
            Column name of the field variable, on which the Hot Spot
            calculation is done.

        limit_frac : float, optional
            Fraction of the max field variable. Example: If you set
            limit_frac = 0.9, the function finds all nodes and regions
            which are >= 90% of the maximum value of the field
            variable.  default: 0.9

        artefact_threshold : float, optional
            If set all the values above the `artefact_threshold` limit are not
            taken into account for the calculation of the maximum value. This
            is meant to be used for numerical artefacts which would take
            the threshold value for hotspot determined by `limit_frac` to such
            a high level, that all the relevant hotspots would "hide" underneath
            it.

        Returns
        -------
        hotspots : pandas.Series
            A Series of integers with the same index of the accessed
            mesh object indicating which mesh point belongs to which hotspot.
            A value 0 means below the `limit_frac`.

        Notes
        -----
        A loop is defined in the following way:
                * Select the node with the maximum stress value
                * Find all elements > `limit_frac` belonging to this node
                * Select all nodes > `limit_frac` belonging to these elements
                * Start loop again until all nodes > `limit_frac` are assigned to a hotspot

        Attention: All stress values are node based, not integration point based
        '''
        max_value = (self._obj[value_key].max() if artefact_threshold is None
                     else self._obj.loc[self._obj[value_key] < artefact_threshold, value_key].max())
        above_limit = self._obj[value_key] >= limit_frac*max_value
        hotspots = pd.Series(0, name='hotspot', index=self._obj.index)

        hs_index = 1
        while above_limit.any():
            hs = self.__hs_sel(above_limit, value_key)
            hotspots.loc[hs] = hs_index
            hs_index += 1
            above_limit ^= hs

        return hotspots

    def __hs_sel(self, remaining, value_key):
        max_index = self._obj.loc[remaining, value_key].idxmax()
        new_hotspot = pd.Series(False, self._obj.index)
        new_hotspot[max_index] = True

        new_entries = True
        while new_entries:
            new_entries = False
            new_nodes_idx = remaining[new_hotspot].index.get_level_values('node_id')
            new_elems_idx = remaining[new_hotspot].index.get_level_values('element_id')
            new_nodes = remaining.loc[remaining.index.isin(new_nodes_idx, level='node_id')] ^ new_hotspot
            new_elems = remaining.loc[remaining.index.isin(new_elems_idx, level='element_id')] ^ new_hotspot
            if new_nodes.any():
                new_entries = True
                new_hotspot[new_nodes] = True
            if new_elems.any():
                new_entries = True
                new_hotspot[new_elems] = True

        return new_hotspot

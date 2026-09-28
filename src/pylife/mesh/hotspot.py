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

"""Cluster connected regions of high scalar values on finite-element meshes.

The module provides the ``hotspot`` DataFrame accessor, which groups mesh
nodes whose scalar value, e.g. a damage sum or a stress in MPa, exceeds a
given fraction of the maximum into connected hotspot regions.  Hotspots
identify the critical locations of a component.
"""

__author__ = "Daniel Christopher Kreuter"
__maintainer__ = "Johannes Mueller"

import pandas as pd
import pylife.mesh.meshsignal as meshsignal


@pd.api.extensions.register_dataframe_accessor('hotspot')
class HotSpot(meshsignal.Mesh):
    """Find connected hotspot regions on finite-element meshes.

    A hotspot is a connected region of high scalar values, for example damage
    or stress in MPa, on a pyLife finite-element mesh.  Connectivity is taken
    from the ``element_id`` and ``node_id`` levels of the mesh index.

    Parameters
    ----------
    pandas_obj : pandas.DataFrame
        Mesh DataFrame with coordinate columns ``x`` and ``y`` and a
        :class:`pandas.MultiIndex` containing ``element_id`` and ``node_id``.
        The scalar field used for clustering must be stored in an additional
        column.

    See Also
    --------
    pylife.mesh.meshsignal.Mesh : Define the finite-element mesh signal
        contract used by this accessor.
    """

    def calc(self, value_key, limit_frac=0.9, artefact_threshold=None):
        r"""Calculate connected hotspots of a scalar field.

        The method labels all connected regions whose value is at least
        ``limit_frac`` times the relevant maximum.  Regions are connected if
        rows share a ``node_id`` or an ``element_id`` in the full mesh
        :class:`pandas.MultiIndex`.

        Parameters
        ----------
        value_key : str
            Column name of the scalar field used for hotspot clustering, for
            example damage or stress in MPa.  Values must be defined for every
            row of the accessed mesh.
        limit_frac : float, optional
            Fraction of the maximum field value used as hotspot threshold.
            With ``limit_frac=0.9``, all connected rows with values greater
            than or equal to 90 percent of the maximum are labeled.  Default
            is ``0.9``.
        artefact_threshold : float, optional
            Upper cutoff for calculating the maximum.  Values above
            ``artefact_threshold`` remain in the mesh but are ignored when the
            reference maximum is determined.  Use this to suppress numerical
            artefacts that would otherwise hide physically relevant hotspots.
            Default is ``None``.

        Returns
        -------
        pandas.Series
            Integer hotspot labels with the same ``element_id``/``node_id``
            index as the accessed mesh.  ``0`` marks rows below the threshold;
            positive integers identify connected hotspot regions and can be
            used for grouping, plotting or selecting critical areas.

        Notes
        -----
        The threshold value is

        .. math::

            v_\mathrm{limit} = \mathrm{limit\_frac} \cdot
            \max(v_i)

        or the same maximum after removing values above
        ``artefact_threshold``.  Starting from the remaining maximum row, the
        algorithm repeatedly adds all above-threshold rows sharing an
        ``element_id`` or ``node_id`` until the connected component is
        complete, then continues with the next component.

        The algorithm assumes node-based values.  Integration-point values
        should be extrapolated or averaged to the node-element mesh rows
        before calling this method.
        """
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

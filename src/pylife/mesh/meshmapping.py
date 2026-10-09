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

"""Map scalar values between coordinate-based meshes.

The module registers the ``meshmapper`` accessor.  It interpolates a value
column from a source point cloud or mesh onto the coordinates of the accessed
target object.  Coordinates are interpreted in mm and the target index,
including ``node_id`` and ``element_id`` for finite-element meshes, is
preserved in the result.
"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import scipy.interpolate as interp
import numpy as np
import pandas as pd
from pylife.mesh.meshsignal import PlainMesh

@pd.api.extensions.register_dataframe_accessor('meshmapper')
class Meshmapper(PlainMesh):
    """Interpolate values from one mesh to another mesh.

    Parameters
    ----------
    pandas_obj : pandas.DataFrame
        Target point cloud or mesh with coordinate columns ``x`` and ``y`` and
        optionally ``z`` in mm.  The target index is preserved in mapped
        results.

    See Also
    --------
    pylife.mesh.meshsignal.PlainMesh : Define the coordinate signal required
        for source and target meshes.
    scipy.interpolate.griddata : Interpolate unstructured point data.

    Notes
    -----
    The accessed DataFrame is the interpolation target.  The source DataFrame
    passed to :meth:`process` must also be accessible as a
    :class:`~pylife.mesh.meshsignal.PlainMesh`.
    """
    def process(self, from_df, value_key, method='linear'):
        """Map a scalar value column from a source mesh to the target mesh.

        Parameters
        ----------
        from_df : pandas.DataFrame
            Source point cloud or mesh with the same coordinate columns and
            spatial dimension as the target.  It must contain ``value_key`` and
            coordinate columns in mm.
        value_key : str
            Name of the scalar column to interpolate, for example stress in
            MPa.
        method : str, optional
            Interpolation method passed to :func:`scipy.interpolate.griddata`.
            Common choices are ``'linear'``, ``'nearest'`` and ``'cubic'``;
            availability depends on the spatial dimension.  Default is
            ``'linear'``.

        Returns
        -------
        pandas.DataFrame
            DataFrame with one column named ``value_key`` and the same index as
            the target mesh.  Values carry the same unit as the source column,
            for example MPa for stress.

        Notes
        -----
        The interpolation is purely geometrical and ignores finite-element
        connectivity.  Points outside the convex hull of the source coordinates
        receive ``NaN`` for methods such as ``'linear'``; use ``'nearest'`` if
        extrapolated nearest-neighbor values are acceptable.
        """
        crd = self._coord_keys
        from_df.plain_mesh
        newvals = interp.griddata(from_df[crd], from_df[value_key], self._obj[crd], method=method)

        return pd.DataFrame({value_key: newvals}).set_index(self._obj.index)

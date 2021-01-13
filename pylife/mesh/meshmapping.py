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

'''
Mesh Mapping
============

Map values of one FEM mesh into another

'''

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import scipy.interpolate as interp
import numpy as np
import pandas as pd
from pylife.mesh.meshsignal import PlainMeshAccessor

@pd.api.extensions.register_dataframe_accessor('meshmapper')
class MeshmapperAccessor(PlainMeshAccessor):
    '''Mapper to map points of one mesh to another

    Notes
    -----
    The accessed DataFrame needs to be accessible by a :class:`PlainMeshAccessor`.
    '''
    def process(self, from_df, value_key, method='linear'):
        '''Performs the mapping

        Parameters
        ----------
        from_df : pandas.DataFrame accessible by a :class:`PlainMeshAccessor`.
            The DataFrame that is to be mapped to the accessed one.
            Needs to have the same dimensions (2D or 3D) as the accessed one
        '''
        crd = self._coord_keys
        from_df.plain_mesh
        newvals = interp.griddata(from_df[crd], from_df[value_key], self._obj[crd], method=method)

        return pd.DataFrame({value_key: newvals}).set_index(self._obj.index)

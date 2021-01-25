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

import pandas as pd

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import cm as colormap

from pylife.mesh import meshsignal


@pd.api.extensions.register_dataframe_accessor('meshplot')
class PlotmeshAccessor(meshsignal.MeshAccessor):
    ''' Plot a value on a 2d mesh

    The accessed DataFrame must be accessible by :class:`meshsignal.MeshAccessor`.
    '''
    def plot(self, axis, value_key, **kwargs):
        '''Plot the accessed dataframe

        Parameters
        ----------
        ax : matplotlib.pyplot.axis
            The axis to plot on

        value_key : str
            The column name in the accessed DataFrame

        **kwargs
            Arguments passed to the :class:`matplotlib.collections.PatchCollection`,
            like for example ``cmap``

        See also
        --------
        :func:`pandas.api.extensions.register_dataframe_accessor()`
        '''
        patches = []
        xmin = self._obj.x.min()
        xmax = self._obj.x.max()
        ymin = self._obj.y.min()
        ymax = self._obj.y.max()

        for _, el in self._obj.groupby('element_id'):
            nds = el.loc[el.index.get_level_values('node_id')[0:el.shape[0]//2], ['x', 'y']].to_numpy()
            patches.append(Polygon(nds, closed=True))

        sf = PatchCollection(patches, **kwargs)
        sf.set_array(self._obj[value_key].groupby('element_id').mean())

        axis.add_collection(sf)
        axis.set_xlim((xmin, xmax))
        axis.set_ylim((ymin, ymax))
        axis.figure.colorbar(sf, ax=axis)

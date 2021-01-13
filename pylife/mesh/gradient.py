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

__author__ = "Mustapha Kassem"
__maintainer__ = "Johannes Mueller"

import numpy as np
import pandas as pd

from .meshsignal import MeshAccessor


@pd.api.extensions.register_dataframe_accessor('gradient')
class Gradient(MeshAccessor):
    '''Computes the gradient of a value in a 3D mesh

    Accesses a `mesh` registered in :mod:`meshsignal`

    Raises
    ------
    AttributeError
        if at least one of the columns `x`, `y` is missing
    AttributeError
        if the index of the DataFrame is not a two level MultiIndex
        with the names `node_id` and `element_id`


    Notes
    -----

    The gradient is calculated by fitting a plane into the nodes of
    each coordinate and the neigbor nodes usine least square fitting.

    The method is described in a `thread on stackoverflow`_.

    .. _thread on stackoverflow:  https://math.stackexchange.com/questions/2627946/how-to-approximate-numerically-the-gradient-of-the-function-on-a-triangular-mesh#answer-2632616
    '''
    def _find_neighbor(self):
        self.neighbors = {}

        sl = pd.IndexSlice
        for node, df in self._obj.groupby('node_id'):
            elidx = df.index.get_level_values('element_id')
            nodes = self._obj.loc[sl[:, elidx], :].index.get_level_values('node_id').to_numpy()
            self.neighbors[node] = np.setdiff1d(np.unique(nodes), [node])

    def _calc_lst_sqr(self):
        self.lst_sqr_grad_dx = {}
        self.lst_sqr_grad_dy = {}
        self.lst_sqr_grad_dz = {}
        groups = self._obj.groupby('node_id')
        self._node_data = np.zeros((len(groups), 4), order='F')
        self._node_data[:, :3] = groups.first()[['x', 'y', 'z']]
        self._node_data[:, 3] = groups[self.value_key].mean()
        for node, row in zip(groups.first().index, np.nditer(self._node_data.T, flags=['external_loop'], order='F')):
            diff = self._node_data[self.neighbors[node]-1, :] - row
            dx, dy, dz = np.linalg.lstsq(diff[:, :3], diff[:, 3], rcond=None)[0]
            self.lst_sqr_grad_dx[node] = dx
            self.lst_sqr_grad_dy[node] = dy
            self.lst_sqr_grad_dz[node] = dz

    def gradient_of(self, value_key):
        ''' returns the gradient

        Parameters
        ----------
        value_key : str
            The key of the value that forms the gradient. Needs to be found in ``df``

        Returns
        -------
        gradient : pd.DataFrame
            A table describing the gradient indexed by ``node_id``.
            The keys for the components of the gradients are
            ``['dx', 'dy', 'dz']``.
        '''
        self.value_key = value_key

        self.nodes_id = np.unique(self._obj.index.get_level_values('node_id'))

        self._find_neighbor()
        self._calc_lst_sqr()

        df_grad = pd.DataFrame({'node_id': self.nodes_id,
                                'dx': [*self.lst_sqr_grad_dx.values()],
                                'dy': [*self.lst_sqr_grad_dy.values()],
                                'dz': [*self.lst_sqr_grad_dz.values()]})
        df_grad = df_grad.set_index('node_id')

        return df_grad

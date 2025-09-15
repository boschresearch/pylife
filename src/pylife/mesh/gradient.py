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

__author__ = "Mustapha Kassem, Benjamin Maier"
__maintainer__ = "Johannes Mueller"

import numpy as np
import pandas as pd
import warnings

from .meshsignal import Mesh


@pd.api.extensions.register_dataframe_accessor('gradient')
class Gradient(Mesh):
    '''Computes the gradient of a value in a triangular 3D mesh

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
    each coordinate and the neighbor nodes using least square fitting.

    The method is described in a `thread on stackoverflow`_.

    .. _thread on stackoverflow:  https://math.stackexchange.com/questions/2627946/how-to-approximate-numerically-the-gradient-of-the-function-on-a-triangular-mesh#answer-2632616
    '''
    def _find_neighbor(self):
        self.neighbors = {}

        for node, df in self._obj.groupby('node_id'):
            elidx = df.index.get_level_values('element_id')
            nodes = self._obj.loc[self._obj.index.isin(elidx, level='element_id')].index.get_level_values('node_id').to_numpy()
            self.neighbors[node] = np.setdiff1d(np.unique(nodes), [node])

    def _calc_lst_sqr(self):
        self.lst_sqr_grad_dx = {}
        self.lst_sqr_grad_dy = {}
        self.lst_sqr_grad_dz = {}
        groups = self._obj.groupby('node_id')
        self._node_data = np.zeros((len(groups), 4), order='F')
        self._node_data[:, :3] = groups.first()[['x', 'y', 'z']]
        self._node_data[:, 3] = groups[self.value_key].mean()
        node_index = groups.first().index
        for node, row in zip(groups.first().index, np.nditer(self._node_data.T, flags=['external_loop'], order='F')):
            idx = node_index.get_indexer_for(self.neighbors[node])
            diff = self._node_data[idx, :] - row
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
            ``['d{value_key}_dx', 'd{value_key}_dy', 'd{value_key}_dz']``.
        '''
        self.value_key = value_key

        self.nodes_id = np.unique(self._obj.index.get_level_values('node_id'))

        self._find_neighbor()
        self._calc_lst_sqr()

        df_grad = pd.DataFrame({'node_id': self.nodes_id,
                                "d%s_dx" % value_key: [*self.lst_sqr_grad_dx.values()],
                                "d%s_dy" % value_key: [*self.lst_sqr_grad_dy.values()],
                                "d%s_dz" % value_key: [*self.lst_sqr_grad_dz.values()]})
        df_grad = df_grad.set_index('node_id')

        return df_grad


@pd.api.extensions.register_dataframe_accessor('gradient_3D')
class Gradient3D(Mesh):
    '''Computes the gradient of a value in a 3D mesh that was imported from Ansys or Abaqus.

    Accesses a `mesh` registered in :mod:`meshsignal`. The accessor for this type of computation
    is `gradient_3D`. Example usage:

    .. code::

        # given a mesh in `pylife_mesh` with column `mises`, compute the gradient
        gradient = pylife_mesh.gradient_3D.gradient_of('mises')

        # add the results back in the mesh
        pylife_mesh = pylife_mesh.join(grad)

    More specifically, the elements in the mesh must be either tetrahedral/simplex or hexahedral elements
    and the order of the nodes per element matters. Linear tetrahedral elements have 4 nodes,
    linear hexahedral elements have 8 nodes. Alternatively, quadratic elements may be used
    with 16 (20) or 10 nodes, respectively. In such a case only the first 4 or 8 nodes are considered
    for the gradient computation. The result contains zeros for all following nodes.

    This is consistent with the node numbering in Ansys/Abaqus, where the first 8 nodes
    of a quadratic hex elements are the same as the respective linear hex elements,
    the same applies for the first 4 nodes of a quadratic simplex element which are the
    same as in the linear simplex element.

    This class detects the type of element (tetrahedral or hexahedral) according to the number of
    nodes of each element and uses the according formula. It also works for mixed meshes that contain
    both tetrahedral and hexahedral elements.

    Note that this gradient computation only works for 3D elements and considers the node order.
    The other gradient computation accessible via `gradient` also works for 2D elements and
    disregards the order of the nodes. However, it is slower and less accurate for tetrahedral
    elements.


    Raises
    ------
    AttributeError
        if at least one of the columns `x`, `y`, `z` is missing
    AttributeError
        if the index of the DataFrame is not a two level MultiIndex
        with the names `node_id` and `element_id`
    '''

    def _initialize_ansatz_function_derivative_hexahedral(self):

        def dphi_a_dxi_j(xi,a,j):
            """Compute derivative of 3D hexahedral ansatz function φ:
            ∂φ_a/∂ξ_j(ξ)
            """
            def phi(xi,a):
                """ 1D linear (hat) function φ, φ0 for left node, φ1 for right node """
                return 1-xi if a == 0 else xi

            def dphi_dxi(a):
                """ derivative of hat function """
                return -1 if a == 0 else 1

            # select 1D ansatz function
            ax = a in [1,2,5,6]
            ay = a in [2,3,6,7]
            az = a in [4,5,6,7]

            if j == 0:
                # ∂φ_a/∂ξ_1
                return dphi_dxi(ax) * phi(xi[1],ay) * phi(xi[2],az)

            elif j == 1:
                # ∂φ_a/∂ξ_2
                return phi(xi[0],ax) * dphi_dxi(ay) * phi(xi[2],az)

            elif j == 2:
                # ∂φ_a/∂ξ_3
                return phi(xi[0],ax) * phi(xi[1],ay) * dphi_dxi(az)

        self._dphi_a_dxi_j = dphi_a_dxi_j

    def _compute_gradient_hexahedral_single_node(self, xi, nodal_values, Jinv):

        dphi_a_dxi_j = self._dphi_a_dxi_j

        # loop over derivative index d/dx_k
        result = [0, 0, 0]
        for k in range(3):

            # loop over node/ansatz function index
            for a,xi_a in enumerate([(0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1)]):
                dphi_dxk = 0

                # loop over internal index j
                for j in range(3):
                    dphi_dxij = dphi_a_dxi_j(xi,a,j)
                    dxi_dx = Jinv[j,k]

                    dphi_dxk += dphi_dxij*dxi_dx

                result[k] += nodal_values.iloc[a] * dphi_dxk
        return result

    def _compute_gradient_hexahedral(self, df):

        df["grad_x"] = 0.0
        df["grad_y"] = 0.0
        df["grad_z"] = 0.0

        # extract node positions xij where i = node number from 1 to 8, j = coordinate axis
        x11,x12,x13 = df.iloc[0,:3]
        x21,x22,x23 = df.iloc[1,:3]
        x31,x32,x33 = df.iloc[2,:3]
        x41,x42,x43 = df.iloc[3,:3]
        x51,x52,x53 = df.iloc[4,:3]
        x61,x62,x63 = df.iloc[5,:3]
        x71,x72,x73 = df.iloc[6,:3]
        x81,x82,x83 = df.iloc[7,:3]

        nodal_values = df.iloc[:8,3]

        self._initialize_ansatz_function_derivative_hexahedral()

        for node_index,xi in enumerate([(0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1)]):

            # compute jacobian matrix of the mapping from reference space [0,1]^3 to world space
            xi1,xi2,xi3 = xi

            J11 = xi3*(-x51*(1 - xi2) + x61*(1 - xi2) + x71*xi2 - x81*xi2) + (1 - xi3)*(-x11*(1 - xi2) + x21*(1 - xi2) + x31*xi2 - x41*xi2)
            J21 = xi3*(-x52*(1 - xi2) + x62*(1 - xi2) + x72*xi2 - x82*xi2) + (1 - xi3)*(-x12*(1 - xi2) + x22*(1 - xi2) + x32*xi2 - x42*xi2)
            J31 = xi3*(-x53*(1 - xi2) + x63*(1 - xi2) + x73*xi2 - x83*xi2) + (1 - xi3)*(-x13*(1 - xi2) + x23*(1 - xi2) + x33*xi2 - x43*xi2)
            J12 = xi3*(-x51*(1 - xi1) - x61*xi1 + x71*xi1 + x81*(1 - xi1)) + (1 - xi3)*(-x11*(1 - xi1) - x21*xi1 + x31*xi1 + x41*(1 - xi1))
            J22 = xi3*(-x52*(1 - xi1) - x62*xi1 + x72*xi1 + x82*(1 - xi1)) + (1 - xi3)*(-x12*(1 - xi1) - x22*xi1 + x32*xi1 + x42*(1 - xi1))
            J32 = xi3*(-x53*(1 - xi1) - x63*xi1 + x73*xi1 + x83*(1 - xi1)) + (1 - xi3)*(-x13*(1 - xi1) - x23*xi1 + x33*xi1 + x43*(1 - xi1))
            J13 = -x11*(1 - xi1)*(1 - xi2) - x21*xi1*(1 - xi2) - x31*xi1*xi2 - x41*xi2*(1 - xi1) + x51*(1 - xi1)*(1 - xi2) + x61*xi1*(1 - xi2) + x71*xi1*xi2 + x81*xi2*(1 - xi1)
            J23 = -x12*(1 - xi1)*(1 - xi2) - x22*xi1*(1 - xi2) - x32*xi1*xi2 - x42*xi2*(1 - xi1) + x52*(1 - xi1)*(1 - xi2) + x62*xi1*(1 - xi2) + x72*xi1*xi2 + x82*xi2*(1 - xi1)
            J33 = -x13*(1 - xi1)*(1 - xi2) - x23*xi1*(1 - xi2) - x33*xi1*xi2 - x43*xi2*(1 - xi1) + x53*(1 - xi1)*(1 - xi2) + x63*xi1*(1 - xi2) + x73*xi1*xi2 + x83*xi2*(1 - xi1)
            J = np.array([[J11,J12,J13],[J21,J22,J23],[J31,J32,J33]])

            # invert jacobian to map from world space to reference domain
            try:
                Jinv = np.linalg.inv(J)
            except np.linalg.LinAlgError:
                continue

            df.iloc[node_index,4:7] \
                = self._compute_gradient_hexahedral_single_node(xi, nodal_values, Jinv)

        return df

    def _initialize_ansatz_function_derivative_simplex(self):

        def dphi_a_dxi_j(node_index,a,j):
            """Compute derivative of 3D hexahedral ansatz function φ:
            ∂φ_a/∂ξ_j(ξ) where ξ is at node_index.
            Note that their derivatives are constant, thus, no dependency on node_index.
            """
            # ansatz functions for tetrahedron:
            #   phi0: 1 - xi1 - xi2 - xi3
            #   phi1: xi1
            #   phi2: xi2
            #   phi3: xi3
            # derivatives:
            #   ∂phi_0/∂ξ_j: -1 for all j
            #   ∂phi_1/∂ξ_0: 1, ∂phi1/∂ξ_j = 0 for j=1,2
            #   ∂phi_2/∂ξ_1: 1, ∂phi1/∂ξ_j = 0 for j=0,2
            #   ∂phi_3/∂ξ_2: 1, ∂phi1/∂ξ_j = 0 for j=0,1

            if a == 0:
                return -1
            return 1 if j == a - 1 else 0

        self._dphi_a_dxi_j = dphi_a_dxi_j

    def _compute_gradient_simplex_single_node(self, node_index, nodal_values, Jinv):

        dphi_a_dxi_j = self._dphi_a_dxi_j

        # loop over derivative index d/dx_k
        result = [0, 0, 0]
        for k in range(3):

            # loop over node/ansatz function index
            for a in range(4):
                dphi_dxk = 0

                # loop over internal index j
                for j in range(3):
                    dphi_dxij = dphi_a_dxi_j(node_index,a,j)
                    dxi_dx = Jinv[j,k]

                    dphi_dxk += dphi_dxij*dxi_dx

                result[k] += nodal_values.iloc[a] * dphi_dxk
        return result


    def _compute_gradient_simplex(self, df):

        df["grad_x"] = 0.0
        df["grad_y"] = 0.0
        df["grad_z"] = 0.0

        # extract node positions xij where i = node number from 1 to 8, j = coordinate axis
        x11,x12,x13 = df.iloc[0,:3]
        x21,x22,x23 = df.iloc[1,:3]
        x31,x32,x33 = df.iloc[2,:3]
        x41,x42,x43 = df.iloc[3,:3]

        nodal_values = df.iloc[:4,3]
        self._initialize_ansatz_function_derivative_simplex()

        # compute jacobian matrix of the mapping from reference space [0,1]^3 to world space
        J11 = -x11 + x21
        J21 = -x12 + x22
        J31 = -x13 + x23
        J12 = -x11 + x31
        J22 = -x12 + x32
        J32 = -x13 + x33
        J13 = -x11 + x41
        J23 = -x12 + x42
        J33 = -x13 + x43
        J = np.array([[J11,J12,J13],[J21,J22,J23],[J31,J32,J33]])

        # invert jacobian to map from world space to reference domain
        try:
            Jinv = np.linalg.inv(J)
        except np.linalg.LinAlgError:
            return df

        for node_index in range(4):

            # loop over derivative index d/dx_k
            df.iloc[node_index,4:7] \
                = self._compute_gradient_simplex_single_node(node_index, nodal_values, Jinv)

        return df

    def _compute_gradient(self, df):

        # if the element is a 8-node hexahedral element (or more nodes)
        if len(df) in [8, 16, 20]:
            return self._compute_gradient_hexahedral(df)

        # if the element is a 4-node simplex/tetrahedral element
        elif len(df) in [4, 10]:
            return self._compute_gradient_simplex(df)

        else:
            warnings.warn(f"Element has {len(df)} nodes, which is not a valid 3D element. "
                         "(Allowed numbers of nodes: 8,16,20 for hexahedral elements, 4,10 for tetrahedral elements)")

        return df

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
            ``['d{value_key}_dx', 'd{value_key}_dy', 'd{value_key}_dz']``.
        '''
        self.value_key = value_key

        assert "x" in self._obj
        assert "y" in self._obj
        assert "z" in self._obj
        assert value_key in self._obj

        # extract only the needed columns, order and sort multi-index
        df = (
            self._obj[["x", "y", "z", value_key]]
            .reorder_levels(["element_id", "node_id"])
            .sort_index(level="element_id", sort_remaining=False)
        )


        # apply the function which computes the gradient to every element separately
        df_grad = df.groupby("element_id", group_keys=False).apply(self._compute_gradient)

        # compile the resulting DataFrame
        result = pd.DataFrame(copy=True, data={
            f"d{value_key}_dx": df_grad.grad_x,
            f"d{value_key}_dy": df_grad.grad_y,
            f"d{value_key}_dz": df_grad.grad_z,
        })

        # remove "element_id" index in multi-index
        result = result.reset_index(level=0, drop=True)

        # remove duplicate indices, keep the first node
        return result[~result.index.duplicated(keep='first')]

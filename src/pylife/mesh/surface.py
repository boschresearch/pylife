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

__author__ = "Benjamin Maier"
__maintainer__ = "Johannes Mueller"

import numpy as np
import pandas as pd
import warnings
import numpy as np
import numpy.linalg

from .meshsignal import Mesh

@pd.api.extensions.register_dataframe_accessor('surface_3D')
class Surface3D(Mesh):
    '''Determines nodes at the surface in a 3D mesh.
    It also computes the outward normal vectors of the surface.
    
    Raises
    ------
    AttributeError
        if at least one of the columns `x`, `y`, `z` is missing
    AttributeError
        if the index of the DataFrame is not a two level MultiIndex
        with the names `node_id` and `element_id`
    '''

    def _solid_angle(self, x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3):

        p0 = np.array([x0, y0, z0]).T
        p1 = np.array([x1, y1, z1]).T
        p2 = np.array([x2, y2, z2]).T
        p3 = np.array([x3, y3, z3]).T

        # all vectors have shape (n,3)
        r0 = p1 - p0
        r1 = p2 - p0
        r2 = p3 - p0

        r0 /= np.linalg.norm(r0, axis=1)[:,None] * np.ones((1,3))
        r1 /= np.linalg.norm(r1, axis=1)[:,None] * np.ones((1,3))
        r2 /= np.linalg.norm(r2, axis=1)[:,None] * np.ones((1,3))

            
        a = np.arccos(np.sum(r0*r1, axis=1))
        b = np.arccos(np.sum(r0*r2, axis=1))
        c = np.arccos(np.sum(r1*r2, axis=1))

        s = (a+b+c) / 2

        # silence invalid values error, the resulting nan values will be masked out at the end
        old_settings = np.geterr()
        np.seterr(invalid="ignore")
        
        sinA = np.sqrt( (np.sin(s-b) * np.sin(s-c)) / (np.sin(b)*np.sin(c)) )
        sinB = np.sqrt( (np.sin(s-a) * np.sin(s-c)) / (np.sin(a)*np.sin(c)) )
        sinC = np.sqrt( (np.sin(s-b) * np.sin(s-a)) / (np.sin(b)*np.sin(a)) )

        cosA = np.sqrt( (np.sin(s) * np.sin(s-a)) / (np.sin(b)*np.sin(c)) )
        cosB = np.sqrt( (np.sin(s) * np.sin(s-b)) / (np.sin(b)*np.sin(c)) )
        cosC = np.sqrt( (np.sin(s) * np.sin(s-c)) / (np.sin(b)*np.sin(c)) )

        # restore silenced warnings
        np.seterr(**old_settings)

        # make large values to 1
        sinA = np.minimum(sinA, np.ones_like(sinA))
        sinB = np.minimum(sinB, np.ones_like(sinB))
        sinC = np.minimum(sinC, np.ones_like(sinC))

        cosA = np.minimum(cosA, np.ones_like(cosA))
        cosB = np.minimum(cosB, np.ones_like(cosB))
        cosC = np.minimum(cosC, np.ones_like(cosC))

        A = np.where(np.isnan(sinA), 2 * np.arccos(cosA), 2 * np.arcsin(sinA))
        B = np.where(np.isnan(sinB), 2 * np.arccos(cosB), 2 * np.arcsin(sinB))
        C = np.where(np.isnan(sinC), 2 * np.arccos(cosC), 2 * np.arcsin(sinC))

        # calculate area
        E = A + B + C - np.pi
        
        return E

    def _compute_normals(self, x0, y0, z0, x1, y1, z1, x2, y2, z2):
        p0 = np.array([x0, y0, z0]).T
        p1 = np.array([x1, y1, z1]).T
        p2 = np.array([x2, y2, z2]).T

        # all vectors have shape (n,3)
        r0 = p1 - p0
        r1 = p2 - p0

        r0 /= np.linalg.norm(r0, axis=1)[:,None] * np.ones((1,3))
        r1 /= np.linalg.norm(r1, axis=1)[:,None] * np.ones((1,3))
        normal = np.cross(r0,r1)
        normal /= np.linalg.norm(normal, axis=1)[:,None] * np.ones((1,3))

        return normal[:,0], normal[:,1], normal[:,2]

    def _determine_is_at_surface(self):
        df = self._obj[["x", "y", "z"]].reorder_levels(["element_id", "node_id"]).sort_index(level="element_id", sort_remaining=False)       
        df["node_id"] = df.index.get_level_values("node_id")

        # add two other nodes for every node
        df_nodes = df.copy()
        df_nodes = df_nodes.merge(df_nodes, on="element_id", how="outer", suffixes=["_n0", "_n1"])
        df_nodes = df_nodes[df_nodes["node_id_n0"] != df_nodes["node_id_n1"]]

        df_nodes = df_nodes.merge(df, on="element_id", how="outer").copy()
        df_nodes = df_nodes[df_nodes["node_id_n1"] < df_nodes["node_id"]]
        df_nodes = df_nodes.rename(columns={"node_id": "node_id_n2", "x": "x_n2", "y": "y_n2", "z": "z_n2"})

        df_nodes = df_nodes.merge(df, on="element_id", how="outer", suffixes=["_n2", "_n3"]).copy()
        df_nodes = df_nodes[df_nodes["node_id_n2"] < df_nodes["node_id"]]
        df0 = df_nodes.rename(columns={"node_id": "node_id_n3", "x": "x_n3", "y": "y_n3", "z": "z_n3"})

        df0.loc[:,"E"] \
            = self._solid_angle(df0.x_n0, df0.y_n0, df0.z_n0, df0.x_n1, df0.y_n1, df0.z_n1, df0.x_n2, df0.y_n2, df0.z_n2, df0.x_n3, df0.y_n3, df0.z_n3)

        max_E = df0.groupby(["element_id", "node_id_n0"]).max()["E"]

        df1 = df0.join(max_E, on=["element_id", "node_id_n0"], how="left", rsuffix="_max")
        df2 = df1[df1["E"] == df1["E_max"]]
        df2 = df2.groupby(["element_id", "node_id_n0"]).first()
        
        index_columns = df2.reset_index()
        index_columns = index_columns[["element_id", "node_id_n0"]]
        df2.index = pd.MultiIndex.from_frame(index_columns, names=["element_id", "node_id"])

        df3 = df2[["x_n0", "y_n0", "z_n0", "E"]]

        df3 = df3.join(df3.groupby(["node_id"]).sum().rename(columns={"E":"Esum"})["Esum"])
        df3.loc[:,"is_at_surface"] = df3["Esum"] < 4*np.pi-1e-5

        df3["is_at_surface"].fillna(False, inplace = True)

        return df3

    def is_at_surface(self):
        ''' Determines for every point in the mesh if it is at the mesh's surface.

        Example usage:
        
        .. code::
        
            # df_mesh is a pandas DataFrame with columns "x", "y", "z" and 
            # one row per node, indexed by a mult-index with levels
            # "element_id" and "node_id".
            is_at_surface_1 = df_mesh.surface_3D.is_at_surface()

            # The result will be a series with values 0 or 1 for every node

        .. note::
            This function is slow for large meshes. A better approach would be
            to determine surface nodes in the commercial solver (Abaqus, Ansys).

        Returns
        -------
        is_at_surface : pd.Series
            A series with the same index as the given DataFrame, indicating 
            whether the node is at a surface of the component or not.
        '''
        assert "x" in self._obj
        assert "y" in self._obj
        assert "z" in self._obj

        # extract only the needed columns, order and sort multi-index
        result = self._determine_is_at_surface()

        return result["is_at_surface"]
        
    def is_at_surface_with_normals(self):
        ''' Determines for every point in the mesh if it is at the mesh's surface,
        additionally calculate the outward normals.

        Example usage:
        
        .. code::
        
            # df_mesh is a pandas DataFrame with columns "x", "y", "z" and 
            # one row per node, indexed by a mult-index with levels
            # "element_id" and "node_id".
            is_at_surface_2 = df_mesh.surface_3D.is_at_surface_with_normals()

            # The result will be a DataFrame with values 0 or 1 for every node

        .. note::
            This function is slow for large meshes. A better approach would be
            to determine surface nodes in the commercial solver (Abaqus, Ansys).

        Returns
        -------
        is_at_surface : pd.Series
            A DataFrame with the same index as the given DataFrame and columns
            "is_at_surface", "normal_x", "normal_y", "normal_z".
            The column is_at_surface determines whether the node is at a surface
            of the component or not. If at the surface, the other columns specify
            the outward normal vector at this point.
        '''

        df = self._determine_is_at_surface()

        df_at_surface = df[df["is_at_surface"]].reset_index().set_index("element_id")

        d = df_at_surface.merge(df_at_surface, on="element_id", how="left", suffixes=["_n0", "_n1"])
        d = d[d["node_id_n0"] != d["node_id_n1"]]

        p0 = d.groupby(["element_id", "node_id_n0"], group_keys=True).nth(0)
        p1 = d.groupby(["element_id", "node_id_n0"], group_keys=True).nth(1)
        d2 = d.groupby(["element_id", "node_id_n0"], group_keys=True).first()

        p0 = p0.reset_index().set_index(["element_id", "node_id_n0"])
        p1 = p1.reset_index().set_index(["element_id", "node_id_n0"])
        d2 = d2.reset_index().set_index(["element_id", "node_id_n0"])

        d2["node_id_p1"] = p0["node_id_n1"]
        d2["x_p1"] = p0["x_n0_n1"]
        d2["y_p1"] = p0["y_n0_n1"]
        d2["z_p1"] = p0["z_n0_n1"]

        d2["node_id_p2"] = p1["node_id_n1"]
        d2["x_p2"] = p1["x_n0_n1"]
        d2["y_p2"] = p1["y_n0_n1"]
        d2["z_p2"] = p1["z_n0_n1"]

        d2.loc[:,"normal_x"], d2.loc[:,"normal_y"], d2.loc[:,"normal_z"] \
        = self._compute_normals(d2.x_n0_n0, d2.y_n0_n0, d2.z_n0_n0, d2.x_p1, d2.y_p1, d2.z_p1, d2.x_p2, d2.y_p2, d2.z_p2)

        df_with_normals = d2[["normal_x", "normal_y", "normal_z"]]
        df_with_normals.index.names = ["element_id", "node_id"]
        
        df_result = df.join(df_with_normals)[["is_at_surface", "normal_x", "normal_y", "normal_z"]]

        return df_result


    
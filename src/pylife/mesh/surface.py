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
"""Evaluate surface information for three-dimensional meshes.

The module registers the ``surface_3D`` accessor.  It identifies surface nodes
and outward surface normals on connected solid meshes.  These quantities are
used together with stress gradients as input for FKM nonlinear support-factor
assessments.
"""

__author__ = "Benjamin Maier"
__maintainer__ = "Johannes Mueller"

import numpy as np
import pandas as pd

from .meshsignal import Mesh


@pd.api.extensions.register_dataframe_accessor('surface_3D')
class Surface3D(Mesh):
    r"""Determine surface nodes and normals of a 3D solid mesh.

    The accessor works on pyLife finite-element meshes with coordinates in mm
    and an ``element_id``/``node_id`` index.  It produces per-row information
    that can be joined to stress and stress-gradient data for FKM nonlinear
    assessments.

    Parameters
    ----------
    pandas_obj : pandas.DataFrame
        Mesh DataFrame with coordinate columns ``x``, ``y`` and ``z`` in mm
        and a :class:`pandas.MultiIndex` containing ``element_id`` and
        ``node_id``.

    Raises
    ------
    AttributeError
        If at least one of the coordinate columns ``x`` or ``y`` is missing.
    AttributeError
        If the index of the DataFrame does not contain the levels ``node_id``
        and ``element_id``.

    See Also
    --------
    pylife.mesh.gradient.Gradient3D : Compute stress gradients used together
        with surface normals.
    pylife.mesh.meshsignal.Mesh : Define the finite-element mesh signal
        contract.

    Notes
    -----
    Surface detection sums the solid angle contributions around each
    ``node_id``.  Interior nodes of a closed volume reach approximately

    .. math::

        \sum \Omega_i = 4\pi

    whereas nodes with a smaller angle sum are classified as surface nodes.

    The implementation is intended for 3D solid meshes and can be slow for
    large industrial models.  If possible, determine surface nodes and normals
    in the FE solver and import them with the result data.
    """

    def _solid_angle(self, df):
        n = len(df)

        p0 = np.array([df.x_n0, df.y_n0, df.z_n0]).T
        p1 = np.array([df.x_n1, df.y_n1, df.z_n1]).T
        p2 = np.array([df.x_n2, df.y_n2, df.z_n2]).T
        p3 = np.array([df.x_n3, df.y_n3, df.z_n3]).T

        # all vectors have shape (n,3)
        r0 = p1 - p0
        r1 = p2 - p0
        r2 = p3 - p0

        # silence invalid values error, the resulting nan values will be masked out at the end
        with np.errstate(invalid="ignore"):
            r0 /= np.broadcast_to(np.linalg.norm(r0, axis=1), (3, n)).T
            r1 /= np.broadcast_to(np.linalg.norm(r1, axis=1), (3, n)).T
            r2 /= np.broadcast_to(np.linalg.norm(r2, axis=1), (3, n)).T

            a = np.arccos(np.sum(r0*r1, axis=1))
            b = np.arccos(np.sum(r0*r2, axis=1))
            c = np.arccos(np.sum(r1*r2, axis=1))

            s = (a+b+c) / 2

            sinA = np.sqrt((np.sin(s-b) * np.sin(s-c)) / (np.sin(b)*np.sin(c)))
            sinB = np.sqrt((np.sin(s-a) * np.sin(s-c)) / (np.sin(a)*np.sin(c)))
            sinC = np.sqrt((np.sin(s-b) * np.sin(s-a)) / (np.sin(b)*np.sin(a)))

            cosA = np.sqrt((np.sin(s) * np.sin(s-a)) / (np.sin(b)*np.sin(c)))
            cosB = np.sqrt((np.sin(s) * np.sin(s-b)) / (np.sin(b)*np.sin(c)))
            cosC = np.sqrt((np.sin(s) * np.sin(s-c)) / (np.sin(b)*np.sin(c)))

        # make large values to 1
        sinA = np.minimum(sinA, 1.0)
        sinB = np.minimum(sinB, 1.0)
        sinC = np.minimum(sinC, 1.0)

        cosA = np.minimum(cosA, 1.0)
        cosB = np.minimum(cosB, 1.0)
        cosC = np.minimum(cosC, 1.0)

        A = np.where(np.isnan(sinA), 2 * np.arccos(cosA), 2 * np.arcsin(sinA))
        B = np.where(np.isnan(sinB), 2 * np.arccos(cosB), 2 * np.arcsin(sinB))
        C = np.where(np.isnan(sinC), 2 * np.arccos(cosC), 2 * np.arcsin(sinC))

        # calculate area
        E = A + B + C - np.pi

        return E

    def _compute_normals(self, df):
        n = len(df)

        p0 = np.array([df.x_n0_n0, df.y_n0_n0, df.z_n0_n0]).T
        p1 = np.array([df.x_p1, df.y_p1, df.z_p1]).T
        p2 = np.array([df.x_p2, df.y_p2, df.z_p2]).T

        # all vectors have shape (n,3)
        r0 = p1 - p0
        r1 = p2 - p0

        r0 /= np.broadcast_to(np.linalg.norm(r0, axis=1), (3, n)).T
        r1 /= np.broadcast_to(np.linalg.norm(r1, axis=1), (3, n)).T

        normal = np.cross(r0, r1)
        normal /= np.broadcast_to(np.linalg.norm(normal, axis=1), (3, n)).T

        return normal

    def _determine_is_at_surface(self):
        df = (
            self.coordinates
            #.reorder_levels(["element_id", "node_id"])
            #.sort_index(level="element_id", sort_remaining=False)
            .assign(node_id=self._obj.index.get_level_values("node_id"))
        )
        # add two other nodes for every node
        df0 = (
            df.merge(df, on="element_id", how="outer", suffixes=["_n0", "_n1"])
            .query("node_id_n0 != node_id_n1")
            .merge(df, on="element_id", how="outer")
            .query("node_id_n1 < node_id")
            .rename(
                columns={"node_id": "node_id_n2", "x": "x_n2", "y": "y_n2", "z": "z_n2"}
            )
            .merge(df, on="element_id", how="outer", suffixes=["_n2", "_n3"])
            .query("node_id_n2 < node_id")
            .rename(
                columns={"node_id": "node_id_n3", "x": "x_n3", "y": "y_n3", "z": "z_n3"}
            )
            .pipe(lambda df: df.assign(E=self._solid_angle(df)))
        )

        max_E = df0.groupby(["element_id", "node_id_n0"]).max()["E"]

        df1 = (
            df0.join(max_E, on=["element_id", "node_id_n0"], how="left", rsuffix="_max")
            .query("E == E_max")
            .groupby(["element_id", "node_id_n0"])
            .first()
            .reset_index()
            .rename(columns={"node_id_n0": "node_id"})
            .set_index(["element_id", "node_id"])
        )

        df3 = df1[["x_n0", "y_n0", "z_n0", "E"]].join(
            df1.groupby(["node_id"]).sum().rename(columns={"E": "Esum"})["Esum"]
        )

        df3.loc[:, "is_at_surface"] = df3["Esum"] < 4*np.pi-1e-5

        return df3

    def is_at_surface(self):
        """Determine whether each mesh row lies on the component surface.

        Returns
        -------
        pandas.Series
            Boolean series with the same ``element_id``/``node_id`` index as
            the accessed mesh.  ``True`` marks rows whose ``node_id`` is on the
            surface of the component.

        Notes
        -----
        The method requires coordinates ``x``, ``y`` and ``z`` in mm.  The
        result has the full finite-element mesh index, not a node-averaged
        index, so duplicated ``node_id`` values can occur for nodes shared by
        several elements.

        This calculation is slow for large meshes.  Prefer importing surface
        flags from the FE solver when they are available.
        """
        assert "x" in self._obj
        assert "y" in self._obj
        assert "z" in self._obj

        # extract only the needed columns, order and sort multi-index
        result = self._determine_is_at_surface()

        return result["is_at_surface"]

    def is_at_surface_with_normals(self):
        """Determine surface membership and outward normal vectors.

        Returns
        -------
        pandas.DataFrame
            DataFrame with the same ``element_id``/``node_id`` index as the
            accessed mesh and the columns ``is_at_surface``, ``normal_x``,
            ``normal_y`` and ``normal_z``.  Normal components are unitless and
            describe the outward normal direction at surface rows; non-surface
            rows contain ``NaN`` normal components.

        Notes
        -----
        Coordinates are interpreted in mm.  The result retains the full
        finite-element mesh index, so a ``node_id`` shared by several elements
        can appear multiple times.

        This calculation is slow for large meshes.  Prefer importing surface
        normal vectors from the FE solver when they are available.
        """

        df = self._determine_is_at_surface()

        df_at_surface = df[df["is_at_surface"]].reset_index("node_id")

        d = df_at_surface.merge(df_at_surface, on="element_id", how="left", suffixes=["_n0", "_n1"])
        d = d[d["node_id_n0"] != d["node_id_n1"]]

        groups = d.groupby(["element_id", "node_id_n0"], group_keys=True)
        p0 = groups.first()
        p1 = groups.nth(1).set_index("node_id_n0", append=True)
        d2 = groups.first()

        d2["node_id_p1"] = p0["node_id_n1"]
        d2["x_p1"] = p0["x_n0_n1"]
        d2["y_p1"] = p0["y_n0_n1"]
        d2["z_p1"] = p0["z_n0_n1"]

        d2["node_id_p2"] = p1["node_id_n1"]
        d2["x_p2"] = p1["x_n0_n1"]
        d2["y_p2"] = p1["y_n0_n1"]
        d2["z_p2"] = p1["z_n0_n1"]

        df_with_normals = pd.DataFrame(
            self._compute_normals(d2),
            columns=["normal_x", "normal_y", "normal_z"],
            index=d2.index,
        )
        df_with_normals.index.names = ["element_id", "node_id"]

        df_result = df.join(df_with_normals)[["is_at_surface", "normal_x", "normal_y", "normal_z"]]

        return df_result

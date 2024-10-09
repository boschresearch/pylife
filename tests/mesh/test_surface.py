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
__maintainer__ = __author__

import pylife.mesh.surface
import pandas as pd
import numpy as np
import pytest


def spherical_mesh(n_elements_phi = 6, n_elements_theta = 4, n_elements_r = 3, offset = np.array([1,2,3])):

    coordinates = []
    for k,r in enumerate(np.linspace(3, 5, n_elements_r+1)):
        for j,theta in enumerate(np.linspace(np.deg2rad(10), np.deg2rad(50), n_elements_theta+1)):    # [0, np.pi]
            for i,phi in enumerate(np.linspace(np.deg2rad(100), np.deg2rad(130), n_elements_phi+1)):   # [-np.pi, np.pi]

                p = offset + np.array([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)])
                coordinates.append(p)

    node_id_list = []
    element_id_list = []
    is_at_surface_list = []

    # fill elements ids and node ids
    for k in range(n_elements_r):
        for j in range(n_elements_theta):
            for i in range(n_elements_phi):

                for _ in range(8):
                    element_id_list.append(k*n_elements_theta*n_elements_phi + j*n_elements_phi + i)

                node_id_list.append(k*(n_elements_theta+1)*(n_elements_phi+1) + j*(n_elements_phi+1) + i)
                node_id_list.append(k*(n_elements_theta+1)*(n_elements_phi+1) + j*(n_elements_phi+1) + i+1)
                node_id_list.append(k*(n_elements_theta+1)*(n_elements_phi+1) + (j+1)*(n_elements_phi+1) + i)
                node_id_list.append(k*(n_elements_theta+1)*(n_elements_phi+1) + (j+1)*(n_elements_phi+1) + i+1)
                node_id_list.append((k+1)*(n_elements_theta+1)*(n_elements_phi+1) + j*(n_elements_phi+1) + i)
                node_id_list.append((k+1)*(n_elements_theta+1)*(n_elements_phi+1) + j*(n_elements_phi+1) + i+1)
                node_id_list.append((k+1)*(n_elements_theta+1)*(n_elements_phi+1) + (j+1)*(n_elements_phi+1) + i)
                node_id_list.append((k+1)*(n_elements_theta+1)*(n_elements_phi+1) + (j+1)*(n_elements_phi+1) + i+1)

                is_at_surface_list.append(k == 0 or j == 0 or i == 0)
                is_at_surface_list.append(k == 0 or j == 0 or i == n_elements_phi-1)
                is_at_surface_list.append(k == 0 or j == n_elements_theta-1 or i == 0)
                is_at_surface_list.append(k == 0 or j == n_elements_theta-1 or i == n_elements_phi-1)
                is_at_surface_list.append(k == n_elements_r-1 or j == 0 or i == 0)
                is_at_surface_list.append(k == n_elements_r-1 or j == 0 or i == n_elements_phi-1)
                is_at_surface_list.append(k == n_elements_r-1 or j == n_elements_theta-1 or i == 0)
                is_at_surface_list.append(k == n_elements_r-1 or j == n_elements_theta-1 or i == n_elements_phi-1)


    # fill coordinates
    x_list = [coordinates[node_id][0] for node_id in node_id_list]
    y_list = [coordinates[node_id][1] for node_id in node_id_list]
    z_list = [coordinates[node_id][2] for node_id in node_id_list]

    df = pd.DataFrame({'node_id':     node_id_list,
                        'element_id': element_id_list,
                        'x': x_list,
                        'y': y_list,
                        'z': z_list}).set_index(['node_id', 'element_id'])

    return df, is_at_surface_list


@pytest.mark.parametrize('n_elements_phi, n_elements_theta, n_elements_r, offset', [
    (3, 3, 3, np.array([0, 0, 0])),
    (3, 3, 3, np.array([0, 1, 0])),
    (3, 3, 3, np.array([1.2, .3, 4.5])),
    (1, 3, 3, np.array([1.2, .3, 4.5])),
    (3, 1, 3, np.array([1.2, .3, 4.5])),
    (3, 3, 1, np.array([1.2, .3, 4.5])),
    (3, 1, 1, np.array([1.2, .3, 4.5])),
    (1, 1, 1, np.array([1.2, .3, 4.5])),
    (5, 4, 3, np.array([1.2, .3, 4.5])),
])
def test_surface_3D(n_elements_phi, n_elements_theta, n_elements_r, offset):

    # mesh is a segment of a sphere
    df_mesh, expected = spherical_mesh(n_elements_phi, n_elements_theta, n_elements_r, offset)

    is_at_surface_1 = df_mesh.surface_3D.is_at_surface()
    is_at_surface_2 = df_mesh.surface_3D.is_at_surface_with_normals()

    np.testing.assert_array_equal(is_at_surface_1, expected)
    np.testing.assert_array_equal(is_at_surface_2["is_at_surface"].values, expected)


@pytest.mark.parametrize('n_elements_phi, n_elements_theta, n_elements_r, offset', [
    (3, 3, 3, np.array([0, 0, 0])),
    (3, 3, 3, np.array([0, 1, 0])),
    (3, 3, 3, np.array([1.2, .3, 4.5])),
    (1, 3, 3, np.array([1.2, .3, 4.5])),
    (3, 1, 3, np.array([1.2, .3, 4.5])),
    (3, 3, 1, np.array([1.2, .3, 4.5])),
    (3, 1, 1, np.array([1.2, .3, 4.5])),
    (1, 1, 1, np.array([1.2, .3, 4.5])),
    (5, 4, 3, np.array([1.2, .3, 4.5])),
])
def test_surface_3D_flipped_indeces(n_elements_phi, n_elements_theta, n_elements_r, offset):

    # mesh is a segment of a sphere
    df_mesh, expected = spherical_mesh(n_elements_phi, n_elements_theta, n_elements_r, offset)

    df_mesh = df_mesh.reorder_levels(["element_id", "node_id"])

    is_at_surface_1 = df_mesh.surface_3D.is_at_surface()
    is_at_surface_2 = df_mesh.surface_3D.is_at_surface_with_normals()

    np.testing.assert_array_equal(is_at_surface_1, expected)
    np.testing.assert_array_equal(is_at_surface_2["is_at_surface"].values, expected)


@pytest.mark.parametrize('n_elements_phi, n_elements_theta, n_elements_r, offset', [
    (3, 3, 3, np.array([0, 0, 0])),
    (3, 3, 3, np.array([0, 1, 0])),
    (3, 3, 3, np.array([1.2, .3, 4.5])),
    (1, 3, 3, np.array([1.2, .3, 4.5])),
    (3, 1, 3, np.array([1.2, .3, 4.5])),
    (3, 3, 1, np.array([1.2, .3, 4.5])),
    (3, 1, 1, np.array([1.2, .3, 4.5])),
    (1, 1, 1, np.array([1.2, .3, 4.5])),
    (5, 4, 3, np.array([1.2, .3, 4.5])),
])
def test_surface_3D_additional_index(n_elements_phi, n_elements_theta, n_elements_r, offset):

    # mesh is a segment of a sphere
    df_mesh, expected = spherical_mesh(n_elements_phi, n_elements_theta, n_elements_r, offset)

    df_mesh = df_mesh.assign(additional=1).set_index("additional", append=True)

    is_at_surface_1 = df_mesh.surface_3D.is_at_surface()
    is_at_surface_2 = df_mesh.surface_3D.is_at_surface_with_normals()

    np.testing.assert_array_equal(is_at_surface_1, expected)
    np.testing.assert_array_equal(is_at_surface_2["is_at_surface"].values, expected)

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
import pytest

import numpy as np
import pandas as pd

import pylife.mesh.meshsignal


def test_plain_mesh_3d():
    df = pd.DataFrame({'x': [1.0], 'y': [2.0], 'z': [3.0], 'a': [9.9]})
    pd.testing.assert_frame_equal(df.plain_mesh.coordinates,
                                  pd.DataFrame({'x': [1.0], 'y': [2.0], 'z': [3.0]}))


def test_plain_mesh_2d():
    df = pd.DataFrame({'x': [1.0], 'y': [2.0], 'b': [3.0], 'a': [9.9]})
    pd.testing.assert_frame_equal(df.plain_mesh.coordinates,
                                  pd.DataFrame({'x': [1.0], 'y': [2.0]}))


def test_plain_mesh_3d_dims():
    df = pd.DataFrame({'x': [1.0, 2.0], 'y': [2.0, 3.0], 'z': [3.0, 4.0], 'b': [3.0, 3.0]})
    assert df.plain_mesh.dimensions == 3
    assert df.plain_mesh.dimensions == 3


def test_plain_mesh_2d_dims():
    df = pd.DataFrame({'x': [1.0, 2.0], 'y': [2.0, 3.0], 'b': [3.0, 3.0]})
    assert df.plain_mesh.dimensions == 2
    assert df.plain_mesh.dimensions == 2


def test_plain_mesh_pseudeo_2d_dims():
    df = pd.DataFrame({'x': [1.0, 2.0], 'y': [2.0, 3.0], 'z': [3.0, 3.0], 'b': [3.0, 3.0]})
    assert df.plain_mesh.dimensions == 2
    assert df.plain_mesh.dimensions == 2


def test_plain_mesh_fail():
    df = pd.DataFrame({'x': [1.0], 't': [2.0], 'b': [3.0], 'a': [9.9]})
    with pytest.raises(AttributeError, match=r'PlainMesh.*Missing y'):
        df.plain_mesh.coordinates


def test_mesh_3d():
    mi = pd.MultiIndex.from_tuples([(1, 1)], names=['element_id', 'node_id'])
    df = pd.DataFrame({'x': [1.0], 'y': [2.0], 'z': [3.0], 'a': [9.9]}).set_index(mi)
    pd.testing.assert_frame_equal(
        df.mesh.coordinates,
        pd.DataFrame({'x': [1.0], 'y': [2.0], 'z': [3.0]}).set_index(mi),
    )


def test_mesh_2d():
    mi = pd.MultiIndex.from_tuples([(1, 1)], names=['element_id', 'node_id'])
    df = pd.DataFrame({'x': [1.0], 'y': [2.0], 'c': [3.0], 'a': [9.9]}).set_index(mi)
    pd.testing.assert_frame_equal(
        df.mesh.coordinates, pd.DataFrame({'x': [1.0], 'y': [2.0]}).set_index(mi)
    )


# GH-111
def test_mesh_additional_index():
    mi = pd.MultiIndex.from_tuples([(1, 1, 1)], names=['element_id', 'node_id', 'additional'])
    df = pd.DataFrame({'x': [1.0], 'y': [2.0], 'c': [3.0], 'a': [9.9]}).set_index(mi)
    pd.testing.assert_frame_equal(
        df.mesh.coordinates, pd.DataFrame({'x': [1.0], 'y': [2.0]}).set_index(mi)
    )


# GH-111
def test_mesh_fail_coordinates():
    mi = pd.MultiIndex.from_tuples([(1, 1)], names=['element_id', 'node_id'])
    df = pd.DataFrame({'x': [1.0], 'e': [2.0], 'c': [3.0], 'a': [9.9]}).set_index(mi)
    with pytest.raises(AttributeError, match=r'Mesh.*Missing y'):
        df.mesh.coordinates


def test_mesh_fail_index_missing_both():
    df = pd.DataFrame({'x': [1.0], 'y': [2.0], 'z': [3.0], 'a': [9.9]})
    with pytest.raises(AttributeError, match=r'.*element_id.*'):
        df.mesh.coordinates


def test_mesh_fail_index_missing_element():
    df = pd.DataFrame(
        {'x': [1.0], 'y': [2.0], 'z': [3.0], 'a': [9.9]},
        index=pd.Index([1], name="node_id"),
    )
    with pytest.raises(AttributeError, match=r'.*element_id.*'):
        df.mesh.coordinates


def test_mesh_fail_index_missing_node():
    df = pd.DataFrame(
        {'x': [1.0], 'y': [2.0], 'z': [3.0], 'a': [9.9]},
        index=pd.Index([1], name="element_id"),
    )
    with pytest.raises(AttributeError, match=r'.*element_id.*'):
        df.mesh.coordinates


def test_connectivity():
    mi = pd.MultiIndex.from_tuples([(1, 2), (1, 1), (1, 3),
                                    (2, 5), (2, 4), (2, 6)], names=['element_id', 'node_id'])
    df = pd.DataFrame({'x': np.arange(1, 7), 'y': np.arange(2, 8)}, index=mi)

    expected = pd.Series([[2, 1, 3], [5, 4, 6]], name='node_id', index=pd.Index([1, 2], name='element_id'))
    pd.testing.assert_series_equal(df.mesh.connectivity, expected)


def test_vtk_grid_return_types():
    mi = pd.MultiIndex.from_tuples([(1, 17), (1, 23), (1, 3)], names=['element_id', 'node_id'])
    df = pd.DataFrame([[0., 0.], [0., 2.], [2., 0.], ], columns=['x', 'y'], index=mi)

    cells, cell_types, points = df.mesh.vtk_data()

    assert isinstance(cells, np.ndarray)
    assert isinstance(cell_types, np.ndarray)
    assert isinstance(points, np.ndarray)


def test_vtk_grid_2d_tri_lin():
    mi = pd.MultiIndex.from_tuples([
        (1, 17), (1, 23), (1, 3),
        (2, 23), (2, 3), (2, 7)
    ], names=['element_id', 'node_id'])

    df = pd.DataFrame([
        [0., 0.],  # 17
        [0., 2.],  # 23
        [2., 0.],  # 3
        [0., 2.],  # 23
        [2., 0.],  # 3
        [2., 2.]   # 7
    ], columns=['x', 'y'], index=mi)

    expected_cells = [
        3, 2, 3, 0,
        3, 3, 0, 1
    ]
    expected_points = [
        [2., 0.],  # 3 : 0
        [2., 2.],  # 7 : 1
        [0., 0.],  # 17 : 2
        [0., 2.],  # 23 : 3
    ]
    expected_cell_types = [5, 5]  # VTK_TRIANGLE

    cells, cell_types, points = df.mesh.vtk_data()

    np.testing.assert_allclose(expected_cells, cells)
    np.testing.assert_allclose(expected_points, points)
    np.testing.assert_allclose(expected_cell_types, cell_types)


def test_vtk_grid_2d_tri_quad():
    mi = pd.MultiIndex.from_tuples([
        (1, 17), (1, 23), (1, 3), (1, 12), (1, 19), (1, 21),
        (2, 23), (2, 3), (2, 7), (2, 13), (2, 19), (2, 33)
    ], names=['element_id', 'node_id'])

    df = pd.DataFrame([
        [0., 0.],  # 17
        [0., 2.],  # 23
        [2., 0.],  # 3
        [7., 7.],
        [7., 7.],
        [7., 7.],
        [0., 2.],  # 23
        [2., 0.],  # 3
        [2., 2.],  # 7
        [7., 7.],
        [7., 7.],
        [7., 7.],
    ], columns=['x', 'y'], index=mi)

    expected_cells = [
        3, 2, 3, 0,
        3, 3, 0, 1
    ]
    expected_points = [
        [2., 0.],  # 3 : 0
        [2., 2.],  # 7 : 1
        [0., 0.],  # 17 : 2
        [0., 2.],  # 23 : 3
    ]
    expected_cell_types = [5, 5]  # VTK_TRIANGLE

    cells, cell_types, points = df.mesh.vtk_data()

    np.testing.assert_allclose(expected_cells, cells)
    np.testing.assert_allclose(expected_points, points)
    np.testing.assert_allclose(expected_cell_types, cell_types)


def test_vtk_grid_2d_squ_lin():
    mi = pd.MultiIndex.from_tuples([
        (1, 17), (1, 23), (1, 7), (1, 3),
        (2, 3), (2, 7), (2, 9), (2, 11)
    ], names=['element_id', 'node_id'])

    df = pd.DataFrame([
        [0., 0.],  # 17
        [0., 2.],  # 23
        [2., 2.],  # 7
        [2., 0.],  # 3
        [2., 0.],  # 3
        [2., 2.],  # 7
        [4., 2.],  # 9
        [4., 0.],  # 11
    ], columns=['x', 'y'], index=mi)

    expected_cells = [
        4, 4, 5, 1, 0,
        4, 0, 1, 2, 3
    ]
    expected_points = [
        [2., 0.],  # 3 : 0
        [2., 2.],  # 7 : 1
        [4., 2.],  # 9 : 2
        [4., 0.],  # 11 : 3
        [0., 0.],  # 17 : 4
        [0., 2.],  # 23 : 5
    ]
    expected_cell_types = [9, 9]  # VTK_QUAD

    cells, cell_types, points = df.mesh.vtk_data()

    np.testing.assert_allclose(expected_cells, cells)
    np.testing.assert_allclose(expected_points, points)
    np.testing.assert_allclose(expected_cell_types, cell_types)


def test_vtk_grid_2d_squ_quad():
    mi = pd.MultiIndex.from_tuples([
        (1, 17), (1, 23), (1, 7), (1, 3), (1, 13), (1, 37), (1, 42), (1, 137),
        (2, 3), (2, 7), (2, 9), (2, 11), (2, 15), (2, 42), (2, 33), (2, 134)
    ], names=['element_id', 'node_id'])

    df = pd.DataFrame([
        [0., 0.],  # 17
        [0., 2.],  # 23
        [2., 2.],  # 7
        [2., 0.],  # 3
        [7., 7.],
        [7., 7.],
        [7., 7.],
        [7., 7.],
        [2., 0.],  # 3
        [2., 2.],  # 7
        [4., 2.],  # 9
        [4., 0.],  # 11
        [7., 7.],
        [7., 7.],
        [7., 7.],
        [7., 7.],
    ], columns=['x', 'y'], index=mi)

    expected_cells = [
        4, 4, 5, 1, 0,
        4, 0, 1, 2, 3
    ]
    expected_points = [
        [2., 0.],  # 3 : 0
        [2., 2.],  # 7 : 1
        [4., 2.],  # 9 : 2
        [4., 0.],  # 11 : 3
        [0., 0.],  # 17 : 4
        [0., 2.],  # 23 : 5
    ]
    expected_cell_types = [9, 9]  # VTK_QUAD

    cells, cell_types, points = df.mesh.vtk_data()

    np.testing.assert_allclose(expected_cells, cells)
    np.testing.assert_allclose(expected_points, points)
    np.testing.assert_allclose(expected_cell_types, cell_types)


def test_vtk_grid_3d_hex_lin():
    mi = pd.MultiIndex.from_tuples([(1, 5), (1, 6), (1, 8), (1, 7), (1, 1), (1, 2), (1, 4), (1, 3),
                                    (2, 9), (2, 10), (2, 12), (2, 11), (2, 5), (2, 6), (2, 8), (2, 7)],
                                   names=['element_id', 'node_id'])

    df = pd.DataFrame([
        [1., 0., 1.],
        [1., 1., 1.],
        [1., 1., 0.],
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 1., 1.],
        [0., 1., 0.],
        [0., 0., 0.],
        [2., 0., 1.],
        [2., 1., 1.],
        [2., 1., 0.],
        [2., 0., 0.],
        [1., 0., 1.],
        [1., 1., 1.],
        [1., 1., 0.],
        [1., 0., 0.],
    ], columns=['x', 'y', 'z'], index=mi)

    expected_cells = [
        8, 4, 5, 7, 6, 0, 1, 3, 2,
        8, 8, 9, 11, 10, 4, 5, 7, 6
    ]
    expected_cell_types = [12, 12]  # VTK_HEXAHEDRON
    expected_points = [
        [0., 0., 1.],
        [0., 1., 1.],
        [0., 0., 0.],
        [0., 1., 0.],
        [1., 0., 1.],
        [1., 1., 1.],
        [1., 0., 0.],
        [1., 1., 0.],
        [2., 0., 1.],
        [2., 1., 1.],
        [2., 0., 0.],
        [2., 1., 0.],
    ]

    cells, cell_types, points = df.mesh.vtk_data()

    np.testing.assert_allclose(expected_cells, cells)
    np.testing.assert_allclose(expected_points, points)
    np.testing.assert_allclose(expected_cell_types, cell_types)


def test_vtk_grid_3d_hex_quad():
    mi = pd.MultiIndex.from_tuples([
        (1,  5), (1,  7), (1,  3), (1,  1), (1,  6), (1,  8), (1,  4), (1,  2),
        (1, 24), (1, 23), (1, 22), (1, 21),
        (1, 25), (1, 26), (1, 27), (1, 28),
        (1, 30), (1, 29), (1, 31), (1, 32),
        (2,  9), (2, 11), (2,  7), (2,  5), (2, 10), (2, 12), (2,  8), (2,  6),
        (2, 35), (2, 34), (2, 24), (2, 33),
        (2, 36), (2, 37), (2, 25), (2, 38),
        (2, 40), (2, 39), (2, 29), (2, 30)
    ], names=['element_id', 'node_id'])

    df = pd.DataFrame([
        [2., 0., 2.],  # 5
        [2., 0., 0.],  # 7
        [0., 0., 0.],  # 3
        [0., 0., 2.],  # 1
        [2., 2., 2.],  # 6
        [2., 2., 0.],  # 8
        [0., 2., 0.],  # 4
        [0., 2., 2.],  # 2
        [2., 0., 1.],
        [1., 0., 0.],
        [0., 0., 1.],
        [1., 0., 2.],
        [2., 2., 1.],
        [1., 2., 0.],
        [0., 2., 1.],
        [1., 2., 2.],
        [2., 1., 2.],
        [2., 1., 0.],
        [0., 1., 0.],
        [0., 1., 2.],
        [4., 0., 2.],  # 9
        [4., 0., 0.],  # 11
        [2., 0., 0.],  # 7
        [2., 0., 2.],  # 5
        [4., 2., 2.],  # 10
        [4., 2., 0.],  # 12
        [2., 2., 0.],  # 8
        [2., 2., 2.],  # 6
        [4., 0., 1.],
        [3., 0., 0.],
        [2., 0., 1.],
        [3., 0., 2.],
        [4., 2., 1.],
        [3., 2., 0.],
        [2., 2., 1.],
        [3., 2., 2.],
        [4., 1., 2.],
        [4., 1., 0.],
        [2., 1., 0.],
        [2., 1., 2.],
    ], columns=['x', 'y', 'z'], index=mi)

    expected_cell_types = [12, 12]  # VTK_HEXAHEDRON
    expected_cells = [
        8, 4, 6, 2, 0, 5, 7, 3, 1,
        8, 8, 10, 6, 4, 9, 11, 7, 5
    ]
    expected_points = [
        [0., 0., 2.],  # 0
        [0., 2., 2.],  # 1
        [0., 0., 0.],  # 2
        [0., 2., 0.],  # 3
        [2., 0., 2.],  # 4
        [2., 2., 2.],  # 5
        [2., 0., 0.],  # 6
        [2., 2., 0.],  # 7
        [4., 0., 2.],  # 8
        [4., 2., 2.],  # 9
        [4., 0., 0.],  # 10
        [4., 2., 0.],  # 11
    ]

    cells, cell_types, points = df.mesh.vtk_data()

    np.testing.assert_allclose(expected_cell_types, cell_types)
    np.testing.assert_allclose(expected_cells, cells)
    np.testing.assert_allclose(expected_points, points)


def test_vtk_grid_3d_tet_lin():
    mi = pd.MultiIndex.from_tuples([
        (1, 25), (1, 33), (1, 14), (1, 1),
        (2, 25), (2, 33), (2, 14), (2, 2)
    ], names=['element_id', 'node_id'])

    df = pd.DataFrame([
        [2., 2., 0.],  # 25
        [4., 2., 2.],  # 33
        [2., 2., 2.],  # 14
        [2., 4., 2.],  # 1
        [2., 2., 0.],  # 25
        [4., 2., 2.],  # 33
        [2., 2., 2.],  # 14
        [2., 0., 2.],  # 2
    ], columns=['x', 'y', 'z'], index=mi)

    expected_cells = [
        4, 3, 4, 2, 0,
        4, 3, 4, 2, 1
    ]
    expected_cell_types = [10, 10]  # VTK_TETRA
    expected_points = [
        [2., 4., 2.], # 0
        [2., 0., 2.], # 1
        [2., 2., 2.], # 2
        [2., 2., 0.], # 3
        [4., 2., 2.], # 4
    ]

    cells, cell_types, points = df.mesh.vtk_data()

    np.testing.assert_allclose(expected_cells, cells)
    np.testing.assert_allclose(expected_points, points)
    np.testing.assert_allclose(expected_cell_types, cell_types)


def test_vtk_grid_3d_tet_quad():
    mi = pd.MultiIndex.from_tuples([
        (1, 25), (1, 33), (1, 14), (1, 1),
        (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18),
        (2, 25), (2, 33), (2, 14), (2, 2),
        (2, 26), (2, 27), (2, 28), (2, 29), (2, 30), (2, 31)
    ], names=['element_id', 'node_id'])

    df = pd.DataFrame([
        [2., 2., 0.],  # 25
        [4., 2., 2.],  # 33
        [2., 2., 2.],  # 14
        [2., 4., 2.],  # 1
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [2., 2., 0.],  # 25
        [4., 2., 2.],  # 33
        [2., 2., 2.],  # 14
        [2., 0., 2.],  # 2
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
    ], columns=['x', 'y', 'z'], index=mi)

    expected_cells = [
        4, 3, 4, 2, 0,
        4, 3, 4, 2, 1
    ]
    expected_cell_types = [10, 10]  # VTK_TETRA
    expected_points = [
        [2., 4., 2.], # 0
        [2., 0., 2.], # 1
        [2., 2., 2.], # 2
        [2., 2., 0.], # 3
        [4., 2., 2.], # 4
    ]

    cells, cell_types, points = df.mesh.vtk_data()

    np.testing.assert_allclose(expected_cells, cells)
    np.testing.assert_allclose(expected_points, points)
    np.testing.assert_allclose(expected_cell_types, cell_types)


def test_vtk_grid_3d_wedge_lin():
    mi = pd.MultiIndex.from_tuples([
        (1, 25), (1, 33), (1, 14), (1, 1), (1, 42), (1, 57),
        (2, 25), (2, 33), (2, 2), (2, 1), (2, 42), (2, 12)
    ], names=['element_id', 'node_id'])

    df = pd.DataFrame([
        [0., 0., 0.],  # 25
        [2., 0., 0.],  # 33
        [2., 0., 2.],  # 14
        [0., 2., 0.],  # 1
        [2., 2., 0.],  # 42
        [2., 2., 2.],  # 57
        [0., 0., 0.],  # 25
        [2., 0., 0.],  # 33
        [0., 0., 2.],  # 2
        [0., 2., 0.],  # 1
        [2., 2., 2.],  # 42
        [0., 2., 2.],  # 12

    ], columns=['x', 'y', 'z'], index=mi)

    expected_cells = [
        6, 4, 5, 3, 0, 6, 7,
        6, 4, 5, 1, 0, 6, 2
    ]
    expected_cell_types = [13, 13]  # VTK_WEDGE
    expected_points = [
        [0., 2., 0.],  # 0
        [0., 0., 2.],  # 1
        [0., 2., 2.],  # 2
        [2., 0., 2.],  # 3
        [0., 0., 0.],  # 4
        [2., 0., 0.],  # 5
        [2., 2., 0.],  # 6
        [2., 2., 2.]   # 7
    ]

    cells, cell_types, points = df.mesh.vtk_data()

    np.testing.assert_allclose(expected_cells, cells)
    np.testing.assert_allclose(expected_points, points)
    np.testing.assert_allclose(expected_cell_types, cell_types)


def test_vtk_grid_3d_wedge_quad():
    mi = pd.MultiIndex.from_tuples([
        (1, 25), (1, 33), (1, 14), (1, 1), (1, 42), (1, 57),
        (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23),
        (2, 25), (2, 33), (2, 2), (2, 1), (2, 42), (2, 12),
        (2, 65), (2, 66), (2, 67), (2, 68), (2, 69), (2, 60), (2, 61), (2, 62), (2, 63),
    ], names=['element_id', 'node_id'])

    df = pd.DataFrame([
        [0., 0., 0.],  # 25
        [2., 0., 0.],  # 33
        [2., 0., 2.],  # 14
        [0., 2., 0.],  # 1
        [2., 2., 0.],  # 42
        [2., 2., 2.],  # 57
        [8., 8., 8.],
        [8., 8., 8.],
        [8., 8., 8.],
        [8., 8., 8.],
        [8., 8., 8.],
        [8., 8., 8.],
        [8., 8., 8.],
        [8., 8., 8.],
        [8., 8., 8.],
        [0., 0., 0.],  # 25
        [2., 0., 0.],  # 33
        [0., 0., 2.],  # 2
        [0., 2., 0.],  # 1
        [2., 2., 2.],  # 42
        [0., 2., 2.],  # 12
        [8., 8., 8.],
        [8., 8., 8.],
        [8., 8., 8.],
        [8., 8., 8.],
        [8., 8., 8.],
        [8., 8., 8.],
        [8., 8., 8.],
        [8., 8., 8.],
        [8., 8., 8.],
    ], columns=['x', 'y', 'z'], index=mi)

    expected_cells = [
        6, 4, 5, 3, 0, 6, 7,
        6, 4, 5, 1, 0, 6, 2
    ]
    expected_cell_types = [26, 26]  # VTK_WEDGE
    expected_points = [
        [0., 2., 0.],  # 0
        [0., 0., 2.],  # 1
        [0., 2., 2.],  # 2
        [2., 0., 2.],  # 3
        [0., 0., 0.],  # 4
        [2., 0., 0.],  # 5
        [2., 2., 0.],  # 6
        [2., 2., 2.]   # 7
    ]

    cells, cell_types, points = df.mesh.vtk_data()

    np.testing.assert_allclose(expected_cells, cells)
    np.testing.assert_allclose(expected_points, points)
    np.testing.assert_allclose(expected_cell_types, cell_types)


def test_vtk_grid():

    mi = pd.MultiIndex.from_tuples([(1, 1), (1, 2), (1, 4), (1, 3), (1, 5), (1, 6), (1, 8), (1, 7),
                                    (2, 2), (2, 9), (2, 6), (2, 3), (2, 10), (2, 7),
                                    (3, 3), (3, 10), (3, 7), (3, 11)],
                                   names=['element_id', 'node_id'])
    df = pd.DataFrame([
        [0., 0., 0., 17.5],  #  1
        [0., 2., 0., 17.5],  #  2
        [2., 2., 0., 17.5],  #  4
        [2., 0., 0., 17.5],  #  3
        [0., 0., 2., 17.5],  #  5
        [0., 2., 2., 17.5],  #  6
        [2., 2., 2., 17.5],  #  8
        [2., 0., 2., 17.5],  #  7
        [2., 0., 0., 27.5],  #  3
        [4., 0., 2., 27.5],  #  9
        [2., 0., 2., 27.5],  #  7
        [2., 2., 0., 27.5],  #  4
        [4., 2., 2., 27.5],  # 10
        [2., 2., 2., 27.5],  #  8
        [2., 2., 0., 37.5],  #  4
        [4., 2., 2., 37.5],  # 10
        [2., 2., 2., 37.5],  #  8
        [2., 4., 2., 37.5],  # 11
    ], columns=['x', 'y', 'z', 'mises'], index=mi)

    expected_cells = [
        8, 0, 1, 3, 2, 4, 5, 7, 6,
        6, 1, 8, 5, 2, 9, 6,
        4, 2, 9, 6, 10
    ]
    expected_points = np.array([
        [0., 0., 0.],
        [0., 2., 0.],
        [2., 0., 0.],
        [2., 2., 0.],
        [0., 0., 2.],
        [0., 2., 2.],
        [2., 0., 2.],
        [2., 2., 2.],
        [4., 0., 2.],
        [4., 2., 2.],
        [2., 4., 2.],
    ])
    expected_cell_types = [12, 13, 10]

    cells, cell_types, points = df.mesh.vtk_data()

    np.testing.assert_allclose(expected_cells, cells)
    np.testing.assert_allclose(expected_points, points)
    np.testing.assert_allclose(expected_cell_types, cell_types)

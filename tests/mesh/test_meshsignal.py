import pytest

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


def test_plain_mesh_fail():
    df = pd.DataFrame({'x': [1.0], 't': [2.0], 'b': [3.0], 'a': [9.9]})
    with pytest.raises(AttributeError, match=r'PlainMeshAccessor.*Missing y'):
        df.plain_mesh.coordinates


def test_mesh_3d():
    mi = pd.MultiIndex.from_tuples([(1, 1)], names=['element_id', 'node_id'])
    df = pd.DataFrame({'x': [1.0], 'y': [2.0], 'z': [3.0], 'a': [9.9]}).set_index(mi)
    pd.testing.assert_frame_equal(df.mesh.coordinates,
                                  pd.DataFrame({'x': [1.0], 'y': [2.0], 'z': [3.0]}).set_index(mi))


def test_mesh_2d():
    mi = pd.MultiIndex.from_tuples([(1, 1)], names=['element_id', 'node_id'])
    df = pd.DataFrame({'x': [1.0], 'y': [2.0], 'c': [3.0], 'a': [9.9]}).set_index(mi)
    pd.testing.assert_frame_equal(df.mesh.coordinates,
                                  pd.DataFrame({'x': [1.0], 'y': [2.0]}).set_index(mi))


def test_mesh_fail_coordinates():
    mi = pd.MultiIndex.from_tuples([(1, 1)], names=['element_id', 'node_id'])
    df = pd.DataFrame({'x': [1.0], 'e': [2.0], 'c': [3.0], 'a': [9.9]}).set_index(mi)
    with pytest.raises(AttributeError, match=r'MeshAccessor.*Missing y'):
        df.mesh.coordinates


def test_mesh_fail_index():
    df = pd.DataFrame({'x': [1.0], 'y': [2.0], 'z': [3.0], 'a': [9.9]})
    with pytest.raises(AttributeError, match=r'.*element_id.*'):
        df.mesh.coordinates

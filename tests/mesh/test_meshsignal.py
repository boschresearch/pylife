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


def test_connectivity():
    mi = pd.MultiIndex.from_tuples([(1, 1), (1, 2), (1, 3),
                                    (2, 4), (2, 5), (2, 6)], names=['element_id', 'node_id'])
    df = pd.DataFrame({'x': np.arange(1, 7), 'y': np.arange(2, 8)}, index=mi)

    expected = pd.Series([[1, 2, 3], [4, 5, 6]], name='node_id', index=pd.Index([1, 2], name='element_id'))
    pd.testing.assert_series_equal(df.mesh.connectivity, expected)


def test_connectivity_iloc():
    mi = pd.MultiIndex.from_tuples([(1, 1), (1, 2), (1, 3), (1, 17),
                                    (2, 4), (2, 5), (2, 6)], names=['element_id', 'node_id'])
    df = pd.DataFrame({'x': np.arange(1, 8), 'y': np.arange(2, 9)}, index=mi)

    expected = pd.Series([[0, 1, 2, 3], [4, 5, 6]], name='node_id', index=pd.Index([1, 2], name='element_id'))
    pd.testing.assert_series_equal(df.mesh.connectivity_iloc, expected)

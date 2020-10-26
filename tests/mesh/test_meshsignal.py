# Copyright (c) 2019-2020 - for information on the respective copyright owner
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

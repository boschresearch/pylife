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

"""Integration tests for the OdbClient

Note that these tests are not part of pyLife's usual CI test pipeline as they
need an Abaqus installation to run.
"""

import os
import pytest

import numpy as np
import pandas as pd

import odbclient

from odbclient.odbclient import OdbServerError

@pytest.fixture
def client():
    return odbclient.OdbClient('tests/beam_3d_hex_quad.odb')


def test_not_existing_odbserver_env():
    with pytest.raises(OSError, match="No odbserver environment found."):
        odbclient.OdbClient('foo.odb', python_env_path='/foo/bar/env')


def test_not_existing_abaqus_path():
    with pytest.raises(FileNotFoundError):
        odbclient.OdbClient('foo.odb', abaqus_bin='/foo/bar/abaqus')


def test_odbclient_instances(client):
    np.testing.assert_array_equal(client.instance_names(), ['PART-1-1'])


def test_odbclient_node_coordinates(client):
    expected = pd.read_csv('tests/node_coordinates.csv', index_col='node_id')
    pd.testing.assert_frame_equal(client.node_coordinates('PART-1-1'), expected)


def test_odbclient_node_ids(client):
    result = client.node_ids('FIX', 'PART-1-1')
    expected = pd.Index([1, 2, 3, 4, 22, 27, 31, 32], dtype='int64', name='node_id')

    pd.testing.assert_index_equal(result, expected)


def test_odbclient_element_ids(client):
    result = client.element_ids('FIX', 'PART-1-1')
    expected = pd.Index([1], dtype='int64', name='element_id')

    pd.testing.assert_index_equal(result, expected)


def test_odbclient_node_coordinates_invalid_instance_name(client):
    with pytest.raises(KeyError, match="Invalid instance name 'nonexistent'."):
        client.node_coordinates('nonexistent')


@pytest.mark.parametrize('instance_name, expected', [
    ('', ' ALL NODES'),
    ('PART-1-1', ['ALL', 'FIX', 'LOAD'])
])
def test_odbclient_nset_names(client, instance_name, expected):
    np.testing.assert_array_equal(client.nset_names(instance_name), expected)


@pytest.mark.timeout(10)
def test_odbclient_nset_names_invalid_instance_name(client):
    with pytest.raises(KeyError, match="Invalid instance name 'nonexistent'."):
        client.nset_names('nonexistent')


@pytest.mark.parametrize('instance_name, expected', [
    ('', ' ALL ELEMENTS'),
    ('PART-1-1', ['ALL', 'FIX'])
])
def test_odbclient_elset_names(client, instance_name, expected):
    np.testing.assert_array_equal(client.elset_names(instance_name), expected)


@pytest.mark.timeout(10)
def test_odbclient_elset_names_invalid_instance_name(client):
    with pytest.raises(KeyError, match="Invalid instance name 'nonexistent'."):
        client.elset_names('nonexistent')


@pytest.mark.skip("to be implemented")
def test_element_connectivity(client):
    expected = pd.read_csv('tests/connectivity.csv', index_col='element_id')
    result = client.element_connectivity('PART-1-1')
    print(result)
    print(expected)
    pd.testing.assert_frame_equal(result, expected)


def test_step_names(client):
    expected = ['Load']
    result = client.step_names()
    np.testing.assert_array_equal(result, expected)


def test_frame_ids(client):
    expected = [0, 1]
    result = client.frame_ids('Load')
    np.testing.assert_array_equal(result, expected)


def test_frame_ids_invalid_step_name(client):
    with pytest.raises(KeyError, match='nonexistent'):
        client.frame_ids('nonexistent')


def test_variable_names(client):
    expected = ['CF', 'COORD', 'E', 'EVOL', 'IVOL', 'RF', 'S', 'U']
    result = client.variable_names('Load', 0)
    np.testing.assert_array_equal(result, expected)


def test_variable_stress_element_nodal(client):
    expected = pd.read_csv('tests/stress_element_nodal.csv', index_col=['node_id', 'element_id'])
    result = client.variable('S', 'PART-1-1', 'Load', 1)

    pd.testing.assert_frame_equal(result, expected)


def test_variable_evol(client):
    result = client.variable('EVOL', 'PART-1-1', 'Load', 1)
    expected = pd.DataFrame({'EVOL': [1000.]}, index=pd.Index([1, 2, 3, 4], name='element_id'))

    pd.testing.assert_frame_equal(result, expected)


def test_variable_invalid_instance_name(client):
    with pytest.raises(KeyError, match="nonexistent"):
        client.variable('S', 'nonexistent', 'Load', 1)


def test_variable_stress_integration_point(client):
    expected = pd.read_csv('tests/stress_integration_point.csv',
                           index_col=['element_id', 'ipoint_id'])
    result = client.variable('S', 'PART-1-1', 'Load', 1, position='INTEGRATION POINTS')
    result.to_csv('tests/stress_integration_point.csv')
    pd.testing.assert_frame_equal(result, expected)

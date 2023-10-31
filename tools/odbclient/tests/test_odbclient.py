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
import json

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

@pytest.fixture
def client_history():
    return odbclient.OdbClient('tests/history_output_test.odb')

def test_history_region_empty(client):
    assert client.history_regions("Load") == ['Assembly ASSEMBLY']

def test_history_region_non_empty(client_history):
    assert client_history.history_regions("Step-1") == ['Assembly ASSEMBLY', 'Element ASSEMBLY.1', 'Node ASSEMBLY.1', 'Node ASSEMBLY.2']

def test_history_outputs(client_history):
    assert client_history.history_outputs("Step-1", 'Element ASSEMBLY.1') == ['CTF1', 'CTF2', 'CTF3', 'CTM1', 'CTM2', 'CTM3', 'CU1', 'CU2', 'CU3', 'CUR1', 'CUR2', 'CUR3']

def test_history_output_values(client_history):
    assert client_history.history_output_values("Step-1", 'Element ASSEMBLY.1', 'CTF1').array[1] == pytest.approx(0.09999854117631912)

def test_history_region_description(client_history):
    assert client_history.history_region_description("Step-1", 'Element ASSEMBLY.1') == "Output at assembly ASSEMBLY instance ASSEMBLY element 1"


def test_history_info(client_history):
    expected = json.loads("""
{"Output at assembly ASSEMBLY instance ASSEMBLY node 1 region RP-1": {"History Outputs": ["RF1", "RF2", "RF3", "RM1", "RM2", "RM3", "U1", "U2", "U3", "UR1", "UR2", "UR3"], "History Region": "Node ASSEMBLY.1", "Steps ": ["Step-1", "Step-2"]}, "Output at assembly ASSEMBLY instance ASSEMBLY element 1": {"History Outputs": ["CTF1", "CTF2", "CTF3", "CTM1", "CTM2", "CTM3", "CU1", "CU2", "CU3", "CUR1", "CUR2", "CUR3"], "History Region": "Element ASSEMBLY.1", "Steps ": ["Step-1", "Step-2"]}, "Output at assembly ASSEMBLY instance ASSEMBLY node 2 region SET-5": {"History Outputs": ["RF1", "RF2", "RF3", "RM1", "RM2", "RM3", "U1", "U2", "U3", "UR1", "UR2", "UR3"], "History Region": "Node ASSEMBLY.2", "Steps ": ["Step-1", "Step-2"]}, "Output at assembly ASSEMBLY": {"History Outputs": ["ALLAE", "ALLCCDW", "ALLCCE", "ALLCCEN", "ALLCCET", "ALLCCSD", "ALLCCSDN", "ALLCCSDT", "ALLCD", "ALLDMD", "ALLDTI", "ALLEE", "ALLFD", "ALLIE", "ALLJD", "ALLKE", "ALLKL", "ALLPD", "ALLQB", "ALLSD", "ALLSE", "ALLVD", "ALLWK", "ETOTAL"], "History Region": "Assembly ASSEMBLY", "Steps ": ["Step-1", "Step-2"]}}
    """)
    assert client_history.history_info() == expected

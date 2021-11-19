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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import os
import sys
import time
import pickle
import subprocess as sp

import threading as THR
import queue as QU

import numpy as np
import pandas as pd


class OdbServerError(Exception):
    pass


class OdbClient:

    def __init__(self, abaqus_bin, python_env_path, odb_file):
        self._proc = None
        env = os.environ
        env['PYTHONPATH'] = self._guess_pythonpath(python_env_path)

        lock_file_exists = os.path.isfile(os.path.splitext(odb_file)[0] + '.lck')

        self._proc = sp.Popen([abaqus_bin, 'python', '-m', 'odbserver', odb_file],
                              stdout=sp.PIPE, stdin=sp.PIPE, stderr=sp.PIPE,
                              env=env)

        if lock_file_exists:
            self._gulp_lock_file_warning()

        self._wait_for_server_ready_sign()

    def _guess_pythonpath(self, python_env_path):
        if sys.platform == 'win32':
            return os.path.join(python_env_path, 'lib', 'site-packages')
        return os.path.join(python_env_path, 'lib', 'python2.7', 'site-packages')

    def _gulp_lock_file_warning(self):
            self._proc.stdout.readline()
            self._proc.stdout.readline()

    def _wait_for_server_ready_sign(self):
        def wait_for_input(stdout, queue):
            sign = stdout.read(5)
            queue.put(sign)

        queue = QU.Queue()
        thread = THR.Thread(target=wait_for_input, args=(self._proc.stdout, queue))
        thread.daemon = True
        thread.start()

        while True:
            self._check_if_process_still_alive()
            try:
                sign = queue.get_nowait()
            except QU.Empty:
                time.sleep(1)
            else:
                if sign != b'ready':
                    raise OdbServerError("Expected ready sign from server, received %s" % sign)
                return

    def instance_names(self):
        return _ascii(_decode, self._query('get_instances'))

    def node_coordinates(self, instance_name, nset_name=''):
        index, node_data = self._query('get_nodes', (instance_name, nset_name))
        return pd.DataFrame(data=node_data, columns=['x', 'y', 'z'],
                            index=pd.Int64Index(index, name='node_id'))

    def element_connectivity(self, instance_name, elset_name=''):
        index, connectivity = self._query('get_connectivity', (instance_name, elset_name))
        return pd.DataFrame({'connectivity': connectivity},
                            index=pd.Int64Index(index, name='element_id'))

    def nset_names(self, instance_name=''):
        return _ascii(_decode, self._query('get_node_sets', instance_name))

    def node_ids(self, nset_name, instance_name=''):
        node_ids = self._query('get_node_set', (instance_name, nset_name))
        return pd.Int64Index(node_ids, name='node_id')

    def elset_names(self, instance_name=''):
        return _ascii(_decode, self._query('get_element_sets', instance_name))

    def element_ids(self, elset_name, instance_name=''):
        element_ids = self._query('get_element_set', (instance_name, elset_name))
        return pd.Int64Index(element_ids, name='element_id')

    def step_names(self):
        return _ascii(_decode, self._query('get_steps'))

    def frame_ids(self, step_name):
        return self._query('get_frames', step_name)

    def variable_names(self, step_name, frame_id):
        return _ascii(_decode, self._query('get_variable_names', (step_name, frame_id)))

    def variable(self, variable_name, instance_name, step_name, frame_id, nset_name='', elset_name='', position=None):
        """Read field data.

        Parameters
        ----------
        ...
        position : string
            Position within element. Terminology as in Abaqus .inp file:
            "INTEGRATION POINTS", "CENTROIDAL", "WHOLE ELEMENT", "NODES",
            "FACES", "AVERAGED AT NODES"
        """
        response = self._query('get_variable', (instance_name, step_name, frame_id, variable_name, nset_name, elset_name, position))
        (labels, index_labels, index_data, values) = response

        index_labels = _ascii(_decode, index_labels)
        if len(index_labels) > 1:
            index = pd.DataFrame(index_data, columns=index_labels).set_index(index_labels).index
        else:
            index = pd.Int64Index(index_data[:, 0], name=index_labels[0])

        column_names = _ascii(_decode, labels)
        return pd.DataFrame(values, index=index, columns=column_names)

    def _query(self, command, args=None):
        args = _ascii(_encode, args)
        self._send_command(command, args)
        self._check_if_process_still_alive()
        array_num, pickle_data = self._parse_response()

        if isinstance(pickle_data, Exception):
            raise pickle_data

        if array_num == 0:
            return _ascii(_decode, pickle_data)

        numpy_arrays = [np.lib.format.read_array(self.proc.stdout) for _ in range(array_num)]

        return _ascii(_decode, pickle_data), numpy_arrays

    def _send_command(self, command, args=None):
        self._check_if_process_still_alive()
        pickle.dump((command, args), self._proc.stdin, protocol=2)
        self._proc.stdin.flush()

    def _parse_response(self):
        pickle_data = b''
        while True:
            line = self._proc.stdout.readline().rstrip() + b'\n'
            pickle_data += line
            if line == b'.\n':
                break
        return pickle.loads(pickle_data, encoding='bytes')

    def __del__(self):
        if self._proc is not None:
            self._send_command('QUIT')

    def _check_if_process_still_alive(self):
        if self._proc.poll() is not None:
            _, error_message = self._proc.communicate()
            self._proc = None

            raise OdbServerError(error_message.decode('ascii'))


def _ascii(fcn, args):
    if isinstance(args, list):
        return [_ascii(fcn, arg) for arg in args]

    if isinstance(args, tuple):
        return tuple(_ascii(fcn, arg) for arg in args)

    return fcn(args)

def _encode(arg):
    return arg.encode('ascii') if isinstance(arg, str) else arg

def _decode(arg):
    return arg.decode('ascii') if isinstance(arg, bytes) else arg

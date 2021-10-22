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

    def instances(self):
        return _decode_ascii_list(self._query('get_instances'))

    def nodes(self, instance_name, node_set_name=b''):
        index, node_data = self._query('get_nodes', (instance_name, node_set_name))
        return pd.DataFrame(data=node_data, columns=['x', 'y', 'z'],
                            index=pd.Int64Index(index, name='node_id'))

    def connectivity(self, instance_name, element_set_name=b''):
        index, connectivity = self._query('get_connectivity', (instance_name, element_set_name))
        return pd.DataFrame({'connectivity': connectivity},
                            index=pd.Int64Index(index, name='element_id'))

    def node_sets(self, instance_name=b''):
        return _decode_ascii_list(self._query('get_node_sets', instance_name))

    def node_set(self, node_set_name, instance_name=b''):
        node_set = self._query('get_node_set', (instance_name, node_set_name))
        return pd.Int64Index(node_set, name='node_id')

    def element_sets(self, instance_name=b''):
        return _decode_ascii_list(self._query('get_element_sets', instance_name))

    def element_set(self, element_set_name, instance_name=b''):
        element_set = self._query('get_element_set', (instance_name, element_set_name))
        return pd.Int64Index(element_set, name='element_id')

    def steps(self):
        return self._query('get_steps')

    def frames(self, step_name):
        return self._query('get_frames', step_name)

    def variable_names(self, step, frame):
        return _decode_ascii_list(self._query('get_variable_names', (step, frame)))

    def variable(self, instance, step, frame, var_name, node_set_name=b'', element_set_name=b''):
        response = self._query('get_variable', (instance, step, frame, var_name, node_set_name, element_set_name))
        (labels, index_labels, index_data, values) = response

        index_labels = _decode_ascii_list(index_labels)
        if len(index_labels) == 2:
            index = pd.DataFrame(index_data, columns=index_labels).set_index(index_labels).index
        else:
            index = pd.Int64Index(index_data[:, 0], name=index_labels[0])

        column_names = _decode_ascii_list(labels)
        return pd.DataFrame(values, index=index, columns=column_names)

    def _query(self, command, args=None):
        self._send_command(command, args)
        self._check_if_process_still_alive()
        array_num, pickle_data = self._parse_response()

        if isinstance(pickle_data, Exception):
            raise pickle_data

        if array_num == 0:
            return pickle_data

        numpy_arrays = [np.lib.format.read_array(self.proc.stdout) for _ in range(array_num)]

        return pickle_data, numpy_arrays

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

def _decode_ascii_list(ascii_list):
    return [item.decode('ascii') for item in ascii_list]

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

from __future__ import print_function

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import sys
import os
import pickle
import struct

import numpy as np

from .interface import OdbInterface

class OdbServer:

    def __init__(self, odbfile):
        self._odb = OdbInterface(odbfile)
        self.command_dict = {
            "get_instances": self.instances,
            "get_steps": self.steps,
            "get_frames": self.frames,
            "get_nodes": self.nodes,
            "get_connectivity": self.connectivity,
            "get_node_sets": self.node_sets,
            "get_element_sets": self.element_sets,
            "get_node_set": self.node_set,
            "get_element_set": self.element_set,
            "get_variable_names": self.variable_names,
            "get_variable": self.variable,
            "get_history_regions": self.history_regions,
            "get_history_outputs": self.history_outputs,
            "get_history_output_values": self.history_output_values,
            "get_history_region_description": self.history_region_description,
            "get_history_info": self.history_info
        }

    def instances(self, _args):
        _send_response(self._odb.instance_names())

    def steps(self, _args):
        _send_response(self._odb.step_names())

    def frames(self, step_name):
        _send_response(self._odb.frame_names(step_name))

    def nodes(self, args):
        instance_name, node_set_name = args
        try:
            nodes = self._odb.nodes(instance_name, node_set_name)
        except Exception as e:
            _send_response(e)
        else:
            _send_response(nodes)

    def connectivity(self, args):
        instance_name, element_set_name = args
        try:
            conn = self._odb.connectivity(instance_name, element_set_name)
        except Exception as e:
            _send_response(e)
        else:
            _send_response(conn)

    def node_sets(self, instance_name):
        _send_response(self._odb.node_sets(instance_name))

    def element_sets(self, instance_name):
        _send_response(self._odb.element_sets(instance_name))

    def node_set(self, args):
        instance_name, node_set_name = args
        _send_response(self._odb.node_set(instance_name, node_set_name))

    def element_set(self, args):
        instance_name, element_set_name = args
        _send_response(self._odb.element_set(instance_name, element_set_name))

    def variable_names(self, args):
        step, frame = args
        _send_response(self._odb.variable_names(step, int(frame)))

    def variable(self, args):
        instance_name, step, frame, var_name, nset, elset, elem_pos = args
        try:
            variable = self._odb.variable(instance_name, step, frame, var_name, nset, elset, elem_pos)
        except Exception as e:
            _send_response(e)
        else:
            _send_response(variable)

    def history_regions(self, step_name):
        _send_response(self._odb.history_regions(step_name))

    def history_outputs(self, args):
        step_name, historyregion_name = args
        _send_response(self._odb.history_outputs(step_name, historyregion_name))

    def history_output_values(self, args):
        step_name, historyregion_name, historyoutput_name = args
        _send_response(self._odb.history_output_values(step_name, historyregion_name, historyoutput_name))

    def history_region_description(self, args):
        step_name, historyregion_name = args
        _send_response(self._odb.history_region_description(step_name, historyregion_name))

    def history_info(self, args):
        _send_response(self._odb.history_info())


def _send_response(pickle_data, numpy_arrays=None):
    stdout = sys.stdout if sys.version_info.major == 2 else sys.stdout.buffer
    numpy_arrays = numpy_arrays or []

    message = pickle.dumps((len(numpy_arrays), pickle_data), protocol=2)
    data_size_8_bytes = struct.pack("Q", len(message))

    stdout.write(data_size_8_bytes)
    stdout.write(message)
    sys.stdout.flush()
    for nparr in numpy_arrays:
        np.lib.format.write_array(sys.stdout, nparr)


def main():

    def decode_strings_if_not_on_python_2(parameters):
        if sys.version_info.major == 2:
            return parameters
        if isinstance(parameters, tuple):
            return tuple(p.decode('ascii') if isinstance(p, bytes) else p for p in parameters)
        if isinstance(parameters, bytes):
            return parameters.decode('ascii')

    def pickle_load_2():
        return pickle.load(sys.stdin)

    def pickle_load_3():
        return pickle.load(sys.stdin.buffer, encoding='ASCII')

    pickle_load = pickle_load_2 if sys.version_info.major == 2 else pickle_load_3

    odbfile = sys.argv[1]

    try:
        server = OdbServer(odbfile)
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    sys.stdout.write('ready')
    sys.stdout.flush()

    command = ''
    while True:
        command, parameters = pickle_load()

        parameters = decode_strings_if_not_on_python_2(parameters)

        if command == 'QUIT':
            break

        func = server.command_dict.get(command)
        if func is not None:
            func(parameters)
            sys.stdout.flush()


if __name__ == "__main__":
    main()

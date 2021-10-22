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

import sys
import os

import pickle

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
            "get_variable": self.variable
        }

    def instances(self, _args):
        _send_response(self._odb.instance_names())

    def steps(self, _args):
        _send_response(self._odb.step_names())

    def frames(self, step_name):
        _send_response(self._odb.frame_names(str(step_name)))

    def nodes(self, args):
        instance_name, node_set_name = args
        try:
            nodes = self._odb.nodes(str(instance_name), str(node_set_name))
        except Exception as e:
            _send_response(e)
        else:
            _send_response(nodes)

    def connectivity(self, args):
        instance_name, element_set_name = args
        try:
            conn = self._odb.connectivity(str(instance_name), str(element_set_name))
        except Exception as e:
            _send_response(e)
        else:
            _send_response(conn)

    def node_sets(self, instance_name):
        _send_response(self._odb.node_sets(str(instance_name)))

    def element_sets(self, instance_name):
        _send_response(self._odb.element_sets(str(instance_name)))

    def node_set(self, args):
        instance_name, node_set_name = args
        _send_response(self._odb.node_set(str(instance_name), str(node_set_name)))

    def element_set(self, args):
        instance_name, element_set_name = args
        _send_response(self._odb.element_set(str(instance_name), str(element_set_name)))

    def variable_names(self, args):
        step, frame = args
        _send_response(self._odb.variable_names(step, int(frame)))

    def variable(self, args):
        instance_name, step, frame, var_name, nset, elset = args
        instance_name = str(instance_name)
        var_name = str(var_name)
        nset = str(nset)
        elset = str(elset)
        try:
            variable = self._odb.variable(instance_name, step, frame, var_name, nset, elset)
        except Exception as e:
            _send_response(e)
        else:
            _send_response(variable)


def _send_response(pickle_data, numpy_arrays=[]):
    s = pickle.dumps((len(numpy_arrays), pickle_data))
    sys.stdout.write(s + '\n')
    sys.stdout.flush()
    for nparr in numpy_arrays:
        np.lib.format.write_array(sys.stdout, nparr)


def main():

    odbfile = sys.argv[1]

    try:
        server = OdbServer(odbfile)
    except Exception as e:
        print >> sys.stderr, e.message
        sys.exit(1)

    sys.stdout.write('ready')
    sys.stdout.flush()

    command = ''
    while True:
        command, parameters = pickle.load(sys.stdin)

        if command == 'QUIT':
            break

        func = server.command_dict.get(command)
        if func is not None:
            func(parameters)
            sys.stdout.flush()


if __name__ == "__main__":
    main()

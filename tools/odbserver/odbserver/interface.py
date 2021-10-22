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
import numpy as np
import odbAccess as ODB


class OdbInterface:

    def __init__(self, odbfile):
        self._odb = ODB.openOdb(odbfile)
        self._asm = self._odb.rootAssembly

        self._index_cache = {}

    def instance_names(self):
        return self._asm.instances.keys()

    def step_names(self):
        return self._odb.steps.keys()

    def frame_names(self, step_name):
        try:
            step = self._odb.steps[step_name]
        except Exception as e:
            return e

        return [frame.frameId for frame in step.frames]

    def nodes(self, instance_name, node_set_name):
        instance = self._instance_or_rootasm(instance_name)

        if node_set_name == b'':
            nodes = instance.nodes
        elif node_set_name in instance.nodeSets.keys():
            nodes = instance.nodeSets[node_set_name].nodes
        elif node_set_name in self._asm.nodeSets.keys():
            nodes = self._asm.nodeSets[node_set_name].nodes[self._asm.instances.keys().index(instance_name)]
        else:
            raise KeyError(node_set_name)

        node_data = np.empty((len(nodes), 3))
        index = np.empty(len(nodes), dtype=np.int32)

        for i, nd in enumerate(nodes):
            index[i] = nd.label
            node_data[i] = nd.coordinates

        return (index, node_data)

    def connectivity(self, instance_name, element_set_name):
        instance = self._instance_or_rootasm(instance_name)

        if element_set_name == b'':
            elements = instance.elements
        elif element_set_name in instance.elementSets.keys():
            elements = instance.elementSets[element_set_name].elements
        elif element_set_name in self._asm.elementSets.keys():
            elements = self._asm.elementSets[element_set_name].elements[self._asm.instances.keys().index(instance_name)]
        else:
            raise KeyError(element_set_name)

        index = np.empty(len(elements), dtype=np.int)
        connectivity = []
        for i, el in enumerate(elements):
            index[i] = el.label
            connectivity.append(list(el.connectivity))

        return (index, connectivity)

    def node_sets(self, instance_name):
        instance = self._instance_or_rootasm(instance_name)
        return instance.nodeSets.keys()

    def element_sets(self, instance_name):
        instance = self._instance_or_rootasm(instance_name)
        return instance.elementSets.keys()

    def node_set(self, instance_name, node_set_name):
        instance = self._instance_or_rootasm(instance_name)

        try:
            node_set = instance.nodeSets[node_set_name]
        except Exception as e:
            return e

        nodes = node_set.nodes[0] if instance_name == b'' else node_set.nodes

        return np.array([node.label for node in nodes], dtype=np.int32)

    def element_set(self, instance_name, element_set_name):
        instance = self._instance_or_rootasm(instance_name)

        try:
            element_set = instance.elementSets[element_set_name]
        except Exception as e:
            return e

        elements = element_set.elements[0] if instance_name == b'' else element_set.elements

        return np.array([element.label for element in elements], dtype=np.int32)

    def variable_names(self, step_name, frame_num):
        try:
            step = self._odb.steps[step_name]
        except Exception as e:
            return e

        try:
            frame = _get_frame(step, frame_num)
        except Exception as e:
            return e

        return frame.fieldOutputs.keys()

    def variable(self, instance_name, step_name, frame_num, variable_name, node_set_name, element_set_name):

        def block_length(block):
            if block.nodeLabels is not None:
                return block.nodeLabels.shape[0]
            if block.elementLabels is not None:
                return block.elementLabels.shape[0]
            return 0

        def index_block_data(block):
            stack = []
            if block.nodeLabels is not None:
                stack.append(block.nodeLabels)
            if block.elementLabels is not None:
                stack.append(block.elementLabels)

            return np.vstack(stack)

        try:
            step = self._odb.steps[step_name]
        except Exception as e:
            return e

        try:
            frame = _get_frame(step, frame_num)
        except Exception as e:
            return e

        try:
            field = frame.fieldOutputs[variable_name]
        except Exception as e:
            return e

        instance = self._asm.instances[instance_name]

        region = None
        if node_set_name != b'':
            if node_set_name in instance.nodeSets.keys():
                node_set = instance.nodeSets[node_set_name]
            elif node_set_name in self._asm.nodeSets.keys():
                node_set = self._asm.nodeSets[str(node_set_name)]
            else:
                raise KeyError(node_set_name)
            region = node_set

        if element_set_name != b'':
            if element_set_name in instance.elementSets.keys():
                element_set = instance.elementSets[element_set_name]
            elif element_set_name in self._asm.elementSets.keys():
                element_set = self._asm.elementSets[str(element_set_name)]
            else:
                raise KeyError(element_set_name)
            region = element_set

        pos = field.locations[0].position
        if pos == ODB.INTEGRATION_POINT:
            pos = ODB.ELEMENT_NODAL

        if pos != "ABQDEFAULT":
            field = field.getSubset(position=pos)
        if region is not None:
            field = field.getSubset(region=region)

        complabels = field.componentLabels if len(field.componentLabels) > 0 else [variable_name]
        blocks = field.bulkDataBlocks

        length = 0
        for block in blocks:
            if block.instance.name != instance_name:
                continue
            length += block_length(block)

        values = np.empty((length, len(complabels)))

        index_dim = 2 if pos == ODB.ELEMENT_NODAL else 1
        index = np.empty((length, index_dim), dtype=np.int32)

        i = 0
        for block in blocks:
            if block.instance.name != instance_name:
                continue

            block_array = block.data
            size = block_array.shape[0]

            index_block = index_block_data(block)

            index[i:i+size, :] = index_block.T
            values[i:i+size, :] = block_array
            i += size

        if pos == ODB.NODAL:
            index_labels = ['node_id']
        elif pos == ODB.WHOLE_ELEMENT or pos == ODB.CENTROID:
            index_labels = ['element_id']
        else:
            index_labels = ['node_id', 'element_id']
        return (complabels, index_labels, index[:i, :], values[:i])

    def _instance_or_rootasm(self, instance_name):
        if instance_name == b'':
            return self._asm
        try:
            return self._asm.instances[instance_name]
        except Exception as e:
            return e


def _get_frame(step, frame_id):
    for frame in step.frames:
        if frame_id == frame.frameId:
            return frame

    raise Exception("Invalid frame id %s", frame_id)

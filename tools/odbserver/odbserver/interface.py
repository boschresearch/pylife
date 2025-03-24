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

        if node_set_name == '':
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

        if element_set_name == '':
            elements = instance.elements
        elif element_set_name in instance.elementSets.keys():
            elements = instance.elementSets[element_set_name].elements
        elif element_set_name in self._asm.elementSets.keys():
            elements = self._asm.elementSets[element_set_name].elements[self._asm.instances.keys().index(instance_name)]
        else:
            raise KeyError(element_set_name)

        index = np.empty(len(elements), dtype=np.int64)
        connectivity = -np.ones((len(elements), 20), dtype=np.int64)
        for i, el in enumerate(elements):
            index[i] = el.label
            conns = list(el.connectivity)
            connectivity[i, :len(conns)] = conns

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

        nodes = node_set.nodes[0] if instance_name == '' else node_set.nodes

        return np.array([node.label for node in nodes], dtype=np.int32)

    def element_set(self, instance_name, element_set_name):
        instance = self._instance_or_rootasm(instance_name)

        try:
            element_set = instance.elementSets[element_set_name]
        except Exception as e:
            return e

        elements = element_set.elements[0] if instance_name == '' else element_set.elements

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

    def variable(self, instance_name, step_name, frame_num, variable_name, node_set_name, element_set_name, position=None):

        def block_length(block):
            if block.nodeLabels is not None:
                return block.nodeLabels.shape[0]
            if block.elementLabels is not None:
                return block.elementLabels.shape[0]
            return 0

        def index_block_data(block):
            stack = []
            index_labels = []

            if getattr(block, 'nodeLabels', None) is not None:
                stack.append(block.nodeLabels)
                index_labels.append('node_id')

            if getattr(block, 'elementLabels', None) is not None:
                stack.append(block.elementLabels)
                index_labels.append('element_id')

            if getattr(block, 'integrationPoints', None) is not None:
                stack.append(block.integrationPoints)
                index_labels.append('ipoint_id')

            if getattr(block, 'faces', None) is not None:
                stack.append(block.faces)
                index_labels.append('face_id')

            return np.vstack(stack), index_labels

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
        if node_set_name != '':
            if node_set_name in instance.nodeSets.keys():
                node_set = instance.nodeSets[node_set_name]
            elif node_set_name in self._asm.nodeSets.keys():
                node_set = self._asm.nodeSets[str(node_set_name)]
            else:
                raise KeyError(node_set_name)
            region = node_set

        if element_set_name != '':
            if element_set_name in instance.elementSets.keys():
                element_set = instance.elementSets[element_set_name]
            elif element_set_name in self._asm.elementSets.keys():
                element_set = self._asm.elementSets[str(element_set_name)]
            else:
                raise KeyError(element_set_name)
            region = element_set

        position_str = position
        position = _set_position(field, user_request=position_str)
        field = field.getSubset(position=position)

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

        if position in [ODB.INTEGRATION_POINT, ODB.ELEMENT_NODAL, ODB.ELEMENT_FACE]:
            index_dim = 2
        elif position in [ODB.CENTROID, ODB.WHOLE_ELEMENT, ODB.NODAL]:
            index_dim = 1
        index = np.empty((length, index_dim), dtype=np.int32)

        i = 0
        for block in blocks:
            if block.instance.name != instance_name:
                continue

            block_array = block.data
            size = block_array.shape[0]

            index_block, index_labels = index_block_data(block)

            index[i:i+size, :] = index_block.T
            values[i:i+size, :] = block_array
            i += size

        return (complabels, index_labels, index[:i, :], values[:i])

    def _instance_or_rootasm(self, instance_name):
        if instance_name == '':
            instance = self._asm
        else:
            instance = self._asm.instances[instance_name]

        element_types = {el.type for el in instance.elements}
        unsupported_types = {et for et in element_types if et[0] != "C"}
        if unsupported_types:
            raise ValueError(
                "Only continuum elements (C...) are supported at this point, sorry. "
                "Please submit an issue to https://github.com/boschresearch/pylife/issues "
                "if you need to support other types. "
                "(Unsupported types %s found in instance %s)" % (
                    ", ".join(unsupported_types), instance_name
                ))

        return instance

    def history_regions(self, step_name):
        """Get history regions, which belongs to the given step.

        Parameters
        ----------
        step_name : Abaqus steps
            It is always required.

        Returns
        -------
        histRegions : history regions, which belong to the given step. In case of error it gives an error message.
            It is a list of hist regions
        """
        try:
            required_step = self._odb.steps[step_name]
            histRegions = required_step.historyRegions.keys()

            return histRegions

        except Exception as e:
            return e

    def history_outputs(self, step_name, historyregion_name):
        """Get history outputs, which belongs to the given step and history region.

        Parameters
        ----------
        step_name : Abaqus steps
            It is always required.
        historyregion_name: Abaqus history region
            It is always required.

        Returns
        -------
        history_data : history data, which belong to the given step and history region. In case of error it gives an error message.
            It is a list of history outputs.

        """
        try:
            required_step = self._odb.steps[step_name]
            history_data = required_step.historyRegions[historyregion_name].historyOutputs.keys()
            return history_data

        except Exception as e:
            return e


    def history_output_values(self, step_name, historyregion_name, historyoutput_name):
        """Get history output values, which belongs to the given step, history region and history output.

        Parameters
        ----------
        step_name : Abaqus steps
            It is always required.
        historyregion_name: Abaqus history region
            It is always required.
        historyoutput_name: Abaqus history output
            It is always required.

        Returns
        -------
        x : time values of a history output. In case of error it gives an error message.
            It is a list of data.
        y : values of a history output. In case of error it gives an error message.
            It is a list of data.

        """
        try:
            required_step = self._odb.steps[step_name]

            history_data = required_step.historyRegions[historyregion_name].historyOutputs[historyoutput_name].data
            step_time = required_step.totalTime

            xdata = []
            ydata = []
            for ith in history_data:
                xdata.append(ith[0]+step_time)
                ydata.append(ith[1])

            x = np.array(xdata)
            y = np.array(ydata)
            return x, y

        except Exception as e:
            return e

    def history_region_description(self, step_name, historyregion_name):
        """Get history region description, which belongs to the given step and history region.

        Parameters
        ----------
        step_name : Abaqus steps
            It is always required.
        historyregion_name: Abaqus history region
            It is always required.

        Returns
        -------
        history_description : str
            The description for history region, which is visible in Abaqus. In case of error it gives an error message.

        """
        try:
            required_step = self._odb.steps[step_name]
            history_description = required_step.historyRegions[historyregion_name].description
            return history_description

        except Exception as e:
            return e


    def history_info(self):
        """Get steps, history regions, history outputs and write into a dictionary.


        Returns
        -------
        A dictionary which contains information about the history of a given odb file.
        In case of error it gives an error message.

        """
        hist_info = {}
        try:
            steps = self._odb.steps.keys()

            for step in steps:
                regions = self.history_regions(step_name=step)

                for reg in regions:
                    description = self.history_region_description(step, reg)

                    outputs = [
                        output
                        for output in self.history_outputs(step, reg)
                        if "Repeated: key" not in output
                    ]

                    steplist = []
                    for istep2 in steps:
                        try:
                            self._odb.steps[istep2].historyRegions[reg].description
                            steplist.append(istep2)
                        except Exception:
                            continue

                    hist_info[description] = {
                        "History Region" : reg,
                        "History Outputs" : outputs,
                        "Steps " : steplist
                    }

            return hist_info
        except Exception as e:
            return e


def _set_position(field, user_request=None):
    """Translate string to symbolic constant and define default behavior.

    Parameters
    ----------
    field : Abaqus field object
        Required if ``user_request=None``.
    user_request : string
        Abaqus .inp file terminology (*ELEMENT OUTPUT, position=...).

    Returns
    -------
    position : symbolic constant
        Abaqus Python interface terminology.
    """
    _position_dict = {'INTEGRATION POINTS': ODB.INTEGRATION_POINT,
                      'CENTROIDAL':         ODB.CENTROID,
                      'WHOLE ELEMENT':      ODB.WHOLE_ELEMENT,
                      'NODES':              ODB.ELEMENT_NODAL,
                      'FACES':              ODB.ELEMENT_FACE,
                      'AVERAGED AT NODES':  ODB.NODAL}

    if user_request is not None:
        return _position_dict[user_request]

    odb_pos = field.locations[0].position

    if odb_pos == ODB.INTEGRATION_POINT:
        return ODB.ELEMENT_NODAL

    return odb_pos

def _get_frame(step, frame_id):
    for frame in step.frames:
        if frame_id == frame.frameId:
            return frame

    raise Exception("Invalid frame id %s", frame_id)

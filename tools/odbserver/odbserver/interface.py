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

"""Wrap the Abaqus ODB API for the server command protocol.

This module runs inside Abaqus, usually under Python 2 for older Abaqus
versions.  It converts Abaqus objects into plain strings, NumPy arrays, and
exceptions that the client process can unpickle.
"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import sys

import numpy as np
import odbAccess as ODB


class OdbInterface:
    """Expose read-only Abaqus ODB queries used by the protocol server.

    Parameters
    ----------
    odbfile : str
        Path to the Abaqus ODB file opened with ``odbAccess.openOdb``.
    """

    def __init__(self, odbfile):
        """Open the ODB file and cache its root assembly.

        Parameters
        ----------
        odbfile : str
            Path to the Abaqus ODB file.
        """
        self._odb = ODB.openOdb(odbfile)
        self._asm = self._odb.rootAssembly
        self._index_cache = {}

    def instance_names(self):
        """Return names of all part instances in the root assembly.

        Returns
        -------
        list of str
            Abaqus instance names.
        """
        return self._asm.instances.keys()

    def step_names(self):
        """Return names of all analysis steps in the ODB.

        Returns
        -------
        list of str
            Abaqus step names.
        """
        return self._odb.steps.keys()

    def frame_names(self, step_name):
        """Return frame IDs for one analysis step.

        Parameters
        ----------
        step_name : str
            Name of the Abaqus analysis step.

        Returns
        -------
        list of int or Exception
            Frame IDs, or the caught Abaqus exception if the step cannot be read.
        """
        try:
            step = self._odb.steps[step_name]
        except Exception as e:
            return e

        return [frame.frameId for frame in step.frames]

    def nodes(self, instance_name, node_set_name):
        """Return node labels and coordinates for an instance or node set.

        Parameters
        ----------
        instance_name : str
            Name of the Abaqus instance, or ``''`` for the root assembly.
        node_set_name : str
            Name of the node set to filter by, or ``''`` for all nodes.

        Returns
        -------
        tuple
            Pair ``(index, node_data)`` with node labels and coordinate array.
        """
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
        """Return element labels and node connectivity.

        Parameters
        ----------
        instance_name : str
            Name of the Abaqus instance, or ``''`` for the root assembly.
        element_set_name : str
            Name of the element set to filter by, or ``''`` for all elements.

        Returns
        -------
        tuple
            Pair ``(index, connectivity)`` with element labels and padded node labels.
        """
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
        """Return node set names for an instance or the root assembly.

        Parameters
        ----------
        instance_name : str
            Instance name, or ``''`` for the root assembly.

        Returns
        -------
        list of str
            Node set names.
        """
        instance = self._instance_or_rootasm(instance_name)
        return instance.nodeSets.keys()

    def element_sets(self, instance_name):
        """Return element set names for an instance or the root assembly.

        Parameters
        ----------
        instance_name : str
            Instance name, or ``''`` for the root assembly.

        Returns
        -------
        list of str
            Element set names.
        """
        instance = self._instance_or_rootasm(instance_name)
        return instance.elementSets.keys()

    def node_set(self, instance_name, node_set_name):
        """Return node labels contained in one node set.

        Parameters
        ----------
        instance_name : str
            Instance name, or ``''`` for an assembly-level set.
        node_set_name : str
            Node set name.

        Returns
        -------
        numpy.ndarray or Exception
            Node labels as integers, or the caught Abaqus exception.
        """
        instance = self._instance_or_rootasm(instance_name)

        try:
            node_set = instance.nodeSets[node_set_name]
        except Exception as e:
            return e

        nodes = node_set.nodes[0] if instance_name == '' else node_set.nodes

        return np.array([node.label for node in nodes], dtype=np.int32)

    def element_set(self, instance_name, element_set_name):
        """Return element labels contained in one element set.

        Parameters
        ----------
        instance_name : str
            Instance name, or ``''`` for an assembly-level set.
        element_set_name : str
            Element set name.

        Returns
        -------
        numpy.ndarray or Exception
            Element labels as integers, or the caught Abaqus exception.
        """
        instance = self._instance_or_rootasm(instance_name)

        try:
            element_set = instance.elementSets[element_set_name]
        except Exception as e:
            return e

        elements = element_set.elements[0] if instance_name == '' else element_set.elements

        return np.array([element.label for element in elements], dtype=np.int32)

    def variable_names(self, step_name, frame_num):
        """Return field output names for one step and frame.

        Parameters
        ----------
        step_name : str
            Name of the Abaqus analysis step.
        frame_num : int
            Abaqus frame ID.

        Returns
        -------
        list of str or Exception
            Field output names, or the caught Abaqus exception.
        """
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
        """Return labels and values for one field output query.

        Parameters
        ----------
        instance_name : str
            Name of the Abaqus instance to query.
        step_name : str
            Name of the Abaqus analysis step.
        frame_num : int
            Abaqus frame ID.
        variable_name : str
            Field output variable name.
        node_set_name : str
            Node set filter, or ``''`` for no node-set filter.
        element_set_name : str
            Element set filter, or ``''`` for no element-set filter.
        position : str, optional
            Requested Abaqus output position in input-file terminology.  Default is
            ``None``, which uses :func:`_set_position`.

        Returns
        -------
        tuple or Exception
            Component labels, index labels, index data, and values, or a caught
            Abaqus exception.
        """

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
        """Return the selected instance and reject unsupported element families."""
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
        """Return history regions that belong to the given step.

        Parameters
        ----------
        step_name : str
            Name of the Abaqus analysis step.

        Returns
        -------
        list of str or Exception
            History region names, or the caught Abaqus exception.
        """
        try:
            required_step = self._odb.steps[step_name]
            histRegions = required_step.historyRegions.keys()

            return histRegions

        except Exception as e:
            return e

    def history_outputs(self, step_name, historyregion_name):
        """Return history outputs for one step and history region.

        Parameters
        ----------
        step_name : str
            Name of the Abaqus analysis step.
        historyregion_name : str
            Name of the Abaqus history region.

        Returns
        -------
        list of str or Exception
            History output names, or the caught Abaqus exception.
        """
        try:
            required_step = self._odb.steps[step_name]
            history_data = required_step.historyRegions[historyregion_name].historyOutputs.keys()
            return history_data

        except Exception as e:
            return e


    def history_output_values(self, step_name, historyregion_name, historyoutput_name):
        """Return abscissa and ordinate arrays for one history output.

        Parameters
        ----------
        step_name : str
            Name of the Abaqus analysis step.
        historyregion_name : str
            Name of the Abaqus history region.
        historyoutput_name : str
            Name of the Abaqus history output.

        Returns
        -------
        x : numpy.ndarray
            Time-like abscissa values shifted by the step total time.
        y : numpy.ndarray
            History output ordinate values.
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
        """Return the Abaqus description of one history region.

        Parameters
        ----------
        step_name : str
            Name of the Abaqus analysis step.
        historyregion_name : str
            Name of the Abaqus history region.

        Returns
        -------
        str or Exception
            History region description, or the caught Abaqus exception.
        """
        try:
            required_step = self._odb.steps[step_name]
            history_description = required_step.historyRegions[historyregion_name].description
            return history_description

        except Exception as e:
            return e


    def history_info(self):
        """Return all history regions, outputs, and step memberships.

        Returns
        -------
        dict or Exception
            Nested history metadata dictionary, or the caught Abaqus exception.
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
    """Translate an output-position string to an Abaqus symbolic constant.

    Parameters
    ----------
    field : Abaqus field output
        Field output whose native location is used when ``user_request`` is
        ``None``.
    user_request : str, optional
        Abaqus input-file position such as ``INTEGRATION POINTS`` or ``NODES``.
        Default is ``None``.

    Returns
    -------
    symbolic constant
        Abaqus output-position constant accepted by ``getSubset``.
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
    """Return the frame with the requested Abaqus frame ID.

    Parameters
    ----------
    step : Abaqus step
        Step object whose frames are searched.
    frame_id : int
        Abaqus frame ID to find.

    Returns
    -------
    Abaqus frame
        Frame object with matching ``frameId``.

    Raises
    ------
    Exception
        If no frame with ``frame_id`` exists in ``step``.
    """
    for frame in step.frames:
        if frame_id == frame.frameId:
            return frame

    raise Exception("Invalid frame id %s", frame_id)

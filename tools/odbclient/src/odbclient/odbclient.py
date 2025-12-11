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
import struct
import subprocess as sp
import shutil

import threading as THR
import queue as QU

import numpy as np
import pandas as pd

import odbclient


class OdbServerError(Exception):
    """Raised when the ODB Server launch fails."""

    pass


class OdbClient:
    """The interface class to access data from odb files provided by the odbserver.

    Parameters
    ----------
    odb_file : string
        The path to the odb file

    abaqus_bin : string, optional
        The path to the abaqus *binary* (no .bat or shell script).
        If not given, ``OdbClient`` will try to get this path from the environment
        variable "ODBSERVER_ABAQUS_BIN".
        If this doesn't exist or is invalid, ``OdbClient`` will try to guess this path.
        Also see https://pylife.readthedocs.io/en/stable/tools/odbserver/index.html.

    python_env_path : string, optional
        The path to the python environmnent to be used by the odbserver.
        If not given, ``OdbClient`` will try to get this path from the environment
        variable "ODBSERVER_PYTHON_ENV_PATH".
        If this doesn't exist or is invalid, ``OdbClient`` will try to guess this path.
        Also see https://pylife.readthedocs.io/en/stable/tools/odbserver/index.html.

    Examples
    --------
    Instantiating and querying instance names

    >>> import odbclient as CL
    >>> client = CL.OdbClient("some_file.odb")
    >>> client.instance_names()
    ['PART-1-1']

    Querying node coordinates

    >>> client.node_coordinates('PART-1-1')
                x     y     z
    node_id
    1       -30.0  15.0  10.0
    2       -30.0  25.0  10.0
    3       -30.0  15.0   0.0
    ...

    Querying step names

    >>> client.step_names()
    ['Load']

    Querying frames of a step

    >>> client.frame_ids('Load')
    [0, 1]

    Querying variable names of a frame and step

    >>> client.variable_names('Load', 1)
    ['CF', 'COORD', 'E', 'EVOL', 'IVOL', 'RF', 'S', 'U']

    Querying variable data of an instance, frame and step

    >>> client.variable('S', 'PART-1-1', 'Load', 1)
                              S11        S22  ...       S13       S23
    node_id element_id                        ...
    5       1          -38.617779   2.705118  ... -3.578981  1.355571
    7       1          -38.617779   2.705118  ...  3.578981 -1.355571
    3       1          -50.749348 -21.749729  ... -7.597347 -0.000003
    1       1          -50.749348 -21.749729  ...  7.597347  0.000003
    6       1           38.643414  -2.588303  ...  3.522046  1.446851
    ...                       ...        ...  ...       ...       ...
    54      4            7.353698  -3.177251  ...  1.775653 -2.608372
    56      4           -6.695759 -17.656754  ...  0.217049 -3.040078
    55      4           -6.695759 -17.656754  ... -0.217049  3.040078
    47      4           -0.226473   1.787100  ...  0.967435 -0.671089
    48      4           -0.226473   1.787100  ... -0.967435  0.671089


    Oftentimes it is desirable to have the node coordinates and multiple field
    variables in one dataframe.  This can be easily achieved by
    :meth:`~pandas.DataFrame.join` operations.

    >>> node_coordinates = client.node_coordinates('PART-1-1')
    >>> stress = client.variable('S', 'PART-1-1', 'Load', 1)
    >>> strain = client.variable('E', 'PART-1-1', 'Load', 1)
    >>> node_coordinates.join(stress).join(strain)
                           x     y     z  ...           E12           E13           E23
    node_id element_id                    ...
    5       1          -20.0  15.0  10.0  ... -2.741873e-11 -4.652675e-11  1.762242e-11
    7       1          -20.0  15.0   0.0  ... -2.741873e-11  4.652675e-11 -1.762242e-11
    3       1          -30.0  15.0   0.0  ... -2.599339e-11 -9.876550e-11 -3.946581e-17
    1       1          -30.0  15.0  10.0  ... -2.599339e-11  9.876550e-11  3.946581e-17
    6       1          -20.0  25.0  10.0  ... -2.689760e-11  4.578660e-11  1.880906e-11
    ...                  ...   ...   ...  ...           ...           ...           ...
    54      4            5.0  25.0  10.0  ... -6.076223e-11  2.308349e-11 -3.390884e-11
    56      4           10.0  20.0  10.0  ... -5.091068e-11  2.821631e-12 -3.952102e-11
    55      4           10.0  20.0   0.0  ... -5.091068e-11 -2.821631e-12  3.952102e-11
    47      4            0.0  20.0   0.0  ... -5.129363e-11  1.257666e-11 -8.724152e-12
    48      4            0.0  20.0  10.0  ... -5.129363e-11 -1.257666e-11  8.724152e-12
    """

    def __init__(self, odb_file, abaqus_bin=None, python_env_path=None):
        self._proc = None

        abaqus_bin = _determine_abaqus_bin(abaqus_bin)
        python_env_path = _determine_python_env_path(python_env_path)

        python_site_packages_path = _determine_site_packages_path(python_env_path, abaqus_bin)
        env = os.environ | {"PYTHONPATH": python_site_packages_path}

        lock_file_exists = os.path.isfile(os.path.splitext(odb_file)[0] + '.lck')

        self._proc = sp.Popen(
            [abaqus_bin, 'python', '-m', 'odbserver', odb_file],
            stdout=sp.PIPE,
            stdin=sp.PIPE,
            stderr=sp.PIPE,
            env=env,
        )

        if lock_file_exists:
            self._gulp_lock_file_warning()

        server_version, server_python_version = self._wait_for_server_ready_sign()
        _raise_if_version_mismatch(server_version)
        if server_python_version == "2":
            self._parse_response = self._parse_response_py2
        else:
            self._parse_response = self._parse_response_py3

    def _gulp_lock_file_warning(self):
        self._proc.stdout.readline()
        self._proc.stdout.readline()

    def _wait_for_server_ready_sign(self):
        def wait_for_input(stdout, queue):
            sign = stdout.read(5)
            queue.put(sign)
            version_sign = stdout.readline()
            queue.put(version_sign)

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

                try:
                    sign = queue.get_nowait()
                    return _ascii(_decode, sign).strip().split()
                except QU.Empty:
                    return "unannounced", "2"
                return

    def instance_names(self):
        """Query the instance names from the odbserver.

        Returns
        -------
        instance_names : list of string
            The names of the instances.
        """
        return _ascii(_decode, self._query('get_instances'))

    def node_coordinates(self, instance_name, nset_name=''):
        """Query the node coordinates of an instance.

        Parameters
        ----------
        instance_name : string
            The name of the instance to be queried
        nset_name : string, optional
            A name of a node set of the instance that the query is to be limited to.

        Returns
        -------
        node_coords : :class:`pandas.DataFrame`
            The node list as a pandas data frame without connectivity.
            The columns are named ``x``, ``y`` and ``z``.
        """
        self._fail_if_instance_invalid(instance_name)
        index, node_data = self._query('get_nodes', (instance_name, nset_name))
        return pd.DataFrame(
            data=node_data,
            columns=['x', 'y', 'z'],
            index=pd.Index(index, name='node_id', dtype=np.int64),
        )

    def element_connectivity(self, instance_name, elset_name=''):
        """Query the element connectivity of an instance.

        Parameters
        ----------
        instance_name : string
            The name of the instance to be queried
        elset_name : string, optional
            A name of an element set of the instance that the query is to be limited to.

        Returns
        -------
        connectivity : :class:`pandas.DataFrame`
            The connectivity as a :class:`pandas.DataFrame`.
            For every element there is list of node ids that the element is connected to.

        """
        index, connectivity = self._query(
            'get_connectivity', (instance_name, elset_name)
        )

        return pd.DataFrame(
            {
                'connectivity': [
                    conn[conn >= -0].tolist() for conn in connectivity
                ]
            },
            index=pd.Index(index, name='element_id', dtype=np.int64),
        )

    def nset_names(self, instance_name=''):
        """Query the available node set names.

        Parameters
        ----------
        instance_name : string, optional
            The name of the instance the node sets are queried from. If not given the
            node sets of all instances are returned.

        Returns
        -------
        instance_names : list of strings
            The names of the instances
        """
        self._fail_if_instance_invalid(instance_name)
        return _ascii(_decode, self._query('get_node_sets', instance_name))

    def node_ids(self, nset_name, instance_name=''):
        """Query the node ids of a certain node set.

        Parameters
        ----------
        nset_name : string
            The name of the node set
        instance_name : string, optional
            The name of the instance the node set is to be taken from. If not given
            node sets from all instances are considered.

        Returns
        -------
        node_ids : :class:`pandas.Index`
            The node ids as :class:`pandas.Index`
        """
        node_ids = self._query('get_node_set', (instance_name, nset_name))
        return pd.Index(node_ids, name='node_id', dtype=np.int64)

    def elset_names(self, instance_name=''):
        """Query the available element set names.

        Parameters
        ----------
        instance_name : string, optional
            The name of the instance the element sets are queried from. If not given the
            element sets of all instances are returned.

        Returns
        -------
        instance_names : list of strings
            The names of the instances
        """
        self._fail_if_instance_invalid(instance_name)
        return _ascii(_decode, self._query('get_element_sets', instance_name))

    def element_ids(self, elset_name, instance_name=''):
        """Query the element ids of a certain element set.

        Parameters
        ----------
        elset_name : string
            The name of the element set
        instance_name : string, optional
            The name of the instance the element set is to be taken from. If not given
            element sets from all instances are considered.

        Returns
        -------
        element_ids : :class:`pandas.Index`
            The element ids as :class:`pandas.Index`
        """
        element_ids = self._query('get_element_set', (instance_name, elset_name))
        return pd.Index(element_ids, name='element_id', dtype=np.int64)

    def step_names(self):
        """Query the step names from the odb file.

        Returns
        -------
        step_names : list of string
            The names of all the steps stored in the odb file.
        """
        return _ascii(_decode, self._query('get_steps'))

    def frame_ids(self, step_name):
        """Query the frames of a given step.

        Parameters
        ----------
        step_name : string
            The name of the step

        Returns
        -------
        step_name : list of ints
            The name of the step the frame ids are expected in.
        """
        return self._query('get_frames', step_name)

    def variable_names(self, step_name, frame_id):
        """Query the variable names of a certain step and frame.

        Parameters
        ----------
        step_name : string
            The name of the step
        frame_id : int
            The index of the frame

        Returns
        -------
        variable_names : list of string
            The names of the variables
        """
        return _ascii(_decode, self._query('get_variable_names', (step_name, frame_id)))

    def variable(self, variable_name, instance_name, step_name, frame_id, nset_name='', elset_name='', position=None):
        """Read field variable data.

        Parameters
        ----------
        variable_name : string
            The name of the variable.
        instance_name : string
            The name of the instance.
        step_name : string
            The name of the step
        frame_id : int
            The index of the frame
        nset_name : string, optional
            The name of the node set to be queried. If not given, the whole instance
        elnset_name : string, optional
            The name of the element set to be queried. If not given, the whole instance
        position : string, optional
            Position within element. Terminology as in Abaqus .inp file:
            ``INTEGRATION POINTS``, ``CENTROIDAL``, ``WHOLE ELEMENT``, ``NODES``,
            ``FACES``, ``AVERAGED AT NODES``

            If not given the native position is taken, except for ``INTEGRATION_POINTS``
            The ``ELEMENT_NODAL`` position is used.
        """
        response = self._query('get_variable', (instance_name, step_name, frame_id, variable_name, nset_name, elset_name, position))
        (labels, index_labels, index_data, values) = response

        index_labels = _ascii(_decode, index_labels)
        if len(index_labels) > 1:
            index = pd.DataFrame(index_data, columns=index_labels, dtype=np.int64).set_index(index_labels).index
        else:
            index = pd.Index(index_data[:, 0], name=index_labels[0], dtype=np.int64)

        column_names = _ascii(_decode, labels)
        return pd.DataFrame(values, index=index, columns=column_names)

    def history_regions(self, step_name):
         """Query the history Regions of a given step.

         Parameters
         ----------
         step_name : string
             The name of the step

         Returns
         -------
         historyRegions : list of strings
             The name of history regions, which are in the required step.
         """
         return self._query('get_history_regions', step_name)

    def history_outputs(self, step_name, history_region_name):
         """Query the history Outputs of a given step in a given history region.

         Parameters
         ----------
         step_name : string
             The name of the step

         history_region_name: string
             The name of the history region

         Returns
         -------
         historyOutputs : list of strings
             The name of the history outputs, which are in the required step and under the required history region
         """
         hisoutputs = self._query("get_history_outputs", (step_name, history_region_name))

         return hisoutputs

    def history_output_values(self, step_name, history_region_name, historyoutput_name):
         """Query the history Regions of a given step.

         Parameters
         ----------
         step_name : string
             The name of the step

         Returns
         -------
         historyRegions : list of strings
             The name of the step the history regions are in.
         """
         hisoutput_valuesx, hisoutput_valuesy = self._query("get_history_output_values", (step_name, history_region_name, historyoutput_name))
         history_region_description = self._query("get_history_region_description", (step_name, history_region_name))
         historyoutput_data = pd.Series(hisoutput_valuesy, index = hisoutput_valuesx, name = history_region_description + ": " + historyoutput_name)

         return historyoutput_data

    def history_region_description(self, step_name, history_region_name):
         """Query the description of a history Regions of a given step.

         Parameters
         ----------
         step_name : string
             The name of the step
         history_region_name: string
             The name of the history region

         Returns
         -------
         historyRegion_description : list of strings
             The description of the history region.
         """
         history_region_description = self._query("get_history_region_description", (step_name, history_region_name))
         return history_region_description

    def history_info(self):
        """Query all the information about the history outputs in a given odb.
         Returns
         -------
         dictionary : ldictionary which contains history information
         """
        dictionary = _decode(self._query("get_history_info"))
        return dictionary

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

    def _parse_response_py2(self):
        pickle_data = b''
        while True:
            line = self._proc.stdout.readline().rstrip() + b'\n'
            pickle_data += line
            if line == b'.\n':
                break
        return pickle.loads(pickle_data, encoding='bytes')

    def _parse_response_py3(self):
        msg = self._proc.stdout.read(8)
        expected_size,  = struct.unpack("Q", msg)
        pickle_data = self._proc.stdout.read(expected_size)
        return pickle.loads(pickle_data)

    def __del__(self):
        if self._proc is not None:
            self._send_command('QUIT')
            time.sleep(1)

    def _check_if_process_still_alive(self):
        if self._proc.poll() is not None:
            _, error_message = self._proc.communicate()
            self._proc = None

            raise OdbServerError(error_message.decode('ascii'))

    def _fail_if_instance_invalid(self, instance_name):
        if instance_name not in self.instance_names() and instance_name != '':
            raise KeyError("Invalid instance name '%s'." % instance_name)


def _ascii(fcn, args):
    if isinstance(args, list):
        return [_ascii(fcn, arg) for arg in args]

    if isinstance(args, tuple):
        return tuple(_ascii(fcn, arg) for arg in args)

    return fcn(args)


def _encode(arg):
    return arg.encode('ascii') if isinstance(arg, str) else arg


def _decode(arg):
    if isinstance(arg, bytes):
        return arg.decode('ascii')
    if isinstance(arg, dict):
        return {_decode(key): _decode(value) for key, value in arg.items()}
    if isinstance(arg, list):
        return [_decode(element) for element in arg]
    return arg


def _determine_abaqus_bin(abaqus_bin=None):
    abaqus_bin = (
        abaqus_bin or
        os.environ.get('ODBSERVER_ABAQUS_BIN') or
        _guess_abaqus_bin()  # never returns an invalid path, only None
        )

    readme_url = 'https://pylife.readthedocs.io/en/stable/tools/odbserver/index.html'

    if abaqus_bin is None:
        raise ValueError(f"Couldn't guess ``abaqus_bin``. Please specify it, see {readme_url}!")
    if abaqus_bin is None or not os.path.exists(abaqus_bin):
        raise FileNotFoundError(f"Couldn't find the specified ``abaqus_bin``: {abaqus_bin}. Please see {readme_url}")

    return abaqus_bin


def _determine_python_env_path(python_env_path=None):
    python_env_path = (
        python_env_path or
        os.environ.get('ODBSERVER_PYTHON_ENV_PATH') or
        _guess_python_env_path(python_env_path)  # never returns an invalid path, only None
        )

    readme_url = 'https://pylife.readthedocs.io/en/stable/tools/odbserver/index.html'

    if python_env_path is None:
        raise ValueError(f"Couldn't guess ``python_env_path``. Please specify a path, see {readme_url}!")
    if not os.path.exists(python_env_path):
        raise FileNotFoundError(f"Couldn't find the specified ``python_env_path``: {python_env_path}. Please see {readme_url}")

    return python_env_path


def _guess_abaqus_bin():
    if sys.platform == 'win32':
        return _guess_abaqus_bin_windows()
    return shutil.which('abaqus')


def _guess_abaqus_bin_windows():
    guesses = [
        r"C:/Program Files/SIMULIA/2024/EstProducts/win_b64/code/bin/SMALauncher.exe",
        r"C:/Program Files/SIMULIA/2023/EstProducts/win_b64/code/bin/SMALauncher.exe",
        r"C:/Program Files/SIMULIA/2022/EstProducts/win_b64/code/bin/SMALauncher.exe",
        r"C:/Program Files/SIMULIA/2021/EstProducts/win_b64/code/bin/ABQLauncher.exe",
        r"C:/Program Files/SIMULIA/2020/Products/win_b64/code/bin/ABQLauncher.exe",
        r"C:/Program Files/SIMULIA/2020/EstProducts/win_b64/code/bin/ABQLauncher.exe",
        r"C:/Program Files/SIMULIA/2018/AbaqusCAE/win_b64/code/bin/ABQLauncher.exe",
    ]
    for guess in guesses:
        if os.path.exists(guess):
            return guess


def _guess_python_env_path(python_env_path):
    home_dir = os.environ.get('HOME') or os.environ.get('USERPROFILE')
    python_env_parent_dir = os.path.dirname(sys.prefix)  # parent dir of python env where odbclient is running

    guesses = [
        os.path.join(python_env_parent_dir, '.venv-odbserver'),
        os.path.join(home_dir, '.conda', 'envs', 'odbserver'),
        os.path.join(home_dir, '.virtualenvs', 'odbserver'),
        ]
    for guess in guesses:
        if os.path.exists(guess):
            return guess


def _determine_site_packages_path(python_env_path, abaqus_bin):
    if sys.platform == 'win32':
        return os.path.join(python_env_path, 'lib', 'site-packages')
    else:
        python_version = _determine_server_python_version(abaqus_bin)
        return os.path.join(python_env_path, 'lib', f'python{python_version}', 'site-packages')


def _determine_server_python_version(abaqus_bin):
    proc = sp.Popen(
        [abaqus_bin, 'python', '--version'],
        stdout=sp.PIPE,
        stdin=sp.PIPE,
        stderr=sp.PIPE,
    )
    msg = proc.stdout.readline() or proc.stderr.readline()
    version_string = msg.decode().split(" ")[1]
    return version_string[:version_string.rfind(".")]


def _raise_if_version_mismatch(server_version):
    def strip_version(version):
        pos = version.find(".post")
        if pos == -1:
            return version
        return version[:pos]

    server_version = strip_version(server_version)
    client_version = strip_version(odbclient.__version__)

    if client_version != server_version:
        raise RuntimeError(
            "Version mismatch: "
            f"odbserver version {server_version} != odbclient version {client_version}"
        )

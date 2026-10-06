# Copyright (c) 2019-2026 - for information on the respective copyright owner
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

"""Query Abaqus ODB result files from a regular Python process.

Abaqus exposes ODB files through its own Python environment.  This module hides
that restriction by launching the matching ``odbserver`` inside Abaqus and by
communicating with it through a small pickle and NumPy-array protocol.
"""

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
    """Report that the Abaqus-backed ODB server could not be used."""

    pass


class OdbClient:
    """Read Abaqus ODB result data through an Abaqus-side server.

    ``OdbClient`` is the user-facing entry point of ``pylife-odbclient``.  It runs
    in a normal Python 3 interpreter, starts ``python -m odbserver`` with the
    Abaqus Python executable, and exchanges commands with that server.  The split
    is necessary because Abaqus ODB access is only available inside the Abaqus
    Python environment, while pyLife analyses typically run in a modern Python
    process.

    Parameters
    ----------
    odb_file : str
        Path to the Abaqus ODB file to open.
    abaqus_bin : str, optional
        Path to the Abaqus executable used to start the server.  Default is
        ``None``, which first reads ``ODBSERVER_ABAQUS_BIN`` and then tries known
        Abaqus installation paths.
    python_env_path : str, optional
        Path to the Python environment that provides the ``odbserver`` package for
        Abaqus.  Default is ``None``, which first reads
        ``ODBSERVER_PYTHON_ENV_PATH`` and then tries common environment locations.

    Raises
    ------
    ValueError
        If the Abaqus executable or server Python environment cannot be guessed.
    FileNotFoundError
        If an explicitly selected executable or environment path does not exist.
    OdbServerError
        If the server process exits or does not announce readiness.
    RuntimeError
        If client and server package versions differ.

    See Also
    --------
    odbclient.OdbClient.variable : Read field output values as a DataFrame.
    odbclient.OdbClient.node_coordinates : Read nodal coordinates as a DataFrame.

    Examples
    --------
    The snippets below need an actual Abaqus installation and ODB file, so they
    are shown as a transcript rather than as executable doctests.

    Open an ODB file and list the part instances it contains.

    .. code-block:: pycon

        >>> import odbclient as CL
        >>> client = CL.OdbClient("some_file.odb")
        >>> client.instance_names()
        ['PART-1-1']

    Query the node coordinates of an instance.

    .. code-block:: pycon

        >>> client.node_coordinates('PART-1-1')
                    x     y     z
        node_id
        1       -30.0  15.0  10.0
        2       -30.0  25.0  10.0
        3       -30.0  15.0   0.0
        ...

    Query the steps, the frames of a step and the field variables written in a
    given frame.

    .. code-block:: pycon

        >>> client.step_names()
        ['Load']
        >>> client.frame_ids('Load')
        [0, 1]
        >>> client.variable_names('Load', 1)
        ['CF', 'COORD', 'E', 'EVOL', 'IVOL', 'RF', 'S', 'U']

    Read the stress tensor of an instance for a given step and frame.

    .. code-block:: pycon

        >>> client.variable('S', 'PART-1-1', 'Load', 1)
                                  S11        S22  ...       S13       S23
        node_id element_id                        ...
        5       1          -38.617779   2.705118  ... -3.578981  1.355571
        7       1          -38.617779   2.705118  ...  3.578981 -1.355571
        3       1          -50.749348 -21.749729  ... -7.597347 -0.000003
        1       1          -50.749348 -21.749729  ...  7.597347  0.000003
        6       1           38.643414  -2.588303  ...  3.522046  1.446851
        ...                       ...        ...  ...       ...       ...

    Oftentimes it is desirable to have the node coordinates and multiple field
    variables in one data frame.  This is easily achieved by
    :meth:`~pandas.DataFrame.join` operations.

    .. code-block:: pycon

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
        ...                  ...   ...   ...  ...           ...           ...           ...
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
        """Return the instance names stored in the ODB root assembly.

        Returns
        -------
        list of str
            Names of all part instances available for mesh and result queries.
        """
        return _ascii(_decode, self._query('get_instances'))

    def node_coordinates(self, instance_name, nset_name=''):
        """Read node coordinates for an instance or node set.

        Parameters
        ----------
        instance_name : str
            Name of the Abaqus part instance to query.
        nset_name : str, optional
            Name of a node set that limits the query.  Default is ``''``, which
            returns all nodes of ``instance_name``.

        Returns
        -------
        pandas.DataFrame
            Node coordinates with columns ``x``, ``y``, and ``z`` and an index named
            ``node_id``.

        Raises
        ------
        KeyError
            If ``instance_name`` is not available in the ODB.
        """
        self._fail_if_instance_invalid(instance_name)
        index, node_data = self._query('get_nodes', (instance_name, nset_name))
        return pd.DataFrame(
            data=node_data,
            columns=['x', 'y', 'z'],
            index=pd.Index(index, name='node_id', dtype=np.int64),
        )

    def element_connectivity(self, instance_name, elset_name=''):
        """Read element connectivity for an instance or element set.

        Parameters
        ----------
        instance_name : str
            Name of the Abaqus part instance to query.
        elset_name : str, optional
            Name of an element set that limits the query.  Default is ``''``, which
            returns all elements of ``instance_name``.

        Returns
        -------
        pandas.DataFrame
            One column named ``connectivity`` containing lists of connected node IDs.
            The index is named ``element_id``.
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
        """Return node set names available in the ODB.

        Parameters
        ----------
        instance_name : str, optional
            Instance whose node sets are requested.  Default is ``''``, which
            requests assembly-level node sets.

        Returns
        -------
        list of str
            Node set names visible at the selected scope.

        Raises
        ------
        KeyError
            If ``instance_name`` is not available in the ODB.
        """
        self._fail_if_instance_invalid(instance_name)
        return _ascii(_decode, self._query('get_node_sets', instance_name))

    def node_ids(self, nset_name, instance_name=''):
        """Read node IDs from a node set.

        Parameters
        ----------
        nset_name : str
            Name of the node set to query.
        instance_name : str, optional
            Instance that owns the node set.  Default is ``''``, which queries an
            assembly-level node set.

        Returns
        -------
        pandas.Index
            Node identifiers with name ``node_id`` and integer dtype.
        """
        node_ids = self._query('get_node_set', (instance_name, nset_name))
        return pd.Index(node_ids, name='node_id', dtype=np.int64)

    def elset_names(self, instance_name=''):
        """Return element set names available in the ODB.

        Parameters
        ----------
        instance_name : str, optional
            Instance whose element sets are requested.  Default is ``''``, which
            requests assembly-level element sets.

        Returns
        -------
        list of str
            Element set names visible at the selected scope.

        Raises
        ------
        KeyError
            If ``instance_name`` is not available in the ODB.
        """
        self._fail_if_instance_invalid(instance_name)
        return _ascii(_decode, self._query('get_element_sets', instance_name))

    def element_ids(self, elset_name, instance_name=''):
        """Read element IDs from an element set.

        Parameters
        ----------
        elset_name : str
            Name of the element set to query.
        instance_name : str, optional
            Instance that owns the element set.  Default is ``''``, which queries an
            assembly-level element set.

        Returns
        -------
        pandas.Index
            Element identifiers with name ``element_id`` and integer dtype.
        """
        element_ids = self._query('get_element_set', (instance_name, elset_name))
        return pd.Index(element_ids, name='element_id', dtype=np.int64)

    def step_names(self):
        """Return analysis step names stored in the ODB.

        Returns
        -------
        list of str
            Names of all Abaqus analysis steps.
        """
        return _ascii(_decode, self._query('get_steps'))

    def frame_ids(self, step_name):
        """Return frame IDs available in an analysis step.

        Parameters
        ----------
        step_name : str
            Name of the Abaqus analysis step.

        Returns
        -------
        list of int
            Abaqus frame IDs contained in ``step_name``.
        """
        return self._query('get_frames', step_name)

    def variable_names(self, step_name, frame_id):
        """Return field output variable names for one frame.

        Parameters
        ----------
        step_name : str
            Name of the Abaqus analysis step.
        frame_id : int
            Abaqus frame ID within ``step_name``.

        Returns
        -------
        list of str
            Field output variable names, for example ``S`` or ``U``.
        """
        return _ascii(_decode, self._query('get_variable_names', (step_name, frame_id)))

    def variable(self, variable_name, instance_name, step_name, frame_id, nset_name='', elset_name='', position=None):
        """Read a field output variable as a pandas DataFrame.

        Parameters
        ----------
        variable_name : str
            Abaqus field output variable name, for example ``S``, ``E``, or ``U``.
        instance_name : str
            Name of the Abaqus part instance to query.
        step_name : str
            Name of the Abaqus analysis step.
        frame_id : int
            Abaqus frame ID within ``step_name``.
        nset_name : str, optional
            Node set that limits the field output query.  Default is ``''``, which
            does not apply a node-set filter.
        elset_name : str, optional
            Element set that limits the field output query.  Default is ``''``,
            which does not apply an element-set filter.
        position : str, optional
            Abaqus output position in input-file terminology, such as
            ``INTEGRATION POINTS``, ``CENTROIDAL``, ``WHOLE ELEMENT``, ``NODES``,
            ``FACES``, or ``AVERAGED AT NODES``.  Default is ``None``, which uses the
            native ODB position except that integration-point data is requested as
            element-nodal data.

        Returns
        -------
        pandas.DataFrame
            Field values with one column per scalar component.  The index is named
            ``node_id`` or ``element_id`` for single-location data and is a MultiIndex
            with labels such as ``node_id``, ``element_id``, ``ipoint_id``, or
            ``face_id`` when Abaqus supplies multiple identifiers.
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
         """Return history region names for one analysis step.

         Parameters
         ----------
         step_name : str
             Name of the Abaqus analysis step.

         Returns
         -------
         list of str
             History region names available in ``step_name``.
         """
         return self._query('get_history_regions', step_name)

    def history_outputs(self, step_name, history_region_name):
         """Return history output names for a history region.

         Parameters
         ----------
         step_name : str
             Name of the Abaqus analysis step.
         history_region_name : str
             Name of the Abaqus history region.

         Returns
         -------
         list of str
             History output names available in the selected region.
         """
         hisoutputs = self._query("get_history_outputs", (step_name, history_region_name))

         return hisoutputs

    def history_output_values(self, step_name, history_region_name, historyoutput_name):
         """Read one history output as a pandas Series.

         Parameters
         ----------
         step_name : str
             Name of the Abaqus analysis step.
         history_region_name : str
             Name of the Abaqus history region.
         historyoutput_name : str
             Name of the history output variable.

         Returns
         -------
         pandas.Series
             History output values indexed by the time-like abscissa returned by
             Abaqus.  The series name combines the region description and output name.
         """
         hisoutput_valuesx, hisoutput_valuesy = self._query("get_history_output_values", (step_name, history_region_name, historyoutput_name))
         history_region_description = self._query("get_history_region_description", (step_name, history_region_name))
         historyoutput_data = pd.Series(hisoutput_valuesy, index = hisoutput_valuesx, name = history_region_description + ": " + historyoutput_name)

         return historyoutput_data

    def history_region_description(self, step_name, history_region_name):
         """Return the descriptive text of a history region.

         Parameters
         ----------
         step_name : str
             Name of the Abaqus analysis step.
         history_region_name : str
             Name of the Abaqus history region.

         Returns
         -------
         str
             Description stored by Abaqus for the selected history region.
         """
         history_region_description = self._query("get_history_region_description", (step_name, history_region_name))
         return history_region_description

    def history_info(self):
        """Return a nested summary of all history output metadata.

        Returns
        -------
        dict
            Mapping from history region descriptions to region names, output names,
            and steps in which the region occurs.
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

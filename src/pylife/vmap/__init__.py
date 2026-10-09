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

"""Import and export VMAP result files for pyLife workflows.

VMAP is an HDF5-based neutral CAE format for exchanging finite-element
geometry and result data between solvers, post-processors, and fatigue
analysis tools.  The :class:`~pylife.vmap.VMAPImport` class reads VMAP
geometries, states, and variables into the pandas mesh representation used by
pyLife.  The :class:`~pylife.vmap.VMAPExport` class writes pyLife meshes and
result fields back to VMAP for post-processing in CAE tools.

See Also
--------
pylife.vmap.VMAPImport : Read VMAP geometry and variables into pandas.
pylife.vmap.VMAPExport : Write pyLife mesh data and variables to VMAP.

Notes
-----
The VMAP demo and tutorial material in pyLife shows the typical workflow:
open a ``.vmap`` file, choose a geometry and state, join coordinates and
variables such as ``STRESS_CAUCHY`` or ``DISPLACEMENT``, and continue with the
``pylife.mesh`` accessors.  Numeric units are not converted by this package;
they follow the unit system used by the originating finite-element model.
"""
from .vmap_import import VMAPImport
from .vmap_export import VMAPExport, VMAPExportError
from .exceptions import *

__all__ = ['VMAPImport', 'VMAPExport', 'VMAPExportError']

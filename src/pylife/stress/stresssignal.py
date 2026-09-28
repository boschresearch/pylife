# Copyright (c) 2019-2023 - for information on the respective copyright owner
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

"""Validate stress tensor signals stored in Voigt notation.

The module provides the ``voigt`` DataFrame accessor for symmetric Cauchy
stress tensors given by the six components ``S11, S22, S33, S12, S13, S23``
in MPa, as they are written by finite-element solvers and consumed by the
equivalent stress and fatigue strength modules of pyLife.
"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import pandas as pd
import numpy as np

from pylife import PylifeSignal


@pd.api.extensions.register_dataframe_accessor("voigt")
class StressTensorVoigt(PylifeSignal):
    r"""Represent a stress tensor stored in Voigt notation.

    The accessor validates a :class:`pandas.DataFrame` that stores one
    symmetric Cauchy stress tensor per row.  The mandatory columns define the
    tensor in Voigt notation ``S11, S22, S33, S12, S13, S23``:

    * ``S11``: Normal stress component in direction 1 in MPa.
    * ``S22``: Normal stress component in direction 2 in MPa.
    * ``S33``: Normal stress component in direction 3 in MPa.
    * ``S12``: Shear stress component in the 1-2 plane in MPa.
    * ``S13``: Shear stress component in the 1-3 plane in MPa.
    * ``S23``: Shear stress component in the 2-3 plane in MPa.

    All components are interpreted as engineering stress components of a
    symmetric tensor,

    .. math::

        \sigma =
        \begin{pmatrix}
        S11 & S12 & S13\\
        S12 & S22 & S23\\
        S13 & S23 & S33
        \end{pmatrix}.

    Parameters
    ----------
    pandas_obj : pandas.DataFrame
        DataFrame containing the mandatory Voigt stress tensor component
        columns.

    See Also
    --------
    pylife.stress.equistress.StressTensorEquistress : Calculate equivalent
        stresses from a Voigt stress tensor DataFrame.

    Notes
    -----
    Accessing ``df.voigt`` raises an :class:`AttributeError` if any mandatory
    component column is missing.  Subclasses, for example
    :class:`~pylife.stress.equistress.StressTensorEquistress`, build on this
    contract to calculate derived stress quantities.

    Examples
    --------
    >>> import pylife.stress.stresssignal
    >>> df = pd.DataFrame({'S11': [1.0], 'S22': [0.0], 'S33': [0.0],
    ...                    'S12': [0.0], 'S13': [0.0], 'S23': [0.0]})
    >>> df.voigt.to_pandas()
       S11  S22  S33  S12  S13  S23
    0  1.0  0.0  0.0  0.0  0.0  0.0
    """
    def _validate(self):
        self.fail_if_key_missing(['S11', 'S22', 'S33', 'S12', 'S13', 'S23'])

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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import pandas as pd
import numpy as np

from pylife import PylifeSignal


@pd.api.extensions.register_dataframe_accessor("voigt")
class StressTensorVoigt(PylifeSignal):
    '''DataFrame accessor class for Voigt noted stress tensors

    Raises
    ------
    AttributeError
        if at least one of the needed columns is missing.

    Notes
    -----
    Base class to access :class:`pandas.DataFrame` objects containing
    Voigt noted stress tensors. The stress tensor components are assumed
    to be in the columns `S11`, `S22`, `S33`, `S12`, `S13`, `S23`.

    See also
    --------
    :func:`pandas.api.extensions.register_dataframe_accessor()`

    Examples
    --------
    For an example see :class:`equistress.StressTensorEquistress`.
    '''
    def _validate(self):
        self.fail_if_key_missing(['S11', 'S22', 'S33', 'S12', 'S13', 'S23'])

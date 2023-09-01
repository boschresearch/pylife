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

import pytest
import pandas as pd
import numpy as np

import pylife.stress.stresssignal


def test_voigt():
    df = pd.DataFrame({'S11': [1.0], 'S22': [1.0], 'S33': [1.0],
                       'S12': [1.0], 'S13': [1.0], 'S23': [1.0]})
    df.voigt


def test_voigt_fail():
    df = pd.DataFrame({'S11': [1.0], 'S22': [1.0], 'S33': [1.0],
                       'S12': [1.0], 'S31': [1.0], 'S23': [1.0]})
    with pytest.raises(AttributeError, match=r'^StressTensorVoigt.*Missing S13'):
        df.voigt

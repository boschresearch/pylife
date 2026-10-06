# Copyright (c) 2024-2026 - for information on the respective copyright owner
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


@pytest.fixture(autouse=True, scope="session")
def use_pandas_cow_behavior():
    if pd.__version__.startswith("2"):
        pd.options.mode.copy_on_write = True
    yield
    if pd.__version__.startswith("2"):
        pd.options.mode.copy_on_write = False

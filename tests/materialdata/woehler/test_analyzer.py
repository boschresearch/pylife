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

import numpy as np
import pandas as pd
import pytest
import unittest.mock as mock

from io import StringIO

from pylife.materialdata import woehler

data = pd.DataFrame(np.array([
        [4.50e+02, 3.40e+04],
        [4.50e+02, 5.40e+04],
        [4.50e+02, 6.00e+04],
        [4.50e+02, 7.60e+04],
        [4.00e+02, 5.30e+04],
        [4.00e+02, 9.40e+04],
        [4.00e+02, 2.07e+05],
        [4.00e+02, 2.27e+05],
        [3.75e+02, 6.80e+04],
        [3.75e+02, 2.34e+05],
        [3.75e+02, 3.96e+05],
        [3.75e+02, 5.00e+05],
        [3.75e+02, 6.00e+05],
        [3.75e+02, 7.09e+05],
        [3.50e+02, 1.70e+05],
        [3.50e+02, 1.87e+05],
        [3.50e+02, 2.20e+05],
        [3.50e+02, 2.89e+05],
        [3.50e+02, 3.09e+05],
        [3.50e+02, 1.00e+07],
        [3.25e+02, 6.75e+05],
        [3.25e+02, 7.51e+05],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.00e+02, 8.95e+05],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07]
]), columns=['load', 'cycles']).sample(frac=1)

data_no_mixed_horizons = pd.DataFrame(np.array([
        [4.50e+02, 3.40e+04],
        [4.50e+02, 5.40e+04],
        [4.50e+02, 6.00e+04],
        [4.50e+02, 7.60e+04],
        [4.00e+02, 5.30e+04],
        [4.00e+02, 9.40e+04],
        [4.00e+02, 2.07e+05],
        [4.00e+02, 2.27e+05],
        [3.75e+02, 6.80e+04],
        [3.75e+02, 2.34e+05],
        [3.75e+02, 3.96e+05],
        [3.75e+02, 5.00e+05],
        [3.75e+02, 6.00e+05],
        [3.75e+02, 7.09e+05],
        [3.50e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07]
]), columns=['load', 'cycles']).sample(frac=1)

no_mixed_horizons_finite_expected = pd.DataFrame(np.array([
        [4.50e+02, 3.40e+04],
        [4.50e+02, 5.40e+04],
        [4.50e+02, 6.00e+04],
        [4.50e+02, 7.60e+04],
        [4.00e+02, 5.30e+04],
        [4.00e+02, 9.40e+04],
        [4.00e+02, 2.07e+05],
        [4.00e+02, 2.27e+05],
        [3.75e+02, 6.80e+04],
        [3.75e+02, 2.34e+05],
        [3.75e+02, 3.96e+05],
        [3.75e+02, 5.00e+05],
        [3.75e+02, 6.00e+05],
        [3.75e+02, 7.09e+05],
]), columns=['load', 'cycles'])

no_mixed_horizons_infinite_expected = pd.DataFrame(np.array([
        [3.50e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07]
]), columns=['load', 'cycles'])

data_pure_runout_horizon_and_mixed_horizons = pd.DataFrame(np.array([
        [4.50e+02, 3.40e+04],
        [4.50e+02, 5.40e+04],
        [4.50e+02, 6.00e+04],
        [4.50e+02, 7.60e+04],
        [4.00e+02, 5.30e+04],
        [4.00e+02, 9.40e+04],
        [4.00e+02, 2.07e+05],
        [4.00e+02, 2.27e+05],
        [3.75e+02, 6.80e+04],
        [3.75e+02, 2.34e+05],
        [3.75e+02, 3.96e+05],
        [3.75e+02, 5.00e+05],
        [3.75e+02, 6.00e+05],
        [3.75e+02, 7.09e+05],
        [3.50e+02, 1.70e+05],
        [3.50e+02, 1.87e+05],
        [3.50e+02, 2.20e+05],
        [3.50e+02, 2.89e+05],
        [3.50e+02, 3.09e+05],
        [3.50e+02, 1.00e+07],
        [3.25e+02, 6.75e+05],
        [3.25e+02, 7.51e+05],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07]
]), columns=['load', 'cycles']).sample(frac=1)

pure_runout_horizon_and_mixed_horizons_finite_expected = pd.DataFrame(np.array([
        [4.50e+02, 3.40e+04],
        [4.50e+02, 5.40e+04],
        [4.50e+02, 6.00e+04],
        [4.50e+02, 7.60e+04],
        [4.00e+02, 5.30e+04],
        [4.00e+02, 9.40e+04],
        [4.00e+02, 2.07e+05],
        [4.00e+02, 2.27e+05],
        [3.75e+02, 6.80e+04],
        [3.75e+02, 2.34e+05],
        [3.75e+02, 3.96e+05],
        [3.75e+02, 5.00e+05],
        [3.75e+02, 6.00e+05],
        [3.75e+02, 7.09e+05]
    ]), columns=['load', 'cycles'])

pure_runout_horizon_and_mixed_horizons_infinite_expected = pd.DataFrame(np.array([
        [3.50e+02, 1.70e+05],
        [3.50e+02, 1.87e+05],
        [3.50e+02, 2.20e+05],
        [3.50e+02, 2.89e+05],
        [3.50e+02, 3.09e+05],
        [3.50e+02, 1.00e+07],
        [3.25e+02, 6.75e+05],
        [3.25e+02, 7.51e+05],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07]
]), columns=['load', 'cycles'])

pure_runout_horizon_and_mixed_horizons_finite_expected_conservative = pd.DataFrame(np.array([
        [4.50e+02, 3.40e+04],
        [4.50e+02, 5.40e+04],
        [4.50e+02, 6.00e+04],
        [4.50e+02, 7.60e+04],
        [4.00e+02, 5.30e+04],
        [4.00e+02, 9.40e+04],
        [4.00e+02, 2.07e+05],
        [4.00e+02, 2.27e+05],
        [3.75e+02, 6.80e+04],
        [3.75e+02, 2.34e+05],
        [3.75e+02, 3.96e+05],
        [3.75e+02, 5.00e+05],
        [3.75e+02, 6.00e+05],
        [3.75e+02, 7.09e+05],
    ]), columns=['load', 'cycles'])

pure_runout_horizon_and_mixed_horizons_infinite_expected_conservative = pd.DataFrame(np.array([
        [3.50e+02, 1.70e+05],
        [3.50e+02, 1.87e+05],
        [3.50e+02, 2.20e+05],
        [3.50e+02, 2.89e+05],
        [3.50e+02, 3.09e+05],
        [3.50e+02, 1.00e+07],
        [3.25e+02, 6.75e+05],
        [3.25e+02, 7.51e+05],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07],
        [2.90e+02, 1.00e+07]
]), columns=['load', 'cycles'])

data_no_runouts = pd.DataFrame(np.array([
        [4.50e+02, 3.40e+04],
        [4.50e+02, 5.40e+04],
        [4.50e+02, 6.00e+04],
        [4.50e+02, 7.60e+04],
        [4.00e+02, 5.30e+04],
        [4.00e+02, 9.40e+04],
        [4.00e+02, 2.07e+05],
        [4.00e+02, 2.27e+05],
        [3.75e+02, 6.80e+04],
        [3.75e+02, 2.34e+05],
        [3.75e+02, 3.96e+05],
        [3.75e+02, 5.00e+05],
        [3.75e+02, 6.00e+05],
        [3.75e+02, 7.09e+05],
        [3.50e+02, 1.70e+05],
        [3.50e+02, 1.87e+05],
        [3.50e+02, 2.20e+05],
        [3.50e+02, 2.89e+05],
        [3.50e+02, 3.09e+05],
        [3.25e+02, 6.75e+05],
        [3.25e+02, 7.51e+05],
]), columns=['load', 'cycles']).sample(frac=1)

no_runouts_infinite_expected = data_no_runouts[:0]
no_runouts_finite_expected = data_no_runouts

data_only_runout_levels = pd.DataFrame(np.array([
        [3.50e+02, 1.70e+05],
        [3.50e+02, 1.87e+05],
        [3.50e+02, 2.20e+05],
        [3.50e+02, 2.89e+05],
        [3.50e+02, 1.00e+07],
        [3.50e+02, 1.00e+07],
        [3.25e+02, 6.75e+05],
        [3.25e+02, 7.51e+05],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.00e+02, 8.95e+05],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07]
]), columns=['load', 'cycles']).sample(frac=1)

only_runout_levels_infinite_expected = data_only_runout_levels
only_runout_levels_finite_expected = data_only_runout_levels[:0]


data_one_runout_load_level = pd.DataFrame(np.array([
    [3.5000e+02, 8.5930e+05],
    [3.5000e+02, 6.5720e+05],
    [3.5000e+02, 5.8650e+05],
    [3.5000e+02, 1.3085e+06],
    [3.5000e+02, 1.4485e+06],
    [3.0000e+02, 1.0000e+07],
    [3.0000e+02, 1.0000e+07],
    [3.0000e+02, 1.0000e+07],
    [3.0000e+02, 1.0000e+07],
    [3.0000e+02, 1.0000e+07],
    [4.5000e+02, 3.2600e+05],
    [4.5000e+02, 2.6280e+05],
    [4.5000e+02, 1.3200e+05],
    [4.5000e+02, 1.9310e+05],
    [4.5000e+02, 2.5670e+05],
    [3.2500e+02, 2.2402e+06],
    [3.2500e+02, 1.1630e+06],
    [3.2500e+02, 1.2132e+06],
    [3.2500e+02, 1.0347e+06],
    [3.2500e+02, 2.2683e+06]
]), columns=['load', 'cycles']).sample(frac=1)


load_sorted = pd.Series(np.array([
        4.50e+02, 4.50e+02, 4.50e+02, 4.50e+02, 4.00e+02, 4.00e+02, 4.00e+02, 4.00e+02, 3.75e+02, 3.75e+02,
        3.75e+02, 3.75e+02, 3.75e+02, 3.75e+02, 3.50e+02, 3.50e+02, 3.50e+02, 3.50e+02, 3.50e+02, 3.50e+02,
        3.25e+02, 3.25e+02, 3.25e+02, 3.25e+02, 3.25e+02, 3.25e+02, 3.25e+02, 3.25e+02, 3.25e+02, 3.25e+02,
        3.00e+02, 3.00e+02, 3.00e+02, 3.00e+02, 3.00e+02, 3.00e+02, 3.00e+02, 3.00e+02, 3.00e+02, 3.00e+02]), name='load').sort_values()

cycles_sorted = pd.Series(np.array([
        3.40e+04, 5.40e+04, 6.00e+04, 7.60e+04, 5.30e+04, 9.40e+04, 2.07e+05, 2.27e+05, 6.80e+04, 2.34e+05,
        3.96e+05, 5.00e+05, 6.00e+05, 7.09e+05, 1.70e+05, 1.87e+05, 2.20e+05, 2.89e+05, 3.09e+05, 1.00e+07,
        6.75e+05, 7.51e+05, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07,
        8.95e+05, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07, 1.00e+07]), name='cycles').sort_values()

finite_expected = pd.DataFrame(np.array([
        [4.50e+02, 3.40e+04],
        [4.50e+02, 5.40e+04],
        [4.50e+02, 6.00e+04],
        [4.50e+02, 7.60e+04],
        [4.00e+02, 5.30e+04],
        [4.00e+02, 9.40e+04],
        [4.00e+02, 2.07e+05],
        [4.00e+02, 2.27e+05],
        [3.75e+02, 6.80e+04],
        [3.75e+02, 2.34e+05],
        [3.75e+02, 3.96e+05],
        [3.75e+02, 5.00e+05],
        [3.75e+02, 6.00e+05],
        [3.75e+02, 7.09e+05],
]), columns=['load', 'cycles']).sort_values(by='load').reset_index(drop=True)

infinite_expected = pd.DataFrame(np.array([
        [3.50e+02, 1.70e+05],
        [3.50e+02, 1.87e+05],
        [3.50e+02, 2.20e+05],
        [3.50e+02, 2.89e+05],
        [3.50e+02, 3.09e+05],
        [3.50e+02, 1.00e+07],
        [3.25e+02, 6.75e+05],
        [3.25e+02, 7.51e+05],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.00e+02, 8.95e+05],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07]
]), columns=['load', 'cycles']).sort_values(by='load').reset_index(drop=True)

finite_expected_conservative = pd.DataFrame(np.array([
        [4.50e+02, 3.40e+04],
        [4.50e+02, 5.40e+04],
        [4.50e+02, 6.00e+04],
        [4.50e+02, 7.60e+04],
        [4.00e+02, 5.30e+04],
        [4.00e+02, 9.40e+04],
        [4.00e+02, 2.07e+05],
        [4.00e+02, 2.27e+05],
        [3.75e+02, 6.80e+04],
        [3.75e+02, 2.34e+05],
        [3.75e+02, 3.96e+05],
        [3.75e+02, 5.00e+05],
        [3.75e+02, 6.00e+05],
        [3.75e+02, 7.09e+05],
]), columns=['load', 'cycles']).sort_values(by='load').reset_index(drop=True)

infinite_expected_conservative = pd.DataFrame(np.array([
        [3.50e+02, 1.70e+05],
        [3.50e+02, 1.87e+05],
        [3.50e+02, 2.20e+05],
        [3.50e+02, 2.89e+05],
        [3.50e+02, 3.09e+05],
        [3.50e+02, 1.00e+07],
        [3.25e+02, 6.75e+05],
        [3.25e+02, 7.51e+05],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.00e+02, 8.95e+05],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07]
]), columns=['load', 'cycles']).sort_values(by='load').reset_index(drop=True)


data_01 = pd.DataFrame({"load": np.array([620, 620, 620, 550, 550, 500, 500, 500, 500, 500, 480, 480]), "cycles": np.array(
    [65783, 89552, 115800, 141826, 190443, 293418, 383341, 525438, 967091, 99505992, 99524024, 199563776])})
data_01_no_pure_runout_horizon = data_01[data_01.load > 480]
data_01_two_fractures = pd.DataFrame({"load": np.array([620, 550,480]), "cycles": np.array([65783, 141826,199563776])})
data_01_one_fracture_level = data_01[data_01.load > 550]


def read_data(s, thres=1e6):
    d = pd.read_csv(StringIO(s), sep="\t", comment="#", names=['cycles', 'load'])
    d.N_threshold = thres
    return d


data_neg_res_1 = read_data("""
127700	450
108000	450
124000	450
127000	450
101000	450
199000	400
213600	400
166000	400
146000	400
140600	400
295000	350
264000	350
330000	350
352000	350
438000	350
645600	300
412000	300
772000	300
501200	300
593600	300
1856000	250
1989900	250
2114500	250
1121400	250
1233600	250
10000000	200
5131700	200
3857300	200
9620900	175
7634000	175
10000000	175
10000000	175
10000000	175""", 10000000)

data_neg_res_2 = read_data("""
109000	377.556025
51000	377.556025
75000	377.556025
111000	377.556025
140000	377.556025
128000	362.84605
75000	362.84605
109000	362.84605
83000	362.84605
68000	362.84605
132000	362.84605
253000	333.4261
987000	333.4261
246000	333.4261
224000	333.4261
159000	333.4261
10000000	304.00615
4576000	304.00615
2139000	304.00615
10000000	304.00615
4576000	304.00615
1191000	289.296175
10000000	289.296175
6337000	289.296175
10000000	289.296175
10000000	289.296175""", 10000000)

all_data = [
    data,
    data_neg_res_1,
    data_neg_res_2
]

def sort_fatigue_data(fd):
    return fd.sort_values(by=['load', 'cycles']).reset_index(drop=True)


def test_fatigue_data_simple_properties():
    fd = woehler.determine_fractures(data, 1e7).sort_index().fatigue_data
    pd.testing.assert_series_equal(fd.load.sort_values(), load_sorted)
    pd.testing.assert_series_equal(fd.cycles.sort_values(), cycles_sorted)

    assert fd.num_runouts == 18
    assert fd.num_fractures == 22


@pytest.mark.parametrize("data, finite_zone_expected, infinite_zone_expected", [
    (data,
     finite_expected,
     infinite_expected),
    (data_no_mixed_horizons,
     no_mixed_horizons_finite_expected,
     no_mixed_horizons_infinite_expected),
    (data_pure_runout_horizon_and_mixed_horizons,
     pure_runout_horizon_and_mixed_horizons_finite_expected,
     pure_runout_horizon_and_mixed_horizons_infinite_expected),
    (data_no_runouts,
     no_runouts_finite_expected,
     no_runouts_infinite_expected),
    (data_only_runout_levels,
     only_runout_levels_finite_expected,
     only_runout_levels_infinite_expected)
])
def test_fatigue_data_finite_infinite_zone(data, finite_zone_expected, infinite_zone_expected):
    fd = woehler.determine_fractures(data, 1e7).sort_index().fatigue_data
    pd.testing.assert_frame_equal(sort_fatigue_data(fd.finite_zone)[['load', 'cycles']],
                                  sort_fatigue_data(finite_zone_expected))
    fd = woehler.determine_fractures(data, 1e7).sort_index().fatigue_data
    pd.testing.assert_frame_equal(sort_fatigue_data(fd.infinite_zone)[['load', 'cycles']],
                                  sort_fatigue_data(infinite_zone_expected))

@pytest.mark.parametrize("data, finite_zone_expected, infinite_zone_expected", [
    (data,
     finite_expected_conservative,
     infinite_expected_conservative),
    (data_no_mixed_horizons,
     no_mixed_horizons_finite_expected,
     no_mixed_horizons_infinite_expected),
    (data_pure_runout_horizon_and_mixed_horizons,
     pure_runout_horizon_and_mixed_horizons_finite_expected_conservative,
     pure_runout_horizon_and_mixed_horizons_infinite_expected_conservative),
    (data_no_runouts,
     no_runouts_finite_expected,
     no_runouts_infinite_expected)
])
def test_fatigue_data_finite_infinite_zone_conservative(data, finite_zone_expected, infinite_zone_expected):
    fd = woehler.determine_fractures(data, 1e7).sort_index().fatigue_data.conservative_fatigue_limit()
    print(sort_fatigue_data(fd.finite_zone))
    print(sort_fatigue_data(finite_zone_expected))
    print(fd.fatigue_limit)
    print(sort_fatigue_data(fd.infinite_zone))
    print(sort_fatigue_data(infinite_zone_expected))
    pd.testing.assert_frame_equal(sort_fatigue_data(fd.finite_zone)[['load', 'cycles']],
                                  sort_fatigue_data(finite_zone_expected))
    fd = woehler.determine_fractures(data, 1e7).sort_index().fatigue_data.conservative_fatigue_limit()
    pd.testing.assert_frame_equal(sort_fatigue_data(fd.infinite_zone)[['load', 'cycles']],
                                  sort_fatigue_data(infinite_zone_expected))


def test_woehler_fracture_determination_given():
    df = pd.DataFrame({
        'load': [1, 2, 3],
        'cycles': [1e6, 1e7, 1e4]
    })

    expected = pd.DataFrame({
        'load': [1, 2, 3],
        'cycles': [1e6, 1e7, 1e4],
        'fracture': [True, False, True]
    })

    expected_runouts = pd.DataFrame({
        'load': [2],
        'cycles': [1e7],
        'fracture': [False]
    }, index=[1])

    expected_fractures = pd.DataFrame({
        'load': [1, 3],
        'cycles': [1e6, 1e4],
        'fracture': [True, True]
    }, index=[0, 2])

    test = woehler.determine_fractures(df, 1e7).sort_index()
    pd.testing.assert_frame_equal(test, expected)

    fd = test.fatigue_data
    pd.testing.assert_frame_equal(fd.fractures, expected_fractures)
    pd.testing.assert_frame_equal(fd.runouts, expected_runouts)


def test_woehler_fracture_determination_infered():
    df = pd.DataFrame({
        'load': [1, 2, 3],
        'cycles': [1e6, 1e7, 1e4]
    })

    expected = pd.DataFrame({
        'load': [1, 2, 3],
        'cycles': [1e6, 1e7, 1e4],
        'fracture': [True, False, True]
    })

    test = woehler.determine_fractures(df).sort_index()
    pd.testing.assert_frame_equal(test, expected)


@pytest.mark.parametrize("data, fatigue_limit_expected", [
    (data, 362.5),
    (data_no_mixed_horizons, 362.5),
    (data_pure_runout_horizon_and_mixed_horizons, 362.5),
    (data_no_runouts, 0.0),
    (data_only_runout_levels, 362.5)
])
def test_woehler_endur_zones(data, fatigue_limit_expected):
    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    assert fd.fatigue_limit == fatigue_limit_expected


@pytest.mark.parametrize("data, fatigue_limit_expected", [
    (data, 325.0),
    (data_no_mixed_horizons, 350.0),
    (data_pure_runout_horizon_and_mixed_horizons, 325.0),
    (data_no_runouts, 0.0),
    (data_only_runout_levels, 325.0)
])
def test_woehler_endur_zones_conservative(data, fatigue_limit_expected):
    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    print(fd.fatigue_limit)
    fd = fd.conservative_fatigue_limit()
    print(fd.fatigue_limit)
    assert fd.fatigue_limit == fatigue_limit_expected


def test_woehler_endure_zones_no_runouts():
    df = data[data.cycles < 1e7]
    fd = woehler.determine_fractures(df, 1e7).fatigue_data
    assert fd.fatigue_limit == 0.0


def test_woehler_elementary():
    expected = pd.Series({
        'SD': 362.5,
        'k_1': 7.0,
        'ND': 3e5,
        'TN': 5.3,
        'TS': 1.27,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    wc = woehler.Elementary(fd).analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)


def test_woehler_elementary_initialize_with_determined_fractures():
    expected = pd.Series({
        'SD': 362.5,
        'k_1': 7.0,
        'ND': 3e5,
        'TN': 5.3,
        'TS': 1.27,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data, 1e7)
    wc = woehler.Elementary(fd).analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)


def test_woehler_elementary_initialize_with_pandas_dataframe():
    expected = pd.Series({
        'SD': 362.5,
        'k_1': 7.0,
        'ND': 3e5,
        'TN': 5.3,
        'TS': 1.27,
        'failure_probability': 0.5
    }).sort_index()

    wc = woehler.Elementary(data).analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)


def test_woehler_elementary_no_runouts():
    expected = pd.Series({
        'SD': 0.0,
        'k_1': 7.0,
        'TN': 5.3,
        'TS': 1.27,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data_no_runouts, 1e7).fatigue_data
    wc = woehler.Elementary(fd).analyze().sort_index().drop('ND')
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)


def test_woehler_elementary_only_one_load_level():
    data = pd.DataFrame(np.array([[350.0, 1e7], [350.0, 1e6]]), columns=['load', 'cycles'])
    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    with pytest.raises(ValueError, match=r"Need at least two load levels to do a Wöhler analysis."):
        woehler.Elementary(fd).analyze().sort_index()


def test_woehler_probit():
    expected = pd.Series({
        'SD': 335,
        'TS': 1.19,
        'k_1': 6.94,
        'ND': 463000.,
        'TN': 5.26,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    wc = woehler.Probit(fd).analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)


def test_woehler_probit_one_runout_load_level():
    fd = woehler.determine_fractures(data_one_runout_load_level, 1e7).fatigue_data
    expected = woehler.Elementary(fd).analyze()
    with pytest.warns(UserWarning, match=r"Probit needs at least two runout load levels. Falling back to Elementary."):
        wc = woehler.Probit(fd).analyze()
    pd.testing.assert_series_equal(wc, expected)


@pytest.mark.filterwarnings("error:invalid")
def test_woehler_probit_data01():
    expected = pd.Series({
        'SD': 490,
        'TS': 1.1,
        'k_1': 8.0,
        'ND': 530e3,
        'TN': 3.0,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data_01, 1e7).fatigue_data
    pb = woehler.Probit(fd)
    wc = pb.analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)


def test_woehler_probit_no_runouts():
    expected = pd.Series({
        'SD': 0.,
        'TS': 1.27,
        'k_1': 6.94,
        'ND': 4.4e30,
        'TN': 5.26,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data_no_runouts, 1e7).fatigue_data
    pb = woehler.Probit(fd)
    with pytest.warns(UserWarning):
        wc = pb.analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)


def test_woehler_max_likelihood_inf_limit():
    expected = pd.Series({
        'SD': 335,
        'TS': 1.19,
        'k_1': 6.94,
        'ND': 463000.,
        'TN': 5.26,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    wc = woehler.MaxLikeInf(fd).analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)


def test_bic_without_analysis():
    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    we = woehler.MaxLikeFull(fd)
    with pytest.raises(ValueError, match="^.*BIC.*"):
        we.bayesian_information_criterion()


def test_woehler_max_likelihood_inf_limit_no_runouts():
    expected = pd.Series({
        'SD': 0.,
        'TS': 1.19,
        'k_1': 6.94,
        'ND': 4.4e30,
        'TN': 5.26,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data_no_runouts, 1e7).fatigue_data
    wc = woehler.MaxLikeInf(fd).analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)


def test_woehler_max_likelihood_full_without_fixed_params():
    expected = pd.Series({
        'SD': 335,
        'TS': 1.19,
        'k_1': 6.94,
        'ND': 463000.,
        'TN': 4.7,
        'failure_probability': 0.5
    }).sort_index()

    bic = 45.35256860035525

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    we = woehler.MaxLikeFull(fd)
    wc = we.analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)
    np.testing.assert_almost_equal(we.bayesian_information_criterion(), bic, decimal=2)


def test_woehler_max_likelihood_full_without_fixed_params_no_runouts():
    expected = pd.Series({
        'SD': 0,
        'TS': 1.,
        'k_1': 6.94,
        'ND': 4.4e30,
        'TN': 5.7,
        'failure_probability': 0.5
    }).sort_index()

    bic = np.inf

    fd = woehler.determine_fractures(data_no_runouts, 1e7).fatigue_data
    we = woehler.MaxLikeFull(fd)
    with pytest.warns(UserWarning, match=r"^.*no runouts are present.*" ):
        wc = we.analyze().sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)
    np.testing.assert_almost_equal(we.bayesian_information_criterion(), bic, decimal=2)


def test_max_likelihood_full_with_fixed_params():
    expected = pd.Series({
        'SD': 335,
        'TS': 1.19,
        'k_1': 8.0,
        'ND': 520000.,
        'TN': 6.0,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    wc = (
        woehler.MaxLikeFull(fd)
        .analyze(fixed_parameters={'TN': 6.0, 'k_1': 8.0})
        .sort_index()
    )
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)
    assert wc['TN'] == 6.0
    assert wc['k_1'] == 8.0


def test_max_likelihood_full_method_with_all_fixed_params():
    """
    Test of woehler curve parameters evaluation with the maximum likelihood method
    """
    fp = {'k_1': 15.7, 'TN': 1.2, 'SD': 280, 'TS': 1.2, 'ND': 10000000}
    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    with pytest.raises(AttributeError, match=r'You need to leave at least one parameter empty!'):
        (
            woehler.MaxLikeFull(fd)
            .analyze(fixed_parameters=fp)
        )


@pytest.mark.parametrize("data,no", [(d, i) for i, d in enumerate(all_data)])
def test_max_likelihood_parameter_sign(data, no):
    def _modify_initial_parameters_mock(fd):
        return fd

    load_cycle_limit = 1e6
    if hasattr(data, "N_threshold"):
        load_cycle_limit = data.N_threshold
    fatdat = woehler.determine_fractures(data, load_cycle_limit=load_cycle_limit)
    ml = woehler.MaxLikeFull(fatigue_data=fatdat.fatigue_data)
    wl = ml.analyze()

    print("Data set number {}".format(no))
    print("Woehler parameters: {}".format(wl))

    def assert_positive_or_nan_but_not_zero(x):
        if np.isfinite(x):
            assert x >= 0
            assert not np.isclose(x, 0.0)

    assert_positive_or_nan_but_not_zero(wl['SD'])
    assert_positive_or_nan_but_not_zero(wl['TS'])
    assert_positive_or_nan_but_not_zero(wl['k_1'])
    assert_positive_or_nan_but_not_zero(wl['ND'])
    assert_positive_or_nan_but_not_zero(wl['TN'])


@pytest.mark.parametrize("invalid_data", [data_01_one_fracture_level, data_01_two_fractures])
def test_max_likelihood_min_three_fractures_on_two_load_levels(invalid_data):
    fd = woehler.determine_fractures(invalid_data, 1e7).fatigue_data
    ml = woehler.MaxLikeFull(fatigue_data=fd)
    with pytest.raises(ValueError, match=r"^.*[N|n]eed at least.*" ):
        ml.analyze()


def test_max_likelihood_one_mixed_horizon():
    expected = pd.Series({
        'SD': 489.3,
        'TS': 1.147,
        'k_1': 7.99,
        'ND': 541e3,
        'TN': 2.51,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data_01, 1e7).fatigue_data
    ml = woehler.MaxLikeFull(fatigue_data=fd)
    with pytest.warns(UserWarning, match=r"^.*less than two mixed load levels.*"):
        wc = ml.analyze().sort_index()
    bic = ml.bayesian_information_criterion()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)

@mock.patch('pylife.materialdata.woehler.bayesian.pm')
def test_bayesian_slope_trace(pm):
    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    bayes = woehler.Bayesian(fd)
    bayes._nsamples = 1000
    bayes._slope_trace()

    formula, data_dict = pm.glm.GLM.from_formula.call_args[0]
    assert formula == 'y ~ x'
    pd.testing.assert_series_equal(data_dict['x'], np.log10(fd.fractures.load))
    np.testing.assert_array_equal(data_dict['y'], np.log10(fd.fractures.cycles.to_numpy()))
    family = pm.glm.GLM.from_formula.call_args[1]['family']  # Consider switch to kwargs property when py3.7 is dropped
    assert family is pm.glm.families.StudentT()

    pm.sample.assert_called_with(1000, target_accept=0.99, random_seed=None, chains=2, tune=1000)


@mock.patch('pylife.materialdata.woehler.bayesian.pm')
def test_bayesian_TN_trace(pm):
    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    bayes = woehler.Bayesian(fd)
    bayes._common_analysis()
    bayes._nsamples = 1000
    bayes._TN_trace()

    pm.HalfNormal.assert_called_with('stdev', sigma=1.3)

    assert pm.Normal.call_count == 2

    expected_mu = 5.294264482012933
    expected_sigma = 0.2621494419382026

    assert pm.Normal.call_args_list[0][0] == ('mu',)
    np.testing.assert_almost_equal(pm.Normal.call_args_list[0][1]['mu'], expected_mu, decimal=9)
    np.testing.assert_almost_equal(pm.Normal.call_args_list[0][1]['sigma'], expected_sigma, decimal=9)

    assert pm.Normal.call_args_list[1][0] == ('y',)
    observed = pm.Normal.call_args_list[1][1]['observed']  # Consider switch to kwargs property when py3.7 is dropped
    np.testing.assert_almost_equal(observed.mean(), expected_mu, decimal=9)
    np.testing.assert_almost_equal(observed.std(), expected_sigma, decimal=9)

    pm.sample.assert_called_with(1000, target_accept=0.99, random_seed=None, chains=3, tune=1000)


@mock.patch('pylife.materialdata.woehler.bayesian.tt')
@mock.patch('pylife.materialdata.woehler.bayesian.pm')
def test_bayesian_SD_TS_trace_mock(pm, tt):
    def check_likelihood(l, var):
        assert var == tt.as_tensor_variable.return_value
        assert isinstance(l.likelihood, woehler.likelihood.Likelihood)
        np.testing.assert_array_equal(l.likelihood._fd, fd)
        return 'foovar'

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    inf_load_mean = fd.infinite_zone.load.mean()
    inf_load_std = fd.infinite_zone.load.std()

    with mock.patch.object(woehler.Bayesian._LogLike, '__call__', autospec=True) as loglike_call:
        loglike_call.side_effect = check_likelihood

        bayes = woehler.Bayesian(fd)
        bayes._nsamples = 1000
        bayes._SD_TS_trace()

    pm.Normal.assert_called_once_with('SD', mu=inf_load_mean, sigma=inf_load_std * 5)
    pm.Lognormal.assert_called_once()
    np.testing.assert_approx_equal(pm.Lognormal.call_args_list[0][1]['mu'], np.log10(1. / 1.1))
    np.testing.assert_approx_equal(pm.Lognormal.call_args_list[0][1]['sigma'], np.log10(0.5))

    tt.as_tensor_variable.assert_called_once_with([pm.Normal.return_value, pm.Lognormal.return_value])

    pm.Potential.assert_called_once_with('likelihood', 'foovar')

    pm.sample.assert_called_with(1000, cores=1,
                                 chains=3,
                                 random_seed=None,
                                 discard_tuned_samples=True,
                                 tune=1000)


@mock.patch('pylife.materialdata.woehler.bayesian.Bayesian._SD_TS_trace')
@mock.patch('pylife.materialdata.woehler.bayesian.Bayesian._TN_trace')
@mock.patch('pylife.materialdata.woehler.bayesian.Bayesian._slope_trace')
def test_bayesian_mock(_slope_trace, _TN_trace, _SD_TS_trace):
    expected = pd.Series({
        'SD': 100.,
        'TS': 1.12,
        'k_1': 7.0,
        'ND': 1e6,
        'TN': 5.3,
        'failure_probability': 0.5
    }).sort_index()

    expected_slope_trace = {
        'x': np.array([0.0, -8.0, -6.0]),
        'Intercept': np.array([0.0, 19., 21.])
    }

    expected_SD_TS_trace = {
        'SD': np.array([0.0, 150., 50]),
        'TS': np.array([0.0, 1.22, 1.02])
    }

    _slope_trace.__call__().get_values.side_effect = lambda key: expected_slope_trace[key]
    _TN_trace.__call__().get_values.return_value = np.array([0.0, 5.4, 5.2])
    _SD_TS_trace.__call__().get_values.side_effect = lambda key: expected_SD_TS_trace[key]

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    wc = woehler.Bayesian(fd).analyze(nsamples=10).sort_index()

    pd.testing.assert_series_equal(wc, expected)


@pytest.mark.slow_acceptance
def test_bayesian_full():
    expected = pd.Series({
        'SD': 340.,
        'TS': 1.12,
        'k_1': 7.0,
        'ND': 400000.,
        'TN': 5.3,
        'failure_probability': 0.5
    }).sort_index()

    fd = woehler.determine_fractures(data, 1e7).fatigue_data
    wc = woehler.Bayesian(fd).analyze(random_seed=4223, progressbar=False).sort_index()
    pd.testing.assert_series_equal(wc, expected, rtol=1e-1)

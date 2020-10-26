# Copyright (c) 2019-2020 - for information on the respective copyright owner
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
import copy

import numpy as np
import pandas as pd
import numpy.testing as testing

import pylife.strength.infinite as INF

woehler_data = {
            'strength_inf': 500.,
            'strength_scatter': 2.
            }
allowed_failure_probability = 0.5

woehler_data02 = {
            'strength_inf': 500.,
            'strength_scatter': 1.0000001
            }
allowed_failure_probability02 = 0.1


woehler_data03 = {
            'strength_inf': 500.,
            'strength_scatter': 1.25
            }
allowed_failure_probability03 = 1e-6


sigma_a_m = pd.DataFrame({'sigma_m': [0], 'sigma_a': [100]})


def test_factors():
    factor01 = sigma_a_m.infinite_security.factors(pd.Series(woehler_data), allowed_failure_probability)
    factor02 = sigma_a_m.infinite_security.factors(pd.Series(woehler_data02), allowed_failure_probability02)
    factor03 = sigma_a_m.infinite_security.factors(pd.Series(woehler_data03), allowed_failure_probability03)
    testing.assert_array_almost_equal(factor01, 5)
    testing.assert_array_almost_equal(factor02, 5)
    testing.assert_array_almost_equal(factor03, 3.3055576111)

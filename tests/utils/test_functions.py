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

import numpy as np
import pylife.utils.functions as F


def test_rossow():
    assert F.rossow_cumfreqs(1) == 0.5
    for N in np.random.randint(1, high=100, size=100):
        cf = F.rossow_cumfreqs(N)
        np.testing.assert_approx_equal(cf.sum(), N/2.0)
        for i in range(N):
            assert cf[i] + cf[N-i-1] == 1.0


def test_scattering_range_2_std():
    np.testing.assert_allclose(F.scattering_range_to_std(1.25), 0.0378096, rtol=1e-5)


def test_std_2_scattering_range():
    np.testing.assert_allclose(F.std_to_scattering_range(0.0378096), 1.25, rtol=1e-5)

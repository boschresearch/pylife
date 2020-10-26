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

from pylife.strength import helpers as hlp

import numpy as np
import pytest


# example collective
# array of classes
S = np.array([
    0.015625, 0.046875, 0.078125, 0.109375, 0.140625, 0.171875,
    0.203125, 0.234375, 0.265625, 0.296875, 0.328125, 0.359375,
    0.390625, 0.421875, 0.453125, 0.484375, 0.515625, 0.546875,
    0.578125, 0.609375, 0.640625, 0.671875, 0.703125, 0.734375,
    0.765625, 0.796875, 0.828125, 0.859375, 0.890625, 0.921875,
    0.953125, 0.984375, 1.015625, 1.046875, 1.078125, 1.109375,
    1.140625, 1.171875, 1.203125, 1.234375, 1.265625, 1.296875,
    1.328125, 1.359375, 1.390625, 1.421875, 1.453125, 1.484375,
    1.515625, 1.546875, 1.578125, 1.609375, 1.640625, 1.671875,
    1.703125, 1.734375, 1.765625, 1.796875, 1.828125, 1.859375,
    1.890625, 1.921875, 1.953125, 1.984375])
# array of continuous cycle count
N = np.array([
    190733.0, 190733.0, 181542.0, 169283.0, 153244.0, 124473.0,
    113752.0, 86802.0, 81117.0, 57718.0, 55726.0, 41750.0, 39033.0,
    30847.0, 28007.0, 23177.0, 20532.0, 17860.0, 15454.0, 13867.0,
    11601.0, 10765.0, 8831.0, 8449.0, 6682.0, 6504.0, 5044.0, 4967.0,
    3801.0, 3720.0, 2775.0, 2671.0, 2014.0, 1825.0, 1403.0, 1195.0,
    954.0, 717.0, 640.0, 418.0, 393.0, 257.0, 256.0, 152.0, 152.0,
    104.0, 104.0, 63.0, 62.0, 34.0, 32.0, 18.0, 18.0, 12.0, 9.0, 5.0,
    3.0, 3.0, 1.0, 1.0, 0, 0, 0, 0]
)

coll = np.stack((S, N), axis=1)
k = 6


def test_solidity_haibach():
    np.testing.assert_almost_equal(hlp.solidity_haibach(coll, k), 0.00115875)


def test_solidity_fkm():
    np.testing.assert_almost_equal(hlp.solidity_fkm(coll, k), 0.32408968)


class TestStressRelations:

    @pytest.mark.parametrize("amplitude, max_stress, R", [(200, 200, -1), (200, 400, 0)])
    def test_get_max_stress_from_amplitude(self, amplitude, max_stress, R):
        max_stress_calculated = hlp.StressRelations.get_max_stress_from_amplitude(
            amplitude=amplitude,
            R=R
        )
        np.testing.assert_almost_equal(max_stress, max_stress_calculated)

    @pytest.mark.parametrize("amplitude, mean_stress, R", [(200, 0, -1), (200, 200, 0)])
    def test_get_mean_stress_from_amplitude(self, amplitude, mean_stress, R):
        mean_stress_calculated = hlp.StressRelations.get_mean_stress_from_amplitude(
            amplitude=amplitude,
            R=R
        )
        np.testing.assert_almost_equal(mean_stress, mean_stress_calculated)

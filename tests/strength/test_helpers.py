# Copyright (c) 2019-2022 - for information on the respective copyright owner
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


def test_irregularity_factor():
    """
    Consider the turning points series with 8 bins from 0 to 7:
    1 - 5 - 4 - 5 - 1 - 2 - 1 - 5 (mean bin = 3)

    The resulting rainflow matrix contains:
        - 5 -> 4
        - 5 -> 1 (2 mean value crossings: 1x upwards, 1x downwards)
        - 1 -> 2

    With the residuals:
        - 1 -> 5 (1 mean value crossing: 1x upwards)

    The two sided irregularity factor is the quotient of
        - the sum of mean value crossings (which is 3) and
        - the sum of turnings points (which is 8)

    So the irregularity factor is 3 / 8 = 0.375
    """
    rfm = np.array([
        #         to
        # 0  1  2  3  4  5  6  7
        [0, 0, 0, 0, 0, 0, 0, 0],  # 0
        [0, 0, 1, 0, 0, 0, 0, 0],  # 1
        [0, 0, 0, 0, 0, 0, 0, 0],  # 2 f
        [0, 0, 0, 0, 0, 0, 0, 0],  # 3 r
        [0, 0, 0, 0, 0, 0, 0, 0],  # 4 o
        [0, 1, 0, 0, 1, 0, 0, 0],  # 5 m
        [0, 0, 0, 0, 0, 0, 0, 0],  # 6
        [0, 0, 0, 0, 0, 0, 0, 0],  # 7
    ])

    residuals = np.array(
        [1, 5]
    )

    duplicated_residuals = np.array(
        [1, 1, 1, 5, 5]
    )

    # Calculated correctly
    np.testing.assert_almost_equal(hlp.irregularity_factor(rfm, residuals), 3./8, 4)

    # Decision bin inferred correctly
    np.testing.assert_almost_equal(hlp.irregularity_factor(rfm, decision_bin=None, residuals=residuals),
                                   hlp.irregularity_factor(rfm, decision_bin=3, residuals=residuals), 4)

    # Decision bin broadcast
    np.testing.assert_almost_equal(hlp.irregularity_factor(rfm, residuals, decision_bin=3.), 3./8, 4)
    with np.testing.assert_raises(ValueError):
        hlp.irregularity_factor(rfm, residuals, decision_bin="three")

    # Duplicates removed correctly
    np.testing.assert_almost_equal(hlp.irregularity_factor(rfm, residuals=residuals),
                                   hlp.irregularity_factor(rfm, residuals=duplicated_residuals), 4)



def test_irregularity_factor_non_square_rainflow():
    rfm = np.array([
        #         to
        # 0  1  2  3  4  5  6  7
        [0, 0, 0, 0, 0, 0, 0, 0],  # 0
        [0, 0, 1, 0, 0, 0, 0, 0],  # 1
        [0, 0, 0, 0, 0, 0, 0, 0],  # 2 f
        [0, 0, 0, 0, 0, 0, 0, 0],  # 3 r
        [0, 0, 0, 0, 0, 0, 0, 0],  # 4 o
        [0, 1, 0, 0, 1, 0, 0, 0],  # 5 m
        [0, 0, 0, 0, 0, 0, 0, 0],  # 6
    ])

    with pytest.raises(ValueError, match=r"Rainflow matrix must be square shaped in order to calculate the irregularity factor."):
        hlp.irregularity_factor(rfm)

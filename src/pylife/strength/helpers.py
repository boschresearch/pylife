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

"""Small helper functions for fatigue analysis

"""

__author__ = "Cedric Philip Wagner"
__maintainer__ = "Johannes Mueller"

import warnings

import numpy as np

warnings.warn(
    FutureWarning(
        "The module pylife.strength.helpers is deprecated and no longer under test. "
        "We will restore the functionality in another module if needed."
    )
)

class StressRelations:
    """Namespace for simple relations of stress / amplitude / R-ratio

    Refer to:
    Haibach (2006), p. 21
    """

    @staticmethod
    def get_max_stress_from_amplitude(amplitude, R):
        return 2 * amplitude / (1 - R)

    @staticmethod
    def get_mean_stress_from_amplitude(amplitude, R):
        return amplitude * (1 + R) / (1 - R)


def irregularity_factor(rainflow_matrix, residuals=np.empty(0), decision_bin=None):
    """
    Calculate the irregularity factor of a turning point sequence based on a rainflow matrix and its residuals.

    Two sided irregularity factor:

        ..math::
        I = N_{mean crossings} / N_{turning points}

    Parameters
    ----------
    rainflow_matrix: np.ndarray[int, int]
        2D-rainflow matrix (must be square shaped)
    residuals: np.ndarray[int], Optional
        1D array of residuals to consider for accurate calculation. Consecutive duplicates are removed beforehand.
        Residuals must be provided as bin numbers.
        Hint: Transformation from physical to binned values possible via np.digitize.
    decision_bin: int, Optional
        Bin number that equals the mean (two-sided). If not provided the decision_bin is inferred by the matrix entries
        as the mean value based on the turning points and will be broadcasted to int-type.

    Todo
    ----
    Future version may provide the one-sided irregularity factor as a second option. Formula would be:

    One sided irregularity factor:

        .. math::

        I = N_{zero bin upwards crossing} / N_{peaks}

    N_{zero bin upwards crossings} equals positive_mean_bin_crossing if `decision_bin` is set to the bin of physical 0.
    Inferring exact amount of peaks from rainflow-matrix and residuals is left to be done.
    """
    # Ensure input types
    assert isinstance(rainflow_matrix, np.ndarray)
    assert isinstance(residuals, np.ndarray)
    if rainflow_matrix.shape[0] != rainflow_matrix.shape[1]:
        raise ValueError("Rainflow matrix must be square shaped in order to calculate the irregularity factor.")

    # Remove duplicates from residuals
    diffs = np.diff(residuals)
    if np.any(diffs == 0.0):
        # Remove the duplicates
        duplicates = np.concatenate([diffs == 0, [False]])
        residuals = residuals[~duplicates]

    # Infer decision bin as mean if necessary
    if decision_bin is None:
        row_sum = 0
        col_sum = 0
        total_counts = 0
        for i in range(rainflow_matrix.shape[0]):
            row = rainflow_matrix[i, :].sum()
            col = rainflow_matrix[:, i].sum()
            total_counts += row + col
            row_sum += i * row
            col_sum += i * col

        total_counts += residuals.shape[0]
        res_sum = residuals.sum()

        decision_bin = int((row_sum + col_sum + res_sum) / total_counts)
    else:
        decision_bin = int(decision_bin)

    # Calculate two sided irregularity factor
    positive_mean_bin_crossing = rainflow_matrix[0:decision_bin, decision_bin:-1].sum()
    negative_mean_bin_crossing = rainflow_matrix[decision_bin:-1, 0:decision_bin].sum()
    total_mean_crossing = 2 * (positive_mean_bin_crossing + negative_mean_bin_crossing)
    amount_of_turning_points = 2 * rainflow_matrix.sum()

    amount_of_turning_points += residuals.shape[0]
    for i in range(residuals.shape[0] - 1):
        if (residuals[i] - decision_bin) * (residuals[i+1] - decision_bin) < 0:
            total_mean_crossing += 1
    return total_mean_crossing / amount_of_turning_points

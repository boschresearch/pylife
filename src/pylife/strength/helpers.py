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

"""Provide deprecated helper functions for fatigue analysis.

The module contains small stress-relation utilities and an irregularity-factor
calculation for rainflow matrices. It is deprecated and no longer under active
test; prefer the maintained accessors in :mod:`pylife.stress` and
:mod:`pylife.materiallaws` for new code.
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
    """Collect simple stress-amplitude-ratio relations.

    The static methods convert between stress amplitude, maximum stress, mean
    stress, and stress ratio ``R = S_min / S_max`` for proportional cyclic
    loads.

    Notes
    -----
    The relations follow the definitions summarized by Haibach
    [Haibach-Helpers]_.

    References
    ----------
    .. [Haibach-Helpers] E. Haibach, "Betriebsfestigkeit", Springer-Verlag,
       2006, p. 21.
    """

    @staticmethod
    def get_max_stress_from_amplitude(amplitude, R):
        r"""Calculate maximum stress from amplitude and stress ratio.

        Parameters
        ----------
        amplitude : float or array_like
            Stress amplitude in MPa or another consistent stress unit.
        R : float or array_like
            Stress ratio ``R = S_min / S_max``, dimensionless.

        Returns
        -------
        float or numpy.ndarray
            Maximum stress in the same unit as ``amplitude``.

        Notes
        -----
        The relation is

        .. math::

            S_{max} = \frac{2 S_a}{1 - R}.
        """
        return 2 * amplitude / (1 - R)

    @staticmethod
    def get_mean_stress_from_amplitude(amplitude, R):
        r"""Calculate mean stress from amplitude and stress ratio.

        Parameters
        ----------
        amplitude : float or array_like
            Stress amplitude in MPa or another consistent stress unit.
        R : float or array_like
            Stress ratio ``R = S_min / S_max``, dimensionless.

        Returns
        -------
        float or numpy.ndarray
            Mean stress in the same unit as ``amplitude``.

        Notes
        -----
        The relation is

        .. math::

            S_m = S_a \frac{1 + R}{1 - R}.
        """
        return amplitude * (1 + R) / (1 - R)


def irregularity_factor(rainflow_matrix, residuals=np.empty(0), decision_bin=None):
    r"""Calculate the two-sided irregularity factor of a rainflow matrix.

    Parameters
    ----------
    rainflow_matrix : numpy.ndarray
        Square two-dimensional rainflow matrix containing cycle counts by
        class index.
    residuals : numpy.ndarray, optional
        One-dimensional residual turning-point sequence represented by class
        indices. Consecutive duplicate residuals are removed before counting
        mean crossings. Default is an empty array.
    decision_bin : int or None, optional
        Class index representing the mean line. If ``None``, infer it from the
        rainflow matrix and residuals. Default is ``None``.

    Returns
    -------
    float
        Two-sided irregularity factor, dimensionless.

    Raises
    ------
    ValueError
        Raised if ``rainflow_matrix`` is not square.

    Notes
    -----
    The two-sided irregularity factor is calculated as

    .. math::

        I = \frac{N_{mean\ crossings}}{N_{turning\ points}}.

    The one-sided definition based on upward zero-bin crossings is not
    implemented by this deprecated helper.
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

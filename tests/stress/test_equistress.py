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

import copy

import numpy as np
import pandas as pd
import pytest
import pylife.stress.equistress as EQS


@pytest.mark.parametrize("s11, s22, s33, s12, s13, s23, eigenvalues_check", [
    (1, 2, 3, 0, 0, 0, [1, 2, 3]),
    (1, 3, 2, 0, 0, 0, [1, 2, 3]),
    (1, -2, 3, 0, 0, 0, [-2, 1, 3]),
    ([1], [-2], [3], [0], [0], [0], [[-2, 1, 3]]),
    (1.12, 2.35, 3.78, -5.41, -3.57, 0.0, [-4.6068, 3.3229, 8.5339]),  # calculated with matlab
    ([1, 10], [3, 30], [2, 20], [0, 0], [0, 0], [0, 0], [[1, 2, 3], [10, 20, 30]]),
])
def test_eigenval(s11, s22, s33, s12, s13, s23, eigenvalues_check):
    eig = EQS.eigenval(s11, s22, s33, s12, s13, s23)
    assert np.allclose(eig, eigenvalues_check)
    assert eig.shape == np.array(eigenvalues_check).shape


@pytest.mark.parametrize("s11, s22, s33", [
    ([1, 2, 3], 2, 3),
    ([1, 2, 3, 4], [1, 2], [3, 4]),
])
def test_sign_trace_error(s11, s22, s33):
    with pytest.raises(AssertionError):
        EQS._sign_trace(s11, s22, s33)


@pytest.mark.parametrize("s11, s22, s33, sgn_check", [
    (1, 2, 3, 1),
    (1, -5, 2, -1),
    (4, -5, 3, 1),
    (0, 0, 0, 1),
    ([1, 2, 3], [-3, -2, -1], [-2, 0, 2], [-1, 1, 1]),
])
def test_sign_trace(s11, s22, s33, sgn_check):
    sgn = EQS._sign_trace(s11, s22, s33)
    assert np.allclose(sgn, sgn_check)
    assert sgn.shape == np.array(sgn_check).shape


@pytest.mark.parametrize("s11, s22, s33, s12, s13, s23, sgn_check", [
    (1, 2, 3, 0, 0, 0, 1),
    (4, -5, 3, 0, 0, 0, -1),
    (0, 0, 0, 0, 0, 0, 1),
    ([1, 2, 3], [-3, -2, -1], [-2, 0, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0], [-1, 1, 1]),
])
def test_sign_abs_max_principal(s11, s22, s33, s12, s13, s23, sgn_check):
    sgn = EQS._sign_abs_max_principal(s11, s22, s33, s12, s13, s23)
    assert np.allclose(sgn, sgn_check)
    assert sgn.shape == np.array(sgn_check).shape


@pytest.mark.parametrize("s11, s22, s33, s12, s13, s23, tresca_check", [
    (1, -2, 3, 0, 0, 0, 5),
    (-1, -2, -5, 0, 0, 0, 4),
    (0, 0, 0, 0, 0, 0, 0),
    (1.12, 2.35, 3.78, -5.41, -3.57, 0.0, 13.1407),  # calculated with matlab
    ([-1, 10], [-5, 50], [-2, 20], [0, 0], [0, 0], [0, 0], [4, 40]),
])
def test_tresca(s11, s22, s33, s12, s13, s23, tresca_check):
    stress = EQS.tresca(s11, s22, s33, s12, s13, s23)
    assert np.allclose(stress, tresca_check)
    assert stress.shape == np.array(tresca_check).shape


@pytest.mark.parametrize("s11, s22, s33, s12, s13, s23, signed_tresca_trace_check", [
    (5, 5, 5, 0, 0, 0, 0),  # hydrostatic case
    (1.12, 2.35, 3.78, -5.41, -3.57, 0.0, 13.1407),  # calculated with matlab
    (-1.12, -2.35, -3.78, 5.41, 3.57, 0.0, -13.1407),  # calculated with matlab
    (0, 0, 0, 1, 2, 3, 7.315),  # case where trace is 0
    (5, 2, -4, 0, 0, 0, 9),
    (5, -1, -4, 0, 0, 0, 9),
    (5, -3, -4, 0, 0, 0, -9),
])
def test_signed_tresca_trace(s11, s22, s33, s12, s13, s23, signed_tresca_trace_check):
    stress = EQS.signed_tresca_trace(s11, s22, s33, s12, s13, s23)
    assert np.allclose(stress, signed_tresca_trace_check)
    assert stress.shape == np.array(signed_tresca_trace_check).shape


@pytest.mark.parametrize("s11, s22, s33, s12, s13, s23, signed_tresca_abs_max_principal_check", [
    (5, 5, 5, 0, 0, 0, 0),  # hydrostatic case
    (1.12, 2.35, 3.78, -5.41, -3.57, 0.0, 13.1407),  # calculated with matlab
    (-1.12, -2.35, -3.78, 5.41, 3.57, 0.0, -13.1407),  # calculated with matlab
    (0, 0, 0, 1, 2, 3, 7.315),  # case where trace is 0
    (5, 2, -4, 0, 0, 0, 9),
    (5, -1, -4, 0, 0, 0, 9),
    (5, -3, -4, 0, 0, 0, 9),
    (5, 2, -6, 0, 0, 0, -11),
])
def test_signed_tresca_abs_max_principal(s11, s22, s33, s12, s13, s23, signed_tresca_abs_max_principal_check):
    stress = EQS.signed_tresca_abs_max_principal(s11, s22, s33, s12, s13, s23)
    assert np.allclose(stress, signed_tresca_abs_max_principal_check)
    assert stress.shape == np.array(signed_tresca_abs_max_principal_check).shape


@pytest.mark.parametrize("s11, s22, s33, s12, s13, s23, principals_check", [
    (1, -2, 3, 0, 0, 0, [-2, 1, 3]),
    (-1, -2, -5, 0, 0, 0, [-5, -2, -1]),
    (1.12, 2.35, 3.78, -5.41, -3.57, 0.0, [-4.6068, 3.3229, 8.5339]),  # calculated with matlab
    ([-1, 10], [-5, 50], [-2, 20], [0, 0], [0, 0], [0, 0], [[-5, -2, -1],[10, 20, 50]]),
])
def test_principals(s11, s22, s33, s12, s13, s23, principals_check):
    stress = EQS.principals(s11, s22, s33, s12, s13, s23)
    assert np.allclose(stress, principals_check)
    assert stress.shape == np.array(principals_check).shape


@pytest.mark.parametrize("s11, s22, s33, s12, s13, s23, abs_max_principal_check", [
    (1, -2, 3, 0, 0, 0, 3),
    (-1, -2, -5, 0, 0, 0, -5),
    (1.12, 2.35, 3.78, -5.41, -3.57, 0.0, 8.5339),  # calculated with matlab
    ([-1, 10], [-5, 50], [-2, 20], [0, 0], [0, 0], [0, 0], [-5, 50]),
])
def test_abs_max_principal(s11, s22, s33, s12, s13, s23, abs_max_principal_check):
    stress = EQS.abs_max_principal(s11, s22, s33, s12, s13, s23)
    assert np.allclose(stress, abs_max_principal_check)
    assert stress.shape == np.array(abs_max_principal_check).shape


@pytest.mark.parametrize("s11, s22, s33, s12, s13, s23, max_principal_check", [
    (1, -2, 3, 0, 0, 0, 3),
    (-1, -2, -5, 0, 0, 0, -1),
    (1.12, 2.35, 3.78, -5.41, -3.57, 0.0, 8.5339),  # calculated with matlab
    ([-1, 10], [-5, 50], [-2, 20], [0, 0], [0, 0], [0, 0], [-1, 50]),
])
def test_max_principal(s11, s22, s33, s12, s13, s23, max_principal_check):
    stress = EQS.max_principal(s11, s22, s33, s12, s13, s23)
    assert np.allclose(stress, max_principal_check)
    assert stress.shape == np.array(max_principal_check).shape


@pytest.mark.parametrize("s11, s22, s33, s12, s13, s23, min_principal_check", [
    (1, -2, 3, 0, 0, 0, -2),
    (-1, -2, -5, 0, 0, 0, -5),
    (1.12, 2.35, 3.78, -5.41, -3.57, 0.0, -4.6068),  # calculated with matlab
    ([-1, 10], [-5, 50], [-2, 20], [0, 0], [0, 0], [0, 0], [-5, 10]),
])
def test_min_principal(s11, s22, s33, s12, s13, s23, min_principal_check):
    stress = EQS.min_principal(s11, s22, s33, s12, s13, s23)
    assert np.allclose(stress, min_principal_check)
    assert stress.shape == np.array(min_principal_check).shape


@pytest.mark.parametrize("s11, s22, s33, s12, s13, s23", [
    ([1, 2, 3], 2, 3, 0, 0, 0),
    ([1, 2, 3, 4], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10]),
])
def test_mises_error(s11, s22, s33, s12, s13, s23):
    with pytest.raises(AssertionError):
        EQS.mises(s11, s22, s33, s12, s13, s23)


@pytest.mark.parametrize("s11, s22, s33, s12, s13, s23, mises_check", [
    (5, 5, 5, 0, 0, 0, 0),  # hydrostatic case
    (1.12, -2.35, -3.78, -5.41, -3.57, 0.0, 12.0452),  # calculated with matlab
    ([0.152893, 1.12], [1.39879, -2.35], [0.041781, -3.78],
     [0.14746, -5.41], [-0.0795342, -3.57], [-0.13885, 0.0],
     [1.35834, 12.0452]),
])
def test_mises(s11, s22, s33, s12, s13, s23, mises_check):
    stress = EQS.mises(s11, s22, s33, s12, s13, s23)
    assert np.allclose(stress, mises_check)
    assert stress.shape == np.array(mises_check).shape


@pytest.mark.parametrize("s11, s22, s33, s12, s13, s23, signed_mises_trace_check", [
    (5, 5, 5, 0, 0, 0, 0),  # hydrostatic case
    (1.12, -2.35, -3.78, -5.41, -3.57, 0.0, -12.0452),  # calculated with matlab
    ([0.152893, 1.12], [1.39879, -2.35], [0.041781, -3.78],
     [0.14746, -5.41], [-0.0795342, -3.57], [-0.13885, 0.0],
     [1.35834, -12.0452]),
    (0, 0, 0, 1, 2, 3, 6.4807)  # case where trace is 0
])
def test_signed_mises_trace(s11, s22, s33, s12, s13, s23, signed_mises_trace_check):
    stress = EQS.signed_mises_trace(s11, s22, s33, s12, s13, s23)
    assert np.allclose(stress, signed_mises_trace_check)
    assert stress.shape == np.array(signed_mises_trace_check).shape


@pytest.mark.parametrize("s11, s22, s33, s12, s13, s23, signed_mises_abs_max_principal_check", [
    (5, 5, 5, 0, 0, 0, 0),  # hydrostatic case
    (1.12, -2.35, -3.78, -5.41, -3.57, 0.0, -12.0452),  # calculated with matlab
    ([0.152893, 1.12], [1.39879, -2.35], [0.041781, -3.78],
     [0.14746, -5.41], [-0.0795342, -3.57], [-0.13885, 0.0],
     [1.35834, -12.0452]),
    (0, 0, 0, 1, 2, 3, 6.4807),  # case where trace is 0
    (10, 5, -9, 0, 0, 0, 17.0587),
    (10, -5, -9, 0, 0, 0, 17.3494),
    (-10, 0, 0, 0, 0, 0, -10),
])
def test_signed_mises_abs_max_principal(s11, s22, s33, s12, s13, s23, signed_mises_abs_max_principal_check):
    stress = EQS.signed_mises_abs_max_principal(s11, s22, s33, s12, s13, s23)
    assert np.allclose(stress, signed_mises_abs_max_principal_check)
    assert stress.shape == np.array(signed_mises_abs_max_principal_check).shape


def test_tresca_pandas():
    dummy_data = np.array([[1, -2, 3, 0, 0, 0],
                           [-1, -2, -5, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [1.12, 2.35, 3.78, -5.41, -3.57, 0.0]])
    df = pd.DataFrame(columns=['S11', 'S22', 'S33', 'S12', 'S13', 'S23'], data=dummy_data)
    eqs = df.equistress.tresca()
    expected = pd.Series([5., 4., 0., 13.1407], name='tresca')
    pd.testing.assert_series_equal(eqs, expected)


def test_signed_tresca_trace_pandas():
    dummy_data = np.array([[5, 5, 5, 0, 0, 0],
                           [1.12, 2.35, 3.78, -5.41, -3.57, 0.0],
                           [-1.12, -2.35, -3.78, 5.41, 3.57, 0.0],
                           [0, 0, 0, 1, 2, 3],
                           [5, 2, -4, 0, 0, 0],
                           [5, -1, -4, 0, 0, 0],
                           [5, -3, -4, 0, 0, 0]])
    df = pd.DataFrame(columns=['S11', 'S22', 'S33', 'S12', 'S13', 'S23'], data=dummy_data)
    eqs = df.equistress.signed_tresca_trace()
    expected = pd.Series([0.0, 13.1407, -13.1407, 7.315, 9., 9., -9.], name='signed_tresca_trace')
    pd.testing.assert_series_equal(eqs, expected)


def test_signed_tresca_abs_max_principal_pandas():
    dummy_data = np.array([[5, 5, 5, 0, 0, 0],
                           [1.12, 2.35, 3.78, -5.41, -3.57, 0.0],
                           [-1.12, -2.35, -3.78, 5.41, 3.57, 0.0],
                           [0, 0, 0, 1, 2, 3],
                           [5, 2, -4, 0, 0, 0],
                           [5, -1, -4, 0, 0, 0],
                           [5, -3, -4, 0, 0, 0],
                           [5, 2, -6, 0, 0, 0]])
    df = pd.DataFrame(columns=['S11', 'S22', 'S33', 'S12', 'S13', 'S23'], data=dummy_data)
    eqs = df.equistress.signed_tresca_abs_max_principal()
    expected = pd.Series([0.0, 13.1407, -13.1407, 7.315,  9.0,  9.0,  9.0,  -11.0],
                         name='signed_tresca_abs_max_principal')
    pd.testing.assert_series_equal(eqs, expected)


def test_principals_pandas():
    dummy_data = np.array([[1, -2, 3, 0, 0, 0],
                           [-1, -2, -5, 0, 0, 0],
                           [1.12, 2.35, 3.78, -5.41, -3.57, 0.0]])
    df = pd.DataFrame(columns=['S11', 'S22', 'S33', 'S12', 'S13', 'S23'], data=dummy_data)
    eqs = df.equistress.principals()
    expected = pd.DataFrame({
        'min_principal': [-2., -5., -4.6068],
        'med_principal': [ 1., -2.,  3.3229],
        'max_principal': [ 3., -1.,  8.5339],
    })
    pd.testing.assert_frame_equal(eqs, expected)


def test_abs_max_principal_pandas():
    dummy_data = np.array([[1, -2, 3, 0, 0, 0],
                           [-1, -2, -5, 0, 0, 0],
                           [1.12, 2.35, 3.78, -5.41, -3.57, 0.0]])
    df = pd.DataFrame(columns=['S11', 'S22', 'S33', 'S12', 'S13', 'S23'], data=dummy_data)
    eqs = df.equistress.abs_max_principal()
    expected = pd.Series([3.0, -5.0, 8.5339], name='abs_max_principal')
    pd.testing.assert_series_equal(eqs, expected)


def test_max_principal_pandas():
    dummy_data = np.array([[1, -2, 3, 0, 0, 0],
                           [-1, -2, -5, 0, 0, 0],
                           [1.12, 2.35, 3.78, -5.41, -3.57, 0.0]])
    df = pd.DataFrame(columns=['S11', 'S22', 'S33', 'S12', 'S13', 'S23'], data=dummy_data)
    eqs = df.equistress.max_principal()
    expected = pd.Series([3.0, -1.0, 8.5339], name='max_principal')
    pd.testing.assert_series_equal(eqs, expected)


def test_min_principal_pandas():
    dummy_data = np.array([[1, -2, 3, 0, 0, 0],
                           [-1, -2, -5, 0, 0, 0],
                           [1.12, 2.35, 3.78, -5.41, -3.57, 0.0]])
    df = pd.DataFrame(columns=['S11', 'S22', 'S33', 'S12', 'S13', 'S23'], data=dummy_data)
    df = pd.DataFrame(columns=['S11', 'S22', 'S33', 'S12', 'S13', 'S23'], data=dummy_data)
    eqs = df.equistress.min_principal()
    expected = pd.Series([-2.0, -5.0, -4.6068], name='min_principal')
    pd.testing.assert_series_equal(eqs, expected)


def test_mises_pandas():
    dummy_data = np.array([[5.0, 5.0, 5.0, 0.0, 0.0, 0.0],
                           [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
                           [1.12, -2.35, -3.78, -5.41, -3.57, 0.0]])
    df = pd.DataFrame(columns=['S11', 'S22', 'S33', 'S12', 'S13', 'S23'], data=dummy_data)
    eqs = df.equistress.mises()
    expected = pd.Series([0.0, 1.73205, 12.0452], name='mises')
    pd.testing.assert_series_equal(eqs, expected)


def test_signed_mises_trace_pandas():
    dummy_data = np.array([[5, 5, 5, 0, 0, 0],
                           [1.12, -2.35, -3.78, -5.41, -3.57, 0.0],
                           [0.152893, 1.39879, 0.041781, 0.14746, -0.0795342, -0.13885],
                           [0, 0, 0, 1, 2, 3]])
    df = pd.DataFrame(columns=['S11', 'S22', 'S33', 'S12', 'S13', 'S23'], data=dummy_data)
    eqs = df.equistress.signed_mises_trace()
    expected = pd.Series([0.0, -12.0452, 1.35834, 6.4807], name='signed_mises_trace')
    pd.testing.assert_series_equal(eqs, expected)


def test_signed_mises_abs_max_principal_pandas():
    dummy_data = np.array([[5, 5, 5, 0, 0, 0],
                           [1.12, -2.35, -3.78, -5.41, -3.57, 0.0],
                           [0.152893, 1.39879,  0.041781, 0.14746,  -0.0795342,  -0.13885],
                           [0, 0, 0, 1, 2, 3],
                           [10, 5, -9, 0, 0, 0],
                           [10, -5, -9, 0, 0, 0],
                           [-10, 0, 0, 0, 0, 0]])
    df = pd.DataFrame(columns=['S11', 'S22', 'S33', 'S12', 'S13', 'S23'], data=dummy_data)
    eqs = df.equistress.signed_mises_abs_max_principal()
    expected = pd.Series([0.0, -12.0452, 1.35834, 6.4807, 17.0587, 17.3494, -10.0], name='signed_mises_abs_max_principal')
    pd.testing.assert_series_equal(eqs, expected)

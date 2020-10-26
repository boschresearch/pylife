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

import pylife.mesh.gradient
import pandas as pd
import numpy as np


def test_grad_constant():
    """
    Test of gradient computation using least square method
    """
    # Setup
    fkt = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt}).set_index(['node_id', 'element_id'])

    gradient_dx = np.zeros(9)
    gradient_dy = np.zeros(9)

    # Exercise
    grad = df.gradient.gradient_of('fct')

    # Verify
    np.testing.assert_array_almost_equal(grad['dx'], gradient_dx, decimal=1)
    np.testing.assert_array_almost_equal(grad['dy'], gradient_dy, decimal=1)


def test_grad_dx():
    """
    Test of gradient computation using least square method
    """
    # Setup
    fkt = [1, 4, 4, 7, 1, 1, 4, 4, 4, 4, 7, 7, 1, 4, 4, 7]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt})
    df = df.set_index(['node_id', 'element_id'])
    gradient_dx = np.ones(9)*3

    # Exercise
    grad = df.gradient.gradient_of('fct')

    # Verify
    np.testing.assert_array_almost_equal(grad['dx'], gradient_dx, decimal=1)


def test_grad_dx_shuffle():
    """
    Test of gradient computation using least square method
    """
    # Setup
    fkt = [1, 4, 4, 7, 1, 1, 4, 4, 4, 4, 7, 7, 1, 4, 4, 7]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt})
    df = df.set_index(['node_id', 'element_id'])
    df = df.sample(frac=1)
    gradient_dx = np.ones(9)*3

    # Exercise
    grad = df.gradient.gradient_of('fct')

    # Verify
    np.testing.assert_array_almost_equal(grad['dx'], gradient_dx, decimal=1)


def test_grad_dy():
    """
    Test of gradient computation using least square method
    """
    # Setup
    fkt = [1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt})
    df = df.set_index(['node_id', 'element_id'])
    df = df.sample(frac=1)
    gradient_dy = np.ones(9)*3

    # Exercise
    grad = df.gradient.gradient_of('fct')

    # Verify
    np.testing.assert_array_almost_equal(grad['dy'], gradient_dy, decimal=1)


def test_grad_dy_shuffle():
    """
    Test of gradient computation using least square method
    """
    # Setup
    fkt = [1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt})
    df = df.set_index(['node_id', 'element_id'])
    gradient_dy = np.ones(9)*3

    # Exercise
    grad = df.gradient.gradient_of('fct')

    # Verify
    np.testing.assert_array_almost_equal(grad['dy'], gradient_dy, decimal=1)


def test_grad_dxy_simple():
    """
    Test of gradient computation using least square method
    """
    # Setup
    fkt = [2, 6, 6, 10, 5, 5, 9, 9, 9, 9, 13, 13, 8, 12, 12, 16]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt})
    df = df.set_index(['node_id', 'element_id'])
    gradient_dx = np.ones(9)*4
    gradient_dy = np.ones(9)*3

    # Exercise
    grad = df.gradient.gradient_of('fct')

    # Verify
    np.testing.assert_array_almost_equal(grad['dx'], gradient_dx, decimal=1)
    np.testing.assert_array_almost_equal(grad['dy'], gradient_dy, decimal=1)


def test_grad_dxy_complex():
    """
    Test of gradient computation using least square method
    """
    # Setup
    fkt = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt})
    df = df.set_index(['node_id', 'element_id'])
    gradient_dx = np.array([1, 0, -1, 1, 0, -1,  1, 0, -1])/3
    gradient_dy = np.array([1, 1, 1, 0,  0, 0, -1, -1, -1])/3

    # Exercise
    grad = df.gradient.gradient_of('fct')

    # Verify
    np.testing.assert_array_almost_equal(grad['dx'], gradient_dx, decimal=2)
    np.testing.assert_array_almost_equal(grad['dy'], gradient_dy, decimal=2)


def test_grad_dxy_simple_shuffle():
    """
    Test of gradient computation using least square method
    """
    # Setup
    fkt = [2, 6, 6, 10, 5, 5, 9, 9, 9, 9, 13, 13, 8, 12, 12, 16]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt})
    df = df.set_index(['node_id', 'element_id'])
    df = df.sample(frac=1)
    gradient_dx = np.ones(9)*4
    gradient_dy = np.ones(9)*3

    # Exercise
    grad = df.gradient.gradient_of('fct')

    # Verify
    np.testing.assert_array_almost_equal(grad['dx'], gradient_dx, decimal=1)
    np.testing.assert_array_almost_equal(grad['dy'], gradient_dy, decimal=1)


def test_grad_dxy_complex_shuffle():
    """
    Test of gradient computation using least square method
    """
    # Setup
    fkt = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt})
    df = df.set_index(['node_id', 'element_id'])
    df = df.sample(frac=1)
    gradient_dx = np.array([1, 0, -1, 1, 0, -1,  1, 0, -1])/3
    gradient_dy = np.array([1, 1, 1, 0,  0, 0, -1, -1, -1])/3

    # Exercise
    grad = df.gradient.gradient_of('fct')

    # Verify
    np.testing.assert_array_almost_equal(grad['dx'], gradient_dx, decimal=2)
    np.testing.assert_array_almost_equal(grad['dy'], gradient_dy, decimal=2)

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

import pylife.mesh.gradient
import pandas as pd
import numpy as np


def test_grad_constant():
    fkt = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt}).set_index(['node_id', 'element_id'])

    expected = pd.DataFrame({
        'dfct_dx': np.zeros(9),
        'dfct_dy': np.zeros(9),
        'dfct_dz': np.zeros(9)
    }, index=pd.RangeIndex(1, 10, name='node_id'))

    grad = df.gradient.gradient_of('fct')

    pd.testing.assert_frame_equal(grad, expected)


def test_grad_dx():
    fkt = [1, 4, 4, 7, 1, 1, 4, 4, 4, 4, 7, 7, 1, 4, 4, 7]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt})
    df = df.set_index(['node_id', 'element_id'])

    expected = pd.DataFrame({
        'dfct_dx': np.full(9, 3.0),
        'dfct_dy': np.zeros(9),
        'dfct_dz': np.zeros(9)
    }, index=pd.RangeIndex(1, 10, name='node_id'))

    grad = df.gradient.gradient_of('fct')

    pd.testing.assert_frame_equal(grad, expected)


def test_grad_dx_shuffle():
    fkt = [1, 4, 4, 7, 1, 1, 4, 4, 4, 4, 7, 7, 1, 4, 4, 7]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt})
    df = df.set_index(['node_id', 'element_id'])
    df = df.sample(frac=1)

    expected = pd.DataFrame({
        'dfct_dx': np.full(9, 3.0),
        'dfct_dy': np.zeros(9),
        'dfct_dz': np.zeros(9)
    }, index=pd.RangeIndex(1, 10, name='node_id'))

    grad = df.gradient.gradient_of('fct')

    pd.testing.assert_frame_equal(grad, expected)


def test_grad_dy():
    fkt = [1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt})
    df = df.set_index(['node_id', 'element_id'])
    df = df.sample(frac=1)

    expected = pd.DataFrame({
        'dfct_dx': np.zeros(9),
        'dfct_dy': np.full(9, 3.0),
        'dfct_dz': np.zeros(9)
    }, index=pd.RangeIndex(1, 10, name='node_id'))

    grad = df.gradient.gradient_of('fct')

    pd.testing.assert_frame_equal(grad, expected)


def test_grad_dy_shuffle():
    fkt = [1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt})
    df = df.set_index(['node_id', 'element_id'])

    expected = pd.DataFrame({
        'dfct_dx': np.zeros(9),
        'dfct_dy': np.full(9, 3.0),
        'dfct_dz': np.zeros(9)
    }, index=pd.RangeIndex(1, 10, name='node_id'))

    grad = df.gradient.gradient_of('fct')

    pd.testing.assert_frame_equal(grad, expected)


def test_grad_dxy_simple():
    fkt = [2, 6, 6, 10, 5, 5, 9, 9, 9, 9, 13, 13, 8, 12, 12, 16]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt})
    df = df.set_index(['node_id', 'element_id'])

    expected = pd.DataFrame({
        'dfct_dx': np.full(9, 4.0),
        'dfct_dy': np.full(9, 3.0),
        'dfct_dz': np.zeros(9)
    }, index=pd.RangeIndex(1, 10, name='node_id'))

    grad = df.gradient.gradient_of('fct')

    pd.testing.assert_frame_equal(grad, expected)


def test_grad_dxy_complex():
    fkt = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt})
    df = df.set_index(['node_id', 'element_id'])

    expected = pd.DataFrame({
        'dfct_dx': np.array([1, 0, -1, 1, 0, -1,  1, 0, -1])/3.,
        'dfct_dy': np.array([1, 1, 1, 0,  0, 0, -1, -1, -1])/3.,
        'dfct_dz': np.zeros(9)
    }, index=pd.RangeIndex(1, 10, name='node_id'))

    grad = df.gradient.gradient_of('fct')

    pd.testing.assert_frame_equal(grad, expected)


def test_grad_dxy_simple_shuffle():
    fkt = [2, 6, 6, 10, 5, 5, 9, 9, 9, 9, 13, 13, 8, 12, 12, 16]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt})
    df = df.set_index(['node_id', 'element_id'])
    df = df.sample(frac=1)

    expected = pd.DataFrame({
        'dfct_dx': np.full(9, 4.0),
        'dfct_dy': np.full(9, 3.0),
        'dfct_dz': np.zeros(9)
    }, index=pd.RangeIndex(1, 10, name='node_id'))

    grad = df.gradient.gradient_of('fct')

    pd.testing.assert_frame_equal(grad, expected)



def test_grad_dxy_complex_shuffle():
    fkt = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt})
    df = df.set_index(['node_id', 'element_id'])
    df = df.sample(frac=1)

    expected = pd.DataFrame({
        'dfct_dx': np.array([1, 0, -1, 1, 0, -1,  1, 0, -1])/3.,
        'dfct_dy': np.array([1, 1, 1, 0,  0, 0, -1, -1, -1])/3.,
        'dfct_dz': np.zeros(9)
    }, index=pd.RangeIndex(1, 10, name='node_id'))

    grad = df.gradient.gradient_of('fct')

    pd.testing.assert_frame_equal(grad, expected)

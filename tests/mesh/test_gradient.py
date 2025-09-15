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

__author__ = "Benjamin Maier"
__maintainer__ = __author__

import pylife.mesh.gradient
import pandas as pd
import numpy as np


def test_grad_constant():

    # 9 nodes and 4 elements used in the following tests
    # 1---2---3
    # |[1]|[2]|
    # 4---5---6
    # |[3]|[4]|
    # 7---8---9

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

    pd.testing.assert_frame_equal(grad, expected, rtol=1e-12)


def test_grad_dx_continous():
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

    pd.testing.assert_frame_equal(grad, expected, rtol=1e-12)


def test_grad_dx_gap_in_elset():
    fkt = [1, 4, 4, 7, 1, 1, 4, 4, 4, 4, 7, 7, 1, 4, 4, 7]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 14, 2, 14, 3, 3, 14, 14],
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

    pd.testing.assert_frame_equal(grad, expected, rtol=1e-12)


def test_grad_dx_gap_in_node_set():
    fkt = [1, 4, 4, 7, 1, 1, 4, 4, 4, 4, 7, 7, 1, 4, 4, 7]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 15, 15, 15, 15, 16, 16, 17, 18, 18, 19],
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
    }, index=pd.Index([1, 2, 3, 4, 15, 16, 17, 18, 19], name='node_id'))

    grad = df.gradient.gradient_of('fct')

    pd.testing.assert_frame_equal(grad, expected, rtol=1e-12)


def test_grad_dx_flipped_index_levels():
    fkt = [1, 4, 4, 7, 1, 1, 4, 4, 4, 4, 7, 7, 1, 4, 4, 7]
    df = pd.DataFrame({'node_id': [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'fct': fkt})
    df = df.set_index(['element_id', 'node_id'])
    expected = pd.DataFrame({
        'dfct_dx': np.full(9, 3.0),
        'dfct_dy': np.zeros(9),
        'dfct_dz': np.zeros(9)
    }, index=pd.RangeIndex(1, 10, name='node_id'))

    grad = df.gradient.gradient_of('fct')

    pd.testing.assert_frame_equal(grad, expected, rtol=1e-12)


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

    pd.testing.assert_frame_equal(grad, expected, rtol=1e-12)


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

    pd.testing.assert_frame_equal(grad, expected, rtol=1e-12)


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

    pd.testing.assert_frame_equal(grad, expected, rtol=1e-12)


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

    pd.testing.assert_frame_equal(grad, expected, rtol=1e-12)


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

    pd.testing.assert_frame_equal(grad, expected, rtol=1e-12)


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

    pd.testing.assert_frame_equal(grad, expected, rtol=1e-12)


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

    pd.testing.assert_frame_equal(grad, expected, rtol=1e-12)


# ---- gradient_3D
def test_gradient_3D_constant():
    fkt = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    df = pd.DataFrame({'node_id':    [1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8, 9],
                       'element_id': [1, 1, 2, 2, 1, 3, 1, 2, 3, 4, 2, 4, 3, 3, 4, 4],
                       'x': [0, 1, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2],
                       'y': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                       'z': [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0],
                       'fct': fkt}).set_index(['node_id', 'element_id'])

    expected = pd.DataFrame({
        'dfct_dx': np.zeros(9),
        'dfct_dy': np.zeros(9),
        'dfct_dz': np.zeros(9)
    }, index=pd.RangeIndex(1, 10, name='node_id'))

    grad = df.gradient_3D.gradient_of('fct').sort_index()

    pd.testing.assert_frame_equal(grad.reset_index(), expected.reset_index(), check_dtype=False, rtol=1e-12)

def test_gradient_3D_is_not_3D():
    fkt = [5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]
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

    grad = df.gradient_3D.gradient_of('fct').sort_index()

    pd.testing.assert_frame_equal(grad.reset_index(), expected.reset_index(), check_dtype=False, rtol=1e-12)


def test_gradient_3D_dx():
    # 18 nodes and 4 3D elements used in the following tests
    # bottom nodes
    # 1---2---3
    # |[1]|[2]|
    # 4---5---6
    # |[3]|[4]|
    # 7---8---9
    #
    # top nodes
    # 11--12--13
    # |[1]| [2]|
    # 14--15--16
    # |[3]| [4]|
    # 17--18--9

    fkt = [1, 4, 4, 1, 1, 4, 4, 1, 4, 7, 7, 4, 4, 7, 7, 4, 1, 4, 4, 1, 1, 4, 4, 1, 4, 7, 7, 4, 4, 7, 7, 4]
    df = pd.DataFrame({'element_id': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
                       'node_id': [1, 2, 5, 4, 11, 12, 15, 14, 2, 3, 6, 5, 12, 13, 16, 15, 4, 5, 8, 7, 14, 15, 18, 17, 5, 6, 9, 8, 15, 16, 19, 18],
                       'x': [0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1],
                       'y': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
                       'z': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,  1, 1, 1, 1, 0, 0, 0, 0,  1, 1, 1, 1],
                       'fct': fkt})

    df = df.set_index(['node_id', 'element_id'])

    expected = pd.DataFrame({
        'dfct_dx': np.full(18, 3.0),
        'dfct_dy': np.zeros(18),
        'dfct_dz': np.zeros(18)
    }, index=pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19], name='node_id'))

    grad = df.gradient_3D.gradient_of('fct').sort_index()

    pd.testing.assert_frame_equal(grad.reset_index(), expected.reset_index(), check_dtype=False, rtol=1e-12)


def test_gradient_3D_dx_flipped_index_levels():
    fkt = [1, 4, 4, 1, 1, 4, 4, 1, 4, 7, 7, 4, 4, 7, 7, 4, 1, 4, 4, 1, 1, 4, 4, 1, 4, 7, 7, 4, 4, 7, 7, 4]
    df = pd.DataFrame({'element_id': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
                       'node_id': [1, 2, 5, 4, 11, 12, 15, 14, 2, 3, 6, 5, 12, 13, 16, 15, 4, 5, 8, 7, 14, 15, 18, 17, 5, 6, 9, 8, 15, 16, 19, 18],
                       'x': [0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1],
                       'y': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
                       'z': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,  1, 1, 1, 1, 0, 0, 0, 0,  1, 1, 1, 1],
                       'fct': fkt})

    df = df.set_index(['element_id', 'node_id'])

    expected = pd.DataFrame({
        'dfct_dx': np.full(18, 3.0),
        'dfct_dy': np.zeros(18),
        'dfct_dz': np.zeros(18)
    }, index=pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19], name='node_id'))

    grad = df.gradient_3D.gradient_of('fct').sort_index()

    pd.testing.assert_frame_equal(grad.reset_index(), expected.reset_index(), check_dtype=False, rtol=1e-12)


def test_gradient_3D_dy():

    fkt = [1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 4, 4, 7, 7, 4, 4, 7, 7, 4, 4, 7, 7, 4, 4, 7, 7]
    df = pd.DataFrame({'element_id': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
                       'node_id': [1, 2, 5, 4, 11, 12, 15, 14, 2, 3, 6, 5, 12, 13, 16, 15, 4, 5, 8, 7, 14, 15, 18, 17, 5, 6, 9, 8, 15, 16, 19, 18],
                       'x': [0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1],
                       'y': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
                       'z': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,  1, 1, 1, 1, 0, 0, 0, 0,  1, 1, 1, 1],
                       'fct': fkt})

    df = df.set_index(['node_id', 'element_id'])

    expected = pd.DataFrame({
        'dfct_dx': np.zeros(18),
        'dfct_dy': np.full(18, 3.0),
        'dfct_dz': np.zeros(18)
    }, index=pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19], name='node_id'))

    grad = df.gradient_3D.gradient_of('fct').sort_index()

    pd.testing.assert_frame_equal(grad.reset_index(), expected.reset_index(), check_dtype=False, rtol=1e-12)

def test_gradient_3D_dy_flipped_index_levels():

    fkt = [1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 4, 4, 7, 7, 4, 4, 7, 7, 4, 4, 7, 7, 4, 4, 7, 7]
    df = pd.DataFrame({'element_id': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
                       'node_id': [1, 2, 5, 4, 11, 12, 15, 14, 2, 3, 6, 5, 12, 13, 16, 15, 4, 5, 8, 7, 14, 15, 18, 17, 5, 6, 9, 8, 15, 16, 19, 18],
                       'x': [0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1],
                       'y': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
                       'z': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,  1, 1, 1, 1, 0, 0, 0, 0,  1, 1, 1, 1],
                       'fct': fkt})

    df = df.set_index(['element_id', 'node_id'])

    expected = pd.DataFrame({
        'dfct_dx': np.zeros(18),
        'dfct_dy': np.full(18, 3.0),
        'dfct_dz': np.zeros(18)
    }, index=pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19], name='node_id'))

    grad = df.gradient_3D.gradient_of('fct').sort_index()

    pd.testing.assert_frame_equal(grad.reset_index(), expected.reset_index(), check_dtype=False, rtol=1e-12)


def test_gradient_3D_dxy():
    fkt_x = np.array([1, 4, 4, 1, 1, 4, 4, 1, 4, 7, 7, 4, 4, 7, 7, 4, 1, 4, 4, 1, 1, 4, 4, 1, 4, 7, 7, 4, 4, 7, 7, 4]) # x: 1 4 7, y: 1 5 9
    fkt_y = np.array([1, 1, 5, 5, 1, 1, 5, 5, 1, 1, 5, 5, 1, 1, 5, 5, 5, 5, 9, 9, 5, 5, 9, 9, 5, 5, 9, 9, 5, 5, 9, 9])
    fkt = fkt_x + fkt_y
    df = pd.DataFrame({'element_id': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
                       'node_id': [1, 2, 5, 4, 11, 12, 15, 14, 2, 3, 6, 5, 12, 13, 16, 15, 4, 5, 8, 7, 14, 15, 18, 17, 5, 6, 9, 8, 15, 16, 19, 18],
                       'x': [0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1],
                       'y': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
                       'z': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,  1, 1, 1, 1, 0, 0, 0, 0,  1, 1, 1, 1],
                       'fct': fkt})
    df = df.set_index(['node_id', 'element_id'])

    expected = pd.DataFrame({
        'dfct_dx': np.full(18, 3.0),
        'dfct_dy': np.full(18, 4.0),
        'dfct_dz': np.zeros(18)
    }, index=pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19], name='node_id'))

    grad = df.gradient_3D.gradient_of('fct').sort_index()

    pd.testing.assert_frame_equal(grad.reset_index(), expected.reset_index(), check_dtype=False, rtol=1e-12)

def test_gradient_3D_dxy_flipped_index_levels():
    fkt_x = np.array([1, 4, 4, 1, 1, 4, 4, 1, 4, 7, 7, 4, 4, 7, 7, 4, 1, 4, 4, 1, 1, 4, 4, 1, 4, 7, 7, 4, 4, 7, 7, 4]) # x: 1 4 7, y: 1 5 9
    fkt_y = np.array([1, 1, 5, 5, 1, 1, 5, 5, 1, 1, 5, 5, 1, 1, 5, 5, 5, 5, 9, 9, 5, 5, 9, 9, 5, 5, 9, 9, 5, 5, 9, 9])
    fkt = fkt_x + fkt_y
    df = pd.DataFrame({'element_id': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
                       'node_id': [1, 2, 5, 4, 11, 12, 15, 14, 2, 3, 6, 5, 12, 13, 16, 15, 4, 5, 8, 7, 14, 15, 18, 17, 5, 6, 9, 8, 15, 16, 19, 18],
                       'x': [0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1],
                       'y': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
                       'z': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,  1, 1, 1, 1, 0, 0, 0, 0,  1, 1, 1, 1],
                       'fct': fkt})
    df = df.set_index(['element_id', 'node_id'])

    expected = pd.DataFrame({
        'dfct_dx': np.full(18, 3.0),
        'dfct_dy': np.full(18, 4.0),
        'dfct_dz': np.zeros(18)
    }, index=pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19], name='node_id'))

    grad = df.gradient_3D.gradient_of('fct').sort_index()

    pd.testing.assert_frame_equal(grad.reset_index(), expected.reset_index(), check_dtype=False, rtol=1e-12)


def test_gradient_3D_dxyz_8nodes():

    df = pd.DataFrame({'element_id': [1, 1, 1, 1, 1, 1, 1, 1],
                       'node_id': [1, 2, 3, 4, 5, 6, 7, 8],
                       'x': np.zeros(8),
                       'y': np.zeros(8),
                       'z': np.zeros(8),
                       'fct': [0-0.2, 1.2-0.4 ,1.1+1.3, 0.2+1.2, 0.4-1.6, 1.5+0.3-1.1, 1.8+1.2-1.2, 0.7+1-1.8]})
    df = df.set_index(['node_id', 'element_id'])

    # set node positions
    offset = np.array([2,3,4])
    df.iloc[0,:3] = np.array([0,0,0.2]) + offset
    df.iloc[1,:3] = np.array([1.2,0,0.4]) + offset
    df.iloc[2,:3] = np.array([1.1,1.3,0]) + offset
    df.iloc[3,:3] = np.array([0.2,1.2,0]) + offset
    df.iloc[4,:3] = np.array([0,0.4,1.6]) + offset
    df.iloc[5,:3] = np.array([1.5,0.3,1.1]) + offset
    df.iloc[6,:3] = np.array([1.8,1.2,1.2]) + offset
    df.iloc[7,:3] = np.array([0.7,1,1.8]) + offset

    expected = pd.DataFrame({
        'dfct_dx': np.full(8, 1.0),
        'dfct_dy': np.full(8, 1.0),
        'dfct_dz': np.full(8, -1.0)
    }, index=pd.Index([1, 2, 3, 4, 5, 6, 7, 8], name='node_id'))

    grad = df.gradient_3D.gradient_of('fct').sort_index()

    pd.testing.assert_frame_equal(grad.reset_index(), expected.reset_index(), rtol=1e-12)

def test_gradient_3D_dxyz_8_nodes_flipped_index_levels():

    df = pd.DataFrame({'element_id': [1, 1, 1, 1, 1, 1, 1, 1],
                       'node_id': [1, 2, 3, 4, 5, 6, 7, 8],
                       'x': np.zeros(8),
                       'y': np.zeros(8),
                       'z': np.zeros(8),
                       'fct': [0-0.2, 1.2-0.4 ,1.1+1.3, 0.2+1.2, 0.4-1.6, 1.5+0.3-1.1, 1.8+1.2-1.2, 0.7+1-1.8]})

    df = df.set_index(['element_id', 'node_id'])

    # set node positions
    offset = np.array([2,3,4])
    df.iloc[0,:3] = np.array([0,0,0.2]) + offset
    df.iloc[1,:3] = np.array([1.2,0,0.4]) + offset
    df.iloc[2,:3] = np.array([1.1,1.3,0]) + offset
    df.iloc[3,:3] = np.array([0.2,1.2,0]) + offset
    df.iloc[4,:3] = np.array([0,0.4,1.6]) + offset
    df.iloc[5,:3] = np.array([1.5,0.3,1.1]) + offset
    df.iloc[6,:3] = np.array([1.8,1.2,1.2]) + offset
    df.iloc[7,:3] = np.array([0.7,1,1.8]) + offset

    expected = pd.DataFrame({
        'dfct_dx': np.full(8, 1.0),
        'dfct_dy': np.full(8, 1.0),
        'dfct_dz': np.full(8, -1.0)
    }, index=pd.Index([1, 2, 3, 4, 5, 6, 7, 8], name='node_id'))

    grad = df.gradient_3D.gradient_of('fct').sort_index()

    pd.testing.assert_frame_equal(grad.reset_index(), expected.reset_index(), rtol=1e-12)

def test_gradient_3D_dxyz_16nodes():

    df = pd.DataFrame({'element_id': [1, 1, 1, 1, 1, 1, 1, 1]*2,
                       'node_id': range(16),
                       'x': np.zeros(16),
                       'y': np.zeros(16),
                       'z': np.zeros(16),
                       'fct': [0-0.2, 1.2-0.4 ,1.1+1.3, 0.2+1.2, 0.4-1.6, 1.5+0.3-1.1, 1.8+1.2-1.2, 0.7+1-1.8]*2})
    df = df.set_index(['node_id', 'element_id'])

    # set node positions
    offset = np.array([2,3,4])
    df.iloc[0,:3] = np.array([0,0,0.2]) + offset
    df.iloc[1,:3] = np.array([1.2,0,0.4]) + offset
    df.iloc[2,:3] = np.array([1.1,1.3,0]) + offset
    df.iloc[3,:3] = np.array([0.2,1.2,0]) + offset
    df.iloc[4,:3] = np.array([0,0.4,1.6]) + offset
    df.iloc[5,:3] = np.array([1.5,0.3,1.1]) + offset
    df.iloc[6,:3] = np.array([1.8,1.2,1.2]) + offset
    df.iloc[7,:3] = np.array([0.7,1,1.8]) + offset

    expected = pd.DataFrame({
        'dfct_dx': [1.0]*8 + [0.0]*8,
        'dfct_dy': [1.0]*8 + [0.0]*8,
        'dfct_dz': [-1.0]*8 + [0.0]*8
    }, index=pd.Index(range(16), name='node_id'))

    grad = df.gradient_3D.gradient_of('fct').sort_index()

    pd.testing.assert_frame_equal(grad.reset_index(), expected.reset_index(), rtol=1e-12)


def test_gradient_3D_dxyz_20nodes():

    df = pd.DataFrame({'element_id': [1, 1, 1, 1, 1, 1, 1, 1]*2 + [1,1,1,1],
                       'node_id': range(20),
                       'x': np.zeros(20),
                       'y': np.zeros(20),
                       'z': np.zeros(20),
                       'fct': [0-0.2, 1.2-0.4 ,1.1+1.3, 0.2+1.2, 0.4-1.6, 1.5+0.3-1.1, 1.8+1.2-1.2, 0.7+1-1.8]*2+[0,0,0,0]})
    df = df.set_index(['node_id', 'element_id'])

    # set node positions
    offset = np.array([2,3,4])
    df.iloc[0,:3] = np.array([0,0,0.2]) + offset
    df.iloc[1,:3] = np.array([1.2,0,0.4]) + offset
    df.iloc[2,:3] = np.array([1.1,1.3,0]) + offset
    df.iloc[3,:3] = np.array([0.2,1.2,0]) + offset
    df.iloc[4,:3] = np.array([0,0.4,1.6]) + offset
    df.iloc[5,:3] = np.array([1.5,0.3,1.1]) + offset
    df.iloc[6,:3] = np.array([1.8,1.2,1.2]) + offset
    df.iloc[7,:3] = np.array([0.7,1,1.8]) + offset

    expected = pd.DataFrame({
        'dfct_dx': [1.0]*8 + [0.0]*12,
        'dfct_dy': [1.0]*8 + [0.0]*12,
        'dfct_dz': [-1.0]*8 + [0.0]*12
    }, index=pd.Index(range(20), name='node_id'))

    grad = df.gradient_3D.gradient_of('fct').sort_index()

    pd.testing.assert_frame_equal(grad.reset_index(), expected.reset_index(), rtol=1e-12)


def test_gradient_3D_tetrahedron():

    df = pd.DataFrame({'element_id': [1, 1, 1, 1],
                       'node_id': range(4),
                       'x': [0.1, 0.4, 0.3, 0.2],
                       'y': [0.5, 0.8, 0.6, 0.6],
                       'z': [0.8, 0.9, 1.0, 2.0],
                       })
    df = df.set_index(['node_id', 'element_id'])
    df["f"] = 1.1*df.x + 2.2*df.y + 3.3*df.z

    expected = pd.DataFrame({
        'df_dx': np.full(4, 1.1),
        'df_dy': np.full(4, 2.2),
        'df_dz': np.full(4, 3.3)
    }, index=pd.Index(range(4), name='node_id'))

    grad = df.gradient_3D.gradient_of('f').sort_index()

    pd.testing.assert_frame_equal(grad.reset_index(), expected.reset_index(), rtol=1e-12)


def test_gradient_3D_tetrahedron_10_nodes():

    df = pd.DataFrame({'element_id': [1, 1, 1, 1] + [1]*6,
                       'node_id': range(10),
                       'x': [0.1, 0.4, 0.3, 0.2] + [0]*6,
                       'y': [0.5, 0.8, 0.6, 0.6] + [0]*6,
                       'z': [0.8, 0.9, 1.0, 2.0] + [0]*6,
                       })
    df = df.set_index(['node_id', 'element_id'])
    df["f"] = 1.1*df.x + 2.2*df.y + 3.3*df.z

    expected = pd.DataFrame({
        'df_dx': [1.1]*4 + [0]*6,
        'df_dy': [2.2]*4 + [0]*6,
        'df_dz': [3.3]*4 + [0]*6
    }, index=pd.Index(range(10), name='node_id'))

    grad = df.gradient_3D.gradient_of('f').sort_index()

    pd.testing.assert_frame_equal(grad.reset_index(), expected.reset_index(), rtol=1e-12)


def test_gradient_3D_tetrahedron_compare():

    df = pd.DataFrame({'element_id': [1, 1, 1, 1],
                       'node_id': range(4),
                       'x': [0.1, 0.4, 0.3, 0.2],
                       'y': [0.5, 0.8, 0.6, 0.6],
                       'z': [0.8, 0.9, 1.0, 2.0],
                       })
    df = df.set_index(['node_id', 'element_id'])
    df["f"] = 1.1*df.x

    expected = pd.DataFrame({
        'df_dx': np.full(4, 1.1),
        'df_dy': np.full(4, 0),
        'df_dz': np.full(4, 0)
    }, index=pd.Index(range(4), name='node_id'))

    grad_3D = df.gradient_3D.gradient_of('f').sort_index()
    grad = df.gradient.gradient_of('f').sort_index()

    # the following fails, which means that the Gradient class is less accurate for tet elements
    #pd.testing.assert_frame_equal(grad_3D.reset_index(), grad.reset_index(), check_dtype=False)
    pd.testing.assert_frame_equal(grad_3D.reset_index(), expected.reset_index(), check_dtype=False, rtol=1e-12)


def test_gradient_3D_hex_compare_1():

    df = pd.DataFrame({'element_id': [1, 1, 1, 1, 1, 1, 1, 1],
                       'node_id': [1, 2, 3, 4, 5, 6, 7, 8],
                       'x': np.zeros(8),
                       'y': np.zeros(8),
                       'z': np.zeros(8)})
    df = df.set_index(['node_id', 'element_id'])

    # set node positions
    offset = np.array([2,3,4])
    df.iloc[0,:3] = np.array([0,0,0.2]) + offset
    df.iloc[1,:3] = np.array([1.2,0,0.4]) + offset
    df.iloc[2,:3] = np.array([1.1,1.3,0]) + offset
    df.iloc[3,:3] = np.array([0.2,1.2,0]) + offset
    df.iloc[4,:3] = np.array([0,0.4,1.6]) + offset
    df.iloc[5,:3] = np.array([1.5,0.3,1.1]) + offset
    df.iloc[6,:3] = np.array([1.8,1.2,1.2]) + offset
    df.iloc[7,:3] = np.array([0.7,1,1.8]) + offset

    df["fct"] = 1.1*df.x

    expected = pd.DataFrame({
        'dfct_dx': np.full(8, 1.1),
        'dfct_dy': np.zeros(8),
        'dfct_dz': np.zeros(8)
    }, index=pd.Index([1, 2, 3, 4, 5, 6, 7, 8], name='node_id'))

    grad_3D = df.gradient_3D.gradient_of('fct').sort_index()
    grad = df.gradient.gradient_of('fct').sort_index()

    pd.testing.assert_frame_equal(grad_3D.reset_index(), grad.reset_index(), check_dtype=False, rtol=1e-12)
    pd.testing.assert_frame_equal(grad_3D.reset_index(), expected.reset_index(), check_dtype=False, rtol=1e-12)


def test_gradient_3D_hex_compare_2():

    df = pd.DataFrame({'element_id': [1, 1, 1, 1, 1, 1, 1, 1],
                       'node_id': [1, 2, 3, 4, 5, 6, 7, 8],
                       'x': np.zeros(8),
                       'y': np.zeros(8),
                       'z': np.zeros(8)})
    df = df.set_index(['node_id', 'element_id'])

    # set node positions
    offset = np.array([2,3,4])
    df.iloc[0,:3] = np.array([0,0,0.2]) + offset
    df.iloc[1,:3] = np.array([1.2,0,0.4]) + offset
    df.iloc[2,:3] = np.array([1.1,1.3,0]) + offset
    df.iloc[3,:3] = np.array([0.2,1.2,0]) + offset
    df.iloc[4,:3] = np.array([0,0.4,1.6]) + offset
    df.iloc[5,:3] = np.array([1.5,0.3,1.1]) + offset
    df.iloc[6,:3] = np.array([1.8,1.2,1.2]) + offset
    df.iloc[7,:3] = np.array([0.7,1,1.8]) + offset

    df["fct"] = 1.1*df.x - 2.5*df.y + 5.4*df.z

    expected = pd.DataFrame({
        'dfct_dx': np.full(8, 1.1),
        'dfct_dy': np.full(8, -2.5),
        'dfct_dz': np.full(8, +5.4),
    }, index=pd.Index([1, 2, 3, 4, 5, 6, 7, 8], name='node_id'))

    grad_3D = df.gradient_3D.gradient_of('fct').sort_index()
    grad = df.gradient.gradient_of('fct').sort_index()

    pd.testing.assert_frame_equal(grad_3D.reset_index(), grad.reset_index(), check_dtype=False, rtol=1e-12)
    pd.testing.assert_frame_equal(grad_3D.reset_index(), expected.reset_index(), check_dtype=False, rtol=1e-12)

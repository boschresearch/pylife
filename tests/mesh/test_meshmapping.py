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
import pytest

import pandas as pd
import numpy as np
import numpy.testing as testing

import pylife.mesh.meshmapping


def points_2d(flen, tlen):
    fx = np.arange(flen, dtype=np.float64)
    fy = np.arange(flen, dtype=np.float64)

    fxx, fyy = np.meshgrid(fx, fy, indexing='ij')
    from_vals = (fyy*2.).flatten()

    tx = np.arange(tlen, dtype=np.float64)
    ty = np.arange(tlen, dtype=np.float64) + 0.5
    txx, tyy = np.meshgrid(tx, ty, indexing='ij')

    fpoints = np.vstack((fxx.flatten(), fyy.flatten())).T
    tpoints = np.vstack((txx.flatten(), tyy.flatten())).T

    from_df = pd.DataFrame(fpoints, columns=['x', 'y'])
    from_df['value'] = from_vals

    to_df = pd.DataFrame(tpoints,  columns=['x', 'y'])

    return from_df, to_df


def points_3d(flen, tlen):
    fx = np.arange(flen, dtype=np.float64)
    fy = np.arange(flen, dtype=np.float64)
    fz = np.arange(flen, dtype=np.float64)

    fxx, fyy, fzz = np.meshgrid(fx, fy, fz, indexing='ij')
    from_vals = (fyy*2.).flatten()

    tx = np.arange(tlen, dtype=np.float64)
    ty = np.arange(tlen, dtype=np.float64) + 0.5
    tz = np.arange(tlen, dtype=np.float64)
    txx, tyy, tzz = np.meshgrid(tx, ty, tz, indexing='ij')

    fpoints = np.vstack((fxx.flatten(), fyy.flatten(), fzz.flatten())).T
    tpoints = np.vstack((txx.flatten(), tyy.flatten(), tzz.flatten())).T

    from_df = pd.DataFrame(fpoints, columns=['x', 'y', 'z'])
    from_df['value'] = from_vals

    to_df = pd.DataFrame(tpoints,  columns=['x', 'y', 'z'])

    return from_df, to_df


def test_mapping_2d_linear():
    flen = 241
    tlen = 21

    from_df, to_df = points_2d(flen, tlen)
    mapper = to_df.meshmapper
    to_vals = mapper.process(from_df, 'value')['value'].to_numpy()
    from_vals = from_df['value'].to_numpy()

    for x in range(tlen):
        for y in range(tlen):
            assert to_vals[x*tlen+y] == (from_vals[x*flen+y] + from_vals[x*flen+y+1])/2.


def test_mapping_2d_linear_out_of_map():
    flen = 241
    tlen = 21

    from_df, to_df = points_2d(flen, tlen)
    mapper = to_df.meshmapper
    from_vals = from_df['value'].to_numpy()

    with pytest.raises(Exception):
        mapper.process(from_vals)


def test_mapping_2d_cubic():
    flen = 241
    tlen = 21

    from_df, to_df = points_2d(flen, tlen)
    mapper = to_df.meshmapper
    to_vals = mapper.process(from_df, 'value', method='cubic')['value'].to_numpy()
    from_vals = from_df['value'].to_numpy()

    for x in range(tlen):
        for y in range(tlen):
            testing.assert_almost_equal(to_vals[x*tlen+y], (from_vals[x*flen+y] + from_vals[x*flen+y+1])/2.)


def test_mapping_3d_linear():
    flen = 23
    tlen = 11

    from_df, to_df = points_3d(flen, tlen)
    mapper = to_df.meshmapper
    to_vals = mapper.process(from_df, 'value')['value'].to_numpy()
    from_vals = from_df['value'].to_numpy()

    for x in range(tlen):
        for y in range(tlen):
            for z in range(tlen):
                assert (to_vals[x*tlen*tlen+y*tlen+z]
                        == (from_vals[x*flen*flen+y*flen+z] + from_vals[x*flen*flen+(y+1)*flen+z])/2.)


def test_mapping_3d_cubic():
    flen = 23
    tlen = 11

    from_df, to_df = points_3d(flen, tlen)
    mapper = to_df.meshmapper

    with pytest.raises(ValueError, match=r'.*cubic.*'):
        mapper.process(from_df, 'value', method='cubic')

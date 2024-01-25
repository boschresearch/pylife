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

import unittest
import numpy as np
import pandas as pd

import pylife.stress.rainflow as RF
import pylife.stress.rainflow.recorders as RFR


def process_signal(signal):
    fr = RFR.FullRecorder()
    dtor = RF.FKMDetector(recorder=fr).process(signal)

    return fr, dtor


class TestFKMMemory1Inner(unittest.TestCase):

    def setUp(self):
        signal = np.array([0., 100., 0., 80., 20., 60., 40., 100., 0., 80., 20., 60., 40., 45.])
        self._fr, self._dtor = process_signal(signal)

    def test_values(self):
        np.testing.assert_array_equal(self._fr.values_from, np.array([60., 80., 100.]))
        np.testing.assert_array_equal(self._fr.values_to, np.array([40., 20., 0.]))

    def test_residuals(self):
        np.testing.assert_array_equal(self._dtor.residuals, np.array([100., 0., 80., 20., 60., 40.]))


class TestFKMMemory1_2_3(unittest.TestCase):

    def setUp(self):
        signal = np.array([0.,
                           1., -1., 1., -2., -1., -2., 2., 0., 2., -2.,
                           1., -1., 1., -2., -1., -2., 2., 0., 2., -2.,
                           -1.8])
        self._fr, self._dtor = process_signal(signal)

    def test_values(self):
        np.testing.assert_array_equal(self._fr.values_from, np.array([1., -2., 2., -2.,  1., -2., -2., 2., -2.]))
        np.testing.assert_array_equal(self._fr.values_to, np.array([-1., -1., 0.,  2., -1.,  1., -1., 0.,  2.]))

    def test_residuals(self):
        np.testing.assert_array_equal(self._dtor.residuals, np.array([1., -2.]))


def test_series_signal_float_index():
    signal = pd.Series([0., 1., 0.], index=[1.0, 2.0, 3.0])
    process_signal(signal)

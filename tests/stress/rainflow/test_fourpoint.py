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

__author__ = "Vishnu Pradeep"
__maintainer__ = "Johannes Mueller"

import unittest
import pytest
import numpy as np

import pylife.stress.rainflow as RF
import pylife.stress.rainflow.recorders as RFR

def process_signal(signal):

    fr = RFR.FullRecorder()
    dtor = RF.FourPointDetector(recorder=fr).process(signal)

    return fr, dtor

class TestFourPointRandomNonPeriodicLoad(unittest.TestCase):
    
    '''Four Point Rainflow Counter method can only recognize the closed cycles and
    excludes any contribution from unpaired reversals.
    '''
    def setUp(self):
        signal = np.array([2.,-1.,3.,-5.,1.,-3.,4.,-4.,2.])
        self._fr, self._dtor = process_signal(signal)

    def test_values(self):
        np.testing.assert_array_equal(self._fr.values_from, np.array([1.]))
        np.testing.assert_array_equal(self._fr.values_to, np.array([-3.]))

    def test_indeces(self):
        np.testing.assert_array_equal(self._fr.index_from, np.array([4]))
        np.testing.assert_array_equal(self._fr.index_to, np.array([5]))

    def test_residuals(self):
        np.testing.assert_array_equal(self._dtor.residuals, np.array([2.,-1.,3.,-5.,
                                                                      4.,-4.,2.]))
        
class TestFourPointRandombyDuplicatingResidualsFromAbove(unittest.TestCase):
    
    '''Four Point Rainflow Counter method can only recognize the closed cycles and
    excludes any contribution from unpaired reversals.
    '''
    def setUp(self):
        signal = np.array([2.,-1,3,-5,4,-4,2,2,-1,3,-5,4,-4,2])
        self._fr, self._dtor = process_signal(signal)

    def test_values(self):
        np.testing.assert_array_equal(self._fr.values_from, np.array([2., -4., 4.]))
        np.testing.assert_array_equal(self._fr.values_to, np.array([-1., 3., -5.]))

    def test_indeces(self):
        np.testing.assert_array_equal(self._fr.index_from, np.array([6,5,4]))
        np.testing.assert_array_equal(self._fr.index_to, np.array([8,9,10]))

    def test_residuals(self):
        np.testing.assert_array_equal(self._dtor.residuals, np.array([2,-1,3,-5,4,-4,2]))
        

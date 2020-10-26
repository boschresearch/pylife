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



import numpy as np
import pandas as pd
from pylife.stress import histogram as hi


def test_combine_hist():
    hist1 = pd.DataFrame(np.array([5, 10]),
                         index=pd.interval_range(start=0, end=2),
                         columns=['data'])
    hist2 = pd.DataFrame(np.array([12, 3, 20]),
                         index=pd.interval_range(start=1, periods=3),
                         columns=['data'])

    expect_sum = pd.DataFrame(np.array([5, 22, 3, 20]),
                              index=pd.interval_range(start=0, periods=4),
                              columns=['data'])
    test_sum = hi.combine_hist([hist1, hist2], method='sum', nbins=4)
    pd.testing.assert_frame_equal(test_sum, expect_sum, check_names=False)

    expect_min = pd.DataFrame(np.array([5, 3]),
                              index=pd.interval_range(
                                  start=0, end=4, periods=2),
                              columns=['data'])
    test_min = hi.combine_hist([hist1, hist2], method='min', nbins=2)
    pd.testing.assert_frame_equal(test_min, expect_min, check_names=False)

    expect_max = pd.DataFrame(np.array([12, 20]),
                              index=pd.interval_range(
                                  start=0, end=4, periods=2),
                              columns=['data'])
    test_max = hi.combine_hist([hist1, hist2], method='max', nbins=2)
    pd.testing.assert_frame_equal(test_max, expect_max, check_names=False)

    expect_mean = pd.DataFrame(np.array([9., 11.5]),
                               index=pd.interval_range(
                                   start=0, end=4, periods=2),
                               columns=['data'])
    test_mean = hi.combine_hist([hist1, hist2], method='mean', nbins=2)
    pd.testing.assert_frame_equal(test_mean, expect_mean, check_names=False)

    expect_std = pd.DataFrame(np.array([np.std(np.array([5, 10, 12])),
                                        np.std(np.array([3, 20]))]),
                              index=pd.interval_range(
                                  start=0, end=4, periods=2),
                              columns=['data'])
    test_std = hi.combine_hist([hist1, hist2], method='std', nbins=2)
    pd.testing.assert_frame_equal(test_std, expect_std, check_names=False)

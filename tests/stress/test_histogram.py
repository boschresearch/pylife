# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:22:00 2020

@author: KRD2RNG
"""


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

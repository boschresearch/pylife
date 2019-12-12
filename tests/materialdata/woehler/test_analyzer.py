# -*- coding: utf-8 -*-

import sys, os, copy
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import numpy.testing as testing
import numpy.ma as ma
from scipy import stats, optimize
import mystic as my

import pylife.materialdata.woehler.analyzer as WCA


def test_maximum_likelihood_5param_method():
    """
    Test of woehler curve parameters evaluation with the maximum likelihood method
    """
    # Setup
    data = np.array([[450, 34000],[450, 54000],[450, 60000],[450, 76000],
                     [400, 53000],[400, 94000],[400, 207000],[400, 227000],
                     [375, 68000],[375, 234000],[375, 396000],
                     [375, 500000],[375, 600000],[375, 709000],
                     [350, 170000],[350, 187000],[350, 220000],[350, 289000],[350, 309000],[350, 1E7],
                     [325, 675000],[325, 751000],[325, 1E7],[325, 1E7],[325, 1E7],[325, 1E7],[325, 1E7],
                     [325, 1E7],[325, 1E7],[325, 1E7],
                     [300, 895000],[300, 1E7], [300, 1E7],[300, 1E7],[300, 1E7],[300, 1E7],[300, 1E7],
                     [300, 1E7],[300, 1E7],[300, 1E7]])
    data = pd.DataFrame(data)
    data.columns=['loads', 'cycles']
    ld_cyc_lim = data.cycles.max()

    SD_50_mali_true = 335.49754284383698
    TS_mali_true = 1.1956125208561188
    k_Mali_mali_true = 6.9451803219153518
    ND_50_mali_true = 463819.1853458205
    TN_mali_true = 4.6980328285392732

    true_mali = [SD_50_mali_true, TS_mali_true, k_Mali_mali_true, ND_50_mali_true, TN_mali_true]

    #Exercise
    WC_data = WCA.WoehlerCurve(data, ld_cyc_lim, {},
                               {'k_1': '', '1/TN': '', 'SD_50': '', '1/TS': '', 'ND_50': ''}
                              )
    result_mali = [*WC_data.Mali_5p_result.values()]

    #Verify
    np.testing.assert_array_almost_equal(result_mali, true_mali, decimal=1)

    #Cleanup - none
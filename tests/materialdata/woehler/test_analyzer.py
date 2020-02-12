# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import numpy.testing as testing
import numpy.ma as ma
from scipy import stats, optimize
import mystic as my

from pylife.materialdata.woehler.creators.woehler_curve_creator import WoehlerCurveCreator
from pylife.materialdata.woehler.fatigue_data import FatigueData

def test_maximum_likelihood_5param_method_with_no_fixed_params():
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

    expected_5p_mali = [SD_50_mali_true, TS_mali_true, k_Mali_mali_true, ND_50_mali_true, TN_mali_true]
    
    #Exercise
    FD = FatigueData(data, ld_cyc_lim)
    WCC = WoehlerCurveCreator(FD)
    WC_data = WCC.maximum_like_procedure({}) 
    result_5p_mali = [WC_data.SD_50, WC_data.TS, WC_data.k, WC_data.ND_50, WC_data.TN]

    #Verify
    np.testing.assert_array_almost_equal(result_5p_mali, expected_5p_mali, decimal=1)

    #Cleanup - none
    
  
def test_maximum_likelihood_5param_method_with_fixed_params():
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

    TN_5p_mali_true = 1.2
    SD_50_5p_mali_true = 280
    TS_5p_mali_true = 1.2
    ND_50_5p_mali_true = 10000000
    k_Mali_5p_mali_true = 6.94518
 

    expected_5p_mali = [SD_50_5p_mali_true, TS_5p_mali_true, k_Mali_5p_mali_true, 
                        ND_50_5p_mali_true, TN_5p_mali_true]

    #Exercise
    FD = FatigueData(data, ld_cyc_lim)
    WCC = WoehlerCurveCreator(FD)
    WC_data = WCC.maximum_like_procedure({'1/TN': 1.2, 'SD_50': 280, 
                                          '1/TS': 1.2, 'ND_50': 10000000}) 
    
    result_5p_mali = [WC_data.SD_50, WC_data.TS, WC_data.k, WC_data.ND_50, WC_data.TN]
    
    #Verify
    np.testing.assert_array_almost_equal(result_5p_mali, expected_5p_mali, decimal=1)
 
    #Cleanup - none   
      
def test_maximum_likelihood_5param_method_with_all_fixed_params():
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

    expected_5p_mali = [0, 0, 0, 0, 0]
    
    SD_50_2p_mali_true = 335.49751
    ND_50_2p_mali_true = 463819.429

    #Exercise
    FD = FatigueData(data, ld_cyc_lim)
    WCC = WoehlerCurveCreator(FD)

    np.testing.assert_raises(AttributeError, WCC.maximum_like_procedure, {'k_1': 15.7, '1/TN': 1.2, 'SD_50': 280, '1/TS': 1.2, 'ND_50': 10000000})

    #Verify
    #np.testing.assert_array_almost_equal(result_5p_mali, expected_5p_mali, decimal = 1)
    #np.testing.assert_array_almost_equal(result_2p_mali, true_2p_mali, decimal = 1)
    #Cleanup - none    
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest

import pylife.materialdata.woehler.fatigue_data

data = pd.DataFrame(np.array([
        [4.50e+02, 3.40e+04],
        [4.50e+02, 5.40e+04],
        [4.50e+02, 6.00e+04],
        [4.50e+02, 7.60e+04],
        [4.00e+02, 5.30e+04],
        [4.00e+02, 9.40e+04],
        [4.00e+02, 2.07e+05],
        [4.00e+02, 2.27e+05],
        [3.75e+02, 6.80e+04],
        [3.75e+02, 2.34e+05],
        [3.75e+02, 3.96e+05],
        [3.75e+02, 5.00e+05],
        [3.75e+02, 6.00e+05],
        [3.75e+02, 7.09e+05],
        [3.50e+02, 1.70e+05],
        [3.50e+02, 1.87e+05],
        [3.50e+02, 2.20e+05],
        [3.50e+02, 2.89e+05],
        [3.50e+02, 3.09e+05],
        [3.50e+02, 1.00e+07],
        [3.25e+02, 6.75e+05],
        [3.25e+02, 7.51e+05],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.25e+02, 1.00e+07],
        [3.00e+02, 8.95e+05],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07],
        [3.00e+02, 1.00e+07]
]), columns=['load', 'cycles'])

def test_max_likelihood_inf_limit():
    """
    Test of woehler curve parameters evaluation with the maximum likelihood method
    """
    SD_50_mali_true = 335.49754284383698
    TS_mali_true = 1.1956125208561188
    k_Mali_mali_true = 6.9451803219153518
    ND_50_mali_true = 463819.1853458205
    TN_mali_true = 5.26

    expected_5p_mali = [SD_50_mali_true, TS_mali_true, k_Mali_mali_true, ND_50_mali_true, TN_mali_true]

    # Exercise
    WC_data = data.fatigue.woehler_max_likelihood_inf_limit()
    result_5p_mali = [WC_data.SD_50, WC_data.TS, WC_data.k, WC_data.ND_50, WC_data.TN]

    np.testing.assert_allclose(result_5p_mali, expected_5p_mali, rtol=1e-1)

def test_max_likelihood_full_without_fixed_params():
    """
    Test of woehler curve parameters evaluation with the maximum likelihood method
    """
    SD_50_mali_true = 335.49754284383698
    TS_mali_true = 1.1956125208561188
    k_Mali_mali_true = 6.9451803219153518
    ND_50_mali_true = 463819.1853458205
    TN_mali_true = 4.6980328285392732

    expected_5p_mali = [SD_50_mali_true, TS_mali_true, k_Mali_mali_true, ND_50_mali_true, TN_mali_true]

    # Exercise
    WC_data = data.fatigue.woehler_max_likelihood()
    result_5p_mali = [WC_data.SD_50, WC_data.TS, WC_data.k, WC_data.ND_50, WC_data.TN]

    # Verify
    np.testing.assert_array_almost_equal(result_5p_mali, expected_5p_mali, decimal=1)


def test_max_likelihood_full_with_fixed_params():
    """
    Test of woehler curve parameters evaluation with the maximum likelihood method
    """
    TN_5p_mali_true = 1.2
    SD_50_5p_mali_true = 280
    TS_5p_mali_true = 1.2
    ND_50_5p_mali_true = 10000000
    k_Mali_5p_mali_true = 6.94518

    expected_5p_mali = [SD_50_5p_mali_true, TS_5p_mali_true, k_Mali_5p_mali_true,
                        ND_50_5p_mali_true, TN_5p_mali_true]

    # Exercise
    WC_data = data.fatigue.woehler_max_likelihood({'1/TN': 1.2, 'SD_50': 280,
                                                   '1/TS': 1.2, 'ND_50': 10000000})

    result_5p_mali = [WC_data.SD_50, WC_data.TS, WC_data.k, WC_data.ND_50, WC_data.TN]

    # Verify
    np.testing.assert_array_almost_equal(result_5p_mali, expected_5p_mali, decimal=1)


def test_max_likelihood_full_method_with_all_fixed_params():
    """
    Test of woehler curve parameters evaluation with the maximum likelihood method
    """
    ld_cyc_lim = data.cycles.max()

    expected_5p_mali = [0, 0, 0, 0, 0]

    SD_50_2p_mali_true = 335.49751
    ND_50_2p_mali_true = 463819.429

    # Exercise
    with pytest.raises(AttributeError, match=r'You need to leave at least one parameter empty!'):
        WC_data = data.fatigue.woehler_max_likelihood({'k_1': 15.7, '1/TN': 1.2, 'SD_50': 280, '1/TS': 1.2, 'ND_50': 10000000})

    #np.testing.assert_raises(AttributeError, WCC.maximum_like_procedure, {
    #                         'k_1': 15.7, '1/TN': 1.2, 'SD_50': 280, '1/TS': 1.2, 'ND_50': 10000000})

    # Verify
    #np.testing.assert_array_almost_equal(result_5p_mali, expected_5p_mali, decimal = 1)
    #np.testing.assert_array_almost_equal(result_2p_mali, true_2p_mali, decimal = 1)
    #Cleanup - none

def test_bayesian():
    SD_50_mali_true = 340.
    TS_mali_true = 1.12
    k_Mali_mali_true = 7.0
    ND_50_mali_true = 400000.
    TN_mali_true = 5.3

    expected = [SD_50_mali_true, TS_mali_true, k_Mali_mali_true, ND_50_mali_true, TN_mali_true]

    # Exercise
    WC_data = data.fatigue.woehler_bayesian(nsamples=200)
    result = [WC_data.SD_50, WC_data.TS, WC_data.k, WC_data.ND_50, WC_data.TN]

    # Verify
    np.testing.assert_allclose(result, expected, rtol=1e-1)

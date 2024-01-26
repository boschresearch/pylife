# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 10:24:26 2021

@author: SIS3SI
"""
import re
import pytest
import numpy as np

from  pylife.materiallaws.true_stress_strain import *

parametrization_data_strain = np.array([
    [0.0, 0.0],
    [1.0, 0.6931471805599453],
    [2.0, 1.0986122886681098]
])


@pytest.mark.parametrize('tech_strain, expected', map(tuple, parametrization_data_strain))
def test_true_strain_scalar(tech_strain, expected):
    np.testing.assert_approx_equal(true_strain(tech_strain), expected, significant=5)


parametrization_data_stress = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 2.0],
    [2.0, 2.0, 6.0]
])

@pytest.mark.parametrize('tech_stress, tech_strain, expected', map(tuple, parametrization_data_stress))
def test_true_stress_scalar(tech_stress, tech_strain, expected):
    np.testing.assert_approx_equal(true_stress(tech_stress, tech_strain), expected, significant=5)


parametrization_fracture_strain= np.array([
    [0.0, 0.0],
    [0.2, 0.22314355131420976],
    [0.3, 0.3566749439387324]
])


@pytest.mark.parametrize('reduction_area_fracture, expected', map(tuple, parametrization_fracture_strain))
def test_true_fracture_strain_scalar(reduction_area_fracture, expected):
    np.testing.assert_approx_equal(true_fracture_strain(reduction_area_fracture), expected, significant=5)


parametrization_fracture_stress= np.array([
    [0.0, 1.0, 0.1, 0],
    [5000.0, 12.0, 0.6, 1041.6666666666665]
   ])


@pytest.mark.parametrize('fracture_force, initial_cross_section, reduction_area_fracture, expected', map(tuple, parametrization_fracture_stress))
def test_true_fracture_stress_scalar(fracture_force, initial_cross_section, reduction_area_fracture, expected):
    np.testing.assert_approx_equal(true_fracture_stress(fracture_force, initial_cross_section, reduction_area_fracture), expected, significant=5)

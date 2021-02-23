import re
import pytest
import numpy as np

from  pylife.materiallaws import RambergOsgood


@pytest.fixture
def ramberg_osgood():
    return RambergOsgood(210.5e9, 1078e6, 0.133)


def test_rambgood_init(ramberg_osgood):
    assert ramberg_osgood._E == 210.5e9
    assert ramberg_osgood._K == 1078e6
    assert ramberg_osgood._n == 0.133


@pytest.mark.parametrize('stress, expected', [
    (0.0, 0.0),
    (700e6, 0.042236),
    (1000e6, 0.57327)
])
def test_rambgood_strain_scalar(ramberg_osgood, stress, expected):
    np.testing.assert_approx_equal(ramberg_osgood.strain(stress), expected, significant=5)


@pytest.mark.parametrize('stress, expected', [
    (np.array([700e6, 1000e6]), np.array([0.042236, 0.57327]))
])
def test_rambgood_strain_array(ramberg_osgood, stress, expected):
    np.testing.assert_allclose(ramberg_osgood.strain(stress), expected, rtol=1e-5)


def test_rambgood_strain_neg_stress_scalar(ramberg_osgood):
    with pytest.raises(ValueError,
                       match=re.escape("Stress value in Ramberg-Osgood equation must not be negative.")):
        ramberg_osgood.strain(-100.0)


def test_rambgood_strain_neg_stress_array(ramberg_osgood):
    with pytest.raises(ValueError,
                       match=re.escape("Stress value in Ramberg-Osgood equation must not be negative.")):
        ramberg_osgood.strain([-100.0, 1000.0])


@pytest.mark.parametrize('delta_stress, expected', [
    (0.0, 0.0),
    (1000e6, 0.01095061),
    (1100e6, 0.01792016)
])
def test_rambgood_delta_strain_scalar(ramberg_osgood, delta_stress, expected):
    np.testing.assert_approx_equal(ramberg_osgood.delta_strain(delta_stress), expected, significant=5)


@pytest.mark.parametrize('delta_stress, expected', [
    (np.array([0.0, 1000e6, 1100e6]), np.array([0.0, 0.01095061, 0.01792016]))
])
def test_rambgood_delta_strain_array(ramberg_osgood, delta_stress, expected):
    np.testing.assert_allclose(ramberg_osgood.delta_strain(delta_stress), expected, rtol=1e-5)


def test_rambgood_delta_strain_neg_delta_stress_scalar(ramberg_osgood):
    with pytest.raises(ValueError,
                       match=re.escape("Stress value in Ramberg-Osgood equation must not be negative.")):
        ramberg_osgood.delta_strain(-100.0)


def test_rambgood_delta_strain_neg_delta_stress_array(ramberg_osgood):
    with pytest.raises(ValueError,
                       match=re.escape("Stress value in Ramberg-Osgood equation must not be negative.")):
        ramberg_osgood.delta_strain(np.array([-100.0, 1000]))


@pytest.mark.parametrize('stress, max_stress, expected', [
    (-700e6, 700e6, -0.042236),
    (-1000e6, 1000e6, -0.57327),
    (0.0, 1000e6, 0.562320)
])
def test_rambgood_lower_hysteresis_scalar(ramberg_osgood, stress, max_stress, expected):
    np.testing.assert_approx_equal(ramberg_osgood.lower_hysteresis(stress, max_stress),
                                   expected, significant=5)


@pytest.mark.parametrize('stress, max_stress, expected', [
    (np.array([-1000e6, 0.0]), 1000e6, np.array([-0.57327, 0.562320])),
])
def test_rambgood_lower_hysteresis_array(ramberg_osgood, stress, max_stress, expected):
    np.testing.assert_allclose(ramberg_osgood.lower_hysteresis(stress, max_stress),
                               expected, rtol=1e-5)

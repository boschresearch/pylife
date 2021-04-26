import re
import pytest
import numpy as np

from  pylife.materiallaws import RambergOsgood


@pytest.fixture
def ramberg_osgood_monotone():
    return RambergOsgood(E=2., K=6., n=0.5)


parametrization_data_monotone = np.array([
    [0.0, 0.0],
    [1.0, 19./36.],
    [2.0, 10./9.],
    [3.0, 7./4.]
])


def test_rambgood_init(ramberg_osgood_monotone):
    assert ramberg_osgood_monotone._E == 2.
    assert ramberg_osgood_monotone._K == 6.
    assert ramberg_osgood_monotone._n == 0.5


@pytest.mark.parametrize('stress, expected', map(tuple, parametrization_data_monotone))
def test_rambgood_strain_scalar(ramberg_osgood_monotone, stress, expected):
    np.testing.assert_approx_equal(ramberg_osgood_monotone.strain(stress), expected, significant=5)


@pytest.mark.parametrize('stress, expected', [
    (parametrization_data_monotone[:, 0], parametrization_data_monotone[:, 1])
])
def test_rambgood_strain_array(ramberg_osgood_monotone, stress, expected):
    np.testing.assert_allclose(ramberg_osgood_monotone.strain(stress), expected, rtol=1e-5)


def test_rambgood_strain_neg_stress_scalar(ramberg_osgood_monotone):
    with pytest.raises(ValueError,
                       match=re.escape("Stress value in Ramberg-Osgood equation must not be negative.")):
        ramberg_osgood_monotone.strain(-100.0)


def test_rambgood_strain_neg_stress_array(ramberg_osgood_monotone):
    with pytest.raises(ValueError,
                       match=re.escape("Stress value in Ramberg-Osgood equation must not be negative.")):
        ramberg_osgood_monotone.strain([-100.0, 1000.0])

parametrization_data_monotone_plastic = np.array([
    [0.0, 0.0],
    [1.0, 1./36.],
    [2.0, 1./9.],
    [3.0, 1./4.]
])


@pytest.mark.parametrize('stress, expected', [
    (parametrization_data_monotone_plastic[:, 0], parametrization_data_monotone_plastic[:, 1])
])
def test_rambgood_plastic_strain_scalar(ramberg_osgood_monotone, stress, expected):
    np.testing.assert_allclose(ramberg_osgood_monotone.plastic_strain(stress), expected, rtol=1e-5)


@pytest.mark.parametrize('stress, expected', map(tuple, parametrization_data_monotone_plastic))
def test_rambgood_plastic_strain_array(ramberg_osgood_monotone, stress, expected):
    np.testing.assert_approx_equal(ramberg_osgood_monotone.plastic_strain(stress), expected, significant=5)


@pytest.fixture
def ramberg_osgood_cyclic():
    return RambergOsgood(E=2., K=3., n=0.5)


parametrization_data_delta = np.array([
    [0.0, 0.0],
    [1.0, 10./18.],
    [2.0, 11./9.]
])


@pytest.mark.parametrize('delta_stress, expected', map(tuple, parametrization_data_delta))
def test_rambgood_delta_strain_scalar(ramberg_osgood_cyclic, delta_stress, expected):
    np.testing.assert_approx_equal(ramberg_osgood_cyclic.delta_strain(delta_stress), expected, significant=5)


@pytest.mark.parametrize('delta_stress, expected', [
    (
        parametrization_data_delta[:, 0],
        parametrization_data_delta[:, 1]
    )
])
def test_rambgood_delta_strain_array(ramberg_osgood_cyclic, delta_stress, expected):
    np.testing.assert_allclose(ramberg_osgood_cyclic.delta_strain(delta_stress), expected, rtol=1e-5)


def test_rambgood_delta_strain_neg_delta_stress_scalar(ramberg_osgood_cyclic):
    with pytest.raises(ValueError,
                       match=re.escape("Stress value in Ramberg-Osgood equation must not be negative.")):
        ramberg_osgood_cyclic.delta_strain(-100.0)


def test_rambgood_delta_strain_neg_delta_stress_array(ramberg_osgood_cyclic):
    with pytest.raises(ValueError,
                       match=re.escape("Stress value in Ramberg-Osgood equation must not be negative.")):
        ramberg_osgood_cyclic.delta_strain(np.array([-100.0, 1000]))


@pytest.fixture
def ramberg_osgood():
    return RambergOsgood(210.5e9, 1078e6, 0.133)


def test_rambgood_char_init(ramberg_osgood):
    assert ramberg_osgood._E == 210.5e9
    assert ramberg_osgood._K == 1078e6
    assert ramberg_osgood._n == 0.133


@pytest.mark.parametrize('stress, expected', [
    (0.0, 0.0),
    (700e6, 0.042236),
    (1000e6, 0.57327)
])
def test_rambgood_char_strain_scalar(ramberg_osgood, stress, expected):
    np.testing.assert_approx_equal(ramberg_osgood.strain(stress), expected, significant=5)


@pytest.mark.parametrize('stress, expected', [
    (np.array([700e6, 1000e6]), np.array([0.042236, 0.57327]))
])
def test_rambgood_char_strain_array(ramberg_osgood, stress, expected):
    np.testing.assert_allclose(ramberg_osgood.strain(stress), expected, rtol=1e-5)


def test_rambgood_char_strain_neg_stress_scalar(ramberg_osgood):
    with pytest.raises(ValueError,
                       match=re.escape("Stress value in Ramberg-Osgood equation must not be negative.")):
        ramberg_osgood.strain(-100.0)


def test_rambgood_char_strain_neg_stress_array(ramberg_osgood):
    with pytest.raises(ValueError,
                       match=re.escape("Stress value in Ramberg-Osgood equation must not be negative.")):
        ramberg_osgood.strain([-100.0, 1000.0])


@pytest.mark.parametrize('delta_stress, expected', [
    (0.0, 0.0),
    (1000e6, 0.01095061),
    (1100e6, 0.01792016)
])
def test_rambgood_char_delta_strain_scalar(ramberg_osgood, delta_stress, expected):
    np.testing.assert_approx_equal(ramberg_osgood.delta_strain(delta_stress), expected, significant=5)


@pytest.mark.parametrize('delta_stress, expected', [
    (np.array([0.0, 1000e6, 1100e6]), np.array([0.0, 0.01095061, 0.01792016]))
])
def test_rambgood_char_delta_strain_array(ramberg_osgood, delta_stress, expected):
    np.testing.assert_allclose(ramberg_osgood.delta_strain(delta_stress), expected, rtol=1e-5)


def test_rambgood_char_delta_strain_neg_delta_stress_scalar(ramberg_osgood):
    with pytest.raises(ValueError,
                       match=re.escape("Stress value in Ramberg-Osgood equation must not be negative.")):
        ramberg_osgood.delta_strain(-100.0)


def test_rambgood_char_delta_strain_neg_delta_stress_array(ramberg_osgood):
    with pytest.raises(ValueError,
                       match=re.escape("Stress value in Ramberg-Osgood equation must not be negative.")):
        ramberg_osgood.delta_strain(np.array([-100.0, 1000]))


@pytest.mark.parametrize('stress, max_stress, expected', [
    (-700e6, 700e6, -0.042236),
    (-1000e6, 1000e6, -0.57327),
    (0.0, 1000e6, 0.562320)
])
def test_rambgood_char_lower_hysteresis_scalar(ramberg_osgood, stress, max_stress, expected):
    np.testing.assert_approx_equal(ramberg_osgood.lower_hysteresis(stress, max_stress),
                                   expected, significant=5)


@pytest.mark.parametrize('stress, max_stress, expected', [
    (np.array([-1000e6, 0.0]), 1000e6, np.array([-0.57327, 0.562320])),
])
def test_rambgood_char_lower_hysteresis_array(ramberg_osgood, stress, max_stress, expected):
    np.testing.assert_allclose(ramberg_osgood.lower_hysteresis(stress, max_stress),
                               expected, rtol=1e-5)

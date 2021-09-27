import re
import pytest
import numpy as np

from pylife.materiallaws import RambergOsgood


@pytest.fixture
def ramberg_osgood_monotone():
    return RambergOsgood(E=2., K=6., n=0.5)


def test_rambgood_init(ramberg_osgood_monotone):
    assert ramberg_osgood_monotone._E == 2.
    assert ramberg_osgood_monotone._K == 6.
    assert ramberg_osgood_monotone._n == 0.5


def test_rambgood_properties(ramberg_osgood_monotone):
    assert ramberg_osgood_monotone.E == ramberg_osgood_monotone._E
    assert ramberg_osgood_monotone.K == ramberg_osgood_monotone._K
    assert ramberg_osgood_monotone.n == ramberg_osgood_monotone._n


parametrization_data_monotone = np.array([
    [0.0, 0.0],
    [1.0, 19./36.],
    [2.0, 10./9.],
    [3.0, 7./4.],
    [-1.0, -19./36.],
    [-2.0, -10./9.],
    [-3.0, -7./4.]
])


@pytest.mark.parametrize('stress, expected', map(tuple, parametrization_data_monotone))
def test_rambgood_strain_scalar(ramberg_osgood_monotone, stress, expected):
    np.testing.assert_approx_equal(ramberg_osgood_monotone.strain(stress), expected, significant=5)


@pytest.mark.parametrize('expected, strain', map(tuple, parametrization_data_monotone))
def test_rambgood_stress_scalar(ramberg_osgood_monotone, expected, strain):
    np.testing.assert_approx_equal(ramberg_osgood_monotone.stress(strain), expected, significant=5)


@pytest.mark.parametrize('stress, expected', [
    (parametrization_data_monotone[:, 0], parametrization_data_monotone[:, 1])
])
def test_rambgood_strain_array(ramberg_osgood_monotone, stress, expected):
    np.testing.assert_allclose(ramberg_osgood_monotone.strain(stress), expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [
    (parametrization_data_monotone[:, 0], parametrization_data_monotone[:, 1])
])
def test_rambgood_stress_array(ramberg_osgood_monotone, expected, strain):
    np.testing.assert_allclose(ramberg_osgood_monotone.stress(strain), expected, rtol=1e-5)


parametrization_data_monotone_plastic = np.array([
    [0.0, 0.0],
    [1.0, 1./36.],
    [2.0, 1./9.],
    [3.0, 1./4.],
    [-1.0, -1./36.],
    [-2.0, -1./9.],
    [-3.0, -1./4.]
])


@pytest.mark.parametrize('stress, expected', [
    (parametrization_data_monotone_plastic[:, 0], parametrization_data_monotone_plastic[:, 1])
])
def test_rambgood_plastic_strain_scalar(ramberg_osgood_monotone, stress, expected):
    np.testing.assert_allclose(ramberg_osgood_monotone.plastic_strain(stress), expected, rtol=1e-5)


@pytest.mark.parametrize('stress, expected', map(tuple, parametrization_data_monotone_plastic))
def test_rambgood_plastic_strain_array(ramberg_osgood_monotone, stress, expected):
    np.testing.assert_approx_equal(ramberg_osgood_monotone.plastic_strain(stress), expected, significant=5)


parametrization_data_monotone_tang_compl = np.array([
    [0., 1./2],
    [1., 5./9],
    [2., 11./18],
    [4., 13./18],
    [-1., 5./9],
    [-2., 11./18],
    [-4., 13./18]
])


@pytest.mark.parametrize('stress, expected', map(tuple, parametrization_data_monotone_tang_compl))
def test_rambgood_tangential_compliance_scalar(ramberg_osgood_monotone, stress, expected):
    np.testing.assert_approx_equal(ramberg_osgood_monotone.tangential_compliance(stress), expected, significant=5)


@pytest.mark.parametrize('stress, expected', [
    (parametrization_data_monotone_tang_compl[:, 0], parametrization_data_monotone_tang_compl[:, 1])
])
def test_rambgood_tangential_compliance_array(ramberg_osgood_monotone, stress, expected):
    np.testing.assert_allclose(ramberg_osgood_monotone.tangential_compliance(stress), expected, rtol=1e-5)


@pytest.mark.parametrize('stress, expected', map(tuple, parametrization_data_monotone_tang_compl))
def test_rambgood_tangential_modulus_scalar(ramberg_osgood_monotone, stress, expected):
    np.testing.assert_approx_equal(ramberg_osgood_monotone.tangential_modulus(stress), 1./expected, significant=5)


@pytest.mark.parametrize('stress, expected', [
    (parametrization_data_monotone_tang_compl[:, 0], parametrization_data_monotone_tang_compl[:, 1])
])
def test_rambgood_tangential_modulus_array(ramberg_osgood_monotone, stress, expected):
    np.testing.assert_allclose(ramberg_osgood_monotone.tangential_modulus(stress), 1./expected, rtol=1e-5)


@pytest.fixture
def ramberg_osgood_cyclic():
    return RambergOsgood(E=2., K=3., n=0.5)


parametrization_data_delta = np.array([
    [0.0, 0.0],
    [1.0, 10./18.],
    [2.0, 11./9.],
    [-1.0, -10./18.],
    [-2.0, -11./9.]
])


@pytest.mark.parametrize('delta_stress, expected', map(tuple, parametrization_data_delta))
def test_rambgood_delta_strain_scalar(ramberg_osgood_cyclic, delta_stress, expected):
    np.testing.assert_approx_equal(ramberg_osgood_cyclic.delta_strain(delta_stress), expected, significant=5)


@pytest.mark.parametrize('expected, delta_strain', map(tuple, parametrization_data_delta))
def test_rambgood_delta_stress_scalar(ramberg_osgood_cyclic, expected, delta_strain):
    np.testing.assert_approx_equal(ramberg_osgood_cyclic.delta_stress(delta_strain), expected, significant=5)


@pytest.mark.parametrize('delta_stress, expected', [(parametrization_data_delta[:, 0], parametrization_data_delta[:, 1])])
def test_rambgood_delta_strain_array(ramberg_osgood_cyclic, delta_stress, expected):
    np.testing.assert_allclose(ramberg_osgood_cyclic.delta_strain(delta_stress), expected, rtol=1e-5)


@pytest.mark.parametrize('expected, delta_strain', [(parametrization_data_delta[:, 0], parametrization_data_delta[:, 1])])
def test_rambgood_delta_stress_array(ramberg_osgood_cyclic, expected, delta_strain):
    np.testing.assert_allclose(ramberg_osgood_cyclic.delta_stress(delta_strain), expected, rtol=1e-5)


@pytest.fixture
def ramberg_osgood():
    return RambergOsgood(210.5e9, 1078e6, 0.133)


def test_rambgood_char_init(ramberg_osgood):
    assert ramberg_osgood._E == 210.5e9
    assert ramberg_osgood._K == 1078e6
    assert ramberg_osgood._n == 0.133


def test_rambgood_properties_real(ramberg_osgood):
    assert ramberg_osgood.E == ramberg_osgood._E
    assert ramberg_osgood.K == ramberg_osgood._K
    assert ramberg_osgood.n == ramberg_osgood._n


parametrization_data_monotone_real = np.array([
    [0.0, 0.0],
    [700e6, 0.042236],
    [1000e6, 0.57327],
    [-700e6, -0.042236],
    [-1000e6, -0.57327]
])


@pytest.mark.parametrize('stress, expected', map(tuple, parametrization_data_monotone_real))
def test_rambgood_char_strain_scalar(ramberg_osgood, stress, expected):
    np.testing.assert_approx_equal(ramberg_osgood.strain(stress), expected, significant=5)


@pytest.mark.parametrize('expected, strain', map(tuple, parametrization_data_monotone_real))
def test_rambgood_char_stress_scalar(ramberg_osgood, expected, strain):
    np.testing.assert_approx_equal(ramberg_osgood.stress(strain), expected, significant=5)


@pytest.mark.parametrize('stress, expected', [(parametrization_data_monotone_real[:, 0], parametrization_data_monotone_real[:, 1])])
def test_rambgood_char_strain_array(ramberg_osgood, stress, expected):
    np.testing.assert_allclose(ramberg_osgood.strain(stress), expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [(parametrization_data_monotone_real[:, 0], parametrization_data_monotone_real[:, 1])])
def test_rambgood_char_stress_array(ramberg_osgood, expected, strain):
    np.testing.assert_allclose(ramberg_osgood.stress(strain), expected, rtol=1e-5)


parametrization_data_delta_real = np.array([
    [0.0, 0.0],
    [1000e6, 0.01095061],
    [1100e6, 0.01792016],
    [-1000e6, -0.01095061],
    [-1100e6, -0.01792016]
])


@pytest.mark.parametrize('delta_stress, expected', map(tuple, parametrization_data_delta_real))
def test_rambgood_char_delta_strain_scalar(ramberg_osgood, delta_stress, expected):
    np.testing.assert_approx_equal(ramberg_osgood.delta_strain(delta_stress), expected, significant=5)


@pytest.mark.parametrize('expected, delta_strain', map(tuple, parametrization_data_delta_real))
def test_rambgood_char_delta_stress_scalar(ramberg_osgood, expected, delta_strain):
    np.testing.assert_approx_equal(ramberg_osgood.delta_stress(delta_strain), expected, significant=5)


@pytest.mark.parametrize('delta_stress, expected', [(parametrization_data_delta_real[:, 0], parametrization_data_delta_real[:, 1])])
def test_rambgood_char_delta_strain_array(ramberg_osgood, delta_stress, expected):
    np.testing.assert_allclose(ramberg_osgood.delta_strain(delta_stress), expected, rtol=1e-5)


parametrization_data_monotone_tang_compl_real = np.array([
    [0., 1/(210.5e9)],
    [700e6, 4.2269651e-10],
    [1000e6, 4.2793411e-09],
    [-700e6, 4.2269651e-10],
    [-1000e6, 4.2793411e-09],
])


@pytest.mark.parametrize('stress, expected', map(tuple, parametrization_data_monotone_tang_compl_real))
def test_rambgood_tangential_compliance_real_scalar(ramberg_osgood, stress, expected):
    np.testing.assert_approx_equal(ramberg_osgood.tangential_compliance(stress), expected, significant=5)


@pytest.mark.parametrize('stress, expected', [
    (parametrization_data_monotone_tang_compl_real[:, 0], parametrization_data_monotone_tang_compl_real[:, 1])
])
def test_rambgood_tangential_compliance_real_array(ramberg_osgood, stress, expected):
    np.testing.assert_allclose(ramberg_osgood.tangential_compliance(stress), expected, rtol=1e-5)


@pytest.mark.parametrize('stress, expected', map(tuple, parametrization_data_monotone_tang_compl_real))
def test_rambgood_tangential_modulus_real_scalar(ramberg_osgood, stress, expected):
    np.testing.assert_approx_equal(ramberg_osgood.tangential_modulus(stress), 1./expected, significant=5)


@pytest.mark.parametrize('stress, expected', [
    (parametrization_data_monotone_tang_compl_real[:, 0], parametrization_data_monotone_tang_compl_real[:, 1])
])
def test_rambgood_tangential_modulus_real_array(ramberg_osgood, stress, expected):
    np.testing.assert_allclose(ramberg_osgood.tangential_modulus(stress), 1./expected, rtol=1e-5)


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


def test_ramgood_lower_hysteresis_above_max_stress_scalar(ramberg_osgood):
    with pytest.raises(ValueError, match=r'Value for \'stress\' must not be higher than \'max_stress\'.'):
        ramberg_osgood.lower_hysteresis(200., 100.)


def test_ramgood_lower_hysteresis_above_max_stress_array(ramberg_osgood):
    with pytest.raises(ValueError, match=r'Value for \'stress\' must not be higher than \'max_stress\'.'):
        ramberg_osgood.lower_hysteresis([50., 200.], 100.)

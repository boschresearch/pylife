import pytest
import numpy as np

from pylife.materiallaws import HookesLaw1D, hookeslaw


@pytest.fixture
def hookeslaw1D():
    return HookesLaw1D(E=2.)


def test_hookeslaw1D_init(hookeslaw1D):
    assert hookeslaw1D._E == 2.


def test_hookeslaw1D_properties(hookeslaw1D):
    assert hookeslaw1D.E == hookeslaw1D._E


def test_hookeslaw1D_eq(hookeslaw1D):
    assert hookeslaw1D == HookesLaw1D(E=2.)
    assert not hookeslaw1D == HookesLaw1D(E=20.)
    assert not hookeslaw1D == 0.
    assert hookeslaw1D != HookesLaw1D(E=20.)


parametrization_data_1D = np.array([
    [0.0, 0.0],
    [1.0, 1./2],
    [2.0, 1.],
    [3.0, 3./2.],
    [-1.0, -1./2],
    [-2.0, -1.],
    [-3.0, -3./2.]
])


@pytest.mark.parametrize('stress, expected', map(tuple, parametrization_data_1D))
def test_hookeslaw1D_strain_scalar(hookeslaw1D, stress, expected):
    np.testing.assert_approx_equal(hookeslaw1D.strain(stress), expected, significant=5)


@pytest.mark.parametrize('expected, strain', map(tuple, parametrization_data_1D))
def test_hookeslaw1D_stress_scalar(hookeslaw1D, expected, strain):
    np.testing.assert_approx_equal(hookeslaw1D.stress(strain), expected, significant=5)


@pytest.mark.parametrize('stress, expected', [
    (parametrization_data_1D[:, 0], parametrization_data_1D[:, 1])
])
def test_hookeslaw1D_strain_array(hookeslaw1D, stress, expected):
    np.testing.assert_allclose(hookeslaw1D.strain(stress), expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [
    (parametrization_data_1D[:, 0], parametrization_data_1D[:, 1])
])
def test_hookeslaw1D_stress_array(hookeslaw1D, expected, strain):
    np.testing.assert_allclose(hookeslaw1D.stress(strain), expected, rtol=1e-5)


@pytest.fixture
def hookeslaw1D_real():
    return HookesLaw1D(E=210.5e9)


def test_hookeslaw1D_init_real(hookeslaw1D_real):
    assert hookeslaw1D_real._E == 210.5e9


def test_hookeslaw1D_properties_real(hookeslaw1D_real):
    assert hookeslaw1D_real.E == hookeslaw1D_real._E


def test_hookeslaw1D_eq_real(hookeslaw1D_real):
    assert hookeslaw1D_real == HookesLaw1D(E=210.5e9)
    assert not hookeslaw1D_real == HookesLaw1D(E=210.5)
    assert not hookeslaw1D_real == 0.
    assert hookeslaw1D_real != HookesLaw1D(E=210.5)


parametrization_data_1D_real = np.array([
    [0.0, 0.0],
    [700e6, 3.325415e-3],
    [1000e6, 4.750593e-3],
    [-700e6, -3.325415e-3],
    [-1000e6, -4.750593e-3]
])


@pytest.mark.parametrize('stress, expected', map(tuple, parametrization_data_1D_real))
def test_hookeslaw1D_strain_scalar_real(hookeslaw1D_real, stress, expected):
    np.testing.assert_approx_equal(hookeslaw1D_real.strain(stress), expected, significant=5)


@pytest.mark.parametrize('expected, strain', map(tuple, parametrization_data_1D_real))
def test_hookeslaw1D_stress_scalar_real(hookeslaw1D_real, expected, strain):
    np.testing.assert_approx_equal(hookeslaw1D_real.stress(strain), expected, significant=5)


@pytest.mark.parametrize('stress, expected', [
    (parametrization_data_1D_real[:, 0], parametrization_data_1D_real[:, 1])
])
def test_hookeslaw1D_strain_array_real(hookeslaw1D_real, stress, expected):
    np.testing.assert_allclose(hookeslaw1D_real.strain(stress), expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [
    (parametrization_data_1D_real[:, 0], parametrization_data_1D_real[:, 1])
])
def test_hookeslaw1D_stress_array_real(hookeslaw1D_real, expected, strain):
    np.testing.assert_allclose(hookeslaw1D_real.stress(strain), expected, rtol=1e-5)

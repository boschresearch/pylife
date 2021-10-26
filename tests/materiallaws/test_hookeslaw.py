import pytest
import numpy as np
import re

from pylife.materiallaws import Hookeslaw1D
from pylife.materiallaws import Hookeslaw2Dstrain
from pylife.materiallaws import Hookeslaw2Dstress
from pylife.materiallaws import Hookeslaw3D


# Hookeslaw1D
@pytest.fixture
def hookeslaw1D():
    return Hookeslaw1D(E=2.)


def test_hookeslaw1D_init(hookeslaw1D):
    assert hookeslaw1D._E == 2.


def test_hookeslaw1D_properties(hookeslaw1D):
    assert hookeslaw1D.E == hookeslaw1D._E


data1D = np.array([
    [0.0, 0.0],
    [1.0, 1./2],
    [2.0, 1.],
    [3.0, 3./2.],
    [-1.0, -1./2],
    [-2.0, -1.],
    [-3.0, -3./2.]
])


@pytest.mark.parametrize('stress, expected', map(tuple, data1D))
def test_hookeslaw1D_strain_scalar(hookeslaw1D, stress, expected):
    np.testing.assert_approx_equal(hookeslaw1D.strain(stress), expected, significant=5)


@pytest.mark.parametrize('expected, strain', map(tuple, data1D))
def test_hookeslaw1D_stress_scalar(hookeslaw1D, expected, strain):
    np.testing.assert_approx_equal(hookeslaw1D.stress(strain), expected, significant=5)


@pytest.mark.parametrize('stress, expected', [(data1D[:, 0], data1D[:, 1])])
def test_hookeslaw1D_strain_array(hookeslaw1D, stress, expected):
    np.testing.assert_allclose(hookeslaw1D.strain(stress), expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [(data1D[:, 0], data1D[:, 1])])
def test_hookeslaw1D_stress_array(hookeslaw1D, expected, strain):
    np.testing.assert_allclose(hookeslaw1D.stress(strain), expected, rtol=1e-5)


@pytest.fixture
def hookeslaw1D_real():
    return Hookeslaw1D(E=210.5e9)


def test_hookeslaw1D_real_init(hookeslaw1D_real):
    assert hookeslaw1D_real._E == 210.5e9


def test_hookeslaw1D_real_properties(hookeslaw1D_real):
    assert hookeslaw1D_real.E == hookeslaw1D_real._E


data1D_real = np.array([
    [0.0, 0.0],
    [100e6, 475.059e-6],
    [500e6, 2375.296e-6],
    [900e6, 4275.553e-6],
    [-100e6, -475.059e-6],
    [-500e6, -2375.296e-6],
    [-900e6, -4275.553e-6]
])


@pytest.mark.parametrize('stress, expected', map(tuple, data1D_real))
def test_hookeslaw1D_real_strain_scalar(hookeslaw1D_real, stress, expected):
    np.testing.assert_approx_equal(hookeslaw1D_real.strain(stress), expected, significant=5)


@pytest.mark.parametrize('expected, strain', map(tuple, data1D_real))
def test_hookeslaw1D_real_stress_scalar(hookeslaw1D_real, expected, strain):
    np.testing.assert_approx_equal(hookeslaw1D_real.stress(strain), expected, significant=5)


@pytest.mark.parametrize('stress, expected', [(data1D_real[:, 0], data1D_real[:, 1])])
def test_hookeslaw1D_real_strain_array(hookeslaw1D_real, stress, expected):
    np.testing.assert_allclose(hookeslaw1D_real.strain(stress), expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [(data1D_real[:, 0], data1D_real[:, 1])])
def test_hookeslaw1D_real_stress_array(hookeslaw1D_real, expected, strain):
    np.testing.assert_allclose(hookeslaw1D_real.stress(strain), expected, rtol=1e-5)


# Hookeslaw2Dstress
@pytest.fixture
def hookeslaw2Dstress():
    return Hookeslaw2Dstress(E=2., nu=0.1)


@pytest.mark.parametrize('nu', [-1.1, 0.6])
def test_hookeslaw2Dstress_init(nu):
    with pytest.raises(ValueError,
                       match=re.escape('Possion\'s ratio nu is %.2f but must be -1 <= nu <= 1./2.' % nu)):
        Hookeslaw2Dstress(E=2., nu=nu)


def test_hookeslaw2Dstress_wrongshapestress(hookeslaw2Dstress):
    with pytest.raises(ValueError,
                       match=re.escape('Components\' shape is not consistent.')):
        hookeslaw2Dstress.stress(e11=np.zeros((1, 2)), e22=np.zeros((1, 2)), e12=np.zeros((2, 1)))


def test_hookeslaw2Dstress_wrongshapestrain(hookeslaw2Dstress):
    with pytest.raises(ValueError,
                       match=re.escape('Components\' shape is not consistent.')):
        hookeslaw2Dstress.strain(s11=np.zeros((1, 2)), s22=np.zeros((1, 2)), s12=np.zeros((2, 1)))


def test_hookeslaw2Dstress_init(hookeslaw2Dstress):
    assert hookeslaw2Dstress._E == 2.
    assert hookeslaw2Dstress._nu == 0.1


def test_hookeslaw2Dstress_properties(hookeslaw2Dstress):
    assert hookeslaw2Dstress.E == hookeslaw2Dstress._E
    assert hookeslaw2Dstress.nu == hookeslaw2Dstress._nu
    np.testing.assert_approx_equal(hookeslaw2Dstress.G, 0.9090909, significant=5)
    np.testing.assert_approx_equal(hookeslaw2Dstress.K, 0.8333333, significant=5)


# (s11, s22, s33, s12), (e11, e22, e33, e12)
data2Dstress = np.array([
    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    [[1.0, 0.0, 0.0, 0.0], [0.5, -0.05, -0.05, 0.0]],
    [[0.0, 1.0, 0.0, 0.0], [-0.05, 0.5, -0.05, 0.0]],
    [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.1]],
    [[1.0, 2.0, 0.0, 0.5], [0.4, 0.95, -0.15, 0.55]],
    [[-0.5, 1.0, 0.0, 2.0], [-0.3, 0.525, -0.025, 2.2]],
    [[0.5, -0.5, 0.0, 1.0], [0.275, -0.275, 0.0, 1.1]]
])


@pytest.mark.parametrize('stress, expected', map(tuple, data2Dstress))
def test_hookeslaw2Dstress_strain_scalar(hookeslaw2Dstress, stress, expected):
    stress = dict(s11=stress[0], s22=stress[1], s12=stress[3])
    np.testing.assert_allclose(hookeslaw2Dstress.strain(**stress), expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', map(tuple, data2Dstress))
def test_hookeslaw2Dstress_stress_scalar(hookeslaw2Dstress, expected, strain):
    strain = dict(e11=strain[0], e22=strain[1], e12=strain[3])
    np.testing.assert_allclose(hookeslaw2Dstress.stress(**strain), expected[[0, 1, 3]], rtol=1e-5)


@pytest.mark.parametrize('stress, expected', [(data2Dstress[:, 0], data2Dstress[:, 1])])
def test_hookeslaw2Dstress_strain_array(hookeslaw2Dstress, stress, expected):
    stress = dict(s11=stress[:, 0], s22=stress[:, 1], s12=stress[:, 3])
    np.testing.assert_allclose(np.array(hookeslaw2Dstress.strain(**stress)).T, expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [(data2Dstress[:, 0], data2Dstress[:, 1])])
def test_hookeslaw2Dstress_stress_array(hookeslaw2Dstress, expected, strain):
    strain = dict(e11=strain[:, 0], e22=strain[:, 1], e12=strain[:, 3])
    np.testing.assert_allclose(np.array(hookeslaw2Dstress.stress(**strain)).T, expected[:, (0, 1, 3)], rtol=1e-5)


@pytest.fixture
def hookeslaw2Dstress_real():
    return Hookeslaw2Dstress(E=205e9, nu=0.3)


def test_hookeslaw2Dstress_real_init(hookeslaw2Dstress_real):
    assert hookeslaw2Dstress_real._E == 205e9
    assert hookeslaw2Dstress_real._nu == 0.3


def test_hookeslaw2Dstress_real_properties(hookeslaw2Dstress_real):
    assert hookeslaw2Dstress_real.E == hookeslaw2Dstress_real._E
    assert hookeslaw2Dstress_real.nu == hookeslaw2Dstress_real._nu
    np.testing.assert_approx_equal(hookeslaw2Dstress_real.G, 78846153e3, significant=5)
    np.testing.assert_approx_equal(hookeslaw2Dstress_real.K, 170838383e3, significant=5)


# (s11, s22, s33, s12), (e11, e22, e33, e12)
data2Dstress_real = np.array([
    [[100e6, 0.0, 0.0, 0.0], [1./2050, -3./20500, -3./20500, 0.0]],
    [[0.0, 100e6, 0.0, 0.0], [-3./20500, 1./2050, -3./20500, 0.0]],
    [[0.0, 0.0, 0.0, 100e6], [0.0, 0.0, 0.0, 13./10250]],
    [[100e6, 200e6, 0.0, 50e6], [1./5125, 17./20500, -9./20500, 13./20500]],
    [[-50e6, 100e6, 0.0, 200e6], [-2./5125, 23./41000, -3./41000, 13./5125]],
    [[50e6, -50e6, 0.0, 100e6], [13./41000, -13./41000, 0.0, 13./10250]]
])


@pytest.mark.parametrize('stress, expected', map(tuple, data2Dstress_real))
def test_hookeslaw2Dstress_real_strain_scalar(hookeslaw2Dstress_real, stress, expected):
    stress = dict(s11=stress[0], s22=stress[1], s12=stress[3])
    np.testing.assert_allclose(hookeslaw2Dstress_real.strain(**stress), expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', map(tuple, data2Dstress_real))
def test_hookeslaw2Dstress_real_stress_scalar(hookeslaw2Dstress_real, expected, strain):
    strain = dict(e11=strain[0], e22=strain[1], e12=strain[3])
    np.testing.assert_allclose(hookeslaw2Dstress_real.stress(**strain), expected[[0, 1, 3]], rtol=1e-5)


@pytest.mark.parametrize('stress, expected', [(data2Dstress_real[:, 0], data2Dstress_real[:, 1])])
def test_hookeslaw2Dstress_real_strain_array(hookeslaw2Dstress_real, stress, expected):
    stress = dict(s11=stress[:, 0], s22=stress[:, 1], s12=stress[:, 3])
    np.testing.assert_allclose(np.array(hookeslaw2Dstress_real.strain(**stress)).T, expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [(data2Dstress_real[:, 0], data2Dstress_real[:, 1])])
def test_hookeslaw2Dstress_real_stress_array(hookeslaw2Dstress_real, expected, strain):
    strain = dict(e11=strain[:, 0], e22=strain[:, 1], e12=strain[:, 3])
    np.testing.assert_allclose(np.array(hookeslaw2Dstress_real.stress(**strain)).T, expected[:, (0, 1, 3)], rtol=1e-5)


# Hookeslaw2Dstrain
@pytest.fixture
def hookeslaw2Dstrain():
    return Hookeslaw2Dstrain(E=2., nu=0.1)


@pytest.mark.parametrize('nu', [-1.1, 0.6])
def test_hookeslaw2Dstrain_init(nu):
    with pytest.raises(ValueError,
                       match=re.escape('Possion\'s ratio nu is %.2f but must be -1 <= nu <= 1./2.' % nu)):
        Hookeslaw2Dstrain(E=2., nu=nu)


def test_hookeslaw2Dstrain_wrongshapestress(hookeslaw2Dstrain):
    with pytest.raises(ValueError,
                       match=re.escape('Components\' shape is not consistent.')):
        hookeslaw2Dstrain.stress(e11=np.zeros((1, 2)), e22=np.zeros((1, 2)), e12=np.zeros((2, 1)))


def test_hookeslaw2Dstrain_wrongshapestrain(hookeslaw2Dstrain):
    with pytest.raises(ValueError,
                       match=re.escape('Components\' shape is not consistent.')):
        hookeslaw2Dstrain.strain(s11=np.zeros((1, 2)), s22=np.zeros((1, 2)), s12=np.zeros((2, 1)))


def test_hookeslaw2Dstrain_init(hookeslaw2Dstrain):
    assert hookeslaw2Dstrain._E == 2.
    assert hookeslaw2Dstrain._nu == 0.1


def test_hookeslaw2Dstrain_properties(hookeslaw2Dstrain):
    assert hookeslaw2Dstrain.E == hookeslaw2Dstrain._E
    assert hookeslaw2Dstrain.nu == hookeslaw2Dstrain._nu
    np.testing.assert_approx_equal(hookeslaw2Dstrain.G, 0.9090909, significant=5)
    np.testing.assert_approx_equal(hookeslaw2Dstrain.K, 0.8333333, significant=5)


# (s11, s22, s33, s12), (e11, e22, e33, e12)
data2Dstrain = np.array([
    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    [[1.0, 0.0, 0.1, 0.0], [99./200, -11./200, 0.0, 0.0]],
    [[0.0, 1.0, 0.1, 0.0], [-11./200, 99./200, 0.0, 0.0]],
    [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.1]],
    [[1.0, 2.0, 0.3, 0.5], [77./200, 187./200, 0.0, 0.55]],
    [[-0.5, 1.0, 0.05, 2.0], [-121./400, 209./400, 0.0, 2.2]],
    [[0.5, -0.5, 0.0, 1.0], [11./40, -11./40, 0.0, 1.1]]
])


@pytest.mark.parametrize('stress, expected', map(tuple, data2Dstrain))
def test_hookeslaw2Dstrain_strain_scalar(hookeslaw2Dstrain, stress, expected):
    stress = dict(s11=stress[0], s22=stress[1], s12=stress[3])
    np.testing.assert_allclose(hookeslaw2Dstrain.strain(**stress), expected[[0, 1, 3]], rtol=1e-5)


@pytest.mark.parametrize('expected, strain', map(tuple, data2Dstrain))
def test_hookeslaw2Dstrain_stress_scalar(hookeslaw2Dstrain, expected, strain):
    strain = dict(e11=strain[0], e22=strain[1], e12=strain[3])
    np.testing.assert_allclose(hookeslaw2Dstrain.stress(**strain), expected, rtol=1e-5)


@pytest.mark.parametrize('stress, expected', [(data2Dstrain[:, 0], data2Dstrain[:, 1])])
def test_hookeslaw2Dstrain_strain_array(hookeslaw2Dstrain, stress, expected):
    stress = dict(s11=stress[:, 0], s22=stress[:, 1], s12=stress[:, 3])
    np.testing.assert_allclose(np.array(hookeslaw2Dstrain.strain(**stress)).T, expected[:, (0, 1, 3)], rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [(data2Dstrain[:, 0], data2Dstrain[:, 1])])
def test_hookeslaw2Dstrain_stress_array(hookeslaw2Dstrain, expected, strain):
    strain = dict(e11=strain[:, 0], e22=strain[:, 1], e12=strain[:, 3])
    np.testing.assert_allclose(np.array(hookeslaw2Dstrain.stress(**strain)).T, expected, rtol=1e-5)


@pytest.fixture
def hookeslaw2Dstrain_real():
    return Hookeslaw2Dstrain(E=205e9, nu=0.3)


def test_hookeslaw2Dstrain_real_init(hookeslaw2Dstrain_real):
    assert hookeslaw2Dstrain_real._E == 205e9
    assert hookeslaw2Dstrain_real._nu == 0.3


def test_hookeslaw2Dstrain_real_properties(hookeslaw2Dstrain_real):
    assert hookeslaw2Dstrain_real.E == hookeslaw2Dstrain_real._E
    assert hookeslaw2Dstrain_real.nu == hookeslaw2Dstrain_real._nu
    np.testing.assert_approx_equal(hookeslaw2Dstrain_real.G, 78846153e3, significant=5)
    np.testing.assert_approx_equal(hookeslaw2Dstrain_real.K, 170838383e3, significant=5)


# (s11, s22, s33, s12), (e11, e22, e33, e12)
data2Dstrain_real = np.array([
    [[100e6, 0.0, 30.0e6, 0.0], [91./205000, -39./205000, 0.0, 0.0]],
    [[0.0, 100e6, 30.0e6, 0.0], [-39./205000, 91./205000, 0.0, 0.0]],
    [[0.0, 0.0, 0.0, 100e6], [0.0, 0.0, 0.0, 13./10250]],
    [[100e6, 200e6, 90.0e6, 50e6], [13./205000, 143./205000, 0.0, 13./20500]],
    [[-50e6, 100e6, 15.0e6, 200e6], [-169./410000, 221./410000, 0.0, 13./5125]],
    [[50e6, -50e6, 0.0, 100e6], [13./41000, -13./41000, 0.0, 13./10250]]
])


@pytest.mark.parametrize('stress, expected', map(tuple, data2Dstrain_real))
def test_hookeslaw2Dstrain_real_strain_scalar(hookeslaw2Dstrain_real, stress, expected):
    stress = dict(s11=stress[0], s22=stress[1], s12=stress[3])
    np.testing.assert_allclose(hookeslaw2Dstrain_real.strain(**stress), expected[[0, 1, 3]], rtol=1e-5)


@pytest.mark.parametrize('expected, strain', map(tuple, data2Dstrain_real))
def test_hookeslaw2Dstrain_real_stress_scalar(hookeslaw2Dstrain_real, expected, strain):
    strain = dict(e11=strain[0], e22=strain[1], e12=strain[3])
    np.testing.assert_allclose(hookeslaw2Dstrain_real.stress(**strain), expected, rtol=1e-5)


@pytest.mark.parametrize('stress, expected', [(data2Dstrain_real[:, 0], data2Dstrain_real[:, 1])])
def test_hookeslaw2Dstrain_real_strain_array(hookeslaw2Dstrain_real, stress, expected):
    stress = dict(s11=stress[:, 0], s22=stress[:, 1], s12=stress[:, 3])
    np.testing.assert_allclose(np.array(hookeslaw2Dstrain_real.strain(**stress)).T, expected[:, (0, 1, 3)], rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [(data2Dstrain_real[:, 0], data2Dstrain_real[:, 1])])
def test_hookeslaw2Dstrain_real_stress_array(hookeslaw2Dstrain_real, expected, strain):
    strain = dict(e11=strain[:, 0], e22=strain[:, 1], e12=strain[:, 3])
    np.testing.assert_allclose(np.array(hookeslaw2Dstrain_real.stress(**strain)).T, expected, rtol=1e-5)


# Hookeslaw3D
@pytest.fixture
def hookeslaw3D():
    return Hookeslaw3D(E=2., nu=0.1)


@pytest.mark.parametrize('nu', [-1.1, 0.6])
def test_hookeslaw3D_init(nu):
    with pytest.raises(ValueError,
                       match=re.escape('Possion\'s ratio nu is %.2f but must be -1 <= nu <= 1./2.' % nu)):
        Hookeslaw2Dstrain(E=2., nu=nu)


def test_hookeslaw3D_wrongshapestress(hookeslaw3D):
    with pytest.raises(ValueError,
                       match=re.escape('Components\' shape is not consistent.')):
        hookeslaw3D.stress(
            e11=np.zeros((1, 2)), e22=np.zeros((1, 2)), e33=np.zeros((2, 1)),
            e12=np.zeros((1, 2)), e13=np.zeros((1, 2)), e23=np.zeros((1, 1)))


def test_hookeslaw3D_wrongshapestrain(hookeslaw3D):
    with pytest.raises(ValueError,
                       match=re.escape('Components\' shape is not consistent.')):
        hookeslaw3D.strain(
            s11=np.zeros((1, 2)), s22=np.zeros((1, 2)), s33=np.zeros((2, 1)),
            s12=np.zeros((1, 2)), s23=np.zeros((1, 1)), s13=np.zeros((2, 1)))


def test_hookeslaw3D_init(hookeslaw3D):
    assert hookeslaw3D._E == 2.
    assert hookeslaw3D._nu == 0.1


def test_hookeslaw3D_properties(hookeslaw3D):
    assert hookeslaw3D.E == hookeslaw3D._E
    assert hookeslaw3D.nu == hookeslaw3D._nu
    np.testing.assert_approx_equal(hookeslaw3D.G, 0.9090909, significant=5)
    np.testing.assert_approx_equal(hookeslaw3D.K, 0.8333333, significant=5)


# (s11, s22, s33, s12, s13, s23), (e11, e22, e33, e12, e13, e23)
data3D = np.array([
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, -0.05, -0.05, 0.0, 0.0, 0.0]],
    [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [-0.05, 0.5, -0.05, 0.0, 0.0, 0.0]],
    [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [-0.05, -0.05, 0.5, 0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.1, 0.0, 0.0]],
    [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.1, 0.0]],
    [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.1]],
    [[1.0, 2.0, 3.0, 0.5, 0.5, 0.5], [0.25, 0.8, 1.35, 0.55, 0.55, 0.55]],
    [[-3.0, 2.0, 1.0, 1., 0.5, 2.0], [-1.65, 1.1, 0.55, 1.1, 0.55, 2.2]],
    [[-1.0, 3.0, 0.5, 2.0, -1.0, 2.0], [-0.675, 1.525, 0.15, 2.2, -1.1, 2.2]],
    [[0.5, -0.5, 0.0, 3.0, 2.0, 1.0], [0.275, -0.275, 0.0, 3.3, 2.2, 1.1]]
])


@pytest.mark.parametrize('stress, expected', map(tuple, data3D))
def test_hookeslaw3D_strain_scalar(hookeslaw3D, stress, expected):
    stress = dict(s11=stress[0], s22=stress[1], s33=stress[2], s12=stress[3], s13=stress[4], s23=stress[5])
    np.testing.assert_allclose(hookeslaw3D.strain(**stress), expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', map(tuple, data3D))
def test_hookeslaw3D_stress_scalar(hookeslaw3D, expected, strain):
    strain = dict(e11=strain[0], e22=strain[1], e33=strain[2], e12=strain[3], e13=strain[4], e23=strain[5])
    np.testing.assert_allclose(hookeslaw3D.stress(**strain), expected, rtol=1e-5)


@pytest.mark.parametrize('stress, expected', [(data3D[:, 0], data3D[:, 1])])
def test_hookeslaw3D_strain_array(hookeslaw3D, stress, expected):
    stress = dict(s11=stress[:, 0], s22=stress[:, 1], s33=stress[:, 2],
                  s12=stress[:, 3], s13=stress[:, 4], s23=stress[:, 5])
    np.testing.assert_allclose(np.array(hookeslaw3D.strain(**stress)).T, expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [(data3D[:, 0], data3D[:, 1])])
def test_hookeslaw3D_stress_array(hookeslaw3D, expected, strain):
    strain = dict(e11=strain[:, 0], e22=strain[:, 1], e33=strain[:, 2],
                  e12=strain[:, 3], e13=strain[:, 4], e23=strain[:, 5])
    np.testing.assert_allclose(np.array(hookeslaw3D.stress(**strain)).T, expected, rtol=1e-5)


@pytest.fixture
def hookeslaw3D_real():
    return Hookeslaw3D(E=205e9, nu=0.3)


def test_hookeslaw3D_real_init(hookeslaw3D_real):
    assert hookeslaw3D_real._E == 205e9
    assert hookeslaw3D_real._nu == 0.3


def test_hookeslaw3D_real_properties(hookeslaw3D_real):
    assert hookeslaw3D_real.E == hookeslaw3D_real._E
    assert hookeslaw3D_real.nu == hookeslaw3D_real._nu
    np.testing.assert_approx_equal(hookeslaw3D_real.G, 78846153e3, significant=5)
    np.testing.assert_approx_equal(hookeslaw3D_real.K, 170838383e3, significant=5)


# (s11, s22, s33, s12, s13, s23), (e11, e22, e33, e12, e13, e23)
data3D_real = np.array([
    [[100e6, 0.0, 0.0, 0.0, 0.0, 0.0], [1./2050, -3./20500, -3./20500, 0.0, 0.0, 0.0]],
    [[0.0, 100e6, 0.0, 0.0, 0.0, 0.0], [-3./20500, 1./2050, -3./20500, 0.0, 0.0, 0.0]],
    [[0.0, 0.0, 100e6, 0.0, 0.0, 0.0], [-3./20500, -3./20500, 1./2050, 0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0, 100e6, 0.0, 0.0], [0.0, 0.0, 0.0, 13./10250, 0.0, 0.0]],
    [[0.0, 0.0, 0.0, 0.0, 100e6, 0.0], [0.0, 0.0, 0.0, 0.0, 13./10250, 0.0]],
    [[0.0, 0.0, 0.0, 0.0, 0.0, 100e6], [0.0, 0.0, 0.0, 0.0, 0.0, 13./10250]],
    [[100e6, 100e6, 100e6, 100e6, 100e6, 100e6], [1./5125, 1./5125, 1./5125, 13./10250, 13./10250, 13./10250]],
    [[-300e6, 200e6, 100e6, 100e6, 50e6, 200e6], [-39./20500, 13./10250, 13./20500, 13./10250, 13./20500, 13./5125]],
    [[-100e6, 300e6, 50e6, 200e6, -100e6, 200e6], [-1./1000, 63./41000, -1./20500, 13./5125, -13./10250, 13./5125]],
    [[50e6, -50e6, 0.0, 300e6, 200e6, 100e6], [13./41000, -13./41000, 0.0, 39./10250, 13./5125, 13./10250]]
])


@pytest.mark.parametrize('stress, expected', map(tuple, data3D_real))
def test_hookeslaw3D_real_strain_scalar(hookeslaw3D_real, stress, expected):
    stress = dict(s11=stress[0], s22=stress[1], s33=stress[2], s12=stress[3], s13=stress[4], s23=stress[5])
    np.testing.assert_allclose(hookeslaw3D_real.strain(**stress), expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', map(tuple, data3D_real))
def test_hookeslaw3D_real_stress_scalar(hookeslaw3D_real, expected, strain):
    strain = dict(e11=strain[0], e22=strain[1], e33=strain[2], e12=strain[3], e13=strain[4], e23=strain[5])
    np.testing.assert_allclose(hookeslaw3D_real.stress(**strain), expected, rtol=1e-5)


@pytest.mark.parametrize('stress, expected', [(data3D_real[:, 0], data3D_real[:, 1])])
def test_hookeslaw3D_real_strain_array(hookeslaw3D_real, stress, expected):
    stress = dict(s11=stress[:, 0], s22=stress[:, 1], s33=stress[:, 2],
                  s12=stress[:, 3], s13=stress[:, 4], s23=stress[:, 5])
    np.testing.assert_allclose(np.array(hookeslaw3D_real.strain(**stress)).T, expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [(data3D_real[:, 0], data3D_real[:, 1])])
def test_hookeslaw3D_real_stress_array(hookeslaw3D_real, expected, strain):
    strain = dict(e11=strain[:, 0], e22=strain[:, 1], e33=strain[:, 2],
                  e12=strain[:, 3], e13=strain[:, 4], e23=strain[:, 5])
    np.testing.assert_allclose(np.array(hookeslaw3D_real.stress(**strain)).T, expected, rtol=1e-5)

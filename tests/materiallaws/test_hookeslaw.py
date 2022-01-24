import pytest
import numpy as np
import re

from pylife.materiallaws import HookesLaw1d
from pylife.materiallaws import HookesLaw2dPlaneStrain
from pylife.materiallaws import HookesLaw2dPlaneStress
from pylife.materiallaws import HookesLaw3d


# HookesLaw1d
@pytest.fixture
def hookeslaw1d():
    return HookesLaw1d(E=2.)


def test_hookeslaw1d_init(hookeslaw1d):
    assert hookeslaw1d._E == 2.


def test_hookeslaw1d_properties(hookeslaw1d):
    assert hookeslaw1d.E == hookeslaw1d._E


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
def test_hookeslaw1d_strain_scalar(hookeslaw1d, stress, expected):
    np.testing.assert_approx_equal(hookeslaw1d.strain(stress), expected, significant=5)


@pytest.mark.parametrize('expected, strain', map(tuple, data1D))
def test_hookeslaw1d_stress_scalar(hookeslaw1d, expected, strain):
    np.testing.assert_approx_equal(hookeslaw1d.stress(strain), expected, significant=5)


@pytest.mark.parametrize('stress, expected', [(data1D[:, 0], data1D[:, 1])])
def test_hookeslaw1d_strain_array(hookeslaw1d, stress, expected):
    np.testing.assert_allclose(hookeslaw1d.strain(stress), expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [(data1D[:, 0], data1D[:, 1])])
def test_hookeslaw1d_stress_array(hookeslaw1d, expected, strain):
    np.testing.assert_allclose(hookeslaw1d.stress(strain), expected, rtol=1e-5)


@pytest.fixture
def hookeslaw1d_real():
    return HookesLaw1d(E=210.5e9)


def test_hookeslaw1d_real_init(hookeslaw1d_real):
    assert hookeslaw1d_real._E == 210.5e9


def test_hookeslaw1d_real_properties(hookeslaw1d_real):
    assert hookeslaw1d_real.E == hookeslaw1d_real._E


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
def test_hookeslaw1d_real_strain_scalar(hookeslaw1d_real, stress, expected):
    np.testing.assert_approx_equal(hookeslaw1d_real.strain(stress), expected, significant=5)


@pytest.mark.parametrize('expected, strain', map(tuple, data1D_real))
def test_hookeslaw1d_real_stress_scalar(hookeslaw1d_real, expected, strain):
    np.testing.assert_approx_equal(hookeslaw1d_real.stress(strain), expected, significant=5)


@pytest.mark.parametrize('stress, expected', [(data1D_real[:, 0], data1D_real[:, 1])])
def test_hookeslaw1d_real_strain_array(hookeslaw1d_real, stress, expected):
    np.testing.assert_allclose(hookeslaw1d_real.strain(stress), expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [(data1D_real[:, 0], data1D_real[:, 1])])
def test_hookeslaw1d_real_stress_array(hookeslaw1d_real, expected, strain):
    np.testing.assert_allclose(hookeslaw1d_real.stress(strain), expected, rtol=1e-5)


# HookesLaw2dPlaneStress
@pytest.fixture
def hookeslaw2dplainstress():
    return HookesLaw2dPlaneStress(E=2., nu=0.1)


@pytest.mark.parametrize('nu', [-1.1, 0.6])
def test_hookeslaw2dplainstress_init(nu):
    with pytest.raises(ValueError,
                       match=re.escape('Possion\'s ratio nu is %.2f but must be -1 <= nu <= 1./2.' % nu)):
        HookesLaw2dPlaneStress(E=2., nu=nu)


def test_hookeslaw2dplainstress_wrongshapestress(hookeslaw2dplainstress):
    with pytest.raises(ValueError,
                       match=re.escape('Components\' shape is not consistent.')):
        hookeslaw2dplainstress.stress(e11=np.zeros((1, 2)), e22=np.zeros((1, 2)), g12=np.zeros((2, 1)))


def test_hookeslaw2dplainstress_wrongshapestrain(hookeslaw2dplainstress):
    with pytest.raises(ValueError,
                       match=re.escape('Components\' shape is not consistent.')):
        hookeslaw2dplainstress.strain(s11=np.zeros((1, 2)), s22=np.zeros((1, 2)), s12=np.zeros((2, 1)))


def test_hookeslaw2dplainstress_init(hookeslaw2dplainstress):
    assert hookeslaw2dplainstress._E == 2.
    assert hookeslaw2dplainstress._nu == 0.1


def test_hookeslaw2dplainstress_properties(hookeslaw2dplainstress):
    assert hookeslaw2dplainstress.E == hookeslaw2dplainstress._E
    assert hookeslaw2dplainstress.nu == hookeslaw2dplainstress._nu
    np.testing.assert_approx_equal(hookeslaw2dplainstress.G, 0.9090909, significant=5)
    np.testing.assert_approx_equal(hookeslaw2dplainstress.K, 0.8333333, significant=5)


# (s11, s22, s33, s12), (e11, e22, e33, g12)
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
def test_hookeslaw2dplainstress_strain_scalar(hookeslaw2dplainstress, stress, expected):
    stress = dict(s11=stress[0], s22=stress[1], s12=stress[3])
    np.testing.assert_allclose(hookeslaw2dplainstress.strain(**stress), expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', map(tuple, data2Dstress))
def test_hookeslaw2dplainstress_stress_scalar(hookeslaw2dplainstress, expected, strain):
    strain = dict(e11=strain[0], e22=strain[1], g12=strain[3])
    np.testing.assert_allclose(hookeslaw2dplainstress.stress(**strain), expected[[0, 1, 3]], rtol=1e-5)


@pytest.mark.parametrize('stress, expected', [(data2Dstress[:, 0], data2Dstress[:, 1])])
def test_hookeslaw2dplainstress_strain_array(hookeslaw2dplainstress, stress, expected):
    stress = dict(s11=stress[:, 0], s22=stress[:, 1], s12=stress[:, 3])
    np.testing.assert_allclose(np.array(hookeslaw2dplainstress.strain(**stress)).T, expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [(data2Dstress[:, 0], data2Dstress[:, 1])])
def test_hookeslaw2dplainstress_stress_array(hookeslaw2dplainstress, expected, strain):
    strain = dict(e11=strain[:, 0], e22=strain[:, 1], g12=strain[:, 3])
    np.testing.assert_allclose(np.array(hookeslaw2dplainstress.stress(**strain)).T, expected[:, (0, 1, 3)], rtol=1e-5)


@pytest.fixture
def hookeslaw2dplainstress_real():
    return HookesLaw2dPlaneStress(E=205e9, nu=0.3)


def test_hookeslaw2dplainstress_real_init(hookeslaw2dplainstress_real):
    assert hookeslaw2dplainstress_real._E == 205e9
    assert hookeslaw2dplainstress_real._nu == 0.3


def test_hookeslaw2dplainstress_real_properties(hookeslaw2dplainstress_real):
    assert hookeslaw2dplainstress_real.E == hookeslaw2dplainstress_real._E
    assert hookeslaw2dplainstress_real.nu == hookeslaw2dplainstress_real._nu
    np.testing.assert_approx_equal(hookeslaw2dplainstress_real.G, 78846153e3, significant=5)
    np.testing.assert_approx_equal(hookeslaw2dplainstress_real.K, 170838383e3, significant=5)


# (s11, s22, s33, s12), (e11, e22, e33, g12)
data2Dstress_real = np.array([
    [[100e6, 0.0, 0.0, 0.0], [1./2050, -3./20500, -3./20500, 0.0]],
    [[0.0, 100e6, 0.0, 0.0], [-3./20500, 1./2050, -3./20500, 0.0]],
    [[0.0, 0.0, 0.0, 100e6], [0.0, 0.0, 0.0, 13./10250]],
    [[100e6, 200e6, 0.0, 50e6], [1./5125, 17./20500, -9./20500, 13./20500]],
    [[-50e6, 100e6, 0.0, 200e6], [-2./5125, 23./41000, -3./41000, 13./5125]],
    [[50e6, -50e6, 0.0, 100e6], [13./41000, -13./41000, 0.0, 13./10250]]
])


@pytest.mark.parametrize('stress, expected', map(tuple, data2Dstress_real))
def test_hookeslaw2dplainstress_real_strain_scalar(hookeslaw2dplainstress_real, stress, expected):
    stress = dict(s11=stress[0], s22=stress[1], s12=stress[3])
    np.testing.assert_allclose(hookeslaw2dplainstress_real.strain(**stress), expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', map(tuple, data2Dstress_real))
def test_hookeslaw2dplainstress_real_stress_scalar(hookeslaw2dplainstress_real, expected, strain):
    strain = dict(e11=strain[0], e22=strain[1], g12=strain[3])
    np.testing.assert_allclose(hookeslaw2dplainstress_real.stress(**strain), expected[[0, 1, 3]], rtol=1e-5)


@pytest.mark.parametrize('stress, expected', [(data2Dstress_real[:, 0], data2Dstress_real[:, 1])])
def test_hookeslaw2dplainstress_real_strain_array(hookeslaw2dplainstress_real, stress, expected):
    stress = dict(s11=stress[:, 0], s22=stress[:, 1], s12=stress[:, 3])
    np.testing.assert_allclose(np.array(hookeslaw2dplainstress_real.strain(**stress)).T, expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [(data2Dstress_real[:, 0], data2Dstress_real[:, 1])])
def test_hookeslaw2dplainstress_real_stress_array(hookeslaw2dplainstress_real, expected, strain):
    strain = dict(e11=strain[:, 0], e22=strain[:, 1], g12=strain[:, 3])
    np.testing.assert_allclose(np.array(hookeslaw2dplainstress_real.stress(**strain)).T,
                               expected[:, (0, 1, 3)], rtol=1e-5)


# HookesLaw2dPlaneStrain
@pytest.fixture
def hookeslaw2dplainstrain():
    return HookesLaw2dPlaneStrain(E=2., nu=0.1)


@pytest.mark.parametrize('nu', [-1.1, 0.6])
def test_hookeslaw2dplainstrain_init(nu):
    with pytest.raises(ValueError,
                       match=re.escape('Possion\'s ratio nu is %.2f but must be -1 <= nu <= 1./2.' % nu)):
        HookesLaw2dPlaneStrain(E=2., nu=nu)


def test_hookeslaw2dplainstrain_wrongshapestress(hookeslaw2dplainstrain):
    with pytest.raises(ValueError,
                       match=re.escape('Components\' shape is not consistent.')):
        hookeslaw2dplainstrain.stress(e11=np.zeros((1, 2)), e22=np.zeros((1, 2)), g12=np.zeros((2, 1)))


def test_hookeslaw2dplainstrain_wrongshapestrain(hookeslaw2dplainstrain):
    with pytest.raises(ValueError,
                       match=re.escape('Components\' shape is not consistent.')):
        hookeslaw2dplainstrain.strain(s11=np.zeros((1, 2)), s22=np.zeros((1, 2)), s12=np.zeros((2, 1)))


def test_hookeslaw2dplainstrain_init(hookeslaw2dplainstrain):
    assert hookeslaw2dplainstrain._E == 2.
    assert hookeslaw2dplainstrain._nu == 0.1


def test_hookeslaw2dplainstrain_properties(hookeslaw2dplainstrain):
    assert hookeslaw2dplainstrain.E == hookeslaw2dplainstrain._E
    assert hookeslaw2dplainstrain.nu == hookeslaw2dplainstrain._nu
    np.testing.assert_approx_equal(hookeslaw2dplainstrain.G, 0.9090909, significant=5)
    np.testing.assert_approx_equal(hookeslaw2dplainstrain.K, 0.8333333, significant=5)


# (s11, s22, s33, s12), (e11, e22, e33, g12)
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
def test_hookeslaw2dplainstrain_strain_scalar(hookeslaw2dplainstrain, stress, expected):
    stress = dict(s11=stress[0], s22=stress[1], s12=stress[3])
    np.testing.assert_allclose(hookeslaw2dplainstrain.strain(**stress), expected[[0, 1, 3]], rtol=1e-5)


@pytest.mark.parametrize('expected, strain', map(tuple, data2Dstrain))
def test_hookeslaw2dplainstrain_stress_scalar(hookeslaw2dplainstrain, expected, strain):
    strain = dict(e11=strain[0], e22=strain[1], g12=strain[3])
    np.testing.assert_allclose(hookeslaw2dplainstrain.stress(**strain), expected, rtol=1e-5)


@pytest.mark.parametrize('stress, expected', [(data2Dstrain[:, 0], data2Dstrain[:, 1])])
def test_hookeslaw2dplainstrain_strain_array(hookeslaw2dplainstrain, stress, expected):
    stress = dict(s11=stress[:, 0], s22=stress[:, 1], s12=stress[:, 3])
    np.testing.assert_allclose(np.array(hookeslaw2dplainstrain.strain(**stress)).T, expected[:, (0, 1, 3)], rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [(data2Dstrain[:, 0], data2Dstrain[:, 1])])
def test_hookeslaw2dplainstrain_stress_array(hookeslaw2dplainstrain, expected, strain):
    strain = dict(e11=strain[:, 0], e22=strain[:, 1], g12=strain[:, 3])
    np.testing.assert_allclose(np.array(hookeslaw2dplainstrain.stress(**strain)).T, expected, rtol=1e-5)


@pytest.fixture
def hookeslaw2dplainstrain_real():
    return HookesLaw2dPlaneStrain(E=205e9, nu=0.3)


def test_hookeslaw2dplainstrain_real_init(hookeslaw2dplainstrain_real):
    assert hookeslaw2dplainstrain_real._E == 205e9
    assert hookeslaw2dplainstrain_real._nu == 0.3


def test_hookeslaw2dplainstrain_real_properties(hookeslaw2dplainstrain_real):
    assert hookeslaw2dplainstrain_real.E == hookeslaw2dplainstrain_real._E
    assert hookeslaw2dplainstrain_real.nu == hookeslaw2dplainstrain_real._nu
    np.testing.assert_approx_equal(hookeslaw2dplainstrain_real.G, 78846153e3, significant=5)
    np.testing.assert_approx_equal(hookeslaw2dplainstrain_real.K, 170838383e3, significant=5)


# (s11, s22, s33, s12), (e11, e22, e33, g12)
data2Dstrain_real = np.array([
    [[100e6, 0.0, 30.0e6, 0.0], [91./205000, -39./205000, 0.0, 0.0]],
    [[0.0, 100e6, 30.0e6, 0.0], [-39./205000, 91./205000, 0.0, 0.0]],
    [[0.0, 0.0, 0.0, 100e6], [0.0, 0.0, 0.0, 13./10250]],
    [[100e6, 200e6, 90.0e6, 50e6], [13./205000, 143./205000, 0.0, 13./20500]],
    [[-50e6, 100e6, 15.0e6, 200e6], [-169./410000, 221./410000, 0.0, 13./5125]],
    [[50e6, -50e6, 0.0, 100e6], [13./41000, -13./41000, 0.0, 13./10250]]
])


@pytest.mark.parametrize('stress, expected', map(tuple, data2Dstrain_real))
def test_hookeslaw2dplainstrain_real_strain_scalar(hookeslaw2dplainstrain_real, stress, expected):
    stress = dict(s11=stress[0], s22=stress[1], s12=stress[3])
    np.testing.assert_allclose(hookeslaw2dplainstrain_real.strain(**stress), expected[[0, 1, 3]], rtol=1e-5)


@pytest.mark.parametrize('expected, strain', map(tuple, data2Dstrain_real))
def test_hookeslaw2dplainstrain_real_stress_scalar(hookeslaw2dplainstrain_real, expected, strain):
    strain = dict(e11=strain[0], e22=strain[1], g12=strain[3])
    np.testing.assert_allclose(hookeslaw2dplainstrain_real.stress(**strain), expected, rtol=1e-5)


@pytest.mark.parametrize('stress, expected', [(data2Dstrain_real[:, 0], data2Dstrain_real[:, 1])])
def test_hookeslaw2dplainstrain_real_strain_array(hookeslaw2dplainstrain_real, stress, expected):
    stress = dict(s11=stress[:, 0], s22=stress[:, 1], s12=stress[:, 3])
    np.testing.assert_allclose(np.array(hookeslaw2dplainstrain_real.strain(**stress)).T,
                               expected[:, (0, 1, 3)], rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [(data2Dstrain_real[:, 0], data2Dstrain_real[:, 1])])
def test_hookeslaw2dplainstrain_real_stress_array(hookeslaw2dplainstrain_real, expected, strain):
    strain = dict(e11=strain[:, 0], e22=strain[:, 1], g12=strain[:, 3])
    np.testing.assert_allclose(np.array(hookeslaw2dplainstrain_real.stress(**strain)).T, expected, rtol=1e-5)


# HookesLaw3d
@pytest.fixture
def hookeslaw3d():
    return HookesLaw3d(E=2., nu=0.1)


@pytest.mark.parametrize('nu', [-1.1, 0.6])
def test_hookeslaw3d_init(nu):
    with pytest.raises(ValueError,
                       match=re.escape('Possion\'s ratio nu is %.2f but must be -1 <= nu <= 1./2.' % nu)):
        HookesLaw2dPlaneStrain(E=2., nu=nu)


def test_hookeslaw3d_wrongshapestress(hookeslaw3d):
    with pytest.raises(ValueError,
                       match=re.escape('Components\' shape is not consistent.')):
        hookeslaw3d.stress(
            e11=np.zeros((1, 2)), e22=np.zeros((1, 2)), e33=np.zeros((2, 1)),
            g12=np.zeros((1, 2)), g13=np.zeros((1, 2)), g23=np.zeros((1, 1)))


def test_hookeslaw3d_wrongshapestrain(hookeslaw3d):
    with pytest.raises(ValueError,
                       match=re.escape('Components\' shape is not consistent.')):
        hookeslaw3d.strain(
            s11=np.zeros((1, 2)), s22=np.zeros((1, 2)), s33=np.zeros((2, 1)),
            s12=np.zeros((1, 2)), s23=np.zeros((1, 1)), s13=np.zeros((2, 1)))


def test_hookeslaw3d_init(hookeslaw3d):
    assert hookeslaw3d._E == 2.
    assert hookeslaw3d._nu == 0.1


def test_hookeslaw3d_properties(hookeslaw3d):
    assert hookeslaw3d.E == hookeslaw3d._E
    assert hookeslaw3d.nu == hookeslaw3d._nu
    np.testing.assert_approx_equal(hookeslaw3d.G, 0.9090909, significant=5)
    np.testing.assert_approx_equal(hookeslaw3d.K, 0.8333333, significant=5)


# (s11, s22, s33, s12, s13, s23), (e11, e22, e33, g12, g13, g23)
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
def test_hookeslaw3d_strain_scalar(hookeslaw3d, stress, expected):
    stress = dict(s11=stress[0], s22=stress[1], s33=stress[2], s12=stress[3], s13=stress[4], s23=stress[5])
    np.testing.assert_allclose(hookeslaw3d.strain(**stress), expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', map(tuple, data3D))
def test_hookeslaw3d_stress_scalar(hookeslaw3d, expected, strain):
    strain = dict(e11=strain[0], e22=strain[1], e33=strain[2], g12=strain[3], g13=strain[4], g23=strain[5])
    np.testing.assert_allclose(hookeslaw3d.stress(**strain), expected, rtol=1e-5)


@pytest.mark.parametrize('stress, expected', [(data3D[:, 0], data3D[:, 1])])
def test_hookeslaw3d_strain_array(hookeslaw3d, stress, expected):
    stress = dict(s11=stress[:, 0], s22=stress[:, 1], s33=stress[:, 2],
                  s12=stress[:, 3], s13=stress[:, 4], s23=stress[:, 5])
    np.testing.assert_allclose(np.array(hookeslaw3d.strain(**stress)).T, expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [(data3D[:, 0], data3D[:, 1])])
def test_hookeslaw3d_stress_array(hookeslaw3d, expected, strain):
    strain = dict(e11=strain[:, 0], e22=strain[:, 1], e33=strain[:, 2],
                  g12=strain[:, 3], g13=strain[:, 4], g23=strain[:, 5])
    np.testing.assert_allclose(np.array(hookeslaw3d.stress(**strain)).T, expected, rtol=1e-5)


@pytest.fixture
def hookeslaw3d_real():
    return HookesLaw3d(E=205e9, nu=0.3)


def test_hookeslaw3d_real_init(hookeslaw3d_real):
    assert hookeslaw3d_real._E == 205e9
    assert hookeslaw3d_real._nu == 0.3


def test_hookeslaw3d_real_properties(hookeslaw3d_real):
    assert hookeslaw3d_real.E == hookeslaw3d_real._E
    assert hookeslaw3d_real.nu == hookeslaw3d_real._nu
    np.testing.assert_approx_equal(hookeslaw3d_real.G, 78846153e3, significant=5)
    np.testing.assert_approx_equal(hookeslaw3d_real.K, 170838383e3, significant=5)


# (s11, s22, s33, s12, s13, s23), (e11, e22, e33, g12, g13, g23)
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
def test_hookeslaw3d_real_strain_scalar(hookeslaw3d_real, stress, expected):
    stress = dict(s11=stress[0], s22=stress[1], s33=stress[2], s12=stress[3], s13=stress[4], s23=stress[5])
    np.testing.assert_allclose(hookeslaw3d_real.strain(**stress), expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', map(tuple, data3D_real))
def test_hookeslaw3d_real_stress_scalar(hookeslaw3d_real, expected, strain):
    strain = dict(e11=strain[0], e22=strain[1], e33=strain[2], g12=strain[3], g13=strain[4], g23=strain[5])
    np.testing.assert_allclose(hookeslaw3d_real.stress(**strain), expected, rtol=1e-5)


@pytest.mark.parametrize('stress, expected', [(data3D_real[:, 0], data3D_real[:, 1])])
def test_hookeslaw3d_real_strain_array(hookeslaw3d_real, stress, expected):
    stress = dict(s11=stress[:, 0], s22=stress[:, 1], s33=stress[:, 2],
                  s12=stress[:, 3], s13=stress[:, 4], s23=stress[:, 5])
    np.testing.assert_allclose(np.array(hookeslaw3d_real.strain(**stress)).T, expected, rtol=1e-5)


@pytest.mark.parametrize('expected, strain', [(data3D_real[:, 0], data3D_real[:, 1])])
def test_hookeslaw3d_real_stress_array(hookeslaw3d_real, expected, strain):
    strain = dict(e11=strain[:, 0], e22=strain[:, 1], e33=strain[:, 2],
                  g12=strain[:, 3], g13=strain[:, 4], g23=strain[:, 5])
    np.testing.assert_allclose(np.array(hookeslaw3d_real.stress(**strain)).T, expected, rtol=1e-5)

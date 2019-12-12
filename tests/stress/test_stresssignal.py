
import pytest
import pandas as pd
import numpy as np

import pylife.stress.stresssignal


def test_voigt():
    df = pd.DataFrame({'S11': [1.0], 'S22': [1.0], 'S33': [1.0],
                       'S12': [1.0], 'S13': [1.0], 'S23': [1.0]})
    df.voigt


def test_voigt_fail():
    df = pd.DataFrame({'S11': [1.0], 'S22': [1.0], 'S33': [1.0],
                       'S12': [1.0], 'S31': [1.0], 'S23': [1.0]})
    with pytest.raises(AttributeError, match=r'^StressTensorVoigtAccessor.*Missing S13'):
        df.voigt


def test_cyclic_sigma_a_sigma_m():
    df = pd.DataFrame({'sigma_a': [1.0], 'sigma_m': [1.0]})
    df.cyclic_stress


def test_cyclic_only_sigma_a():
    df = pd.DataFrame({'sigma_a': [1.0]})
    df.cyclic_stress
    np.testing.assert_equal(df['sigma_m'].to_numpy(), np.zeros_like(df['sigma_a'].to_numpy()))


def test_cyclic_sigma_a_R():
    df = pd.DataFrame({'sigma_a': [1.0, 1.0], 'R': [0.0, -1.0]})
    df.cyclic_stress
    np.testing.assert_equal(df['sigma_m'].to_numpy(), [1.0, 0.0])


@pytest.mark.parametrize('R, sigma_m_check', [
    (0.0, 1.0),
    (-1.0, 0.0)
])
def test_cyclic_sigma_constant_R(R, sigma_m_check):
    df = pd.DataFrame({'sigma_a': [1.0], 'R': [R]})
    df.cyclic_stress
    np.testing.assert_equal(df['sigma_m'].to_numpy(), sigma_m_check)


def test_cyclic_no_sigma_a():
    df = pd.DataFrame({'sigma_m': [0.0]})
    with pytest.raises(AttributeError):
        df.cyclic_stress

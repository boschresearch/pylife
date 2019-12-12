

import numpy as np
import numpy.random as rnd
from scipy.stats import norm

import pylife.strength.failure_probability as FP

import numpy.testing as testing

def test_simple_load():
    pf = FP.FailureProbability(100., 5.)
    r = pf.pf_simple_load(100.)
    assert r == 0.5

def test_norm_load():
    pf = FP.FailureProbability(100., 5.)
    r = pf.pf_norm_load(100., 1e-32)
    testing.assert_almost_equal(r, 0.5)

def test_arbitrary_load():
    strength_median = 100.
    strength_std = 5.

    load_median = 70.
    load_std = 13.

    pf = FP.FailureProbability(strength_median, strength_std)
    expected = pf.pf_norm_load(load_median, load_std)

    lower = np.log10(load_median) - 13*load_std
    upper = np.log10(load_median) + 7*load_std
    load = lower + np.sort(rnd.rand(1500)*(upper-lower))

    pdf = norm.pdf(load, loc=np.log10(load_median), scale=load_std)

    r = pf.pf_arbitrary_load(load, pdf)

    testing.assert_almost_equal(r, expected, decimal=3)

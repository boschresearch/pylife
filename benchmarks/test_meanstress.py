import numpy as np
from pylife.strength import meanstress


def test_meanstress(benchmarker_meanstress):

    amplitude = np.random.uniform(0.1,100,100000)
    mean_value = np.random.uniform(-100,-0.1,100000)
    R_goal = -1
    M = 0.5
    elapsed1, elapsed2 = benchmarker_meanstress(mean_value, amplitude,R_goal, M, meanstress.fkm_goodman, meanstress.fkm_goodman_cython)

    assert (elapsed2 < elapsed1 / 10
   ), f"Cython implementation must be at least 10x faster. Python: {elapsed1:.4f}s, Cython: {elapsed2:.4f}s (speedup: {elapsed1/elapsed2:.2f}x)"

import pytest

import time
import numpy as np


@pytest.fixture
def benchmarker():

    def the_benchmarker(rainflow_counter):
        load_signal = np.loadtxt('load_signal.csv')

        tic = time.perf_counter()

        rainflow_counter.process(load_signal)

        toc = time.perf_counter()
        elapsed = toc - tic

        classname = rainflow_counter.__class__.__name__
        print(f"Processing {classname} took {elapsed:0.4f} seconds")

        return elapsed

    return the_benchmarker

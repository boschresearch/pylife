import pytest
import time
import numpy as np


@pytest.fixture
def benchmarker_fkm():
    def the_benchmarker_fkm(
        series,
        df,
        calc_input_parameters_material,
        calc_input_parameters_stress,
        fatigue_limit_local,
    ):

        # Process whole Fkm toolchain and measure the time needed
        tic = time.perf_counter()
        assessment_parameters = calc_input_parameters_material(series, df)
        assessment_parameters = calc_input_parameters_stress(series, assessment_parameters)
        assessment_parameters = fatigue_limit_local(assessment_parameters)

        toc = time.perf_counter()
        elapsed = toc - tic
        print(f"Processing Fkm-linear took {elapsed:0.4f} seconds")

        return elapsed

    return the_benchmarker_fkm


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
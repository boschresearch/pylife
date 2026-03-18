import numpy as np
import pandas as pd

from pylife.strength.fkm_linear.fkm_linear_factors import (
    calc_input_parameters_material,
    calc_input_parameters_stress,
    fatigue_limit_local_chap4,
    fatigue_limit_local_chap5,
)


def test_fkm_linear_chap4(benchmarker_fkm):

    experiment_settings = pd.Series(
        {
            "fkm_chapter": "chap4",
            "Profile": "Rod",
            "Diameter": 6.0,
            "sup_method": "Stieler",
            "Condition": None,
            "MatGroupFKM": "GJL",
            "MatGroupFKM_Temp": "GJL",
        },
    )
    # Initialize data frame
    ap = pd.DataFrame(
        {
            "Rm": 210.0,
            "G0": 0.5,
            "amplitude": 1.0,
            "meanstress": 0.0,
            "Temperature": 50.0,
            "GJL_Mat": "GJL-250",
            "Kf_method": "Equation",
            "Rz": 200,
            "S_Type": "normal",
            "Finish": None,
            "HardProc": "None",
        },
        index=pd.Index(np.arange(4e4, dtype=np.int64), name="node_id"),
    )

    ap.loc[(4e4 / 2) :, "Temperature"] = 140.0
    ap.loc[(4e4 / 2) :, "Kf_method"] = "Table"

    benchmark_time = 0.3  # seconds
    elapsed = benchmarker_fkm(
        experiment_settings,
        ap,
        calc_input_parameters_material,
        calc_input_parameters_stress,
        fatigue_limit_local_chap4,
    )

    assert (
        elapsed < benchmark_time
    ), f"Benchmark time of {benchmark_time} s not exceeded. Needed {elapsed:0.4f} s."


def test_fkm_linear_chap5(benchmarker_fkm):

    experiment_settings = pd.Series(
        {
            "fkm_chapter": "chap5.5",
            "Profile": "Rod",
            "Diameter": 6.0,
            "sup_method": "Stieler",
            "Condition": "Hardened",
            "MatGroupFKM": "Steel",
            "MatGroupFKM_Temp": None,
        },
    )

    ap = pd.DataFrame(
        {
            "Rm": 2392.5,
            "G0": 1.225,
            "amplitude": 1.0,
            "meanstress": 0.0,
            "Temperature": 24.0,
            "GJL_Mat": None,
            "Kf_method": "Table",
            "Rz": 10,
            "S_Type": "shear",
            "Finish": None,
            "HardProc": "Inductive hardening",
            "HV": 725.0,
            "HV_core": 230.0,
        },
        index=pd.Index(np.arange(4e4, dtype=np.int64), name="node_id"),
    )

    benchmark_time = 0.3  # seconds
    elapsed = benchmarker_fkm(
        experiment_settings,
        ap,
        calc_input_parameters_material,
        calc_input_parameters_stress,
        fatigue_limit_local_chap5,
    )

    assert (
        elapsed < benchmark_time
    ), f"Benchmark time of {benchmark_time} s not exceeded. Needed {elapsed:0.4f} s."
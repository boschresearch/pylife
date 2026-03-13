import os

import numpy as np
import pandas as pd
import pytest

from pylife.strength.fkm_linear.fkm_functions import FkmLinearFunctions as fkm_class
from pylife.strength.fkm_linear.fkm_linear_factors import (
    calc_input_parameters_material,
    calc_input_parameters_stress,
    fatigue_limit_local_chap5,
)


@pytest.fixture
def path_to_data():
    return os.path.join(os.path.dirname(__file__), "data", "FKM")


@pytest.mark.parametrize(
    "experiment_settings,assessment_parameters, size",
    [
        (
            pd.Series(
                {
                    "fkm_chapter": "chap5.5",
                    "Profile": "Rod",
                    "Diameter": 6.0,
                    "sup_method": "Stieler",
                    "Condition": "Hardened",
                    "MatGroupFKM": "Steel",
                    "MatGroupFKM_Temp": None,

                },
            ),
            pd.DataFrame(
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
                    "Profile": "Rod",
                    "Finish": None,
                    "HardProc": "Inductive hardening",
                    "HV": 725.0,
                    "HV_core": 230.0,
                },
                index=pd.Index(np.arange(4e3, dtype=np.int64), name="node_id"),
            ),
            4e3,
        )
    ],
)
def test_calc_input_parameters_chap5(experiment_settings,assessment_parameters, size):

    # Calc and test parameters related to the specific material
    assessment_parameters = calc_input_parameters_material(experiment_settings,assessment_parameters)

    np.testing.assert_equal(
        np.round(assessment_parameters["Kr"].values, 6), 0.863181 * np.ones(int(size))
    )
    np.testing.assert_equal(
        np.round(assessment_parameters["SE_trans"].values, 6),
        238.5 * np.ones(int(size)),
    )
    np.testing.assert_equal(
        np.round(assessment_parameters["SW_RS"].values, 6), 733.0 * np.ones(int(size))
    )
    np.testing.assert_equal(
        np.round(assessment_parameters["M_trans"].values, 6),
        0.09558 * np.ones(int(size)),
    )

    # Calc and test parameters related to stress and stress gradient
    assessment_parameters = calc_input_parameters_stress(experiment_settings,assessment_parameters)
    np.testing.assert_equal(
        np.round(assessment_parameters["n_RS"].values, 6), 1.061065 * np.ones(int(size))
    )

    # Compute component fatigue strength with chapter 5.5
    assessment_parameters = fatigue_limit_local_chap5(assessment_parameters)

    # Test results
    np.testing.assert_equal(
        np.round(assessment_parameters["SW_trans"].values, 6),
        349.6 * np.ones(int(size)),
    )
    np.testing.assert_equal(
        np.round(assessment_parameters["Kak_trans"].values, 6),
        0.934795 * np.ones(int(size)),
    )
    np.testing.assert_equal(
        np.round(assessment_parameters["SDFKM_trans"].values, 6),
        326.804158 * np.ones(int(size)),
    )
    np.testing.assert_equal(
        np.round(assessment_parameters["SDFKM_RS"].values, 6),
        924.890885 * np.ones(int(size)),
    )


@pytest.mark.parametrize(
    "method, expected",
    [
        (
            "Case hardening",
            pd.DataFrame(
                {
                    "Swk_RS": 599,
                    "Kak_RS": 1.149,
                    "SDFKM_RS": 689,
                    "SW_trans": 286,
                    "SE_trans": 53,
                    "Kak_trans": 0.9821,
                    "SDFKM_trans": 280,
                },
                index=[0],
            ),
        ),
        (
            "Nitriding",
            pd.DataFrame(
                {
                    "Swk_RS": 678,
                    "Kak_RS": 1.1791,
                    "SDFKM_RS": 799,
                    "SW_trans": 366,
                    "SE_trans": 81,
                    "Kak_trans": 0.9788,
                    "SDFKM_trans": 358,
                },
                index=[0],
            ),
        ),
        (
            "Carbonitriding",
            pd.DataFrame(
                {
                    "Swk_RS": 678,
                    "Kak_RS": 1.1186,
                    "SDFKM_RS": 758,
                    "SW_trans": 366,
                    "SE_trans": 53,
                    "Kak_trans": 0.986,
                    "SDFKM_trans": 360,
                },
                index=[0],
            ),
        ),
        (
            "Inductive hardening",
            pd.DataFrame(
                {
                    "Swk_RS": 720,
                    "Kak_RS": 1.2834,
                    "SDFKM_RS": 924,
                    "SW_trans": 349,
                    "SE_trans": 238,
                    "Kak_trans": 0.9348,
                    "SDFKM_trans": 326,
                },
                index=[0],
            ),
        ),
        (
            "Shot peening",
            pd.DataFrame(
                {
                    "Swk_RS": 1058,
                    "Kak_RS": 1.0797,
                    "SDFKM_RS": 1142,
                    "SW_trans": 341,
                    "SE_trans": 84,
                    "Kak_trans": 0.9764,
                    "SDFKM_trans": 333,
                },
                index=[0],
            ),
        ),
        (
            "Cold rolling",
            pd.DataFrame(
                {
                    "Swk_RS": 1058,
                    "Kak_RS": 1.0308,
                    "SDFKM_RS": 1091,
                    "SW_trans": 341,
                    "SE_trans": 54,
                    "Kak_trans": 0.9848,
                    "SDFKM_trans": 336,
                },
                index=[0],
            ),
        ),
    ],
)
def test_fatigue_limit_local_chap5(path_to_data, method, expected):

    # Read the CSV file with specified data types
    assessment_parameters = pd.read_csv(
        os.path.join(path_to_data, "example_surface_hardening.csv")
    ).replace(np.nan, None)
    assessment_parameters["HardProc"] = method

    experiment_settings = pd.Series({"fkm_chapter": "chap5.5",
                                    "Profile": "Rod",
                                    "Diameter": 6.0,
                                    "sup_method": "Stieler",
                                    "Condition": "Hardened",
                                    "MatGroupFKM": "Steel",
                                    "MatGroupFKM_Temp": None,
                                },
                            )

    assessment_parameters = calc_input_parameters_material(experiment_settings,assessment_parameters)

    assessment_parameters = calc_input_parameters_stress(experiment_settings,assessment_parameters)

    res = fatigue_limit_local_chap5(assessment_parameters)

    test_fkm = pd.DataFrame(
        {
            "Swk_RS": int(res["Swk_RS"].iloc[0]),
            "Kak_RS": np.round(res["Kak_RS"], 4),
            "SDFKM_RS": int(res["SDFKM_RS"].iloc[0]),
            "SW_trans": int(res["SW_trans"].iloc[0]),
            "SE_trans": int(res["SE_trans"].iloc[0]),
            "Kak_trans": np.round(res["Kak_trans"], 4),
            "SDFKM_trans": int(res["SDFKM_trans"].iloc[0]),
        },
        index=[0],
    )

    # Compare relevant FKM measures at surface and at core
    pd.testing.assert_frame_equal(test_fkm, expected, check_dtype=False)

    assert np.round(res["n_RS"].values, 4) == 1.0611
    assert np.round(res["Kwk_RS"].values, 4) == 1.0171


@pytest.mark.parametrize(
    "G, HV_RS, expected",
    [
        (4.0, 80.0, 2.0),
        (4.0, 800.0, 1.1),
        (100.0, 400.0, 2.0),
    ],
)
def test_support_factor_chap5(G, HV_RS, expected):
    fkm = fkm_class()
    ap = pd.DataFrame(data={"G0": G, "HV": HV_RS}, index=[0])
    assert fkm.support_chap5_cython(ap["G0"], ap["HV"]) == pytest.approx(expected)


@pytest.mark.parametrize(
    "HardProc, expected",
    [
        ("Case hardening", 0.5),
        ("Nitriding", 0.45),
        ("Carbonitriding", 0.45),
        ("Inductive hardening", 0.4),
        ("Shot peening", 0.3),
        ("Cold rolling", 0.18),
    ],
)
def test_mean_stress_sens(HardProc, expected):
    fkm = fkm_class()
    assert fkm.proc_const[HardProc]["M_RS"] == expected


@pytest.mark.parametrize(
    "MatGroupFKM, expected",
    [
        ("CaseHard_Steel", 2.0),
        ("Stainless_Steel", 2.0),
        ("Forg_Steel", 2.0),
        ("Steel", 2.0),
        ("Al_wrought", 2.0),
        ("GS", 2.0),
        ("GJS", 1.5),
        ("GJM", 1.2),
        ("Al_cast", 1.2),
        ("GJL", 1.0),
    ],
)
def test_Kf_factor_constant(MatGroupFKM, expected):
    fkm = fkm_class()
    df = pd.DataFrame(data={"MatGroupFKM": MatGroupFKM}, index=[0])
    assert fkm.kf_constant_cython(df["MatGroupFKM"]) == expected

@pytest.mark.parametrize(
    "Rm, mat, S_type, method, HV, Proc, expected",
    [
        (
            None,
            "CaseHard_Steel",
            "normal",
            "chap5.5",
            499.0,
            "Case hardening",
            608.8,
        ),
        (
            None,
            "CaseHard_Steel",
            "normal",
            "chap5.5",
            550.0,
            "Case hardening",
            610.0,
        ),
        (None, "Steel", "normal", "chap5.5", 100.0, "Shot peening", 148.5),
        (500.0, "Steel", "normal", "chap5.5", None, "Shot peening", 500 * 0.45),
        (500.0, "Steel", "shear", "chap5.5", None, "Shot peening", 500 * 0.45 * 0.577),
        (500.0, "Steel", "shear", "chap5.5", 100.0, "None", [np.nan]),
    ],
)
def test_reversed_mat_strength(Rm, mat, S_type, method, HV, Proc, expected):
    fkm = fkm_class()
    ap = pd.DataFrame(
        data={
            "MatGroupFKM": mat,
            "Rm": Rm,
            "S_Type": S_type,
            "HV": HV,
            "HardProc": Proc,
        },
        index=[0],
    )

    # Replace None with Nan for each column
    ap["Rm"] = ap["Rm"].astype(float)
    ap["HV"] = ap["HV"].astype(float)

    df_consts, df_fw_t = fkm.get_material_constants(ap["MatGroupFKM"], ap["S_Type"])
    df_proc = fkm.get_material_constants_chap5_5(ap["HardProc"])
    res = fkm.reversed_mat_strength_chap5_5_cython(
        ap["Rm"], df_consts, df_fw_t, ap["S_Type"], ap["HV"], df_proc, ap["HardProc"]
    )

    if np.all(np.isnan(res)) and np.all(np.isnan(expected)):
        print("Arrays are equal with NaN values")
    else:
        assert res == pytest.approx(expected)


# 'hard_procs': ['Case hardening', 'Nitriding', 'Carbonitriding',
# 'Inductive hardening', 'Shot peening', 'Cold rolling' ],
@pytest.mark.parametrize(
    "HV_RS, HV_core, HardProc, Rm, expected",
    [
        (500.0, 100.0, "Case hardening", None, -250.0),
        (
            600.0,
            100.0,
            "Case hardening",
            None,
            -175.0,
        ),  # Carburizing = Case hardening
        (499.0, 100.0, "Nitriding", None, -269.325),
        (501.0, 100.0, "Nitriding", None, -270.0),
        (500.0, 100.0, "Inductive hardening", None, -320.0),
        (600.0, 100.0, "Inductive hardening", None, -820.0),
        (1.0, 499.0 / 3.3, "Cold rolling", None, 0.0),
        (888.0, 1499.0 / 3.3, "Cold rolling", None, -699.3),
        (888.0, 1501.0 / 3.3, "Cold rolling", None, -700.0),
        (888.0, 1501.0 / 3.3, "Cold rolling", 500.0, -700.0),
        (888.0, None, "Cold rolling", 500.0, 0.0),
        (1.0, 1599.0 / 3.3, "Shot peening", None, -499.74),
        (888.0, 1601.0 / 3.3, "Shot peening", None, -500.0),
        (888.0, 1601.0 / 3.3, "Shot peening", 500.0, -500.0),
        (888.0, None, "Shot peening", 1000.0, -344.0),
        (888.0, None, "None", 1000.0, -344.0),
    ],
)
def test_eigenstress_RS(HV_RS, HV_core, HardProc, Rm, expected):
    fkm = fkm_class()
    ap = pd.DataFrame(
        data={"Rm": Rm, "HV": HV_RS, "HV_core": HV_core, "HardProc": HardProc},
        index=[0],
    )

    ap["Rm"] = ap["Rm"].astype(float)
    ap["HV_core"] = ap["HV_core"].astype(float)

    res = fkm.eigenstress_RS_cython(ap["Rm"], ap["HV"], ap["HV_core"], ap["HardProc"])
    if np.all(np.isnan(res)):
        print("Arrays are equal with NaN values")
    else:
        assert res == pytest.approx(expected)


@pytest.mark.parametrize(
    "Rm_trans, M, SE, Swk, sL, Rm_norm, expected",
    [
        (1000.0, 0.5, -300.0, 300.0, 0.25, 8888.0, 4 / 3),  # Transition point BG1
        (None, 0.5, -300.0, 300.0, 0.25, 1000.0, 4 / 3),  # Transition point BG1
        (1000.0, 0.5, -300.0, 300.0, 2.5, 8888.0, 2 / 3),  # Transition point BG2
        (
            1000.0,
            0.5,
            -300.0,
            300.0,
            3 + 6.75 / 3.5,
            8888.0,
            0.518518519,
        ),  # Transition point BG3
        (1000.0, 0.5, -300.0, 300.0, 1.0, 8888.0, 1.0),  # Position at R=-1
        (
            1000.0,
            0.5,
            -300.0,
            300.0,
            2.0,
            8888.0,
            3 / 4,
        ),  # Position at R=0.33333
        (1000.0, 0.5, -300.0, 300.0, 0.2, 8888.0, 4 / 3),  # Position at R=-0.6666
        (1000.0, 0.5, -300.0, 300.0, 3.0, 8888.0, 0.62962963),  # Position at R=0.5
        (
            1000.0,
            0.5,
            -300.0,
            300.0,
            3.5,
            8888.0,
            0.596491228,
        ),  # Position at R=0.555555
        (1000.0, 0.5, -300.0, 300.0, 5.0, 8888.0, 0.518518519),  # Position at R=0.66666
    ],
)
def test_meanstress_shift_chap5_5(Rm_trans, M, SE, Swk, sL, Rm_norm, expected):
    fkm = fkm_class()
    ap = pd.DataFrame(
        data={
            "Rm_trans": Rm_trans,
            "M_trans": M,
            "SE_trans": SE,
            "Swk_trans": Swk,
            "Rm": Rm_norm,
            "SmSa": sL,
        },
        index=[0],
    )
    ap["Rm_trans"] = ap["Rm_trans"].astype(float)

    assert fkm.sm_factor_chap5_cython(
        ap["Rm_trans"],
        ap["M_trans"],
        ap["SE_trans"],
        ap["Swk_trans"],
        ap["SmSa"],
        ap["Rm"],
    ) == pytest.approx(expected)


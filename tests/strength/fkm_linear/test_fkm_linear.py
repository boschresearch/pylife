import os
import sys

import numpy as np
import pandas as pd
import pytest

import pylife.strength.meanstress as MST
from pylife.strength.fkm_linear.fkm_functions import FkmLinearFunctions as fkm_class
from pylife.strength.fkm_linear.fkm_linear_factors import (
    calc_input_parameters_material,
    calc_input_parameters_stress,
    fatigue_limit_local_chap4,
)


# sm_factor for testing mean stress conversion from pylife
def _sm_factor(M, R, SmSa):
    """Calculate mean stress sensitivity factor

    Parameters:
    -----------
    M: float
            mean stress sensitivity factor
    R: float
            stress ratio
    SmSa: float
            mean stress to stress amplitude ratio
    Returns:
    --------
    Kak: float
            mean stress influence factor
    """

    if R > 1:
        return 1 / (1 - M)
    elif R >= -sys.float_info.max and R <= 0:
        return 1 / (1 + M * SmSa)
    elif R > 0 and R <= 0.5:
        return (3 + M) / ((1 + M) * (3 + M * SmSa))
    else:
        return (3 + M) / (3 * (1 + M) ** 2)


@pytest.fixture
def path_to_data():
    return os.path.join(os.path.dirname(__file__), "data", "FKM")


@pytest.mark.parametrize(
    "experiment_settings,assessment_parameters, size",
    [
        (
            pd.Series(
                {
                    "fkm_chapter": "chap4",
                    "Profile": "Rod",
                    "Diameter": 6.0,
                    "sup_method": "Stieler",
                    "Condition": None,
                    "MatGroupFKM": "GJL",
                    "MatGroupFKM_Temp": "GJL",

                },
            ),
            pd.DataFrame(
                {
                    "Rm": 210.0,
                    "G0": 0.5,
                    "amplitude": 1.0,
                    "meanstress": 0.0,
                    "Temperature": 50.0,
                    "GJL_Mat": "GJL-250",
                    "Kf_method": "Table",
                    "Rz": 200,
                    "S_Type": "normal",
                    "Finish": None,
                    "HardProc": "None",
                    "Rm_trans": None,
                },
                index=pd.Index(np.arange(4e3, dtype=np.int64), name="node_id"),
            ),
            4e3,
        )
    ],
)
def test_calc_parameters(experiment_settings,assessment_parameters, size):

    # Calc and test parameters related to the specific material
    assessment_parameters = calc_input_parameters_material(experiment_settings,assessment_parameters)

    np.testing.assert_equal(
        np.round(assessment_parameters["Rm_trans"].values, 6),
        np.nan * np.ones(int(size)),
    )
    np.testing.assert_equal(
        np.round(assessment_parameters["Kr"].values, 6), 0.913953 * np.ones(int(size))
    )
    np.testing.assert_equal(
        np.round(assessment_parameters["Knle"].values, 6), 1.05 * np.ones(int(size))
    )
    np.testing.assert_equal(
        np.round(assessment_parameters["M"].values, 6), 0.5 * np.ones(int(size))
    )

    # Calc and test parameters related to stress and stress gradient
    assessment_parameters = calc_input_parameters_stress(experiment_settings,assessment_parameters)

    np.testing.assert_equal(
        np.round(assessment_parameters["n"].values, 6), 1.682119 * np.ones(int(size))
    )

    # Compute component fatigue strength with chapter 4
    assessment_parameters = fatigue_limit_local_chap4(assessment_parameters)

    # Test results
    np.testing.assert_equal(
        np.round(assessment_parameters["Kwk"].values, 6), 0.619484 * np.ones(int(size))
    )
    np.testing.assert_equal(
        np.round(assessment_parameters["Kak"].values, 6), np.ones(int(size))
    )
    np.testing.assert_equal(
        np.round(assessment_parameters["SDFKM"].values, 6),
        115.257201 * np.ones(int(size)),
    )


@pytest.mark.parametrize(
    "experiment_settings,experiment, expected",
    [
        (
            pd.Series(
                {
                    "fkm_chapter": "chap4",
                    "Profile": "Rod",
                    "Diameter": 6.0,
                    "sup_method": "Stieler",
                    "Condition": None,
                    "MatGroupFKM": "GJL",
                    "MatGroupFKM_Temp": "GJL",

                },
            ),
            "example_6_2.csv",
            pd.DataFrame(
                {
                    "SW": 71,
                    "Kr": 0.914,
                    "n": 1.682,
                    "Kf": 1,
                    "Kv": 1,
                    "Ks": 1,
                    "Knle": 1.05,
                    "Swk": 115,
                    "Kwk": 0.619,
                    "Kak": 0.879,
                    "SDFKM": 101,
                },
                index=[0],
            ),
        ),
        (
            pd.Series(
                {
                    "fkm_chapter": "chap4",
                    "Profile": "Rod",
                    "Diameter": 150.0,
                    "sup_method": "Stieler",
                    "Condition": None,
                    "MatGroupFKM": "Steel",
                    "MatGroupFKM_Temp": "None",

                },
            ),
            "example_IMA_FKM_training.csv",
            pd.DataFrame(
                {
                    "SW": 213,
                    "Kr": 0.885,
                    "n": 1.088,
                    "Kf": 2,
                    "Kv": 1,
                    "Ks": 1,
                    "Knle": 1,
                    "Swk": 217,  # Original Value 218.
                    "Kwk": 0.979,
                    "Kak": 0.996,
                    "SDFKM": 216,
                },
                index=[0],
            ),
        ),
        (
            pd.Series(
                {
                    "fkm_chapter": "chap4",
                    "Profile": "Rod",
                    "Diameter": 22.0,
                    "sup_method": "A90",
                    "Condition": None,
                    "MatGroupFKM": "Steel",
                    "MatGroupFKM_Temp": "None",

                },
            ),
            "example_FKM_FN_chapter_4_2.csv",
            pd.DataFrame(
                {
                    "SW": 390,
                    "Kr": 0.804,
                    "n": 1.09,
                    "Kf": 2,
                    "Kv": 1,
                    "Ks": 1,
                    "Knle": 1,
                    "Swk": 379,
                    "Kwk": 1.029,
                    "Kak": 0.937,
                    "SDFKM": 354,
                    "n_st": 1.03,
                    "n_vm": 1.05,
                    "n_bm": 1.0,
                },
                index=[0],
            ),
        ),
    ],
)
def test_fatigue_limit_local_chap4(experiment_settings,experiment, path_to_data, expected):

    # Read the CSV file with specified data types
    assessment_parameters = pd.read_csv(os.path.join(path_to_data, experiment)).replace(
        np.nan, None
    )
    assessment_parameters["HardProc"] = "Nones"

    # Compute further quantities for fkm linear (e.g. roughness, kak)
    assessment_parameters = calc_input_parameters_material(experiment_settings,assessment_parameters)

    assessment_parameters = calc_input_parameters_stress(experiment_settings,assessment_parameters)

    # Execute FKM-Guideline
    res = fatigue_limit_local_chap4(assessment_parameters)

    # compare relevant FKM measues
    test_fkm = pd.DataFrame(
        {
            "SW": np.round(res["SW"]),
            "Kr": np.round(res["Kr"], 3),
            "n": np.round(res["n"], 3),
            "Kf": np.round(res["Kf"], 3),
            "Kv": np.round(res["Kv"]),
            "Ks": np.round(res["Ks"]),
            "Knle": np.round(res["Knle"], 3),
            "Swk": np.round(res["Swk"]),
            "Kwk": np.round(res["Kwk"], 3),
            "Kak": np.round(res["Kak"], 3),
            "SDFKM": int(res["SDFKM"].iloc[0]),
        },
        index=[0],
    )

    if experiment == "example_FKM_FN_chapter_4_2.csv":

        fkm_add = pd.DataFrame(
            {
                "n_st": np.round(res["n_st"], 2),
                "n_vm": np.round(res["n_vm"], 2),
                "n_bm": np.round(res["n_bm"], 2),
            },
            index=[0],
        )

        test_fkm = pd.concat([test_fkm, fkm_add], axis=1)

    pd.testing.assert_frame_equal(test_fkm, expected, check_dtype=False)


@pytest.mark.parametrize(
    "meanstress, expected",
    [
        (np.array([0.0]), 1.0),
        (np.array([1.1]), 0.657),
        (np.array([4.0]), 0.519),
        (np.array([-3.0]), 2.0),
    ],
)
def test_KAK_pylife_fkm(meanstress, expected):

    # compare mean stress conversion from pylife with sm-factor from fkm-linear
    # only difference between the two functions is in the range 0.5 < R < 1
    amplitude = np.array([1.0])
    m_sensitivity = 0.5
    sm_sa = meanstress / amplitude
    R = (-amplitude + meanstress) / (amplitude + meanstress)

    if 0.5 < R < 1:
        meanstress = np.array([3.0])

    # Compute KAK factor via pylife
    res1 = MST.fkm_goodman(
        amplitude=amplitude,
        meanstress=meanstress,
        M=m_sensitivity,
        M2=m_sensitivity / 3,
        R_goal=-1,
    )

    # Compute KAK factor via fkm linear
    res2 = _sm_factor(m_sensitivity, R, sm_sa)
    assert np.round(res2, 3) == expected
    assert np.round(1 / res1, 3) == np.round(res2, 3)


# Some expected values seem to be wrong
@pytest.mark.parametrize(
    "mat, temperature, expected",
    [
        ("GJL", 100.0, 1.0),
        ("GJL", 200.0, 0.96),
        ("GJL", 1000.0, 0.75),
        ("Stainless_Steel", 100.0, 1.0),
        ("Stainless_Steel", 200.0, 1.0),
        ("Stainless_Steel", 1000.0, 1.0),
        ("Aluminum", 50.0, 1.0),
        ("Aluminum", 60.0, 0.988),
        ("Aluminum", 200.0, 0.82),
        ("Aluminum", None, 1.0),
        ("None", None, 1.0),
    ],
)
def test_KTD_factor(mat, temperature, expected):
    fkm = fkm_class()

    ap = pd.DataFrame(
        {"MatGroupFKM": mat, "Temperature": temperature}, index=[0]
    )
    ap["Temperature"] = ap["Temperature"].astype(float)

    df_temperature = fkm.get_temperature_constants(ap["MatGroupFKM"])
    assert fkm.temperature_model(
        df_temperature, ap["Temperature"],ap["MatGroupFKM"]
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "G, S_type, Rm, expected",
    [
        (0.0, "normal", 1350.0, 1.0),
        (0.1, "normal", 2700.0, 1.01),
        (1.0, "normal", 1350.0, 1.1),
        (100.0, "normal", 1350.0, 1.0),
        (16.0, "normal", 1350.0, 1.2),
        (1.0, "shear", (1 - 0.5) * 2700 / 0.577, 1.1),
    ],
)
def test_support_factor_stieler(G, S_type, Rm, expected):
    fkm = fkm_class()
    ap = pd.DataFrame(
        {"MatGroupFKM": "Steel", "G0": G, "Rm": Rm, "S_Type": S_type}, index=[0]
    )
    df_consts, df_fw_t = fkm.get_material_constants(ap["MatGroupFKM"], ap["S_Type"])
    assert fkm.stieler_support(
        df_consts, df_fw_t, ap["S_Type"], ap["G0"], ap["Rm"]
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "mat, G0, A90, Rm, SW, expected",
    [
        ("Steel", 25.0, 500.0, 340.0, 200, [1.0, 1.1, 10 / 6.75, 10 / 6.75 * 1.1 * 1]),
        ("Steel", 25.0, 500.0, 945, 162.5, [1.0, 1.1, 1.114297, 1.114297 * 1.1 * 1]),
        ("Steel", 25.0, 500.0, 680, 194.047619, [1.0, 1.1, 1.25, 1.25 * 1.1 * 1]),
        ("Steel", 1.1, 500.0, 1200, 270.731707, [1.0, 1.05, 1.0, 1.05 * 1.0 * 1]),
        ("GS", 1.1, 500.0, 1200, 270.731707, [1.0, 1.0, 1.0, 1.0 * 1.0 * 1]),
        ("Al_cast", 25.0, 80.752791, 216, 80, [1.2, 1.0, 1.25, 1.2 * 1.0 * 1.25]),
    ],
)
def test_support_factor_mat_mech(mat, G0, A90, Rm, SW, expected):
    fkm = fkm_class()
    assert fkm.support_fkm2012_local_surf(mat, G0, A90, Rm, SW) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "mat, G0, V90_Mises, Rm, SW, expected",
    [
        ("Steel", 25.0, 1000.0, 340.0, 200, [1.0, 1.1, 10 / 6.75, 10 / 6.75 * 1.1 * 1]),
        ("Steel", 25.0, 1000.0, 945, 162.5, [1.0, 1.1, 1.114297, 1.114297 * 1.1 * 1]),
        ("Steel", 25.0, 1000.0, 680, 194.047619, [1.0, 1.1, 1.25, 1.25 * 1.1 * 1]),
        ("Steel", 1.1, 1000.0, 1200, 270.731707, [1.0, 1.05, 1.0, 1.05 * 1.0 * 1]),
        ("GS", 1.1, 1000.0, 1200, 270.731707, [1.0, 1.0, 1.0, 1.0 * 1.0 * 1]),
        ("Al_cast", 25.0, 80.752791 * 2.0, 216, 80, [1.2, 1.0, 1.25, 1.2 * 1.0 * 1.25]),
    ],
)
def test_support_factor_mat_mech_vol(mat, G0, V90_Mises, Rm, SW, expected):
    fkm = fkm_class()
    assert fkm.support_fkm2012_local_vol(mat, G0, V90_Mises, Rm, SW) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "mat, Rm, Rz, S_type, Finish_type, expected",
    [
        ("Steel", 502.37727, 100.0, "normal", None, 0.824),
        ("Steel", 502.37727, 100.0, "shear", None, 0.898448),
        ("GJL", 500.0, 100.0, "normal", None, 0.88),
        ("Al_cast", 210.291465, 10.0, "normal", "None", 0.9),
        ("Steel", 1000.0, 1.0, "normal", "Polieren", 1.0),
    ],
)
def test_roughness_factor(mat, Rm, Rz, S_type, Finish_type, expected):
    fkm = fkm_class()
    ap = pd.DataFrame(
        {
            "MatGroupFKM": mat,
            "Rz": Rz,
            "Rm": Rm,
            "S_Type": S_type,
            "Finish": Finish_type,
        },
        index=[0],
    )
    df_consts, df_fw_t = fkm.get_material_constants(ap["MatGroupFKM"], ap["S_Type"])

    assert fkm.rough_factor(
        ap["Rm"], ap["Rz"], df_consts, df_fw_t, ap["S_Type"], ap["Finish"]
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "Proc, G0, Deff, expected",
    [
        ("Case hardening", 2.1, 22.0, 2.0),
        ("Nitriding", 2.0, 22.0, 1.2),
        ("Carbonitriding", 2.1, 22.5, 1.8),
        ("Inductive hardening", 2.0, 22.5, 1.125),
        ("Shot peening", 2.1, 22.5, 1.55),
        ("Cold rolling", 2.0, 22.0, 1.2),
        ("None", 2.0, 22.0, 1.0),
    ],
)
def test_layer_factor(Proc, G0, Deff, expected):
    fkm = fkm_class()

    ap = pd.DataFrame({"HardProc": Proc, "G0": G0, "Deff": Deff}, index=[0])
    df_proc = fkm.get_material_constants_chap5_5(ap["HardProc"])
    assert fkm.surf_layer_factor(
        df_proc, ap["G0"], ap["Deff"], ap["HardProc"]
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "GJL_Mat, expected",
    [
        ("GJL-100", 1.075),
        ("GJL-150", 1.075),
        ("GJL-200", 1.05),
        ("GJL-250", 1.05),
        ("GJL-300", 1.025),
        ("GJL-350", 1.025),
        ("Something else", 1.0),
    ],
)
def test_GJL_bending_factor(GJL_Mat, expected):
    fkm = fkm_class()
    ap = pd.DataFrame({"GJL_Mat": GJL_Mat}, index=[0])
    assert fkm.GJL_bending_factor(ap["GJL_Mat"]) == pytest.approx(expected)


@pytest.mark.parametrize(
    "G0, b, stress, n, expected",
    [
        (0.0, 1.0, "normal", 1.5, 1.0),
        (2.773533, 10.0, "normal", 1.5, 2.0),
        (2.577495, 10.0, "shear", 1.5, 2.5),
        (0.25, 5.0, "normal", 1.2, 1.0),
    ],
)
def test_Kf_factor(G0, b, stress, n, expected):
    fkm = fkm_class()
    ap = pd.DataFrame({"G0": G0, "b": b, "S_Type": stress, "n": n}, index=[0])
    assert fkm.kf_local(
        ap["G0"], ap["b"], ap["n"], ap["S_Type"]
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "Kf, Kr, Kv, Ks, Knle, n, expected",
    [
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        (2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        (2.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.5),
        (1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 2.0),
        (1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.5),
        (1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 2.0),
        (1.0, 1.0, 1.0, 1.0, 1.05, 1.0, 1.0 / 1.05),
        (1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5),
    ],
)
def test_design_factor(Kf, Kr, Kv, Ks, Knle, n, expected):
    fkm = fkm_class()
    ap = pd.DataFrame(
        {"n": n, "Kf": Kf, "Kr": Kr, "Kv": Kv, "Ks": Ks, "Knle": Knle}, index=[0]
    )
    assert fkm.design_factor(
        ap["n"], ap["Kf"], ap["Kr"], ap["Kv"], ap["Ks"], ap["Knle"]
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "Rm, mat, S_type, expected",
    [
        (1.0, "CaseHard_Steel", "normal", 0.40),
        (1.0, "Stainless_Steel", "normal", 0.40),
        (1.0, "Forg_Steel", "normal", 0.40),
        (1.0, "Steel", "normal", 0.45),
        (1.0, "GS", "normal", 0.34),
        (1.0, "GJS", "normal", 0.34),
        (1.0, "GJM", "normal", 0.30),
        (1.0, "GJL", "normal", 0.34),
        (1.0, "Al_wrought", "normal", 0.30),
        (1.0, "Al_cast", "normal", 0.30),
        (1.0, "CaseHard_Steel", "shear", 0.40 * 0.577),
        (1.0, "Stainless_Steel", "shear", 0.40 * 0.577),
        (1.0, "Forg_Steel", "shear", 0.40 * 0.577),
        (1.0, "Steel", "shear", 0.45 * 0.577),
        (1.0, "GS", "shear", 0.34 * 0.577),
        (1.0, "GJS", "shear", 0.34 * 0.65),
        (1.0, "GJM", "shear", 0.30 * 0.75),
        (1.0, "GJL", "shear", 0.34 * 1.0),
        (1.0, "Al_wrought", "shear", 0.30 * 0.577),
        (1.0, "Al_cast", "shear", 0.30 * 0.75),
    ],
)
def test_reversed_mat_strength(Rm, mat, S_type, expected):
    fkm = fkm_class()

    ap = pd.DataFrame({"Rm": Rm, "MatGroupFKM": mat, "S_Type": S_type}, index=[0])
    df_consts, df_fw_t = fkm.get_material_constants(ap["MatGroupFKM"], ap["S_Type"])
    assert fkm.reversed_mat_strength_chap4(
        ap["Rm"], df_consts, df_fw_t, ap["S_Type"]
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "experiment, profile, condition, expected",
    [
        (
            "example_dummy.csv",
            "Rod",
            None,
            pd.Series(
                {
                    "Deff": 10.0,
                    "Profile": "Rod",
                    "Condition": None,
                    "b": 10.0,
                    "Kf_method": "Equation",
                },
            ),
        ),
        (
            "example_dummy.csv",
            "Tube",
            None,
            pd.Series(
                {
                    "Deff": 10.0,
                    "Profile": "Tube",
                    "Condition": None,
                    "b": 1.0,
                    "Kf_method": "Equation",
                },
            ),
        ),
        (
            "example_dummy.csv",
            "Wide sheet",
            None,
            pd.Series(
                {
                    "Deff": 1.0,
                    "Profile": "Wide sheet",
                    "Condition": None,
                    "b": 1.0,
                    "Kf_method": "Equation",
                },
            ),
        ),
        (
            "example_dummy.csv",
            "Rectangle",
            None,
            pd.Series(
                {
                    "Deff": 1.0,
                    "Profile": "Rectangle",
                    "Condition": None,
                    "b": 1,
                    "Kf_method": "Equation",
                },
            ),
        ),
        (
            "example_dummy.csv",
            "Square",
            None,
            pd.Series(
                {
                    "Deff": 1.0,
                    "Profile": "Square",
                    "Condition": None,
                    "b": 1,
                    "Kf_method": "Equation",
                },
            ),
        ),
        (
            "example_dummy.csv",
            "Rod",
            "Hardened",
            pd.Series(
                {
                    "Deff": 10.0,
                    "Profile": "Rod",
                    "Condition": "Hardened",
                    "b": 10 / 2.0,
                    "Kf_method": "Equation",
                },
            ),
        ),
        (
            "example_dummy.csv",
            "Tube",
            "Hardened",
            pd.Series(
                {
                    "Deff": 10.0,
                    "Profile": "Tube",
                    "Condition": "Hardened",
                    "b": 2.0 / 2.0,
                    "Kf_method": "Equation",
                },
            ),
        ),
        (
            "example_dummy.csv",
            "Wide sheet",
            "Hardened",
            pd.Series(
                {
                    "Deff": 1.0,
                    "Profile": "Wide sheet",
                    "Condition": "Hardened",
                    "b": 2.0 / 2.0,
                    "Kf_method": "Equation",
                },
            ),
        ),
        (
            "example_dummy.csv",
            "Rectangle",
            "Hardened",
            pd.Series(
                {
                    "Deff": 1.0,
                    "Profile": "Rectangle",
                    "Condition": "Hardened",
                    "b": 1 / 2.0,
                    "Kf_method": "Equation",
                },
            ),
        ),
        (
            "example_dummy.csv",
            "Square",
            "Hardened",
            pd.Series(
                {
                    "Deff": 1.0,
                    "Profile": "Square",
                    "Condition": "Hardened",
                    "b": 1 / 2.0,
                    "Kf_method": "Equation",
                },
            ),
        ),
    ],
)
def test_char_size_and_fictive_width_b(
    experiment, profile, condition, path_to_data, expected
):

    # Read the CSV file with specified data types
    assessment_parameters = pd.read_csv(os.path.join(path_to_data, experiment)).replace(
        np.nan, None
    )
    #assessment_parameters["Condition"] = condition
    #assessment_parameters["Profile"] = profile

    experiment_settings =  pd.Series({"fkm_chapter": "chap4",
                                    "Profile": profile,
                                    "Diameter": 10.0,
                                    "Width": 1.0,
                                    "Thickness": 1.0,
                                    "sup_method": "Stieler",
                                    "Condition": condition,
                                    "MatGroupFKM": "Steel",
                                    "MatGroupFKM_Temp": "None",

                                },
                            )

    # Compute further quantities for fkm linear (e.g. roughness, kak)
    assessment_parameters = calc_input_parameters_material(experiment_settings,assessment_parameters)
    assessment_parameters = calc_input_parameters_stress(experiment_settings,assessment_parameters)

    # Output values
    np.testing.assert_equal(
        np.round(assessment_parameters["Deff"].values, 6), expected["Deff"]
    )
    np.testing.assert_equal(
        assessment_parameters["Profile"].iloc[0], expected["Profile"]
    )
    np.testing.assert_equal(
        np.round(assessment_parameters["b"].values, 6), expected["b"]
    )
    np.testing.assert_equal(
        assessment_parameters["Condition"].iloc[0], expected["Condition"]
    )
    np.testing.assert_equal(
        assessment_parameters["Kf_method"].iloc[0], expected["Kf_method"]
    )

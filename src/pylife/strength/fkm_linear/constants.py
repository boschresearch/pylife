# Copyright (c) 2019-2024 - for information on the respective copyright owner
# see the NOTICE file and/or the repository
# https://github.com/boschresearch/pylife
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FKM guideline material constants and coefficients.

This module contains material-specific constants and coefficients used in the FKM guideline (2012) for fatigue strength assessment.
"""

import numpy as np
import pandas as pd

# Material constants for different material groups according to FKM guideline
# Table references: FKM 2012
consts_fkm = pd.DataFrame(
    {
        "CaseHard_Steel": {
            "fw_s": 0.40,
            "aG": 0.50,
            "bG": 2700,
            "aR_s": 0.22,
            "RmNmin": 400,
            "a_M": 0.35,
            "b_M": -0.10,
            "k_st": 30,
            "E": 2.1,
            "Kf_loc": 2.0,
        },
        "Stainless_Steel": {
            "fw_s": 0.40,
            "aG": 0.40,
            "bG": 2400,
            "aR_s": 0.22,
            "RmNmin": 400,
            "a_M": 0.35,
            "b_M": -0.10,
            "k_st": 30,
            "E": 2.1,
            "Kf_loc": 2.0,
        },
        "Forg_Steel": {
            "fw_s": 0.40,
            "aG": 0.50,
            "bG": 2700,
            "aR_s": 0.22,
            "RmNmin": 400,
            "a_M": 0.35,
            "b_M": -0.10,
            "k_st": 30,
            "E": 2.1,
            "Kf_loc": 2.0,
        },
        "Steel": {
            "fw_s": 0.45,
            "aG": 0.50,
            "bG": 2700,
            "aR_s": 0.22,
            "RmNmin": 400,
            "a_M": 0.35,
            "b_M": -0.10,
            "k_st": 30,
            "E": 2.1,
            "Kf_loc": 2.0,
        },
        "GS": {
            "fw_s": 0.34,
            "aG": 0.25,
            "bG": 2000,
            "aR_s": 0.20,
            "RmNmin": 400,
            "a_M": 0.35,
            "b_M": 0.05,
            "k_st": 15,
            "E": 2.1,
            "Kf_loc": 2.0,
        },
        "GJS": {
            "fw_s": 0.34,
            "aG": 0.05,
            "bG": 3200,
            "aR_s": 0.16,
            "RmNmin": 400,
            "a_M": 0.35,
            "b_M": 0.08,
            "k_st": 10,
            "E": 1.7,
            "Kf_loc": 1.5,
        },
        "GJM": {
            "fw_s": 0.30,
            "aG": -0.05,
            "bG": 3200,
            "aR_s": 0.12,
            "RmNmin": 350,
            "a_M": 0.35,
            "b_M": 0.13,
            "k_st": 10,
            "E": 1.8,
            "Kf_loc": 1.2,
        },
        "GJL": {
            "fw_s": 0.34,
            "aG": -0.05,
            "bG": 3200,
            "aR_s": 0.06,
            "RmNmin": 100,
            "a_M": 0.00,
            "b_M": 0.50,
            "k_st": 10,
            "E": 1.0,
            "Kf_loc": 1.0,
        },
        "Al_wrought": {
            "fw_s": 0.30,
            "aG": 0.05,
            "bG": 850,
            "aR_s": 0.22,
            "RmNmin": 133,
            "a_M": 1.00,
            "b_M": -0.04,
            "k_st": 20,
            "E": 0.7,
            "Kf_loc": 2.0,
        },
        "Al_cast": {
            "fw_s": 0.30,
            "aG": -0.05,
            "bG": 3200,
            "aR_s": 0.20,
            "RmNmin": 133,
            "a_M": 1.00,
            "b_M": 0.2,
            "k_st": 10,
            "E": 0.7,
            "Kf_loc": 1.2,
        },
    }
)

# FKM factor for shear material strength at R=-1 (Schubwechselfestigkeitsfaktor)
# Table 4.2.1
fw_t = pd.DataFrame(
    {
        "CaseHard_Steel": {"shear": 0.577, "normal": 1},
        "Stainless_Steel": {"shear": 0.577, "normal": 1},
        "Forg_Steel": {"shear": 0.577, "normal": 1},
        "Steel": {"shear": 0.577, "normal": 1},
        "GS": {"shear": 0.577, "normal": 1},
        "GJS": {"shear": 0.650, "normal": 1},
        "GJM": {"shear": 0.750, "normal": 1},
        "GJL": {"shear": 1.000, "normal": 1},
        "Al_wrought": {"shear": 0.577, "normal": 1},
        "Al_cast": {"shear": 0.750, "normal": 1},
    }
)

# Surface hardening procedure constants
proc_const = pd.DataFrame(
    {
        # Einsatzhärten (Case hardening)
        "Case hardening": {
            "a": 1.2,
            "b": 10,
            "Sw_zd_RS_max": 610,
            "M_RS": 0.5,
            "Kv_gr": 1.3,
            "Kv_sm": 1.6,
            "Kv_gr_notched": 1.6,
            "Kv_sm_notched": 2.0,
        },
        # Nitrieren (Nitriding)
        "Nitriding": {
            "a": 1.2,
            "b": 90,
            "Sw_zd_RS_max": 690,
            "M_RS": 0.45,
            "Kv_gr": 1.125,
            "Kv_sm": 1.2,
            "Kv_gr_notched": 1.65,
            "Kv_sm_notched": 2.45,
        },
        # Karbonitierung (Carbonitriding)
        "Carbonitriding": {
            "a": 1.2,
            "b": 90,
            "Sw_zd_RS_max": 690,
            "M_RS": 0.45,
            "Kv_gr": 1.8,
            "Kv_sm": 1.8,
            "Kv_gr_notched": 1.8,
            "Kv_sm_notched": 1.8,
        },
        # Induktives Härten (Inductive hardening)
        "Inductive hardening": {
            "a": 1.42,
            "b": 23,
            "Sw_zd_RS_max": 733,
            "M_RS": 0.4,
            "Kv_gr": 1.125,
            "Kv_sm": 1.2,
            "Kv_gr_notched": 1.65,
            "Kv_sm_notched": 2.45,
        },
        # Kugelstrahlen (Shot peening)
        "Shot peening": {
            "a": np.nan,
            "b": np.nan,
            "Sw_zd_RS_max": np.nan,
            "M_RS": 0.3,
            "Kv_gr": 1.175,
            "Kv_sm": 1.3,
            "Kv_gr_notched": 1.55,
            "Kv_sm_notched": 1.85,
        },
        # Festwalzen (Cold rolling)
        "Cold rolling": {
            "a": np.nan,
            "b": np.nan,
            "Sw_zd_RS_max": np.nan,
            "M_RS": 0.18,
            "Kv_gr": 1.15,
            "Kv_sm": 1.2,
            "Kv_gr_notched": 1.3,
            "Kv_sm_notched": 1.95,
        },
        # No hardening procedure
        "None": {
            "a": np.nan,
            "b": np.nan,
            "Sw_zd_RS_max": np.nan,
            "M_RS": np.nan,
            "Kv_gr": np.nan,
            "Kv_sm": np.nan,
            "Kv_gr_notched": np.nan,
            "Kv_sm_notched": np.nan,
        },
    }
)

# Temperature coefficients for different material groups
temp_coeff = pd.DataFrame(
    {
        "fine grain structural steel": {
            "T_threshold": 60.0,
            "delta_T": 0.0,
            "a_TD": 1.0,
            "T_max": 500.0,
        },
        "other kinds of steel": {
            "T_threshold": 100.0,
            "delta_T": 100.0,
            "a_TD": 1.4,
            "T_max": 500.0,
        },
        "Steel": {"T_threshold": 100.0, "delta_T": 100.0, "a_TD": 1.4, "T_max": 500.0},
        # FKM: No values are known for stainless steel
        "Stainless_Steel": {
            "T_threshold": 100.0,
            "delta_T": 0.0,
            "a_TD": 0.0,
            "T_max": 101.0,
        },
        "GS": {"T_threshold": 100.0, "delta_T": 100.0, "a_TD": 1.2, "T_max": 500.0},
        "GJS": {"T_threshold": 100.0, "delta_T": 0.0, "a_TD": 1.6, "T_max": 500.0},
        "GJM": {"T_threshold": 100.0, "delta_T": 0.0, "a_TD": 1.3, "T_max": 500.0},
        "GJL": {"T_threshold": 100.0, "delta_T": 0.0, "a_TD": 1.0, "T_max": 500.0},
        "Aluminum": {"T_threshold": 50.0, "delta_T": 50.0, "a_TD": 1.2, "T_max": 200.0},
        "None": {
            "T_threshold": 1e16,
            "delta_T": np.nan,
            "a_TD": np.nan,
            "T_max": np.nan,
        },
    }
)

# Additional constants and material classifications
consts = {
    # Surface hardening procedures (excluding 'Cyaniding' which is commented out)
    "hard_procs": [
        "Case hardening",
        "Nitriding",
        "Carbonitriding",
        "Inductive hardening",
        "Shot peening",
        "Cold rolling",
    ],
    # Mechanical surface hardening procedures
    "hard_proc_mechanical": ["Shot peening", "Cold rolling"],
    # Thermal/chemical surface hardening procedures
    "hard_proc_others": [
        "Case hardening",
        "Nitriding",
        "Carbonitriding",
        "Inductive hardening",
    ],
    # Brittle materials
    "brittle": ["GS", "GJS", "GJL", "GJM", "Al_cast"],
    # Steel materials
    "steels": ["CaseHard_Steel", "Stainless_Steel", "Forg_Steel", "Steel"],
    # Aluminum materials
    "alu": ["Al_wrought", "Al_cast"],
    # Reference loaded surface for statistical support factor [mm²]
    "A90_0": 500,
    # Reference loaded volume for statistical support factor [mm³]
    "V90_0": 1000,
}


def _fkm_consts():
    """Get all FKM material constants.

    Returns
    -------
    list of pd.DataFrame
        List containing:

        * [0] : Material constants (consts_fkm)
        * [1] : Shear/normal stress factors (fw_t)
        * [2] : Surface hardening procedure constants (proc_const)
        * [3] : Temperature coefficients (temp_coeff)
    """
    fkm_consts = [consts_fkm, fw_t, proc_const, temp_coeff]
    return fkm_consts


def _additional_consts(material_class):
    """Get additional material constants by material class.

    Parameters
    ----------
    material_class : str
        Material class identifier. One of:

        * 'hard_procs' : List of all hardening procedures
        * 'hard_proc_mechanical' : Mechanical hardening procedures
        * 'hard_proc_others' : Thermal/chemical hardening procedures
        * 'brittle' : Brittle materials
        * 'steels' : Steel materials
        * 'alu' : Aluminum materials
        * 'A90_0' : Reference loaded surface [mm²]
        * 'V90_0' : Reference loaded volume [mm³]

    Returns
    -------
    various types
        Material class data (list, float, or str if not found)
    """
    return consts.get(material_class, "Material class not found in list")

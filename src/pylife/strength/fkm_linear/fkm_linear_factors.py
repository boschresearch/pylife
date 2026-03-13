"""Collection of functions for computational proof for machine elements.


FKM-guidelines according to chapter 4 and 5.5 (FKM guideline 2012).
"""

import numpy as np
import pandas as pd

from pylife.strength.fkm_linear.fkm_functions import FkmLinearFunctions as fkm_class

HARDENING_PROCEDURES = [
    "Inductive hardening",
    "Flame hardening",
    "Case hardening",
    "Carburizing",
    "Nitriding",
    "Cyaniding",
    "Carbonitriding",
    "Deep rolling",
    "Shot peening",
    "Cold rolling",
]

TEMPERATURE_GROUPS = [
    "GS",
    "GJL",
    "GS",
    "GJM",
    "Steel",
    "Stainless Steel",
    "Aluminum",
    "fine grain structural steel",
    "other kinds of steel",
]


def calc_input_parameters_material(experiment_settings, assessment_parameters_):
    """Compute relevant material-dependent factors for FKM guideline.

    Calculates factors that depend on the design and material of the component
    and not on the applied stress (e.g. mean stress sensitivity or roughness factor).

    Parameters
    ----------
    experiment_settings : pd.Series
        Pandas Series with following columns:

        * fkm_chapter : str
            Chapter of the FKM guideline, one of {'chap4', 'chap5.5'}
        * MatGroupFKM : str
            Material group. One of:

            * 'CaseHard_Steel' : Case-hardened steels
            * 'Stainless_Steel' : Stainless steels
            * 'Forg_Steel' : Forged steels
            * 'Steel' : All other steel groups
            * 'GS' : Cast steels and tempering cast steels
            * 'GJS' : Cast iron with spheroidal graphite (old GGG)
            * 'GJM' : Heart fittings (Temperguss, old: GT)
            * 'GJL' : Cast iron with lamellar graphite (old GG)
            * 'Al_wrought' : Wrought aluminium alloys
            * 'Al_cast' : Cast aluminium alloys

        * MatGroupFKM_Temp : str
            Material temperature group, one of {'fine grain structural steel',
            'Stainless Steel', 'other kinds of steel', 'GJL', 'GS', 'GJS',
            'GJM', 'Aluminum', 'None'}
        * Profile : str
            Profile/Geometry of the specimen, one of {'Rod', 'Tube',
            'Wide sheet', 'Rectangle', 'Square'}
        * Diameter : float
            Diameter of the specimen [mm]
        * Width : float
            Width of the specimen [mm] (only needed if profile is
            'Rectangle' or 'Square')
        * Thickness : float
            Thickness of the specimen [mm] (only needed if profile is
            'Rectangle', 'Sheet', or 'Wide sheet')
        * Condition : str or None
            Heat treatment condition for fictive width b calculation,
            one of {'Hardened', 'Annealed'} or None

    assessment_parameters_ : pd.DataFrame
        DataFrame with following columns:

        * Rm : float
            Tensile strength [MPa]
        * Rz : float
            Surface roughness [µm]
        * S_Type : {'normal', 'shear'}
            Stress type
        * Temperature : float
            Temperature of the component [°C]
        * Finish : {'polished'}, optional
            Surface finish polished (if used, no Rz value necessary)
        * GJL_mat : str, optional
            The GJL material group (if GJL is used)
        * HV : float, optional
            Vickers hardness (only needed for chap5.5)
        * HV_core : float, optional
            Core hardness (only needed for chap5.5)
        * HardProc : str, optional
            Surface hardening method, one of {'Inductive hardening',
            'Flame hardening', 'Case hardening', 'Carburizing', 'Nitriding',
            'Cyaniding', 'Carbonitriding', 'Deep rolling', 'Shot peening'}

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns including relevant FKM quantities
    """
    ap = assessment_parameters_.copy()

    # Initialize class object
    fkm = fkm_class()

    # Set parameters that are equal for all nodes
    ap["MatGroupFKM"] = experiment_settings["MatGroupFKM"]
    ap["MatGroupFKM_Temp"] = experiment_settings["MatGroupFKM_Temp"]
    ap["Profile"] = experiment_settings["Profile"]
    ap["Condition"] = experiment_settings["Condition"]

    # Adapt data frame such that it matches the desired format
    ap = _adapt_data_frame(ap)

    # Get material constants
    df_consts, df_fw_t = fkm.get_material_constants(ap["MatGroupFKM"], ap["S_Type"])
    df_temperature = fkm.get_temperature_constants(ap["MatGroupFKM_Temp"])

    # Calculate temperature factor K_D_T
    ap["K_D_T"] = fkm.temperature_model_cython(
        df_temperature, ap["Temperature"], ap["MatGroupFKM"]
    )

    if experiment_settings["fkm_chapter"] == "chap4":

        # Compute reversed material strength
        ap["Sw"] = fkm.reversed_mat_strength_chap4_cython(
            ap["Rm"], df_consts, df_fw_t, ap["S_Type"]
        )

        # Roughness factor
        ap["Kr"] = fkm.rough_factor_cython(
            ap["Rm"], ap["Rz"], df_consts, df_fw_t, ap["S_Type"], ap["Finish"]
        )

        # Coating factor KS is set to 1 by default,
        # adapt if needed before using calc_input_parameters_stress
        ap["Ks"] = 1.0

        fictive_width = _fictive_width_b(experiment_settings)
        ap["b"] = fictive_width

        ap["Knle"] = fkm.GJL_bending_factor_cython(ap["GJL_Mat"])

        # Compute characteristic size
        characteristic_size = _characteristic_size(experiment_settings)
        ap["Deff"] = characteristic_size

        # Mean stress sensitivity M
        ap["M"] = fkm.sm_sensitivity_cython_chap4(
            ap["Rm"], df_consts, df_fw_t, ap["S_Type"]
        )

    elif experiment_settings["fkm_chapter"] == "chap5.5":

        df_proc = fkm.get_material_constants_chap5_5(ap["HardProc"])
        ap["Rm"] = 3.3 * ap["HV"]
        ap["SW_RS"] = fkm.reversed_mat_strength_chap5_5_cython(
            ap["Rm"],
            df_consts,
            df_fw_t,
            ap["S_Type"],
            ap["HV"],
            df_proc,
            ap["HardProc"],
        )

        ap["Rm_RS"] = 3.3 * ap["HV"]

        # Roughness factor
        ap["Kr"] = fkm.rough_factor_cython(
            ap["Rm"], ap["Rz"], df_consts, df_fw_t, ap["S_Type"], ap["Finish"]
        )

        # Notch strength reduction factor (Kerbwirkungszahl) (FKM Chapter 4.3.1.2)
        ap["Kf"] = fkm.kf_constant_cython(ap["MatGroupFKM"])

        # Surface factor for case-hardened materials are initialized with default values,
        # adapt if needed before using calc_input_parameters_stress
        ap["Deff"] = None
        ap["Kv"] = 1.0
        ap["Ks"] = 1.0
        ap["Knle"] = 1.0

        # Mean stress sensitivity
        ap["M_RS"] = df_proc["M_RS"].values

        # Eigenstress for surface layer SE
        ap["SE_RS"] = fkm.eigenstress_RS_cython(
            ap["Rm"],
            ap["HV"],
            ap["HV_core"],
            ap["HardProc"],
        )

        # SDFKM at transition point from hardened surface layer to core
        # Reversed material strength
        ap["SW_trans"] = fkm.reversed_mat_strength_chap5_5_cython(
            ap["Rm"],
            df_consts,
            df_fw_t,
            ap["S_Type"],
            ap["HV_core"],
            df_proc,
            ap["HardProc"],
        )

        # Calculate mean stress sensitivity M_trans for transition point
        ap["Rm_trans"] = fkm.sm_sensitivity_cython_Rm_trans(ap["HV_core"])
        ap["M_trans"] = fkm.sm_sensitivity_cython_M_trans(
            ap["Rm"], df_consts, df_fw_t, ap["S_Type"], ap["Rm_trans"]
        )

        ap["SE_trans"] = -0.3 * ap["SE_RS"]

    return ap


def calc_input_parameters_stress(experiment_settings, assessment_parameters_):
    """Compute relevant stress-dependent factors for FKM guideline.

    Calculates factors that depend on the stress applied to the component,
    i.e., gradient, amplitude and meanstress (e.g. design factor, support factor).

    Note: Requires executing calc_input_parameters_material() before this function.

    Parameters
    ----------
    experiment_settings : pd.Series
        Pandas Series with following columns:

        * fkm_chapter : str
            Chapter of the FKM guideline, one of {'chap4', 'chap5.5'}
        * sup_method : str, optional
            Support factor calculation method. One of {'Stieler', 'V90_Mises', 'A90'}.
            Default is Stieler.

    assessment_parameters_ : pd.DataFrame
        DataFrame with the following columns:

        * Rm : float
            Tensile strength [MPa]
        * G0 : float
            Relative stress gradient [1/mm]. Sum of local & global relative
            stress gradients
        * A90 : float, optional
            Maximum loaded surface according to FKM2012 [mm²]
        * V90_Mises : float, optional
            Maximum loaded volume according to von Mises support factor [mm³]
        * MatGroupFKM : str
            Material group. One of:

            * 'CaseHard_Steel' : Case-hardened steels
            * 'Stainless_Steel' : Stainless steels
            * 'Forg_Steel' : Forged steels
            * 'Steel' : All other steel groups
            * 'GS' : Cast steels and tempering cast steels
            * 'GJS' : Cast iron with spheroidal graphite (old GGG)
            * 'GJM' : Heart fittings (Temperguss, old: GT)
            * 'GJL' : Cast iron with lamellar graphite (old GG)
            * 'Al_wrought' : Wrought aluminium alloys
            * 'Al_cast' : Cast aluminium alloys

        * amplitude : float
            Stress amplitude [MPa] (can be set to 1.0)
        * meanstress : float
            Mean stress [MPa]
        * S_Type : {'normal', 'shear'}
            Stress type
        * Kf_method : {'Table', 'Equation'}
            Method for estimating the fatigue notch factor (chap. 4.3.1.2)
        * FKM_chap : {'chap4', 'chap5.5'}
            Chapter of the FKM guideline
        * HV : float, optional
            Vickers hardness (only needed for chap5.5)

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns including relevant FKM quantities
    """
    ap = assessment_parameters_.copy()

    fkm = fkm_class()

    ap["R_ratio"] = (ap["meanstress"] - ap["amplitude"]) / (
        ap["meanstress"] + ap["amplitude"]
    )

    # Get material constants
    df_consts, df_fw_t = fkm.get_material_constants(ap["MatGroupFKM"], ap["S_Type"])

    df_proc = fkm.get_material_constants_chap5_5(ap["HardProc"])

    if experiment_settings["fkm_chapter"] == "chap4":

        # Compute support factor
        if experiment_settings["sup_method"] == "Stieler":
            ap["n"] = fkm.stieler_support_cython(
                df_consts, df_fw_t, ap["S_Type"], ap["G0"], ap["Rm"]
            )

            (
                ap["n_st"],
                ap["n_vm"],
                ap["n_bm"],
            ) = (np.nan, np.nan, np.nan)

        elif experiment_settings["sup_method"] == "V90_Mises":

            # Calculate support factors using V90_Mises (maximum stressed volume)
            ap_temp = ap.copy()
            res = fkm.support_fkm2012_local_vol_frame(ap_temp)

            # Create a new DataFrame from the list of tuples
            df_temp = pd.DataFrame(res.tolist(), columns=["n_st", "n_vm", "n_bm", "n"])

            # Set support values
            ap["n_st"] = df_temp["n_st"].to_numpy()
            ap["n_vm"] = df_temp["n_vm"].to_numpy()
            ap["n"] = df_temp["n"].to_numpy()
            ap["n_bm"] = df_temp["n_bm"].to_numpy()

        else:

            # Calculate support factors using A90 (maximum stressed surface)
            ap_temp = ap.copy()
            res = fkm.support_fkm2012_local_surf_frame(ap_temp)

            # Create a new DataFrame from the list of tuples
            df_temp = pd.DataFrame(res.tolist(), columns=["n_st", "n_vm", "n_bm", "n"])

            # Set support values
            ap["n_st"] = df_temp["n_st"].to_numpy()
            ap["n_vm"] = df_temp["n_vm"].to_numpy()
            ap["n"] = df_temp["n"].to_numpy()
            ap["n_bm"] = df_temp["n_bm"].to_numpy()

        # Notch strength reduction factor (Kerbwirkungszahl) (FKM Chapter 4.3.1.2)
        ap["Kf"] = fkm.kf_factor_cython(
            ap["Kf_method"], ap["MatGroupFKM"], ap["G0"], ap["b"], ap["n"], ap["S_Type"]
        )

        # Surface factor for case-hardened materials
        ap["Kv"] = fkm.surf_layer_factor_cython(
            df_proc, ap["G0"], ap["Deff"], ap["HardProc"]
        )

    elif experiment_settings["fkm_chapter"] == "chap5.5":

        # Support factor for surface layer (chap 5.5 specific)
        ap["n_RS"] = fkm.support_chap5_cython(ap["G0"], ap["HV"])

    return ap


def fatigue_limit_local_chap4(assessment_parameters_):
    """Calculate fatigue limit for non-welded materials using local stress concept.

    According to chapter 4 of FKM guideline 2012.

    Parameters
    ----------
    assessment_parameters_ : pd.DataFrame
        DataFrame with the following columns:

        * Rm : float
            Tensile strength [MPa]
        * G0 : float, optional
            Relative stress gradient [1/mm]. Sum of local & global relative
            stress gradients
        * A90 : float, optional
            Maximum loaded surface according to FKM2012 [mm²]
        * MatGroupFKM : str
            Material group. One of:

            * 'CaseHard_Steel' : Case-hardened steels
            * 'Stainless_Steel' : Stainless steels
            * 'Forg_Steel' : Forged steels
            * 'Steel' : All other steel groups
            * 'GS' : Cast steels and tempering cast steels
            * 'GJS' : Cast iron with spheroidal graphite (old GGG)
            * 'GJM' : Heart fittings (Temperguss, old: GT)
            * 'GJL' : Cast iron with lamellar graphite (old GG)
            * 'Al_wrought' : Wrought aluminium alloys
            * 'Al_cast' : Cast aluminium alloys

        * R_ratio : float
            Stress ratio
        * Rz : float
            Surface roughness [µm]
        * S_Type : {'normal', 'shear'}
            Stress type

    Returns
    -------
    pd.DataFrame
        DataFrame containing input data and all calculated data for FKM prognoses
        with predicted fatigue limit [MPa]
    """
    ap = assessment_parameters_.copy()
    fkm = fkm_class()

    # Compute temperature-adjusted reversed material strength
    ap["SW"] = ap["K_D_T"] * ap["Sw"]

    # Compute design factor
    ap["Kwk"] = fkm.design_factor_cython(
        ap["n"],
        ap["Kf"],
        ap["Kr"],
        ap["Kv"],
        ap["Ks"],
        ap["Knle"],
    )

    # Reversed strength of design element
    ap["Swk"] = ap["SW"] / ap["Kwk"]

    SmSa = (1 + ap["R_ratio"]) / (1 - ap["R_ratio"])

    # Mean stress factor
    ap["Kak"] = fkm.sm_factor_cython(ap["R_ratio"], ap["M"], SmSa)

    # Fatigue limit
    ap["SDFKM"] = ap["Swk"] * ap["Kak"]

    return ap


def fatigue_limit_local_chap5(assessment_parameters_):
    """Calculate fatigue limit for surface-hardened materials using local stress concept.

    For surface layer according to chapter 5.5 of FKM guideline 2012.

    Parameters
    ----------
    assessment_parameters_ : pd.DataFrame
        DataFrame with the following columns:

        * Rm : float
            Tensile strength [MPa]
        * G0 : float, optional
            Relative stress gradient [1/mm]. Sum of local & global relative
            stress gradients
        * A90 : float, optional
            Maximum loaded surface according to FKM2012 [mm²]
        * MatGroupFKM : str
            Material group. One of:

            * 'CaseHard_Steel' : Case-hardened steels
            * 'Stainless_Steel' : Stainless steels
            * 'Forg_Steel' : Forged steels
            * 'Steel' : All other steel groups
            * 'GS' : Cast steels and tempering cast steels
            * 'GJS' : Cast iron with spheroidal graphite (old GGG)
            * 'GJM' : Heart fittings (Temperguss, old: GT)
            * 'GJL' : Cast iron with lamellar graphite (old GG)
            * 'Al_wrought' : Wrought aluminium alloys
            * 'Al_cast' : Cast aluminium alloys

        * R_ratio : float
            Stress ratio
        * Rz : float
            Surface roughness [µm]
        * S_Type : {'normal', 'shear'}
            Stress type

    Returns
    -------
    pd.DataFrame
        DataFrame containing input data and all calculated data for FKM prognoses
        with predicted fatigue limits at surface (RS) and core [MPa]
    """
    ap = assessment_parameters_.copy()

    fkm = fkm_class()

    if (ap["HardProc"].isin(fkm.hard_procs)).all():

        # Design factor
        ap["Kwk_RS"] = fkm.design_factor_cython(
            ap["n_RS"],
            ap["Kf"],
            ap["Kr"],
            ap["Kv"],
            ap["Ks"],
            ap["Knle"],
        )

        # Reversed strength of design element
        ap["Swk_RS"] = ap["SW_RS"] / ap["Kwk_RS"]

        # Mean stress factor for overload case F2 and design factor at inner point
        SmSa = (1 + ap["R_ratio"]) / (1 - ap["R_ratio"])

        ap["Kak_RS"] = fkm.sm_factor_chap5_cython(
            ap["Rm_RS"], ap["M_RS"], ap["SE_RS"], ap["Swk_RS"], SmSa, ap["Rm"]
        )

        # Fatigue limit at surface
        ap["SDFKM_RS"] = ap["Swk_RS"] * ap["Kak_RS"]

        # Reversed strength of design element in the core
        ap["Kwk_trans"] = 1.0  # Design factor Kwk = 1
        ap["Swk_trans"] = ap["SW_trans"] * ap["Kwk_trans"]

        # Mean stress factor for overload case F2 in the core
        ap["Kak_trans"] = fkm.sm_factor_chap5_cython(
            ap["Rm_trans"],
            ap["M_trans"],
            ap["SE_trans"],
            ap["Swk_trans"],
            SmSa,
            ap["Rm"],
        )

        # Fatigue limit at transition point in the core
        ap["SDFKM_trans"] = ap["Swk_trans"] * ap["Kak_trans"]

        return ap


def _adapt_data_frame(df):
    """Adapt DataFrame format to match FKM function requirements.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame including FKM parameters

    Returns
    -------
    pd.DataFrame
        DataFrame with adapted columns matching required format
    """
    # Convert all numeric columns to floats
    numeric_columns = df.select_dtypes(include=["int", "float"]).columns
    df[numeric_columns] = df[numeric_columns].astype(float)

    # Replace non-existing names of hardening procedures and temperature groups with 'None'
    df["HardProc"] = df["HardProc"].apply(
        lambda x: x if x in HARDENING_PROCEDURES else "None"
    )
    df["MatGroupFKM_Temp"] = df["MatGroupFKM_Temp"].apply(
        lambda x: x if x in TEMPERATURE_GROUPS else "None"
    )
    df["Finish"] = df["Finish"].apply(lambda x: x if x in ["polished"] else "None")

    # Replace the parameters Rm, Rm_trans, HV, HV_core with np.nan if they are None
    for col in ["Rm", "Rm_trans", "HV", "HV_core"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


def _fictive_width_b(series):
    """Determine fictive width of the component.

    Parameters
    ----------
    series : pd.Series
        Series including the geometry of the component with columns:

        * Profile : str
            One of {'Rod', 'Tube', 'Wide sheet', 'Rectangle', 'Square'}
        * Diameter : float
            Diameter [mm] (for Rod, Tube)
        * Thickness : float
            Thickness [mm] (for Tube, Wide sheet, Rectangle)
        * Width : float
            Width [mm] (for Rectangle, Square)
        * Condition : str
            Heat treatment condition
        * MatGroupFKM : str
            Material group

    Returns
    -------
    float
        Fictive width of the component [mm]
    """
    if series["Profile"] == "Rod":
        Deff1 = series["Diameter"]
        Deff2 = series["Diameter"]
    elif series["Profile"] == "Tube":
        Deff1 = 2 * series["Thickness"]
        Deff2 = series["Thickness"]
    elif series["Profile"] == "Wide sheet":
        Deff1 = 2 * series["Thickness"]
        Deff2 = series["Thickness"]
    elif series["Profile"] == "Rectangle":
        Deff1 = (
            2
            * series["Width"]
            * series["Thickness"]
            / (series["Width"] + series["Thickness"])
        )
        Deff2 = series["Thickness"]
    elif series["Profile"] == "Square":
        Deff1 = series["Width"]
        Deff2 = series["Width"]
    else:
        raise ValueError(f"Unknown profile type: {series['Profile']}")

    if series["Condition"] == "Hardened" or series["MatGroupFKM"] in [
        "CaseHard_Steel",
        "GJS",
        "GJM",
        "GJL",
    ]:
        return Deff1 / 2
    else:
        return Deff2


def _characteristic_size(series):
    """Determine characteristic size of the component.

    Parameters
    ----------
    series : pd.Series
        Series including the geometry-specific size of the component with columns:

        * Profile : str
            One of {'Rod', 'Tube', 'Wide sheet', 'Rectangle', 'Square'}
        * Diameter : float
            Diameter [mm] (for Rod, Tube)
        * Thickness : float
            Thickness [mm] (for Wide sheet, Rectangle)
        * Width : float
            Width [mm] (for Square)

    Returns
    -------
    float
        Characteristic size of the component [mm]
    """
    if series["Profile"] == "Rod":
        return series["Diameter"]
    elif series["Profile"] == "Tube":
        return series["Diameter"]
    elif series["Profile"] == "Wide sheet":
        return series["Thickness"]
    elif series["Profile"] == "Rectangle":
        return series["Thickness"]
    elif series["Profile"] == "Square":
        return series["Width"]
    else:
        raise ValueError(f"Unknown profile type: {series['Profile']}")

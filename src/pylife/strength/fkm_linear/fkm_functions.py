"""Module for implementing useful FKM functions."""

import numpy as np

import pylife._fkm_linear_functions as F
from pylife.strength.fkm_linear.constants import _additional_consts, _fkm_consts


class FkmLinearFunctions:
    """Class to represent FKM functions.

    This class provides methods for calculating various factors according to
    the FKM guideline for fatigue strength assessment.
    """

    # Define valid hardening procedures as class constant
    VALID_HARDENING_PROCEDURES = [
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

    def __init__(self):
        """Initialize FKM linear functions with material constants."""
        consts_data = _fkm_consts()
        self.consts = consts_data[0]
        self.fw_t = consts_data[1]
        self.proc_const = consts_data[2]
        self.Temp_coeff = consts_data[3]

        self.hard_procs = _additional_consts("hard_procs")
        self.hard_proc_mechanical = _additional_consts("hard_proc_mechanical")
        self.brittle = _additional_consts("brittle")
        self.steels = _additional_consts("steels")
        self.alu = _additional_consts("alu")
        self.A90_0 = _additional_consts("A90_0")
        self.V90_0 = _additional_consts("V90_0")

    def get_material_constants(self, mat, S_type):
        """Get material constants for direct usage in other functions.

        Parameters
        ----------
        mat : str
            FKM material group. One of:

            * 'CaseHard_Steel' : Case-hardened steels
            * 'Stainless_Steel' : Stainless steels
            * 'Forg_Steel' : Forged steels
            * 'Steel' : All other steel groups
            * 'GS' : Cast steels and tempering cast steels
            * 'GJS' : Cast iron with spheroidal graphite (old GGG)
            * 'GJM' : Heart fittings (Temperguss, old: GT)
            * 'GJL' : Cast iron with lamellar graphite (old GG)
            * 'Al_wrought' : Wrought aluminum alloys
            * 'Al_cast' : Cast aluminum alloys

        S_type : {'normal', 'shear'}
            Stress type

        Returns
        -------
        df_consts : pd.DataFrame
            DataFrame including the material constants
        df_fw_t : pd.DataFrame
            DataFrame including the fw_t factors
        """
        df_consts = self.consts[mat].T[
            ["fw_s", "aG", "bG", "aR_s", "RmNmin", "a_M", "b_M", "k_st", "E", "Kf_loc"]
        ]
        df_fw_t = self.fw_t[mat].T[["shear", "normal"]]

        return df_consts, df_fw_t

    def get_temperature_constants(self, mat):
        """Get temperature constants for the specified material.

        Parameters
        ----------
        mat : str
            FKM material group. One of:

            * 'CaseHard_Steel' : Case-hardened steels
            * 'Stainless_Steel' : Stainless steels
            * 'Forg_Steel' : Forged steels
            * 'Steel' : All other steel groups
            * 'GS' : Cast steels and tempering cast steels
            * 'GJS' : Cast iron with spheroidal graphite (old GGG)
            * 'GJM' : Heart fittings (Temperguss, old: GT)
            * 'GJL' : Cast iron with lamellar graphite (old GG)
            * 'Al_wrought' : Wrought aluminum alloys
            * 'Al_cast' : Cast aluminum alloys

        Returns
        -------
        df_temperature : pd.DataFrame
            DataFrame including the temperature factors
        """
        return self.Temp_coeff[mat].T[["T_threshold", "T_max", "a_TD", "delta_T"]]

    def get_material_constants_chap5_5(self, Proc):
        """Get material constants for chapter 5.5.

        Parameters
        ----------
        Proc : pd.Series
            Series with hardening procedure names

        Returns
        -------
        pd.DataFrame
            DataFrame including all constants for the hardening procedure
        """
        Proc = Proc.apply(_replace_proc_string)
        return self.proc_const[Proc].T[
            [
                "a",
                "b",
                "Sw_zd_RS_max",
                "M_RS",
                "Kv_gr",
                "Kv_sm",
                "Kv_gr_notched",
                "Kv_sm_notched",
            ]
        ]

    def reversed_mat_strength_chap4(self, Rm, df_consts, df_fw_t, S_type):
        """Calculate material strength for R=-1 axial/shear stress.

        According to FKM 2012 local approach (Chapter 4.2.1.1) using Cython.

        Parameters
        ----------
        Rm : np.ndarray
            Tensile strength [MPa]
        df_consts : pd.DataFrame
            DataFrame with material constants
        df_fw_t : pd.DataFrame
            DataFrame including stress type constants
        S_type : str
            Stress type, one of {'shear', 'normal'}

        Returns
        -------
        np.ndarray
            Fully reversed material strengths [MPa]
        """
        fw_t = df_fw_t.iloc[0].loc[S_type].values
        return F.reversed_mat_strength_chap4(Rm, fw_t, df_consts.fw_s)

    def reversed_mat_strength_chap5_5(
        self, Rm, df_consts, df_fw_t, S_type, HV, df_proc, Proc
    ):
        """Calculate material strength for R=-1 axial/shear stress.

        According to FKM 2012 local approach (Chapter 4.2.1.1) for chapter 5.5
        using Cython.

        Parameters
        ----------
        Rm : np.ndarray
            Tensile strength [MPa]
        df_consts : pd.DataFrame
            DataFrame with material constants
        df_fw_t : pd.DataFrame
            DataFrame including stress type constants
        S_type : str
            Stress type, one of {'shear', 'normal'}
        HV : np.ndarray
            Vickers hardness values
        df_proc : pd.DataFrame
            DataFrame including hardening constants
        Proc : str
            Hardening procedure

        Returns
        -------
        np.ndarray
            Fully reversed material strengths [MPa]
        """
        fw_t = df_fw_t.iloc[0].loc[S_type].values
        return F.reversed_mat_strength_chap5_5(
            Rm,
            fw_t,
            df_consts.fw_s,
            HV,
            df_proc.a,
            df_proc.b,
            df_proc.Sw_zd_RS_max,
            Proc,
        )

    def stieler_support(self, df_consts, df_fw_t, S_type, G, Rm):
        """Calculate support factors according to Stieler's equation.

        According to FKM 2012 local approach (Chapter 4.3.1.3.1) using Cython.

        Parameters
        ----------
        df_consts : pd.DataFrame
            DataFrame including material constants
        df_fw_t : pd.DataFrame
            DataFrame including stress type constants
        S_type : np.ndarray
            Stress types, one of {'shear', 'normal'}
        G : np.ndarray
            Stress gradients [1/mm]
        Rm : np.ndarray
            Tensile strength [MPa]

        Returns
        -------
        np.ndarray
            Stieler's support factors
        """
        fw_t = df_fw_t.iloc[0].loc[S_type].values
        return F.stieler_support(fw_t, df_consts.aG, df_consts.bG, G, Rm)

    def stat_support_surf(self, A90, k_st):
        """Calculate statistical support factor n_st.

        According to FKM 2012 local approach (Chapter 4.3.1.3.2).

        Parameters
        ----------
        A90 : float
            Maximum loaded surface according to FKM2012 [mm²]
        k_st : float
            Weibull exponent for statistical support factor

        Returns
        -------
        float
            Statistical support factor
        """
        return (self.A90_0 / A90) ** (1 / k_st)

    def stat_support_vol(self, V90_Mises, k_st):
        """Calculate statistical support factor n_st based on volume.

        Not official part of FKM GL, see FKM Heft 306, Vorhaben 282:
        Verbessertes Berechnungskonzept FKM-Richtlinie.

        Parameters
        ----------
        V90_Mises : float
            Maximum loaded volume according to FKM2012 [mm³]
        k_st : float
            Weibull exponent for statistical support factor

        Returns
        -------
        float
            Statistical support factor
        """
        return (self.V90_0 / V90_Mises) ** (1 / k_st)

    def mech_support(self, n_st, sw, mat, Rm):
        """Calculate the mechanical support factor n_vm.

        According to FKM 2012 local approach (Chapter 4.3.1.3.2).

        Parameters
        ----------
        n_st : float
            Statistical support factor
        sw : float
            Material strength at R=-1 [MPa]
        mat : str
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

        Rm : float
            Tensile strength [MPa]

        Returns
        -------
        float
            Mechanical support factor
        """
        Emod = self.consts[mat]["E"]

        # Brittle materials
        if mat in self.brittle:
            return 1.0

        # Steel materials
        if mat in self.steels and Rm <= 630:
            e_plW = 0.0002
            n_ = 0.15
        elif mat in self.steels and Rm > 630:
            e_plW = 0.0002 * (1 - 0.375 * (Rm / 630 - 1))
            n_ = 0.15
        # Aluminum wrought
        elif mat == "Al_wrought":
            e_plW = 0.000016
            n_ = 0.11
        # Cast materials
        elif mat in ["GS", "GJS", "GJL", "GJM"]:
            e_plW = 0.0
            n_ = 0.15
        else:
            return 1.0

        return np.sqrt(1 + (Emod * 100000 * e_plW * n_st ** (1 / n_ - 1)) / sw)

    def fract_mech_support_local(self, n_st, n_vm, G0, Rm, mat):
        """Calculate the fracture mechanical support factor n_bm.

        According to FKM 2012 local approach (Chapter 4.3.1.3.2).

        Parameters
        ----------
        n_st : float
            Statistical support factor
        n_vm : float
            Mechanical support factor
        G0 : float
            Related stress gradient [1/mm]
        Rm : float
            Tensile strength [MPa]
        mat : str
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

        Returns
        -------
        float
            Fracture mechanical support factor
        """
        if mat not in self.steels + self.alu:
            return 1.0

        if mat in self.steels:
            a_ = (Rm / 680) * np.sqrt((7.5 + np.sqrt(G0)) / (1 + 0.2 * np.sqrt(G0)))
        else:  # aluminum
            a_ = (Rm / 270) * np.sqrt((7.5 + np.sqrt(G0)) / (1 + 0.2 * np.sqrt(G0)))

        n_bm = (5 + np.sqrt(G0)) / (5 * n_vm * n_st + a_)
        return max(n_bm, 1.0)

    def support_fkm2012_local_surf_frame(self, df):
        """Apply surface-based support factor calculation to DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns: 'MatGroupFKM', 'G0', 'A90', 'Rm', 'Sw'

        Returns
        -------
        pd.Series
            Series with support factors (n_st, n_vm, n_bm, n)
        """
        res = df.apply(
            lambda r: self.support_fkm2012_local_surf(
                r["MatGroupFKM"], r["G0"], r["A90"], r["Rm"], r["Sw"]
            ),
            axis=1,
        )
        return res

    def support_fkm2012_local_vol_frame(self, df):
        """Apply volume-based support factor calculation to DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns: 'MatGroupFKM', 'G0', 'V90_Mises', 'Rm', 'Sw'

        Returns
        -------
        pd.Series
            Series with support factors (n_st, n_vm, n_bm, n)
        """
        res = df.apply(
            lambda r: self.support_fkm2012_local_vol(
                r["MatGroupFKM"], r["G0"], r["V90_Mises"], r["Rm"], r["Sw"]
            ),
            axis=1,
        )
        return res

    def support_fkm2012_local_surf(self, mat, G0, A90, Rm, SW):
        """Calculate support factor according to FKM 2012 local method.

        Uses surface-based approach (Chapter 4.3.1.3.2).

        Parameters
        ----------
        mat : str
            FKM material group. One of:

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

        G0 : float
            Relative stress gradient [1/mm]
        A90 : float
            Maximum loaded surface according to FKM2012 [mm²]
        Rm : float
            Tensile strength [MPa]
        SW : float
            Cyclic strength of material [MPa]

        Returns
        -------
        n_st : float
            Statistical support factor
        n_vm : float
            Mechanical support factor
        n_bm : float
            Fracture mechanical support factor
        n : float
            Combined support factor (n_st * n_vm * n_bm)
        """
        n_st = self.stat_support_surf(A90, self.consts[mat]["k_st"])
        n_vm = 1.0 if mat in self.brittle else self.mech_support(n_st, SW, mat, Rm)
        n_bm = self.fract_mech_support_local(n_st, n_vm, G0, Rm, mat)
        n = n_st * n_vm * n_bm
        return n_st, n_vm, n_bm, n

    def support_fkm2012_local_vol(self, mat, G0, V90_Mises, Rm, SW):
        """Calculate support factor according to FKM 2012 local method.

        Uses volume-based approach (Chapter 4.3.1.3.2).

        Parameters
        ----------
        mat : str
            FKM material group. One of:

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

        G0 : float
            Relative stress gradient [1/mm]
        V90_Mises : float
            Maximum loaded volume according to FKM2012 [mm³]
        Rm : float
            Tensile strength [MPa]
        SW : float
            Cyclic strength of material [MPa]

        Returns
        -------
        n_st : float
            Statistical support factor
        n_vm : float
            Mechanical support factor
        n_bm : float
            Fracture mechanical support factor
        n : float
            Combined support factor (n_st * n_vm * n_bm)
        """
        n_st = self.stat_support_vol(V90_Mises, self.consts[mat]["k_st"])
        n_vm = 1.0 if mat in self.brittle else self.mech_support(n_st, SW, mat, Rm)
        n_bm = self.fract_mech_support_local(n_st, n_vm, G0, Rm, mat)
        n = n_st * n_vm * n_bm
        return n_st, n_vm, n_bm, n

    def support_chap5(self, G0, HV_RS):
        """Calculate support factor of surface layer for case-hardened parts.

        According to local FKM 2012 chapter 5.5 method using Cython.

        Parameters
        ----------
        G0 : np.ndarray
            Relative stress gradients [1/mm]
        HV_RS : np.ndarray
            Vickers hardness of the surface layer

        Returns
        -------
        np.ndarray
            Support factors of the surface layer
        """
        return F.support_chap5(G0, HV_RS)

    def kf_local(self, G0, b, n, S_type):
        """Calculate the Kf factor for local FKM 2012 approach.

        According to Chapter 4.3.1.2 using Cython.

        Parameters
        ----------
        G0 : np.ndarray
            Relative stress gradients [1/mm]
        b : np.ndarray
            Fictive specimen width [mm]
        n : np.ndarray
            Notch support factors (FKM2012/Stieler)
        S_type : np.ndarray
            Acting stress types, one of {'normal', 'shear'}

        Returns
        -------
        np.ndarray
            Notch strength reduction factors (Kerbwirkungszahl)
        """
        return F.kf_local(G0, b, n, S_type)

    def kf_constant(self, mat_group):
        """Select Kf factor based on FKM material group from FKM 2012.

        Uses Cython implementation.

        Parameters
        ----------
        mat_group : np.ndarray
            Material groups as strings. One of:

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

        Returns
        -------
        np.ndarray
            Notch strength reduction factors (Kerbwirkungszahl)
        """
        return F.kf_constant(mat_group)

    def kf_factor(self, kf_method, mat_group, G0, b, n, S_type):
        """Select Kf factor based on method and FKM material group from FKM 2012.

        Uses Cython implementation.

        Parameters
        ----------
        kf_method : str
            Method for Kf calculation
        mat_group : np.ndarray
            Material groups as strings. One of:

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

        G0 : np.ndarray
            Relative stress gradients [1/mm]
        b : np.ndarray
            Fictive specimen width [mm]
        n : np.ndarray
            Notch support factors
        S_type : np.ndarray
            Stress types, one of {'normal', 'shear'}

        Returns
        -------
        np.ndarray
            Notch strength reduction factors (Kerbwirkungszahl)
        """
        return F.kf_factor(kf_method, mat_group, G0, b, n, S_type)

    def surf_layer_factor(self, df_proc, G0, Deff, Proc):
        """Select surface treatment factor Kv based on surface process.

        According to FKM 2012 (Chapter 4.3.3) using Cython.

        Parameters
        ----------
        df_proc : pd.DataFrame
            DataFrame including hardening constants
        G0 : np.ndarray
            Relative stress gradients [1/mm]
        Deff : np.ndarray
            Effective specimen sizes [mm]
        Proc : np.ndarray
            Surface hardening methods, one of:

            * 'Inductive hardening'
            * 'Flame hardening'
            * 'Case hardening'
            * 'Carburizing'
            * 'Nitriding'
            * 'Cyaniding'
            * 'Carbonitriding'
            * 'Deep rolling'
            * 'Shot peening'

        Returns
        -------
        np.ndarray
            Surface treatment factors
        """
        return F.surf_layer_factor(
            df_proc.Kv_sm_notched,
            df_proc.Kv_gr_notched,
            df_proc.Kv_sm,
            df_proc.Kv_gr,
            G0,
            Deff,
            Proc,
        )

    def GJL_bending_factor(self, GJL_Mat):
        """Select factor KNL,E for non-linear elastic behavior of GJL in bending.

        According to FKM 2012 (Chapter 4.3.5) using Cython.

        Parameters
        ----------
        GJL_Mat : np.ndarray
            GJL material identifiers

        Returns
        -------
        np.ndarray
            GJL bending factors (KNL,E)
        """
        return F.GJL_bending_factor(GJL_Mat)

    def rough_factor(self, Rm, Rz, df_consts, df_fw_t, S_type, Finish):
        """Calculate roughness influence factor Kr_sig.

        According to FKM 2012 (Chapter 4.3.1.4) using Cython.

        Parameters
        ----------
        Rm : np.ndarray
            Tensile strength [MPa]
        Rz : np.ndarray
            Surface roughness Rz [µm]
        df_consts : pd.DataFrame
            DataFrame including material constants
        df_fw_t : pd.DataFrame
            DataFrame including stress type related constants
        S_type : np.ndarray
            Stress types, one of {'normal', 'shear'}
        Finish : np.ndarray
            Finish procedures, one of {'polished', 'None'}

        Returns
        -------
        np.ndarray
            Roughness influence factors
        """
        fw_t = df_fw_t.iloc[0].loc[S_type].values
        return F.rough_factor(Rm, Rz, fw_t, df_consts.aR_s, df_consts.RmNmin, Finish)

    def design_factor(self, n, Kf, Kr, Kv, Ks, Knle):
        """Calculate design influence factor Kwk.

        According to FKM 2012 (Chapter 4.3.1.1) using Cython.

        Parameters
        ----------
        n : np.ndarray
            Support factors
        Kf : np.ndarray
            Notch strength reduction factors (Kerbwirkungszahl)
        Kr : np.ndarray
            Roughness influence factors
        Kv : np.ndarray
            Surface hardening factors
        Ks : np.ndarray
            Protective layer influence factors
        Knle : np.ndarray
            Factors for GJL materials at bending (Knle=1 for all other cases)

        Returns
        -------
        np.ndarray
            Design influence factors
        """
        return F.design_factor(n, Kf, Kr, Kv, Ks, Knle)

    def sm_sensitivity_chap4(self, Rm, df_consts, df_fw_t, S_type):
        """Calculate mean stress sensitivity factor.

        According to FKM 2012 (Chapter 4.4.2.1.2) using Cython.

        Parameters
        ----------
        Rm : np.ndarray
            Tensile strengths [MPa]
        df_consts : pd.DataFrame
            DataFrame including material constants
        df_fw_t : pd.DataFrame
            DataFrame including stress-type related constants
        S_type : np.ndarray
            Stress types, one of {'shear', 'normal'}

        Returns
        -------
        np.ndarray
            Mean stress sensitivity factors
        """
        fw_t = df_fw_t.iloc[0].loc[S_type].values
        return F.sm_sensitivity_chap4(Rm, df_consts.a_M, df_consts.b_M, fw_t)

    def sm_sensitivity_Rm_trans(self, HV_core):
        """Calculate tensile strength Rm at the core.

        According to chapter 5.5 using Cython.

        Parameters
        ----------
        HV_core : np.ndarray
            Core hardness [HV]

        Returns
        -------
        np.ndarray
            Tensile strengths at core [MPa]
        """
        return F.sm_sensitivity_Rm_trans(HV_core)

    def sm_sensitivity_M_trans(self, Rm, df_consts, df_fw_t, S_type, Rm_trans):
        """Calculate mean stress sensitivity factor for chapter 5.5.

        According to FKM 2012 using Cython.

        Parameters
        ----------
        Rm : np.ndarray
            Tensile strengths [MPa]
        df_consts : pd.DataFrame
            DataFrame including material constants
        df_fw_t : pd.DataFrame
            DataFrame including stress-type related constants
        S_type : np.ndarray
            Stress types, one of {'shear', 'normal'}
        Rm_trans : np.ndarray
            Tensile strengths at the core [MPa]

        Returns
        -------
        np.ndarray
            Mean stress sensitivity factors
        """
        fw_t = df_fw_t.iloc[0].loc[S_type].values
        return F.sm_sensitivity_M_trans(
            Rm, df_consts.a_M, df_consts.b_M, fw_t, Rm_trans
        )

    def eigenstress_RS(self, Rm, HV, HV_core, HardProc):
        """Calculate eigenstress of surface layer for surface treated components.

        According to FKM 2012 (Chapter 5.5.2.1) using Cython.

        Parameters
        ----------
        Rm : np.ndarray
            Strengths of material in initial state [MPa]
        HV : np.ndarray
            Hardness of surface layer [HV]
        HV_core : np.ndarray
            Hardness of core [HV]
        HardProc : np.ndarray
            Hardening processes, one of:

            * 'Carburizing'
            * 'Nitriding'
            * 'Inductive hardening'

        Returns
        -------
        np.ndarray
            Eigenstresses for the surface layer [MPa]
        """
        return F.eigenstress_RS(Rm, HV, HV_core, HardProc)

    def sm_factor(self, R, M, SmSa):
        """Calculate mean stress sensitivity factor KAK for overload case 2.

        R=const. according to Chapter 4 using Cython.

        Parameters
        ----------
        R : np.ndarray
            Stress ratios
        M : np.ndarray
            Mean stress sensitivities
        SmSa : np.ndarray
            Ratios between mean stress and amplitudes

        Returns
        -------
        np.ndarray
            KAK factors
        """
        return F.sm_factor(R, M, SmSa)

    def sm_factor_chap5(self, Rm_trans, M, SE, Swk, sL, Rm_norm):
        """Calculate mean stress sensitivity factor KAK for overload case 2.

        R=const. according to FKM 2021 (Chapter 5.5.1.2) using Cython.

        Parameters
        ----------
        Rm_trans : np.ndarray
            Tensile strengths based on HV_Core [MPa]
        M : np.ndarray
            Mean stress sensitivity factors
        SE : np.ndarray
            Eigen stresses [MPa]
        Swk : np.ndarray
            Reversed strengths of design element [MPa]
        sL : np.ndarray
            Load ratio parameters
        Rm_norm : np.ndarray
            Non-heat treated tensile strengths [MPa]

        Returns
        -------
        np.ndarray
            Mean stress influence factors (KAK)
        """
        return F.sm_factor_chap5(Rm_trans, M, SE, Swk, sL, Rm_norm)

    def temperature_model(self, df, temperature, mat):
        """Compute temperature factor.

        Uses Cython implementation.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame including temperature constants depending on temperature group
        temperature : np.ndarray
            Temperature values [°C]
        mat : np.ndarray
            Material groups. One of:

            * 'CaseHard_Steel' : Case-hardened steels
            * 'Stainless_Steel' : Stainless steels
            * 'Forg_Steel' : Forged steels
            * 'Steel' : All other steel groups
            * 'GS' : Cast steels and tempering cast steels
            * 'GJS' : Cast iron with spheroidal graphite (old GGG)
            * 'GJM' : Heart fittings (Temperguss, old: GT)
            * 'GJL' : Cast iron with lamellar graphite (old GG)
            * 'Al_wrought' : Wrought aluminum alloys
            * 'Al_cast' : Cast aluminum alloys

        Returns
        -------
        np.ndarray
            Temperature factors
        """
        return F.temperature_model(
            df.T_threshold, df.T_max, df.a_TD, df.delta_T, temperature, mat
        )


def _replace_proc_string(entry):
    """Replace hardening procedure string with valid value or 'None'.

    Parameters
    ----------
    entry : str
        Hardening procedure name

    Returns
    -------
    str
        Valid procedure name or 'None'
    """
    ls1 = [
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
    if entry in ls1:
        return entry
    else:
        return "None"

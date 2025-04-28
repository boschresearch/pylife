# Copyright (c) 2019-2023 - for information on the respective copyright owner
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

__author__ = ["Benjamin Maier", "Sebastian Bucher", "Kristina Lepper"]
__maintainer__ = __author__

import numpy as np
import pandas as pd


class FKMNLConstants:
    """A singleton class that contains all the FKM non-linear constants."""


    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FKMNLConstants, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Constants for various formulas used in the FKM nonlinear procedure."""
        self._all_constants = pd.DataFrame({
            "Steel": {
                # general values and values for P_RAM
                "E": 206e3,             # table 2.7 (2.16)
                "n_prime": 0.187,       # table 2.8 (2.17)
                "a_sigma": 3.1148,
                "a_epsilon": 1033,
                "b_sigma": 0.897,
                "b_epsilon": -1.235,
                "epsilon_grenz": 0.338,
                "f_25percent_damage_woehler": 0.86,    # table 2.9
                "a_PZ_RAM": 20.0,           # table 2.10 (2.18), note that values for P_RAJ are different
                "b_PZ_RAM": 0.587,
                "a_PD_RAM": 0.82,
                "b_PD_RAM": 0.92,
                "d_1": -0.302,
                "d_2": -0.197,
                "f_25percent_material_woehler_FKM_nonlinear_RAM": 0.71,     # table 2.18
                "f_25percent_material_woehler_FKM_roughness_RAM": 0.73,     # table 35 in report "Rauheit und Randschicht 2022"
                "k_st": 30,             # table 2.11 (2.19)
                "a_RP": 0.27,           # table 2.12 (2.20)
                "b_RP": 0.43,
                "R_m_N_min": 400,
                "a_M": 0.35,            # table 2.14 (2.22)
                "b_M": -0.1,
                "R_m_bm": 680,          # eq. (2.5-33)

                # values for P_RAJ
                "d_RAJ": -0.63,         # table 2.33 (2.41) Note, this is -0.56 in the FKM nonlinear document, however due to an error by the authors, the value was corrected later (e-mail of Moritz Hupka 15.09.22 10:27)
                "f_25percent_material_woehler_FKM_nonlinear_RAJ": 0.39,   # This value is listed as 0.39, table 2.33 in FKM nonlinear 2019, and as 0.35 in table 35 in report "Rauheit und Randschicht 2022".
                "f_25percent_material_woehler_FKM_roughness_RAJ": 0.25,   # table 35 in report "Rauheit und Randschicht 2022"
                "a_PZ_RAJ": 10,         # Note, this is 1.173 in the FKM nonlinear document, however due to an error by the authors, the value was corrected later (e-mail of Moritz Hupka 15.09.22 10:27)
                "b_PZ_RAJ": 0.826,      # Note, this is 1 in the FKM nonlinear document, however due to an error by the authors, the value was corrected later (e-mail of Moritz Hupka 15.09.22 10:27)
                "a_PD_RAJ": 3.33e-5,
                "b_PD_RAJ": 1.55,
            },
            "SteelCast": {
                # general values and values for P_RAM
                "E": 206e3,             # table 2.7 (2.16)
                "n_prime": 0.176,       # table 2.8 (2.17)
                "a_sigma": 1.732,
                "a_epsilon": 0.847,
                "b_sigma": 0.982,
                "b_epsilon": -0.181,
                "epsilon_grenz": np.inf,
                "f_25percent_damage_woehler": 0.68,    # table 2.9
                "a_PZ_RAM": 25.56,      # table 2.10 (2.18), note that values for P_RAJ are different
                "b_PZ_RAM": 0.519,
                "a_PD_RAM": 0.46,
                "b_PD_RAM": 0.96,
                "d_1": -0.289,
                "d_2": -0.189,
                "f_25percent_material_woehler_FKM_nonlinear_RAM": 0.51,
                "k_st": 15,             # table 2.11 (2.19)
                "a_RP": 0.25,           # table 2.12 (2.20)
                "b_RP": 0.42,
                "R_m_N_min": 400,
                "a_M": 0.35,            # table 2.14 (2.22)
                "b_M": 0.05,
                "R_m_bm": 680,          # eq. (2.5-33)

                # values for P_RAJ
                "d_RAJ": -0.66,         # table 2.33 (2.41)
                "f_25percent_material_woehler_FKM_nonlinear_RAJ": 0.40,
                "a_PZ_RAJ": 10.03,
                "b_PZ_RAJ": 0.695,
                "a_PD_RAJ": 5.15e-6,
                "b_PD_RAJ": 1.63,
            },
            "Al_wrought": {
                # general values and values for P_RAM
                "E": 70e3,              # table 2.7 (2.16)
                "n_prime": 0.128,       # table 2.8 (2.17)
                "a_sigma": 9.12,
                "a_epsilon": 895.9,
                "b_sigma": 0.742,
                "b_epsilon": -1.183,
                "epsilon_grenz": np.inf,
                "f_25percent_damage_woehler": 0.88,    # table 2.9
                "a_PZ_RAM": 16.71,      # table 2.10 (2.18), note that values for P_RAJ are different
                "b_PZ_RAM": 0.537,
                "a_PD_RAM": 0.30,
                "b_PD_RAM": 1.00,
                "d_1": -0.238,
                "d_2": -0.167,
                "f_25percent_material_woehler_FKM_nonlinear_RAM": 0.61,
                "k_st": 20,             # table 2.11 (2.19)
                "a_RP": 0.27,           # table 2.12 (2.20)
                "b_RP": 0.43,
                "R_m_N_min": 133,
                "a_M": 1.0,             # table 2.14 (2.22)
                "b_M": -0.04,
                "R_m_bm": 270,          # eq. (2.5-33)

                # values for P_RAJ
                "d_RAJ": -0.61,         # table 2.33 (2.41)
                "f_25percent_material_woehler_FKM_nonlinear_RAJ": 0.36,
                "a_PZ_RAJ": 101.7,
                "b_PZ_RAJ": 0.26,
                "a_PD_RAJ": 5.18e-7,
                "b_PD_RAJ": 2.04,
            }
        })

    def for_material_group(self, assessment_parameters):
        """
        Retrieve the constants for one of the three material groups that are defined in FKM nonlinear.

        .. note::

            The constants for all material groups can be accessed as
            ``pylife.strength.fkm_nonlinear.constants.all_constants``.

        Parameters
        ----------
        assessment_parameters : pandas Series
            A Series with at least the item ``MatGroupFKM``, which has to be one of
            ``Steel``, ``SteelCast``, ``Al_wrought``.

        Returns
        -------
        pandas Series
            All constants that are defined by FKM nonlinear for the given material group.

        """
        # select set of constants according to given material group
        assert "MatGroupFKM" in assessment_parameters

        material_group = assessment_parameters["MatGroupFKM"]

        resulting_constants = self._all_constants[material_group]

        # Rename the key for the safety factor f_25%
        resulting_constants["f_25percent_material_woehler_RAM"] \
            = resulting_constants["f_25percent_material_woehler_FKM_nonlinear_RAM"]
        resulting_constants["f_25percent_material_woehler_RAJ"] \
            = resulting_constants["f_25percent_material_woehler_FKM_nonlinear_RAJ"]

        return resulting_constants

    def add_custom_material(self, material_group_name, material_constants):
        """Add a custom material to the global FKMNL constants.


        Paramters
        ---------
        material_group_name : str
            The name of the custom material

        material_constants : pd.Series | dict
            The constants for the custom material

        Return
        ------
        self

        Raises
        ------
        ValueError if the material already exists.
        """
        if material_group_name in self._all_constants:
            raise ValueError(f"Material `{material_group_name}` already exists.")
        self._all_constants[material_group_name] = pd.Series(material_constants)
        return self

    def __iter__(self):
        for mg in self._all_constants:
            yield mg

    def __getitem__(self, material_group_name):
        return self._all_constants[material_group_name].copy()

    def to_pandas(self):
        """Optiain a copy of all material constants."""
        return self._all_constants.copy()

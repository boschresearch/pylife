# Copyright (c) 2019-2021 - for information on the respective copyright owner
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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import pandas as pd

from pylife.materiallaws import WoehlerCurve


@pd.api.extensions.register_series_accessor('fatigue')
@pd.api.extensions.register_dataframe_accessor('fatigue')
class Fatigue(WoehlerCurve):
    """Extension for `WoehlerCurve` accessor class for fatigue calculations.

    Note
    ----
    This class is accessible by the `fatigue` accessor attribute.
    """

    def damage(self, load_collective):
        """Calculate the damage to the material caused by a given load collective.

        Parameters
        ----------
        load_collective : pandas object or object behaving like a load collective
            The given load collective

        Returns
        -------
        damage : pd.Series
            The calculated damage values. The index is the broadcast between
            `load_collective` and `self`.
        """
        cycles = self.basquin_cycles(load_collective.amplitude)
        return pd.Series(load_collective.cycles / cycles, name='damage')

    def security_load(self, load_collective, allowed_failure_probability):
        """Calculate the security factor in load direction for given load collective.

        Parameters
        ----------
        load_collective : pandas object or object behaving like a load collective
            The given load collective

        Returns
        -------
        security_factor : pd.Series
            The calculated security_factors. The index is the broadcast between
            `load_collective` and `self`.
        """
        allowed_load = self.basquin_load(load_collective.cycles, allowed_failure_probability)
        return pd.Series(allowed_load / load_collective.amplitude, name='security_factor')

    def security_cycles(self, load_collective, allowed_failure_probability):
        """Calculate the security factor in cycles direction for given load collective.

        Parameters
        ----------
        load_collective : pandas object or object behaving like a load collective
            The given load collective

        Returns
        -------
        security_factor : pd.Series
            The calculated security_factors. The index is the broadcast between
            `load_collective` and `self`.
        """
        allowed_cycles = self.basquin_cycles(load_collective.amplitude, allowed_failure_probability)
        return pd.Series(allowed_cycles / load_collective.cycles, name='security_factor')

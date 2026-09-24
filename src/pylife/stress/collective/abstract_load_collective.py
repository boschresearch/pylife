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

"""Define the common interface for pyLife load collective accessors."""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import pandas as pd

from abc import ABC, abstractmethod


class AbstractLoadCollective(ABC):
    r"""Define the contract shared by load collective implementations.

    Alternative implementations must provide load amplitude, mean load, and
    number of cycles for each collective entry.  The base class derives the
    upper and lower turning load from these quantities.

    Notes
    -----
    The common quantities are related by

    .. math::

        S_\mathrm{upper} = S_\mathrm{mean} + S_\mathrm{a}

        S_\mathrm{lower} = S_\mathrm{mean} - S_\mathrm{a}

    where ``S_a`` is the load amplitude and ``S_mean`` is the mean load.  In
    pyLife these load values are commonly stresses in MPa, but the interface is
    unit-agnostic as long as all load-like quantities use the same unit.
    """

    @property
    @abstractmethod
    def amplitude(self):
        """Calculate the load amplitude for each collective entry.

        Returns
        -------
        pandas.Series
            Load amplitude in the same unit as the source load values,
            typically MPa.
        """
        pass

    @property
    @abstractmethod
    def meanstress(self):
        """Calculate the mean load for each collective entry.

        Returns
        -------
        pandas.Series
            Mean load in the same unit as the source load values, typically
            MPa.
        """
        pass

    @property
    @abstractmethod
    def cycles(self):
        """Return the number of cycles for each collective entry.

        Returns
        -------
        pandas.Series
            Number of cycles represented by each entry.
        """
        pass

    @property
    def upper(self):
        """Calculate the upper turning load for each collective entry.

        Returns
        -------
        pandas.Series
            Upper load in the same unit as the source load values, typically
            MPa.
        """
        return pd.Series(self.meanstress + self.amplitude, name='upper')

    @property
    def lower(self):
        """Calculate the lower turning load for each collective entry.

        Returns
        -------
        pandas.Series
            Lower load in the same unit as the source load values, typically
            MPa.
        """
        return pd.Series(self.meanstress - self.amplitude, name='lower')

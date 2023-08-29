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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import pandas as pd

from abc import ABC, abstractmethod


class AbstractLoadCollective(ABC):

    @property
    @abstractmethod
    def amplitude(self):
        pass

    @property
    @abstractmethod
    def meanstress(self):
        pass

    @property
    @abstractmethod
    def cycles(self):
        pass

    @property
    def upper(self):
        """Calculate the upper load values of the load collective.

        Returns
        -------
        upper : pd.Series
            The upper load values of the load collective
        """
        return pd.Series(self.meanstress + self.amplitude, name='upper')

    @property
    def lower(self):
        """Calculate the lower load values of the load collective.

        Returns
        -------
        lower : pd.Series
            The lower load values of the load collective
        """
        return pd.Series(self.meanstress - self.amplitude, name='lower')

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
import numpy as np
import scipy.stats as stats

from pylife.stress import stresssignal
from pylife.utils.functions import scatteringRange2std


@pd.api.extensions.register_dataframe_accessor('infinite_security')
class InfiniteSecurityAccessor(stresssignal.CyclicStressAccessor):
    """ Compute Security factor for infinite cyclic stress.

    """

    def factors(self, woehler_data, allowed_failure_probability):
        """compute security factor between strength and existing stress.

        Parameters
        ----------
        woehler_data : dict
            strength_inf: double - value for the allowed infinite cyclic strength
            strength_scatter: double - scattering of strength
        allowed_failure_probability : double
            Failure Probability we allow

        Returns
        -------
        double
            quotient of strength and existing stress

        """
        wd = woehler_data
        std_dev = scatteringRange2std(wd.strength_scatter)
        allowed_stress = 10**stats.norm.ppf(allowed_failure_probability,
                                            loc=np.log10(wd.strength_inf),
                                            scale=std_dev)
        factors = allowed_stress / self._obj.sigma_a

        return pd.DataFrame({'security_factor': factors}, index=self._obj.index)

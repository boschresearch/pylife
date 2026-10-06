# Copyright (c) 2019-2026 - for information on the respective copyright owner
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

"""Provide solidity factors for load collectives.

Solidity, also called the collective shape factor, describes how strongly a
load collective fills the range between zero load and the maximum amplitude.
It is used by Gassner and Miner calculations to scale S-N curve life.
"""

__author__ = "Cedric Philip Wagner"
__maintainer__ = "Johannes Mueller"


import pandas as pd
import numpy as np

import pylife.stress.collective as CL

@pd.api.extensions.register_series_accessor('solidity')
class SolidityAccessor(CL.LoadHistogram):
    """Provide solidity calculations as a load histogram accessor.

    The accessor is registered as ``.solidity`` on :class:`pandas.Series` load
    histograms that satisfy the :class:`pylife.stress.collective.LoadHistogram`
    contract.

    Parameters
    ----------
    pandas_obj : pandas.Series
        Load histogram data passed by the pandas accessor machinery.

    See Also
    --------
    pylife.strength.solidity.haibach : Calculate Haibach solidity.
    pylife.strength.solidity.fkm : Calculate FKM solidity.
    """

    def haibach(self, k):
        """Calculate Haibach solidity for the histogram.

        Parameters
        ----------
        k : float
            Wöhler slope used to weight normalized amplitudes, dimensionless.

        Returns
        -------
        float
            Haibach solidity, dimensionless.
        """
        return haibach(self, k)

    def fkm(self, k):
        """Calculate FKM solidity for the histogram.

        Parameters
        ----------
        k : float
            Wöhler slope used to weight normalized amplitudes, dimensionless.

        Returns
        -------
        float
            FKM solidity, dimensionless.
        """
        return fkm(self, k)


def haibach(collective, k):
    r"""Calculate the Haibach solidity of a load collective.

    Parameters
    ----------
    collective : pylife.stress.collective.LoadCollective
        Load collective or load histogram accessor exposing stress amplitudes
        and cycle counts. Amplitudes may be stress amplitudes or load
        amplitudes, but must use one consistent unit.
    k : float
        Wöhler slope used to weight normalized amplitudes, dimensionless.

    Returns
    -------
    float
        Haibach solidity ``V``, dimensionless.

    Notes
    -----
    Following Haibach [Haibach-Solidity]_, with cycle counts ``n_i`` and
    normalized amplitudes ``x_i = S_{a,i} / S_{a,max}``, Haibach solidity is

    .. math::

        V = \sum_i \frac{n_i}{\sum_j n_j} x_i^k.

    References
    ----------
    .. [Haibach-Solidity] E. Haibach, "Betriebsfestigkeit", Springer-Verlag,
       2006, p. 271.
    """

    S = collective.amplitude
    hi = collective.cycles

    xi = S / S[hi > 0].max()
    V = np.sum((hi * (xi**k)) / hi.sum())

    return V


def fkm(collective, k):
    r"""Calculate the FKM solidity of a load collective.

    Parameters
    ----------
    collective : pylife.stress.collective.LoadCollective
        Load collective or load histogram accessor exposing stress amplitudes
        and cycle counts. Amplitudes may be stress amplitudes or load
        amplitudes, but must use one consistent unit.
    k : float
        Wöhler slope used to weight normalized amplitudes, dimensionless.

    Returns
    -------
    float
        FKM solidity ``V_FKM``, dimensionless.

    Notes
    -----
    According to the FKM guideline [FKM-Solidity]_, the solidity is derived
    from the Haibach solidity ``V_H`` as

    .. math::

        V_{FKM} = V_H^{1/k}.

    References
    ----------
    .. [FKM-Solidity] Forschungskuratorium Maschinenbau, "Rechnerischer
       Festigkeitsnachweis für Maschinenbauteile", 6th ed., 2012,
       Eq. 2.4.55.
    """

    V_haibach = haibach(collective, k)
    V = V_haibach**(1./k)

    return V

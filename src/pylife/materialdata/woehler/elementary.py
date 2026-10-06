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

"""Estimate Wöhler curve start parameters from finite-life test data.

The elementary analyzer provides the common first estimate for all Wöhler
analyzers in this package.  It fits the finite-life slope and estimates the
scatter from the pearl chain method.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

from .likelihood import (
    LikelihoodAllFractures,
    LikelihoodHighestMixedLevel,
    LikelihoodPureFiniteZone,
    LikelihoodLegacy
)
from .pearl_chain import PearlChainProbability
import pylife.utils.functions as functions
from . import FatigueData, determine_fractures
import warnings


class Elementary:
    """Estimate finite-life Wöhler parameters from fracture data.

    ``Elementary`` is the base analyzer for Wöhler test data.  It estimates
    ``k_1``, the finite-life slope, ``SD``, the endurance limit load at the
    knee point, ``ND``, the cycle number at the knee point, and ``TN``, the
    scatter in cycle direction.  ``TS`` is derived from ``TN`` and ``k_1``.

    Choose this analyzer when only a robust first estimate is needed or when
    the data in the endurance-limit region is insufficient for Probit or
    maximum-likelihood evaluation.  Derived analyzers use its result as their
    start value.

    Parameters
    ----------
    fatigue_data : pandas.DataFrame or FatigueData
        Wöhler test data.  A data frame must contain ``load`` and ``cycles``;
        when ``fracture`` is missing, the maximum cycle count is interpreted as
        the runout limit.

    See Also
    --------
    pylife.materialdata.woehler.Probit : Estimate endurance-limit parameters with the Probit method.
    pylife.materialdata.woehler.MaxLikeInf : Refine ``SD`` and ``TS`` by maximum likelihood.
    pylife.materialdata.woehler.MaxLikeFull : Fit all Wöhler parameters by maximum likelihood.

    Notes
    -----
    The finite-life slope is fitted in double-logarithmic load-cycle space.
    The scatter ``TN`` is evaluated with the DIN 50100 pearl chain method,
    which shifts fracture points to a common load level before fitting their
    failure probabilities.
    """

    def __init__(self, fatigue_data):
        """Create an analyzer for Wöhler fatigue data.

        Parameters
        ----------
        fatigue_data : pandas.DataFrame or FatigueData
            Wöhler test data to be analyzed.
        """
        self._fd = self._get_fatigue_data(fatigue_data)
        self.use_highest_mixed_level()

    def use_old_likelihood_estimation(self):
        """Select the likelihood formulation used up to pyLife 2.1.x.

        The legacy formulation uses all fractures for the finite-life
        likelihood and only the infinite zone for the endurance-limit
        likelihood.

        Returns
        -------
        Elementary
            The same analyzer configured with the legacy likelihood.
        """
        self._lh = LikelihoodLegacy(self._fd)
        return self

    def use_highest_mixed_level(self):
        """Select pure finite levels and the highest mixed load level.

        This default formulation uses fractures from pure fracture levels and
        from the highest mixed load level for the finite-life likelihood.

        Returns
        -------
        Elementary
            The same analyzer configured with the default likelihood.
        """
        self._lh = LikelihoodHighestMixedLevel(self._fd)
        return self

    def use_all_fractures(self):
        """Select all fractures for the finite-life likelihood.

        Returns
        -------
        Elementary
            The same analyzer configured to use every fracture test.
        """
        self._lh = LikelihoodAllFractures(self._fd)
        return self

    def use_only_pure_fracture_levels(self):
        """Select only pure fracture levels for the finite-life likelihood.

        Returns
        -------
        Elementary
            The same analyzer configured to ignore mixed levels in the
            finite-life likelihood.
        """
        self._lh = LikelihoodPureFiniteZone(self._fd)
        return self

    def use_custom_likelihood_estimation(self, likelihood_class):
        """Select a custom likelihood calculation class.

        Parameters
        ----------
        likelihood_class : type
            Class implementing the likelihood interface of
            :class:`~pylife.materialdata.woehler.likelihood.AbstractLikelihood`.

        Returns
        -------
        Elementary
            The same analyzer configured with ``likelihood_class``.
        """
        self._lh = likelihood_class(self._fd)
        return self

    def _get_fatigue_data(self, fatigue_data):
        if isinstance(fatigue_data, pd.DataFrame):
            if hasattr(fatigue_data, "fatigue_data"):
                params = fatigue_data.fatigue_data
            else:
                params = determine_fractures(fatigue_data).fatigue_data
        elif isinstance(fatigue_data, FatigueData):
            params = fatigue_data
        else:
            raise ValueError("fatigue_data of type {} not understood: {}".format(type(fatigue_data), fatigue_data))
        params.sanitize_check()
        params = params.irrelevant_runouts_dropped()

        return params

    def analyze(self, **kwargs):
        """Analyze the Wöhler test data.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments forwarded to the specific analyzer
            implementation.

        Returns
        -------
        pandas.Series
            Wöhler curve parameters ``k_1``, ``ND``, ``SD``, ``TN``, ``TS``,
            and ``failure_probability`` for 50 % failure probability.
        """
        if len(self._fd.load.unique()) < 2:
            raise ValueError(
                "Need at least two different load levels in the finite zone to do a Wöhler slope analysis."
            )
        self._raise_if_no_cycle_variance_in_finite_zone()
        if len(self._fd.finite_zone.load.unique()) < 2:
            warnings.warn(
                UserWarning(
                    "Need at least two different load levels in the finite zone to do a Wöhler slope analysis."
                )
            )
            if len(self._fd.finite_zone.load.unique()) == 1:
                wc = pd.Series({
                    'k_1': np.nan,
                    'ND': np.nan,
                    'SD': np.nan,
                    'TN': np.nan,
                    'TS': np.nan
                    })
            else:
                wc = pd.Series({
                    'k_1': np.inf,
                    'ND': np.nan,
                    'SD': np.nan,
                    'TN': 1.0,
                    'TS': np.nan
                    })
            wc = self._specific_analysis(wc, **kwargs)
            wc['failure_probability'] = 0.5
            return wc

        self._finite_fractures = self._fd.finite_zone.loc[self._fd.finite_zone.fracture == True]
        wc = self._common_analysis()
        wc = self._specific_analysis(wc, **kwargs)
        self.__calc_bic(wc)
        wc['failure_probability'] = 0.5

        return wc

    def _raise_if_no_cycle_variance_in_finite_zone(self):
        finite_zone = self._fd.finite_zone
        finite_fractures_cycles = finite_zone.loc[finite_zone['fracture'], 'cycles']
        if finite_fractures_cycles.max() == finite_fractures_cycles.min():
            raise ValueError(
                "Cycle numbers must spread in finite zone to do a Wöhler slope analysis."
            )

    def _common_analysis(self):
        self._slope, self._lg_intercept = self._fit_slope()
        TN, TS = self._pearl_chain_method()
        return pd.Series({
            'k_1': -self._slope,
            'ND': self._transition_cycles(self._fd.finite_infinite_transition),
            'SD': self._fd.finite_infinite_transition,
            'TN': TN,
            'TS': TS
        })

    def _specific_analysis(self, wc):
        return wc

    def bayesian_information_criterion(self):
        """Return the Bayesian information criterion of the last analysis.

        Returns
        -------
        float
            Bayesian information criterion value.  Lower values indicate a
            better fit for the same likelihood formulation.

        Raises
        ------
        ValueError
            Raised when :meth:`analyze` has not been called yet.

        Notes
        -----
        The BIC is not suitable for comparing results from different
        likelihood formulations because the underlying model definition
        changes.
        """
        if not hasattr(self,"_bic"):
            raise ValueError("BIC value undefined. Analysis has not been conducted.")
        return self._bic

    def pearl_chain_estimator(self):
        """Return the pearl chain probability estimator of the last analysis.

        Returns
        -------
        PearlChainProbability
            Probability fit created during the pearl chain scatter estimate.
        """
        return self._pearl_chain_estimator

    def __calc_bic(self, wc):
        
        param_num = 5  # SD, TS, k_1, ND, TN
        log_likelihood = self._lh.likelihood_total(wc['SD'], wc['TS'], wc['k_1'], wc['ND'], wc['TN'])
        self._bic = (-2 * log_likelihood) + (param_num * np.log(self._fd.num_tests))

    def _fit_slope(self):
        slope, lg_intercept, _, _, _ = stats.linregress(np.log10(self._finite_fractures.load),
                                                        np.log10(self._finite_fractures.cycles))

        return slope, lg_intercept

    def _transition_cycles(self, finite_infinite_transition):
        # FIXME Elementary means finite_infinite_transition == 0 -> np.inf
        if finite_infinite_transition == 0:
            finite_infinite_transition = 0.1
        return 10**(self._lg_intercept + self._slope * (np.log10(finite_infinite_transition)))

    def _pearl_chain_method(self):
        self._pearl_chain_estimator = PearlChainProbability(self._finite_fractures, self._slope)

        TN = functions.std_to_scattering_range(1./self._pearl_chain_estimator.slope)
        TS = TN**(1./-self._slope)

        return TN, TS

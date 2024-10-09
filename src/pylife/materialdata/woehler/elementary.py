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

import numpy as np
import pandas as pd
import scipy.stats as stats

from .likelihood import Likelihood
from .pearl_chain import PearlChainProbability
import pylife.utils.functions as functions
from . import FatigueData, determine_fractures
import warnings

class Elementary:
    """Base class to analyze SN-data.

    The common base class for all SN-data analyzers calculates the first
    estimation of a Wöhler curve in the finite zone of the SN-data. It
    calculates the slope `k`, the fatigue limit `SD`, the transition cycle
    number `ND` and the scatter in load direction `1/TN`.

    The result is just meant to be a first guess. Derived classes are supposed
    to use those first guesses as starting points for their specific
    analysis. For that they should implement the method `_specific_analysis()`.
    """

    def __init__(self, fatigue_data):
        """The constructor.

        Parameters
        ----------
        fatigue_data : pd.DataFrame or FatigueData
           The SN-data to be analyzed.
        """
        self._fd = self._get_fatigue_data(fatigue_data)
        self._lh = self._get_likelihood()

    def _get_likelihood(self):
        return Likelihood(self._fd)

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
        params = params.irrelevant_runouts_dropped()

        return params

    def analyze(self, **kwargs):
        """Analyze the SN-data.

        Parameters
        ----------
        **kwargs : kwargs arguments
            Arguments to be passed to the derived class
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
        """The Bayesian Information Criterion

        Bayesian Information Criterion is a criterion for model selection among
        a finite set of models; the model with the lowest BIC is preferred.
        https://www.statisticshowto.datasciencecentral.com/bayesian-information-criterion/

        Basically the lower the better the fit.
        """
        if not hasattr(self,"_bic"):
            raise ValueError("BIC value undefined. Analysis has not been conducted.")
        return self._bic

    def pearl_chain_estimator(self):
        return self._pearl_chain_estimator

    def __calc_bic(self, wc):
        '''         '''
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

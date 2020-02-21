
import numpy as np
import pandas as pd
import scipy.stats as stats

import pylife
from .likelihood import Likelihood
import pylife.utils.functions as functions


class Elementary:
    def __init__(self, df):
        self._fd = df.fatigue_data
        self._df = df
        self._lh = Likelihood(self._fd)

    def analyze(self, **kw):
        self._slope, self._lg_intercept = self._fit_slope()
        TN_inv, TS_inv = self._pearl_chain_method()
        wc = pd.Series({
            'k_1': -self._slope,
            'ND_50': self._transition_cycles(self._fd.fatigue_limit),
            'SD_50': self._fd.fatigue_limit, '1/TN': TN_inv, '1/TS': TS_inv
        })

        wc = self._specific_analysis(wc, **kw)
        self.__calc_bic(wc)
        return wc

    def _specific_analysis(self, wc):
        return wc

    def bayesian_information_criterion(self):
        return self._bic

    def __calc_bic(self, wc):
        ''' Bayesian Information Criterion: is a criterion for model selection among a finite set of models;
        the model with the lowest BIC is preferred.
        https://www.statisticshowto.datasciencecentral.com/bayesian-information-criterion/
        '''
        param_num = 5 # SD_50, 1/TS, k_1, ND_50, 1/TN
        log_likelihood = self._lh.likelihood_total(wc['SD_50'], wc['1/TS'], wc['k_1'], wc['ND_50'], wc['1/TN'])
        self._bic = (-2*log_likelihood)+(param_num*np.log(self._df.shape[0]))


    def _fit_slope(self):
        '''Computes the slope of the finite zone with the help of a linear regression function
        '''
        slope, lg_intercept, _, _, _ = stats.linregress(np.log10(self._fd.fractures.load),
                                                        np.log10(self._fd.fractures.cycles))

        return slope, lg_intercept

    def _transition_cycles(self, fatigue_limit):
        # FIXME Elementary means fatigue_limit == 0 -> np.inf
        if fatigue_limit == 0:
            fatigue_limit = 0.1
        return 10**(self._lg_intercept + self._slope*(np.log10(fatigue_limit)))

    def _pearl_chain_method(self):
        '''
        Pearl chain method: consists of shifting the fractured data to a median load level.
        The shifted data points are assigned to a Rossow failure probability.The scatter in load-cycle
        direction can be computed from the probability net.
        '''
        fr = self._fd.fractures

        mean_fr_load = fr.load.mean()
        self._normed_cycles = np.sort(fr.cycles * ((mean_fr_load/fr.load)**(self._slope)))

        fp = functions.rossow_cumfreqs(len(self._normed_cycles))
        percentiles = stats.norm.ppf(fp)
        prob_slope, prob_intercept, _, _, _ = stats.linregress(np.log10(self._normed_cycles), percentiles)
        # Scatter in load cycle direction
        TN_inv = functions.std2scatteringRange(1./prob_slope)
        # Scatter in load direction
        # Empirical method "following Koeder" to estimate the scatter in load direction '
        TS_inv = TN_inv**(1./-self._slope)

        return TN_inv, TS_inv

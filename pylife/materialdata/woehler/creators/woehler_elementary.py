
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import optimize
import mystic as my

import pylife
import pylife.materialdata.woehler as woehler
from pylife.materialdata.woehler.creators.likelihood import Likelihood
import pylife.utils.functions as functions


class WoehlerElementary:
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


class WoehlerProbit(WoehlerElementary):
    def _specific_analysis(self, wc):
        wc['1/TS'], wc['SD_50'], wc['ND_50'] = self.__probit_analysis()
        return wc

    def __probit_rossow_estimation(self):
        inf_groups = self._fd.infinite_zone.groupby('load')
        frac_num = inf_groups.fracture.sum().astype(int).to_numpy()
        tot_num = inf_groups.fracture.count().to_numpy()

        fprobs = np.empty_like(frac_num, dtype=np.float)
        c1 = frac_num == 0
        w = np.where(c1)
        fprobs[w] = 1. - 0.5**(1./tot_num[w])

        c2 = frac_num == tot_num
        w = np.where(c2)
        fprobs[w] = 0.5**(1./tot_num[w])

        c3 = np.logical_and(np.logical_not(c1), np.logical_not(c2))
        w = np.where(c3)
        fprobs[w] = (3*frac_num[w] - 1) / (3*tot_num[w] + 1)
        return fprobs, inf_groups.load.mean()

    def __probit_analysis(self):
        fprobs, load = self.__probit_rossow_estimation()
        ppf = stats.norm.ppf(fprobs)
        probit_slope, probit_intercept, _, _, _ = stats.linregress(np.log10(load), ppf)

        TS_inv = functions.std2scatteringRange(1./probit_slope)

        SD_50 = 10**(-probit_intercept/probit_slope)
        ND_50 = self._transition_cycles(SD_50)

        return TS_inv, SD_50, ND_50


class WoehlerMaxLikeInf(WoehlerElementary):
    def _specific_analysis(self, wc):
        SD_50, TS_inv = self.__max_likelihood_inf_limit()

        wc['SD_50'] = SD_50
        wc['1/TS'] = TS_inv
        wc['ND_50'] = self._transition_cycles(SD_50)

        return wc

    def __max_likelihood_inf_limit(self):
        ''' This maximum likelihood procedure estimates the load endurance limit SD50_mali_2_param and the
        scatter in load direction TS_mali_2_param.
        Moreover, the load cycle endurance is computed by the interesecting endurance limit line with the
        line of slope k_1
        '''
        SD_start = self._fd.fatigue_limit
        TS_start = 1.2

        var_opt = optimize.fmin(lambda p: -self._lh.likelihood_infinite(p[0], p[1]),
                                [SD_start, TS_start], disp=False, full_output=True)

        SD_50 = var_opt[0][0]
        TS_inv = var_opt[0][1]

        return SD_50, TS_inv


class WoehlerMaxLikeFull(WoehlerElementary):
    def _specific_analysis(self, wc, fixed_parameters={}):
        return pd.Series(self.__max_likelihood_full(wc, fixed_parameters))

    def __max_likelihood_full(self, initial_wcurve, fixed_prms):
        """
        Maximum likelihood is a method of estimating the parameters of a distribution model by maximizing
        a likelihood function, so that under the assumed statistical model the observed data is most probable.
        This procedure consists of estimating the Woehler curve parameters, where some of these paramters may
        be fixed by the user. The remaining paramters are then fitted to produce the best possible outcome.
        The procedure uses the function Optimize.fmin
        Optimize.fmin iterates over the function mali_sum_lolli values till it finds the minimum

        https://en.wikipedia.org/wiki/Maximum_likelihood_estimation

        Parameters
        ----------
        self.p_opt: Start values of the Mali estimated parameters if none are fixed by the user.

        self.dict_bound: Boundary values of the Mali estimated parameters if none are fixed by the user.
        This forces the optimizer to search for a minimum solution within a given area.


        Returns
        -------
        self.Mali_5p_result: The estimated parameters computed using the optimizer.

        """

        p_opt = initial_wcurve.to_dict()
        for k in fixed_prms:
            p_opt.pop(k)

        if not p_opt:
            raise AttributeError('You need to leave at least one parameter empty!')
        var_opt = my.scipy_optimize.fmin(
            self.__likelihood_wrapper, [*p_opt.values()],
            args=([*p_opt], fixed_prms),
            full_output=True,
            disp=True,
            maxiter=1e4,
            maxfun=1e4,
        )
        res = {}
        res.update(fixed_prms)
        res.update(zip([*p_opt], var_opt[0]))

        return res

    def __likelihood_wrapper(self, var_args, var_keys, fix_args):
        ''' 1) Finds the start values to be optimized. The rest of the paramters are fixed by the user.
            2) Calls function mali_sum_lolli to calculate the maximum likelihood of the current
            variable states.
        '''
        args = {}
        args.update(fix_args)
        args.update(zip(var_keys, var_args))

        return -self._lh.likelihood_total(args['SD_50'], args['1/TS'], args['k_1'], args['ND_50'], args['1/TN'])

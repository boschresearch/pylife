# Copyright (c) 2019 - for information on the respective copyright owner
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

__author__ = "Mustapha Kassem"
__maintainer__ = "Johannes Mueller"

import numpy as np
import pandas as pd
import numpy.ma as ma
from scipy import stats, optimize
import mystic as my


class WoehlerCurve:

    load_cycle_limit = None

    def __init__(self, data, load_cycle_limit, param_fix, param_estim):

        self.data = data
        self.param_fix = param_fix
        self.param_estim = param_estim
        self.load_cycle_limit = load_cycle_limit
        
    def __data_sort(self):

        self.loads_max = self.data.loads.max()
        self.loads_min = self.data.loads.min()

        self.cycles_max = self.data.cycles.max()
        self.cycles_min = self.data.cycles.min()

        self.fractures = self.data[self.data.cycles < self.load_cycle_limit]
        self.runouts = self.data[self.data.cycles >= self.load_cycle_limit]

        self.__calc_ld_endur_zones()

        self.zone_inf_fractures = self.fractures[self.fractures.loads < self.fatg_lim]

        self.ld_lvls = np.unique(self.data.loads, return_counts=True)
        self.ld_lvls_fin = np.unique(self.zone_fin.loads, return_counts=True)
        self.ld_lvls_inf = np.unique(self.zone_inf.loads, return_counts=True)
        self.ld_lvls_inf_frac = np.unique(self.zone_inf_fractures.loads, return_counts=True)


    def __calc_ld_endur_zones(self):
        '''
        Computes the start value of the load endurance limit. This is done by searching for the lowest load
        level before the appearance of a runout data point, and the first load level where a runout appears.
        Then the median of the two load levels is the start value.
        '''

        self.zone_fin = self.fractures[self.fractures.loads > self.runouts.loads.max()]
        zone_fin_min = self.zone_fin.loads.min()
        if zone_fin_min == 0:
            self.fatg_lim = self.runouts.loads.max()
        else:
            self.fatg_lim = np.mean([zone_fin_min, self.runouts.loads.max()])
        self.zone_inf = self.data[self.data.loads<= self.fatg_lim]


    '# Evaluation of the finite zone'
    def __slope(self):
        '# Computes the slope of the finite zone with the help of a linear regression function'

        self.a_wl, self.b_wl, _, _, _ = stats.linregress(np.log10(self.fractures.loads),
                                                         np.log10(self.fractures.cycles)
                                                         )

        '# Woehler Slope'
        self.k = -self.a_wl
        '# Cycle for load = 1'
        self.N0 = 10**self.b_wl
        '# Load-cycle endurance start value relative to the load endurance start value'
        self.N_E = 10**(self.b_wl + self.a_wl*(np.log10(self.fatg_lim)))


    def __deviation(self):
        '''
        Pearl chain method: consists of shifting the fractured data to a median load level.
        The shifted data points are assigned to a Rossow failure probability.The scatter in load-cycle
        direction can be computed from the probability net.
        '''
        # Mean load level:
        self.Sa_shift = np.mean(self.fractures.loads)

        # Shift probes to the mean load level
        self.N_shift = self.fractures.cycles * ((self.Sa_shift/self.fractures.loads)**(-self.k))
        self.N_shift = np.sort(self.N_shift)

        self.fp = self.rossow_fail_prob(self.N_shift)
        self.u = stats.norm.ppf(self.fp)

        self.a_pa, self.b_pa, _, _, _ = stats.linregress(np.log10(self.N_shift), self.u)

        # Scatter in load cycle direction
        self.TN = 10**(2.5631031311*(1./self.a_pa))

        # Scatter in load direction
        '# Empirical method "following Koeder" to estimate the scatter in load direction '
        self.TS = self.TN**(1./self.k)


    '# Evaluation of the infinite zone:'
    def __probit_procedure(self):
        '''
        Probit procedure uses the probit function to compute the failure probability of the infinite
        zone, in order to estimate the endurance parameters as well as the scatter in load direction
        '''
        # Probaility regression plot
        self.inv_cdf = stats.norm.ppf(self.FP)
        self.a_ue, self.b_ue, _, _, _ = stats.linregress(np.log10(self.ld_lvls_inf[0]), self.inv_cdf)
        # Deviation TS in load-cycle direction
        TS_probit = 10**(2.5631031311*(1./self.a_ue))
        # Average fatigue strength
        SD50_probit = 10**((0-self.b_ue)/self.a_ue)
        # Average endurance load cycle
        ND50_probit = 10**(self.b_wl+self.a_wl*np.log10(SD50_probit))

        self.Probit_result = {'SD_50':SD50_probit, '1/TS':TS_probit,'ND_50':ND50_probit}


    def __probit(self):
        """
        Probit, probability unit, is the inverse cumulative distribution function (CDF).
        Describes the failure probability of the infinite zone.

        Parameters
        ----------
        data_probit:
            - A three column table containing:
                1st column: Load levels in the infinite zone
                2nd column: The quantity of data points found in the respective load level
                3rd column: Of these data points, the quantity of fractured data points

        Returns
        -------
        self.FP: The failure probability following rossow's method for the infinite zone
        """
        data_probit = np.zeros((len(self.ld_lvls_inf[0]), 3))

        data_probit[:, 0] = self.ld_lvls_inf[0]
        data_probit[:, 1] = self.ld_lvls_inf[1]

        if len(self.ld_lvls_inf[0]) != len(self.ld_lvls_inf_frac[1]):
            x = {k:v for k,v in enumerate(~np.in1d(self.ld_lvls_inf[0], self.ld_lvls_inf_frac[0]))
                 if v == True
                }
            if len([*x.keys()]) > 1:
                fracs = list(self.ld_lvls_inf_frac[1])
                for keys in np.arange(len([*x.keys()])):
                    fracs.insert([*x.keys()][keys], 0)
                data_probit[:, 2] = np.asarray(fracs)
            else:
                    fracs = list(self.ld_lvls_inf_frac[1])
                    fracs.insert([*x.keys()][0], 0)
                    data_probit[:, 2] = np.asarray(fracs)
        else:
            data_probit[:, 2] = self.ld_lvls_inf_frac[1]

        # Rossow failure probability for the transition zone
        self.FP = []

        for i in data_probit:
            '#Number of specimen'
            n = i[1]
            '#Number of fractures'
            r = i[2]

            if r == 0:
                self.FP.append(1. - 0.5**(1./n))
            elif r == n:
                self.FP.append(0.5**(1./n))
            else:
                self.FP.append((3*r-1)/(3*n+1))

        self.FP = np.asarray(self.FP)


    def __maximum_like_procedure(self):
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
        self.p_opt = {'SD_50': self.fatg_lim, '1/TS': 1.2, 'k_1': self.k,
                      'ND_50': self.N_E, '1/TN': self.TN}

        self.dict_bound = {'SD_50':(-np.inf, self.fatg_lim*0.95), '1/TS':(0.5, 2),
                           'k_1':(self.k*0.9, self.k*1.1), 'ND_50':(self.N_E*0.8, self.load_cycle_limit*0.8),
                           '1/TN':(self.TN*0.1, self.TN*10)
                           }

        for k in self.param_fix:
            self.p_opt.pop(k)
            self.dict_bound.pop(k)
        self.Mali_5p_result = {}
        if self.dict_bound: 
            var_opt = my.scipy_optimize.fmin(self.mali_sum_lolli_wrapper, [*self.p_opt.values()],
                                            bounds=[*self.dict_bound.values()],
                                            args=([*self.p_opt], self.param_fix, self.fractures, self.zone_inf,
                                                self.load_cycle_limit
                                                ),
                                            disp=True,
                                            maxiter=1e4,
                                            maxfun=1e4,
                                            )
            self.Mali_5p_result.update(self.param_fix)
            self.Mali_5p_result.update(zip([*self.p_opt], var_opt))
        else:
            print('You need to have at least one parameter empty!')

    def calc_woehler_curve_parameters(self):
        self.__data_sort()

        self.__slope()
        self.__deviation()

        if len(self.ld_lvls_inf[0])<2:
            self.Probit_result = {}
        else:
            self.__probit()
            self.__probit_procedure()

        self.__maximum_like_procedure()
        self.__maximum_like_procedure_2_param()

    def __maximum_like_procedure_2_param(self):
        ''' This maximum likelihood procedure estimates the load endurance limit SD50_mali_2_param and the
        scatter in load direction TS_mali_2_param.
        Moreover, the load cycle endurance is computed by the interesecting endurance limit line with the
        line of slope k_1
        '''
        SD_start = self.fatg_lim
        TS_start = 1.2

        var = optimize.fmin(self.Mali_SD_TS, [SD_start, TS_start],
                                           args=(self.zone_inf, self.load_cycle_limit),
                                           disp=False)

        ND50 = 10**(self.b_wl + self.a_wl*np.log10(var[0]))
        self.Mali_2p_result = {'SD_50':var[0], '1/TS':var[1],'ND_50':ND50}


    def mali_sum_lolli(self, SD, TS, k, N_E, TN, fractures, zone_inf, load_cycle_limit):
        """
        Produces the likelihood functions that are needed to compute the parameters of the woehler curve.
        The likelihood functions are represented by probability and cummalative distribution functions.
        The likelihood function of a runout is 1-Li(fracture). The functions are added together, and the
        negative value is returned to the optimizer.

        Parameters
        ----------
        SD:
            Endurnace limit start value to be optimzed, unless the user fixed it.
        TS:
            The scatter in load direction 1/TS to be optimzed, unless the user fixed it.
        k:
            The slope k_1 to be optimzed, unless the user fixed it.
        N_E:
            Load-cycle endurance start value to be optimzed, unless the user fixed it.
        TN:
            The scatter in load-cycle direction 1/TN to be optimzed, unless the user fixed it.
        fractures:
            The data that our log-likelihood function takes in. This data represents the fractured data.
        zone_inf:
            The data that our log-likelihood function takes in. This data is found in the infinite zone.
        load_cycle_limit:
            The dependent variable that our model requires, in order to seperate the fractures from the
            runouts.

        Returns
        -------
        neg_sum_lolli :
            Sum of the log likelihoods. The negative value is taken since optimizers in statistical
            packages usually work by minimizing the result of a function. Performing the maximum likelihood
            estimate of a function is the same as minimizing the negative log likelihood of the function.

        """
        # Likelihood functions of the fractured data
        x_ZF = np.log10(fractures.cycles * ((fractures.loads/SD)**(k)))
        Mu_ZF = np.log10(N_E)
        Sigma_ZF = np.log10(TN)/2.5631031311
        Li_ZF = stats.norm.pdf(x_ZF, Mu_ZF, abs(Sigma_ZF))
        LLi_ZF = np.log(Li_ZF)

        # Likelihood functions of the data found in the infinite zone
        std_log = np.log10(TS)/2.5631031311
        runouts = ma.masked_where(zone_inf.cycles >= load_cycle_limit, zone_inf.cycles)
        t = runouts.mask.astype(int)
        Li_DF = stats.norm.cdf(np.log10(zone_inf.loads/SD), loc=np.log10(1), scale=abs(std_log))
        LLi_DF = np.log(t+(1-2*t)*Li_DF)

        sum_lolli = LLi_DF.sum() + LLi_ZF.sum()
        neg_sum_lolli = -sum_lolli

        return neg_sum_lolli

    def Mali_SD_TS(self, variables, zone_inf, load_cycle_limit):
        """
        Produces the likelihood functions that are needed to compute the endurance limit and the scatter
        in load direction. The likelihood functions are represented by a cummalative distribution function.
        The likelihood function of a runout is 1-Li(fracture).

        Parameters
        ----------
        variables:
            The start values to be optimized. (Endurance limit SD, Scatter in load direction 1/TS)
        zone_inf:
            The data that our log-likelihood function takes in. This data is found in the infinite zone.
        load_cycle_limit:
            The dependent variable that our model requires, in order to seperate the fractures from the
            runouts.

        Returns
        -------
        neg_sum_lolli :
            Sum of the log likelihoods. The negative value is taken since optimizers in statistical
            packages usually work by minimizing the result of a function. Performing the maximum likelihood
            estimate of a function is the same as minimizing the negative log likelihood of the function.

        """

        SD = variables[0]
        TS = variables[1]

        std_log = np.log10(TS)/2.5631031311
        runouts = ma.masked_where(zone_inf.cycles >= load_cycle_limit, zone_inf.cycles)
        t = runouts.mask.astype(int)
        Li_DF = stats.norm.cdf(np.log10(zone_inf.loads/SD), loc=np.log10(1), scale=abs(std_log))
        LLi_DF = np.log(t+(1-2*t)*Li_DF)

        sum_lolli = LLi_DF.sum()
        neg_sum_lolli = -sum_lolli

        return neg_sum_lolli


    def rossow_fail_prob(self, x):
        """ Failure Probability estimation formula of Rossow

        'Statistics of Metal Fatigue in Engineering' page 16

        https://books.google.de/books?isbn=3752857722
        """
        i = np.arange(len(x))+1
        pa = (3.*(i)-1.)/(3.*len(x)+1.)

        return pa


    def bayes_inf_crit(self, LLi, dict_opt, n_data_pt):
        ''' Bayesian Information Criterion: is a criterion for model selection among a finite set of models;
        the model with the lowest BIC is preferred.
        https://www.statisticshowto.datasciencecentral.com/bayesian-information-criterion/
        '''

        param_est = len([*dict_opt.values()])
        bic = (-2*LLi)+(param_est*np.log(n_data_pt))
        return bic


    def life_cyc_eval(self, method):

        if method == 'Mali':
            # Allowed load amplitude with a certain load and 50% failure probability
            N_allowed_50 = self.ND50_mali*(self.Load_ampl_goal/
                                                   self.Param_SD50_mali)**(-self.Param_k_mali)
            # Allowed load relative to the goal failure probability
            N_allowed_pa_goal = (N_allowed_50/(10**(-stats.norm.ppf(self.Pa_goal)*
                                                    np.log10(self.Param_TN_mali)/2.56
                                                   )
                                              )
                                )
        else:
            N_allowed_50 = self.ND50_mali*(self.Load_ampl_goal/
                                                                self.SD50_probit)**(-WoehlerCurve.k)

            N_allowed_pa_goal = (N_allowed_50/(10**(-stats.norm.ppf(self.Pa_goal)*
                                                    np.log10(self.TN)/2.56
                                                   )
                                              )
                                )

        return N_allowed_50, N_allowed_pa_goal

    def mali_sum_lolli_wrapper(self, var_args, var_keys, fix_args, fractures, zone_inf, load_cycle_limit):
        ''' 1) Finds the start values to be optimized. The rest of the paramters are fixed by the user.
            2) Calls function mali_sum_lolli to calculate the maximum likelihood of the current
            variable states.
        '''
        args = {}
        args.update(fix_args)
        args.update(zip(var_keys, var_args))

        return self.mali_sum_lolli(args['SD_50'], args['1/TS'], args['k_1'], args['ND_50'],
                                          args['1/TN'], fractures, zone_inf, load_cycle_limit)
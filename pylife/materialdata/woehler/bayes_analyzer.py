# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:24:22 2019

@author: KSS7RNG
"""
import pymc3 as pm
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import theano
theano.gof.compilelock.set_lock_status(False)
from scipy import stats
import theano
import theano.tensor as tt


def ld_lvl_sample(WoehlerCurve, data, n_N):

    data_choice = {}
    lds = {}
    last = len(WoehlerCurve.ld_lvls_fin[0])-1
    ld_lvl_1 = np.log10(data.cycles[WoehlerCurve.ld_lvls_fin[0][last] == data.loads])
    data_choice['data_1'] = np.random.choice(ld_lvl_1, n_N[0])
    lds['x_1'] = np.ones(len(data_choice['data_1']))*np.log10(WoehlerCurve.ld_lvls_fin[0][last])
    y_small_size = data_choice['data_1']
    x_small_size = lds['x_1']

    for ld_lvl in np.arange(len(n_N)-1)+1:
        N_ld_lvl = np.log10(data.cycles[WoehlerCurve.ld_lvls_fin[0][-ld_lvl] == data.loads])
        data_choice['data_%d' % ld_lvl] = np.random.choice(N_ld_lvl, n_N[ld_lvl])
        lds['x_%d' % ld_lvl] =  np.ones(len(data_choice['data_%d' % ld_lvl]))*np.log10(WoehlerCurve.ld_lvls_fin[0][-ld_lvl])
        y_small_size = np.concatenate((y_small_size, data_choice['data_%d' % ld_lvl]))
        x_small_size = np.concatenate((x_small_size, lds['x_%d' % ld_lvl]))

    data_small = dict(x=x_small_size, y=y_small_size)

    return data_small
#np.random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])

def bayesian_slope(data_dict, model_name, samples, chains):

    with pm.Model() as model_name:
        family = pm.glm.families.StudentT()
        pm.glm.GLM.from_formula('y ~ x', data_dict, family=family)
        trace_robust = pm.sample(samples, nuts_kwargs={'target_accept': 0.99}, chains=chains, tune=1000)

    return trace_robust


def bayesian_slope_norm(data_dict, model_name, samples, chains):

    with pm.Model() as model_name:
        pm.glm.GLM.from_formula('y ~ x', data_dict)
        trace_robust = pm.sample(samples, nuts_kwargs={'target_accept': 0.99}, chains=chains, tune=1000)

    return trace_robust


def bayesian_slope_plot(data_dict, burned_trace, slope_small, intercept_small, slope_full, intercept_full):
    plt.figure(figsize=(7, 5))
    plt.plot(data_dict['x'], data_dict['y'], 'x', label='data')
    pm.plot_posterior_predictive_glm(burned_trace,
                                     lm = lambda x,
                                     sample: sample['Intercept'] + sample['x'] * data_dict['x'],
                                     eval = data_dict['x'], samples=500,
                                     label='posterior predictive regression lines')

    plt.plot(data_dict['x'], intercept_small + slope_small*data_dict['x'], 'r', lw=3., label='fitted line')
    plt.plot(data_dict['x'], intercept_full + slope_full*data_dict['x'], 'b', lw=3., label='true line')
    matplotlib.pyplot.xlim(data_dict['x'].min()-0.0025, data_dict['x'].max()*1.0005)
    matplotlib.pyplot.ylim(data_dict['y'].min()-0.1, data_dict['y'].max()*1.1)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc=0);


def trace_quantiles(x):

    return pd.DataFrame(pm.quantiles(x, [10, 50, 90]))


def trace_sd(x):

    return pd.Series(np.std(x, 0), name='sd')


def monte(x):

    return pd.Series(pm.stats.mc_error(x,len(x)), name='mc_error')

# define a theano Op for our likelihood function
'#http://mattpitkin.github.io/samplers-demo/pages/pymc3-blackbox-likelihood/'

def mali_sum_lolli(var, data_S, data_N, load_cycle_limit):
    '''Maximum likelihood is a method of estimating the parameters of a distribution model by maximizing
    a likelihood function, so that under the assumed statistical model the observed data is most probable.

    https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
    '''
    SD = var[0]
    TS = var[1]
    # Likelihood functions of the infinite zone
    std_log = TS/2.5631031311
    durchlaefer = ma.masked_where(data_N >= np.log10(load_cycle_limit), data_N)
    t = durchlaefer.mask.astype(int)
    Li_DF = stats.norm.cdf(data_S/SD, loc=np.log10(1), scale=abs(std_log))
    LLi_DF = np.log(t+(1-2*t)*Li_DF)

    loglike = LLi_DF.sum()

    return loglike


class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)

    http://mattpitkin.github.io/samplers-demo/pages/pymc3-blackbox-likelihood/
    """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data_S, data_N, load_cycle_limit):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'load_cycle_limit') that our model requires

        """
        # add inputs as class attributes
        self.likelihood = loglike
        self.data_S = data_S
        self.data_N = data_N
        self.load_cycle_limit = load_cycle_limit

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        var, = inputs  # this will contain my variables
        #mali_sum_lolli(var, self.zone_inf, self.load_cycle_limit):
        # call the log-likelihood function
        logl = self.likelihood(var, self.data_S, self.data_N, self.load_cycle_limit)

        outputs[0][0] = np.array(logl) # output the log-likelihood


def fail_runout_sampler(lvl, lvl_N_small, probability_runout, n_N_inf, n_frac):
    lvl_N_small['ld_lvl_%d' % (lvl+1)] = np.random.choice(2, n_N_inf[lvl], p=[1-probability_runout[lvl], probability_runout[lvl]])
    # Length of sampler vector (2 = 0&1, 1 = 0|1)
    n_frac_len=len(np.unique(lvl_N_small['ld_lvl_%d' % (lvl+1)], return_counts=True)[0])
    if n_frac_len == 2:
        # Case lvl_N_small has 0 and 1
        n_frac['%d'%(lvl+1)] = np.unique(lvl_N_small['ld_lvl_%d' % (lvl+1)], return_counts=True)[1][1]
    else:
        # Case only 1
        if np.unique(lvl_N_small['ld_lvl_%d' % (lvl+1)], return_counts=True)[0][0]==1:
            n_frac['%d'%(lvl+1)]=n_N_inf[lvl]
        else:
            #Case oney 0
            n_frac['%d'%(lvl+1)] = 0

    return n_frac

#def failure_sampler(data_inf, N_values_frac, n_fracures, n_N_infinity, lvl):
#    # replace 1 with a value from random norm. and 0 with load cycle limit
#    if n_fracures!=0:
#        if n_fracures!=n_N_infinity:
#            data_inf = np.concatenate((data_inf, np.ones(n_N_infinity-n_fracures)*WoehlerCurve.load_cycle_limit))
#            data_inf['data_%d' % (lvl+1)] = data_inf
#        else:
#            data_inf['data_%d' % (lvl+1)] = np.random.choice(N_values_frac, n_N_infinity)
#    else:
#        data_inf['data_%d' % (lvl+1)] = np.ones(n_N_infinity)*WoehlerCurve.load_cycle_limit
#
#    return data_inf['data_%d' % (lvl+1)]



def ld_lvl_sample_inf(WoehlerCurve, n_N_inf):

    lvl_N_small = {}
    data_inf_dict = {}
    n_frac = {}
    y_small_size=[]
    x_small_size=[]
    x_inf_dict={}

    s_fp = np.log10(WoehlerCurve.ld_lvls_inf[0])
    s_fp[::-1].sort()
    # Using SD_50 and TS_50 to deduce the probability of getting a fracture depending on the load level we are testing:
    probability_runout = stats.norm.cdf((s_fp-np.log10(WoehlerCurve.Mali_5p_result['SD_50']))*(2.56/np.log10(WoehlerCurve.Mali_5p_result['1/TS'])))

    # find the 0-1 distribution
    for lvl in np.arange(len(WoehlerCurve.ld_lvls_inf[0])):
        # Check if user put a 0 in this load level
        if n_N_inf[lvl] != 0:
            n_frac = fail_runout_sampler(lvl, lvl_N_small, probability_runout, n_N_inf, n_frac)

            N_values_frac = WoehlerCurve.zone_inf_fractures.cycles[WoehlerCurve.ld_lvls_inf[0][-(lvl+1)] == WoehlerCurve.zone_inf_fractures.loads]
            # replace 1 with a value from random norm. and 0 with load cycle limit
            if n_frac['%d'%(lvl+1)]!=0:
                data_inf = np.random.choice(N_values_frac, n_frac['%d'%(lvl+1)])
                if n_frac['%d'%(lvl+1)]!=n_N_inf[lvl]:
                    data_inf = np.concatenate((data_inf, np.ones(n_N_inf[lvl]-n_frac['%d'%(lvl+1)])*WoehlerCurve.load_cycle_limit))
                    data_inf_dict['data_%d' % (lvl+1)] = data_inf
                else:
                    data_inf_dict['data_%d' % (lvl+1)] = data_inf
            else:
                data_inf_dict['data_%d' % (lvl+1)] = np.ones(n_N_inf[lvl]-n_frac['%d'%(lvl+1)])*WoehlerCurve.load_cycle_limit
#            data_inf_dict = failure_sampler(data_inf_dict, N_values_frac, n_frac['%d'%(lvl+1)], n_N_inf[lvl], lvl)
            x_inf_dict['x_%d' % (lvl+1)] = np.ones(n_N_inf[lvl])*np.log10(WoehlerCurve.ld_lvls_inf[0][-(lvl+1)])

            y_small_size = np.concatenate((y_small_size, np.log10([*data_inf_dict['data_%d' % (lvl+1)]])))
            x_small_size = np.concatenate((x_small_size, [*x_inf_dict['x_%d' % (lvl+1)]]))

    data_small_inf = dict(x=x_small_size, y=y_small_size)

    return data_small_inf, data_inf_dict, n_frac



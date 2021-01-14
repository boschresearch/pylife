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

__author__ = "Mustapha Kassem"
__maintainer__ = "Johannes Mueller"

import numpy as np
import pandas as pd
import theano.tensor as tt
import pymc3 as pm

from .elementary import Elementary
from .likelihood import Likelihood


class Bayesian(Elementary):

    class _LogLike(tt.Op):
        """
        Specify what type of object will be passed and returned to the Op when it is
        called. In our case we will be passing it a vector of values (the parameters
        that define our model) and returning a single "scalar" value (the
        log-likelihood)

        http://mattpitkin.github.io/samplers-demo/pages/pymc3-blackbox-likelihood/
        """
        itypes = [tt.dvector]  # expects a vector of parameter values when called
        otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

        def __init__(self, likelihood):
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
            self.likelihood = likelihood

        def perform(self, node, inputs, outputs):
            # the method that is used when calling the Op
            var, = inputs  # this will contain my variables
            # mali_sum_lolli(var, self.zone_inf, self.load_cycle_limit):
            # call the log-likelihood function
            logl = self.likelihood.likelihood_infinite(var[0], var[1])

            outputs[0][0] = np.array(logl)  # output the log-likelihood

    def _specific_analysis(self, wc, nsamples=1000, **kw):
        self._nsamples = nsamples
        nburn = self._nsamples // 10

        tune = kw.pop('tune', 1000)
        random_seed = kw.pop('random_seed', None)

        slope_trace = self._slope_trace(tune=tune, random_seed=random_seed, **kw)
        TN_trace = self._TN_trace(tune=tune, random_seed=random_seed, **kw)
        SD_TS_trace = self._SD_TS_trace(tune=tune, random_seed=random_seed, **kw)

        slope = slope_trace.get_values('x')[nburn:].mean()
        intercept = slope_trace.get_values('Intercept')[nburn:].mean()
        SD_50 = SD_TS_trace.get_values('SD_50')[nburn:].mean()
        ND_50 = np.power(10., np.log10(SD_50) * slope + intercept)

        res = {
            'SD_50': SD_50,
            '1/TS': SD_TS_trace.get_values('TS_50')[nburn:].mean(),
            'ND_50': ND_50,
            'k_1': -slope,
            '1/TN': TN_trace.get_values('mu')[nburn:].mean(),
        }

        return pd.Series(res)

    def _slope_trace(self, chains=2, random_seed=None, tune=1000, **kw):
        data_dict = {
            'x': np.log10(self._fd.fractures.load),
            'y': np.log10(self._fd.fractures.cycles.to_numpy())
        }
        with pm.Model():
            family = pm.glm.families.StudentT()
            pm.glm.GLM.from_formula('y ~ x', data_dict, family=family)
            trace_robust = pm.sample(self._nsamples,
                                     target_accept=0.99,
                                     random_seed=random_seed,
                                     chains=chains,
                                     tune=tune,
                                     **kw)

            return trace_robust

    def _TN_trace(self, chains=3, random_seed=None, tune=1000, **kw):
        with pm.Model():
            log_N_shift = np.log10(self._pearl_chain_estimator.normed_cycles)
            stdev = pm.HalfNormal('stdev', sigma=1.3)  # sigma standard wert (log-normal/ beat Verteilung/exp lambda)
            mu = pm.Normal('mu', mu=log_N_shift.mean(), sigma=log_N_shift.std())
            _ = pm.Normal('y', mu=mu, sigma=stdev, observed=log_N_shift)

            trace_TN = pm.sample(self._nsamples,
                                 target_accept=0.99,
                                 random_seed=random_seed,
                                 chains=chains,
                                 tune=tune,
                                 **kw)

        return trace_TN

    def _SD_TS_trace(self, chains=3, random_seed=None, tune=1000, **kw):
        loglike = self._LogLike(Likelihood(self._fd))

        with pm.Model():
            inf_load = self._fd.infinite_zone.load
            SD = pm.Normal('SD_50', mu=inf_load.mean(), sigma=inf_load.std()*5)
            TS = pm.Lognormal('TS_50', mu=np.log10(1.1), sigma=np.log10(0.5))

            # convert m and c to a tensor vector
            var = tt.as_tensor_variable([SD, TS])

            pm.Potential('likelihood', loglike(var))

            trace_SD_TS = pm.sample(self._nsamples,
                                    cores=1,
                                    chains=chains,
                                    random_seed=random_seed,
                                    discard_tuned_samples=True,
                                    tune=tune,
                                    **kw)

        return trace_SD_TS

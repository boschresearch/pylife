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
import theano.tensor as tt
import pymc3 as pm

from pylife.materialdata.woehler.creators.likelihood import Likelihood

class Bayesian:
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

    def __init__(self, fatigue_data):
        self._loglike = self._LogLike(Likelihood(fatigue_data))
        self._fd = fatigue_data

    def slope(self, nsamples=5000, chains=2):
        data_dict = {
            'x': np.log10(self._fd.fractures.load),
            'y': np.log10(self._fd.fractures.cycles.to_numpy())
        }
        with pm.Model():
            family = pm.glm.families.StudentT()
            pm.glm.GLM.from_formula('y ~ x', data_dict, family=family)
            trace_robust = pm.sample(nsamples, nuts_kwargs={'target_accept': 0.99}, chains=chains, tune=1000)

            return trace_robust

    def TN(self, nsamples=5000, chains=3):
        with pm.Model():
            log_N_shift = np.log10(self._fd.N_shift)
            stdev = pm.HalfNormal('stdev', sd=1.3)  # sd standard wert (log-normal/ beat Verteilung/exp lambda)
            mu = pm.Normal('mu', mu=log_N_shift.mean(), sd=log_N_shift.std())  # mu könnte von FKM gegeben
            _ = pm.Normal('y', mu=mu, sd=stdev, observed=log_N_shift)  # lognormal

            trace_TN = pm.sample(nsamples, nuts_kwargs={'target_accept': 0.99}, chains=chains, tune=1000)

        return trace_TN

    def SD_TS(self, nsamples=5000, chains=3):
        with pm.Model():
            SD = pm.Normal('SD_50', mu=self._fd.zone_inf.load.mean(), sd=self._fd.zone_inf.load.std()*5)
            TS = pm.Lognormal('TS_50', mu=np.log10(1.1), sd=np.log10(0.5))

            # convert m and c to a tensor vector
            var = tt.as_tensor_variable([SD, TS])

            pm.DensityDist('likelihood', lambda v: self._loglike(v), observed={'v': var})

            trace_SD_TS = pm.sample(nsamples, tune=1000, chains=chains, discard_tuned_samples=True)

        return trace_SD_TS

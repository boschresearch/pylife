# Copyright (c) 2019-2023 - for information on the respective copyright owner
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

"""
Until version 2.1.0 pyLife had a module for Bayesian Wöhler analysis.  The
Bayesian Wöhler analysis is not well established in the community, so it was a
bit ahead of its time.  Now that we have collected experiences with it, it
turns out, that its result are too often inaccurate.  Therefore we decided to
disable it.

That means, that the code remains in the code base, we are only preventing you
from importing it.  That means, if you want to experiment with the code, you
can grab the module and delete the exception raising.  We do not recommend to
use it for production, though.

If you want to propose an improvement of the code that leads to better results,
we are open for PRs (see :doc:`/CONTRIBUTING`) and might eventually enable it
again.
"""

__author__ = "Mustapha Kassem"
__maintainer__ = "Johannes Mueller"


raise NotImplementedError(
    "pyLife's Bayesian Wöhler analyzer has been shutdown. "
    "See documentation for details."
)

import sys
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import pymc as pm
import bambi

from .elementary import Elementary
from .likelihood import Likelihood


class Bayesian(Elementary):
    """A Wöhler analyzer using Bayesian optimization

    Warning
    -------

    We are considering switching from pymc to GPyOpt as calculation engine in the
    future.  Maybe this will lead to breaking changes without new major release.
    """


    class _LogLike(pt.Op):
        """
        Specify what type of object will be passed and returned to the Op when it is
        called. In our case we will be passing it a vector of values (the parameters
        that define our model) and returning a single "scalar" value (the
        log-likelihood)

        http://mattpitkin.github.io/samplers-demo/pages/pymc-blackbox-likelihood/
        """
        itypes = [pt.dvector]  # expects a vector of parameter values when called
        otypes = [pt.dscalar]  # outputs a single scalar value (the log likelihood)

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

        SD_TS_trace = self._SD_TS_trace(tune=tune, random_seed=random_seed, **kw)
        slope_trace = self._slope_trace(tune=tune, random_seed=random_seed, **kw)
        slope = slope_trace.get('x').values[-1, nburn:].mean()
        intercept = slope_trace.get('Intercept').values[-1, nburn:].mean()

        SD = SD_TS_trace.get('SD').values[-1, nburn:].mean()
        ND = np.power(10., np.log10(SD) * slope + intercept)

        TN_trace = self._TN_trace(tune=tune, random_seed=random_seed, **kw)

        return pd.Series({
            'SD': SD,
            'TS': SD_TS_trace.get('TS').values[-1, nburn:].mean(),
            'ND': ND,
            'k_1': -slope,
            'TN': TN_trace.get('mu').values[-1, nburn:].mean(),
        })

    def _slope_trace(self, chains=2, random_seed=None, tune=1000, **kw):
        data_dict = pd.DataFrame({
            'x': np.log10(self._fd.fractures.load),
            'y': np.log10(self._fd.fractures.cycles.to_numpy())
        })
        with pm.Model():
            model = bambi.Model('y ~ x', data_dict, family='t')
            fitted = model.fit(
                draws=self._nsamples,
                cores=self._core_num(),
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                **kw
            )
            return fitted.posterior

    def _TN_trace(self, chains=3, random_seed=None, tune=1000, **kw):
        with pm.Model():
            log_N_shift = np.log10(self._pearl_chain_estimator.normed_cycles)
            stdev = pm.HalfNormal('stdev', sigma=1.3)  # sigma standard wert (log-normal/ beat Verteilung/exp lambda)
            mu = pm.Normal('mu', mu=log_N_shift.mean(), sigma=log_N_shift.std())
            _ = pm.Normal('y', mu=mu, sigma=stdev, observed=log_N_shift)

            trace_TN = pm.sample(self._nsamples,
                                 cores=self._core_num(),
                                 target_accept=0.99,
                                 random_seed=random_seed,
                                 chains=chains,
                                 tune=tune,
                                 **kw)

        return trace_TN.posterior

    def _SD_TS_trace(self, chains=3, random_seed=None, tune=1000, **kw):
        loglike = self._LogLike(Likelihood(self._fd))

        with pm.Model():
            inf_load = self._fd.infinite_zone.load
            SD = pm.Normal('SD', mu=inf_load.mean(), sigma=inf_load.std()*5)
            TS_inv = pm.Lognormal('TS', mu=np.log10(1.1), sigma=0.3)

            # convert m and c to a tensor vector
            print(type(SD))
            print(type(TS_inv))
            var = pt.as_tensor_variable([SD, TS_inv])

            pm.Potential('likelihood', loglike(var))

            trace_SD_TS = pm.sample(self._nsamples,
                                    cores=self._core_num(),
                                    chains=chains,
                                    random_seed=random_seed,
                                    discard_tuned_samples=True,
                                    tune=tune,
                                    **kw)

        return trace_SD_TS.posterior

    def _core_num(self):
        return 1 if sys.platform.startswith('win') else None

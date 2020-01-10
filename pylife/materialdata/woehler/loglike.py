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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import numpy as np
import theano
import theano.tensor as tt

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

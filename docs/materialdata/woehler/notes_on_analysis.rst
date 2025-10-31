Notes on Wöhler analysis
########################

Even though there are established Wöhler analysis algorithms they all leave
some degrees of freedom.  Therfore we document the way we perform the analysis
in pyLife in this section.


Basic principles of analysis
============================

There are some minimal requirements for the data to perform a Wöhler analysis
in a meaningful way. Some of them are mandatory and we outright refuse the
analysis if there are not met. Some we do not mandate but warn, because it is
likely that the result is not meaningful and should not be used for any serious
purpose.


The concept of load levels
--------------------------

In general, fatigue data is the cyclic load plotted over the cycles endured.
In order to perform an analysis, we need what we call load levels.  So the load
values chosen for the fatigue data is not arbitrary, there should always be
multiple tests – ideally five – on a single exact same load value. Those we
call the load levels.


The finite and infinite zone
----------------------------

We separate the Wöhler data in two zones. The finite zone is all the load
levels at which all specimens fracture. The infinite zone is all the load
levels at which there is at least one runout. Usually in order to perform a
proper Wöhler analysis you need at least two load levels in each zone. For the
infinite zone that means at least two mixed levels that has at least one runout
and at least one fracture.

The analysis of Wöhler data basically works in two steps. First we calculate
the finite part, that is the sloped part with the fractures to calculate the
slope `k` the scatter in life time direction `TN` and the imaginary 50% failure
load at one cycle. Then we calculate the infinite values, those are the
endurance limit in load direction `SD` and the scatter in load direction `SD`.


Calculating the finite values
-----------------------------

Calculating the slope `k` is basically a linear regression of all the fractures
in the finite zone.


Steps of the analysis
=====================

The analysis of Wöhler data basically works in two steps. First we calculate
the finite part, that is the sloped part with the fractures to calculate the
slope `k` the scatter in life time direction `TN` and the imaginary 50% failure
load at one cycle. Then we calculate the infinite values, those are the
endurance limit in load direction `SD` and the scatter in load direction `SD`.


Calculating the finite values
-----------------------------

Calculating the slope `k` is in the easiest case a linear regression of all the
fractures in the finite zone. That's what is done by
:class:`~pylife.materialdata.woehler.Elementary`.


Calculating the infinite values
-------------------------------

The infinite values cannot be calculated by a linear regression. There are two
established methods to calculate them:

* the maximum likelihood method implemented in
  :class:`~pylife.materialdata.woehler.MaxLikeFull` and
  :class:`~pylife.materialdata.woehler.MaxLikeInf`.
* the probit method implemented in :class:`~pylife.materialdata.woehler.Probit`

Please refer to the documentation of the implementing modules for details.


Likelihood estimation strategy
==============================

The maximum likelihood methods basically try to find a solution that maximizes
the likelihood that the solution is correct.  So it needs to a way of
estimating the likelihood for a solution in question.  Again we do that
separately for the finite and the infinite values.  That raises the question,
which load levels to take into account to evaluate the likelihood of the finite
resp. infinite zone.  We have three options implemented for that, all of which
use all fractures to evaluate the infinite zone.  For the finite zone they do
the following

1. By default we use all fractures on pure fracture levels and the fractures of
   the highest mixed level.
   :class:`~pylife.materialdata.woehler.likelihood.LikelihoodHighestMixedLevel`

2. The second option is to use only the fractures of the pure fracture levels.
   :class:`~pylife.materialdata.woehler.likelihood.LikelihoodPureFiniteZone`

3. The third option is to use all fractures.
   :class:`~pylife.materialdata.woehler.likelihood.LikelihoodAllFractures`


You can also implement your own likelihood estimator by subclassing
:class:`~pylife.materialdata.woehler.likelihood.AbstractLikelihood` and
then

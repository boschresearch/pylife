# pyLife – a general library for fatigue and reliability

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/boschresearch/pylife/develop?labpath=demos%2Findex.ipynb)
[![Documentation Status](https://readthedocs.org/projects/pylife/badge/?version=latest)](https://pylife.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/pylife)](https://pypi.org/project/pylife/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pylife)
[![testsuite](https://github.com/boschresearch/pylife/workflows/testsuite/badge.svg)](https://github.com/boschresearch/pylife/actions?query=workflow%3Atestsuite)

pyLife is an Open Source Python library for state of the art algorithms used in
lifetime assessment of mechanical components subject to fatigue load.


## Purpose of the project

This library was originally compiled at [Bosch
Research](https://www.bosch.com/research/) to collect algorithms needed by
different in house software projects, that deal with lifetime prediction and
material fatigue on a component level. In order to further extent and
scrutinize it we decided to release it as Open Source.  Read [this
article](https://www.bosch.com/stories/bringing-open-source-to-mechanical-engineering/)
about pyLife's origin.

So we are welcoming collaboration not only from science and education but also
from other commercial companies dealing with the topic. We commend this library
to university teachers to use it for education purposes.

The company [Viktor](https://viktor.ai) has set up a [web application for Wöhler
test analysis](https://cloud.viktor.ai/public/wohler-fatigue-test-analysis)
based on pyLife code.


## Status

pyLife-2.0.3 has been released.  That means that for the time being we hope
that we will not introduce *breaking* changes.  That does not mean that the
release is stable finished and perfect.  We will do small improvements,
especially with respect to documentation in the upcoming months and release
them as 2.0.x releases.  Once we have noticeable feature additions we will come
up with a 2.x.0 release.  No ETA about that.

## Contents

There are/will be the following subpackages:

* `stress` everything related to stress calculation
	* equivalent stress
	* stress gradient calculation
	* rainflow counting
	* ...
* `strength` everything related to strength calculation
	* failure probability estimation
	* S-N-calculations
	* ...
* `mesh` FEM mesh related stuff
    * stress gradients
	* FEM-mapping
	* hotspot detection
* `util` all the more general utilities
	* ...
* `materialdata` analysis of material testing data
    * Wöhler (SN-curve) data analysis

* `materiallaws` modeling material behavior
    * Ramberg Osgood
    * Wöhler curves

* `vmap` a interface to [VMAP](https://www.vmap.eu.com/)


## License

pyLife is open-sourced under the Apache-2.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in pyLife, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).

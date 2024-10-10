# pyLife – a general library for fatigue and reliability

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/boschresearch/pylife/develop?labpath=demos%2Findex.ipynb)
[![Documentation Status](https://readthedocs.org/projects/pylife/badge/?version=latest)](https://pylife.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/pylife)](https://pypi.org/project/pylife/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pylife)
[![Testsuite](https://github.com/boschresearch/pylife/actions/workflows/pytest.yml/badge.svg)](https://github.com/boschresearch/pylife/actions/workflows/pytest.yml)

pyLife is an Open Source Python library for state of the art algorithms used in
lifetime assessment of mechanical components subjected to fatigue.


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

pyLife-2.1.x is the current release the you get by default.  We are doing small
improvements, in the pyLife-2.1.x branch (`master`) while developing the more
vast features in the 2.2.x branch (`develop`).

The main new features of the 2.2.x branch is about FKM functionality. As that
is quite a comprehensive addition we would need some time to get it right
before we can release it as default release.

Once 2.2.x is released we will probably stick to a one branch development.

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
	* local strain concept: FKM guideline nonlinear
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

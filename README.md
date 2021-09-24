# pyLife – a general library for fatigue and reliability

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/boschresearch/pylife/master?filepath=demos%2Findex.ipynb)
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


## Disclaimer

The `develop` branch is at the moment undergoing heavy development for the
`pylife-2.0` release.  Check out [NEWS-2.0.md](NEWS-2.0.md) for details.  That
means that breaking changes are likely to occur.  If you are new to pyLife, it
is nevertheless worthwhile following the `develop` branch, as it stabilizes in
the upcoming months.  The way things are done are and will be a huge
improvement over `pylife-1.x`.


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

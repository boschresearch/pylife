# pyLife – a general library for fatigue and reliability

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/boschresearch/pylife/master?filepath=demos%2Findex.ipynb)
[![Documentation Status](https://readthedocs.org/projects/pylife/badge/?version=latest)](https://pylife.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/pylife)](https://pypi.org/project/pylife/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pylife)
[![testsuite](https://github.com/boschresearch/pylife/workflows/testsuite/badge.svg)](https://github.com/boschresearch/pylife/actions?query=workflow%3Atestsuite)

pyLife is an Open Source Python library for state of the art algorithms used in
lifetime assessment of mechanical components subject to fatigue load.


## Purpose of the project

This library was originally compiled at Bosch Research to collect algorithms
needed by different in house software projects, that deal with lifetime
prediction and material fatigue on a component level. In order to further
extent and scrutinize it we decided to release it as Open Source.

So we are welcoming collaboration not only from science and education but also
from other commercial companies dealing with the topic. We commend this library
to university teachers to use it for education purposes.


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


## Disclaimer

*pyLife is in continuous development.* We hope to keep the interfaces more or
less stable. However depending on the practical use of pyLife in the future
interface changes might occur. If that happens, we probably won't be able to
put too much effort into backwards compatibility. So be prepared to react to
deprecations.


## License

pyLife is open-sourced under the Apache-2.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in pyLife, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).

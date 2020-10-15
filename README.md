[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/boschresearch/pylife/master?filepath=demos%2Findex.ipynb)

# pyLife – a general library for fatigue and reliability

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


## Disclaimer

*This is work in progress at very early stage.* That means that interfaces will
most likely change so any application you link against this library will most
probably break once you pull updates. So don't do it unless you can live with
that. This also means that if you have an application that could make use of a
general reliability library you can contribute to the development and also have a
say in the interface design.


## License

pyLife is open-sourced under the Apache-2.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in pyLife, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).

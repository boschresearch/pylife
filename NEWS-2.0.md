# What is new and what has changed in pyLife-2.0

This document lists changes and new features of pyLife-2.0.


## General changes

In pyLife-1.x the individual modules often where not playing together really
well, sometimes we had even the same concept multiple times.  For the
pyLife-2.0 release we aim to improve that.  The concept of the [accessor
classes](https://pandas.pydata.org/pandas-docs/stable/development/extending.html#registering-custom-accessors)
will be used more extensively.


## New features

### Rainflow counting

The [rainflow counting module](docs/stress/rainflow.rst) has been vastly redesigned
in order to get more flexibility.  New possibilities are:

* [Four point rainflow counting](docs/stress/rainflow/fourpointdetector.rst)

* Recording of the hysteresis loop information is in a separate class to allow
  the recording in a customized way.

See docs of the [rainflow counting module](docs/stress/rainflow.rst) for details.


## Restructuring the code

We are now using [PyScaffold](https://pyscaffold.org) to handle the packaging
files.  That's why we have restructured the code base.  Basically the only
notable things that have changed is that all the code has been moved from
`pylife` to `src/pylife` and the documentation has been moved from `doc/source`
to `docs`.  Both are the common locations for Python 3.x packages.


## Changes that affect your code

* Strength scattering is now stored as `TS` and `TN`, no longer by `1/TS` and
  `1/TN`.  This only concerns the naming, the underlying values are still the
  same.  With this we are following the newer conventions in DIN 50100:2016-12.


* `self._validate()` is no longer called with arguments.  The arguments `obj`
  and `validator` are no longer needed.  `obj` is now accessible by
  `self._obj`. The methods of `DataValidator` are now accessible as methods of
  `PylifeSignal` directly.

* Signal accessor class names are no longer suffixed with `Accessor`

* The `PyLifeSignal` is promoted to the toplevel of the `pylife` package.  That
  means that you have to change

  ```python

  from pylife import signal

  ...

  class Foo(signal.PylifeSignal):
      ...
  ```

  to

  ```python

  from pylife import PylifeSignal

  ...

  class Foo(PylifeSignal):
      ...
  ```

* The name of a rainflow matrix series is no longer `frequency` but `cycles`.

* The names of the functions `scatteringRange2std` and `std2scatteringRange`
  have been adjusted to the naming conventions and are now
  `scattering_range_to_std` and `std_to_scattering_range`.

* The accessor class `CyclicStress` with the accessor `cyclic_stress` is gone.
  Use `pylife.LoadCollective` instead.


## Variable names

Currently we are brainstorming on guidelines about variable names.  See the
article [in the docs](docs/variable_names.rst) about it.  It will be
continuously updated.

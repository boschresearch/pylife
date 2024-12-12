# pyLife's change log file

In this file noteworthy changes of new releases of pyLife are documented since
2.0.0.

## Unreleased


### Breaking changes

* Drop support for python-3.8 ast it is end of life


### Minor improvements

* Connectivity information is now transferred from odbserver to odbclient as
  ints.


### Bugfixes

* Fix unnecessary and harmful copy of pandas object in Wöhler classes (#146)


## pylife-2.1.3

### Improvements

* Massive performance improvement when reading element connectivity by
  `odbclient.` Note that also `odbserver` needs to be updated in its
  environment. (#121)

* Fix some warnings on import


## pylife-2.1.2

### New features

* New method `LoadCollective.histogram()` (#107)

### Improvements

* Sanitize checks for Wöhler analysis (#108)
* Error messages when odbclient gets unsupported element types (#64)
* Improved documentation

### Bugfixes

* `MeshSignal` now allows for additional indeces (#111)


## pylife-2.1.1

### Breaking changes

* Change fracture load levels used for slope `k_1` and scatter `T_N`
  estimation. Now only fractures in the `finite_zone` are used for estimation
  and not all fractures (in `finite_zone` and `infinite_zone`). Change based on
  findings in DIN50100:2022-12. (#80, #101)

  (see [this discussion](https://github.com/boschresearch/pylife/discussions/104))

* Rename `FatigueData.fatigue_limit` to `finite_infinite_transition`

* The Bayesian Wöhler analyzer has been shutdown (#74) (see [this
  discussion](https://github.com/boschresearch/pylife/discussions/104))


### New features

* Add option to manually set `fatigue_limit` (now renamed) for Woehler curve
  estimation. (#73)


### Improvements / bug fixes

* Rainflow counters work with `pd.Series` of all index type (#69)
* Improved documentation
* Fixed confusing load matrix after mean stress transformation (#105)
* Massive performance improvements of three point and four point rainflow
  counters.
* The Wöhler analyzer now ignores irrelevant pure runout levels (#100)
* Support numpy>=2.0.0


## pylife-2.1.0

### New features

* History output for `odbclient`

* Introduce `WoehlerCurve.miner_original()`

### Breaking changes

* Non-destructive miner modifiers of `WoehlerCurve`

  The methods `WoehlerCurve.miner_elementary()` and
  `WoehlerCurve.miner_haibach()` now return modified copies of the original
  WoehlerCurve object, rather than modifying the original.


## pylife-2.0.4

Minor bugfix release

* switch to vtk > 9
* fix two index bugs in gradient.py and hotspot.py
* support for python 3.11
* pandas-2.0 compliant


## pylife-2.0.3

A minor release, mostly dependency related updates and documentation
improvements.


## pylife-2.0.2

Minor bugfix release

* Fix bug for detection of Abaqus binaries in odbserver


## pylife-2.0.1

A minor release, mostly dependency related updates and documentation
improvements.


### New features

* The `LoadCollective` accessor class honors the `cycles` column in the give
  `DataFrame`.  This is useful for manually created load collectives.

* Support for python 3.10


### Bug fixes

* Fix off-by-one error in meanstress transformation rebinning

* Correct the index of the detected turns when NaNs have been dropped in
  rainflow counting


### Other changes

* Documentation improved

* Switch from pymc3 to pymc version 4 for Bayesian Wöhler analysis. The
  extension is not installable via pip as the current bambi release pulls a
  numpy version that is incompatible with our demandas. If you need it, please
  install bambi manually from their current git repo and then pymc via pip.

* Lift numpy version restrictions

# pyLife's change log file

In this file noteworthy changes of new releases of pyLife are documented since
2.0.0.


## pylife-2.0.1

### New features

* The `LoadCollective` accessor class honors the `cycles` column in the give
  `DataFrame`.  This is useful for manually created load collectives.

* Support for python 3.10

### Bug fixes

* Fix off-by-one error in meanstress transformation rebinning

* Correct the index of the detected turns when NaNs have been dropped in
  rainflow counting


### Other changes

* Switch from pymc3 to pymc version 4 for Bayesian Wöhler analysis. The
  extension is not installable via pip as the current bambi release pulls a
  numpy version that is incompatible with our demandas. If you need it, please
  install bambi manually from their current git repo and then pymc via pip.

* Lift numpy version restrictions
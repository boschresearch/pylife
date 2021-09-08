# What is new and what has changed in pyLife-2.0

The pyLife-2.0 release is planned for end of 2021.  No promises, though.  This
document lists changes and new features of pyLife-2.0.


## General changes

In pyLife-1.x the individual modules often where not playing together really
well, sometimes we had even the same concept multiple times.  For the
pyLife-2.0 release we aim to improve that.  The concept of the [accessor
classes](https://pandas.pydata.org/pandas-docs/stable/development/extending.html#registering-custom-accessors)
will be used more extensively.


## Changes that affect your code

* Strength scattering is now stored as `TS` and `TN`, no longer by `1/TS` and
  `1/TN`.
  That means you will have to adjust your code where you deal with those kinds
  of scatters.

* `self._validate()` is no longer called with arguments.  The arguments `obj`
  and `validator` are no longer needed.  `obj` is now accessible by
  `self._obj`. The methods of `DataValidator` are now accessible as methods of
  `PylifeSignal` directly.

* Signal accessor class names are no longer suffixed with `Accessor`

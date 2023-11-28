The notch approximation law classes
======================================

The following classes are available: ``ExtendedNeuber``, ``SeegerBeste``, and ``Binned``.

The ``ExtendedNeuber`` and ``SeegerBeste`` classes implement the respective notch approximations.

The ``Binned`` class implements binning to a predefined number of bins.
It contains an object of either ``ExtendedNeuber`` or ``SeegerBeste``.
The respective stress and strain values from the notch approximation for equi-spaced load values get precomputed and stored in a lookup table.
This speeds up the computation as only the initialization step is compute intense.

.. autoclass:: pylife.materiallaws.notch_approximation_law.ExtendedNeuber
	:undoc-members:
	:members:
	:inherited-members:

.. autoclass:: pylife.materiallaws.notch_approximation_law_seegerbeste.SeegerBeste
	:undoc-members:
	:members:
	:inherited-members:

.. autoclass:: pylife.materiallaws.notch_approximation_law.Binned
	:undoc-members:
	:members:
	:inherited-members:

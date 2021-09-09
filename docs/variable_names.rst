pyLife's variable name conventions
**********************************

Preamble
========

In order for source code to be readable and maintainable, variable names should
be expressive, i.e. they should imply what the variable represents.  By doing
that, documenting variable names becomes virtually unnecessary.

However, in scientific programming we often need to deal with fairly complex
mathematical equations.  Then it is tempting to use the same or at least
similar symbols as we find in the equation in the text book.  While these
symbols are obvious to people with domain knowledge, for programmers focusing
on software optimization and industrialization these symbols are often hard to
read.

Out of these considerations we decided that in pyLife for physical quantities
the variable names as described in this document are *mandatory*.  For physical
quantities not described in this document, you can either use an expressive
variable name or you can document a symbol in your module documentation.


General rules
=============

Letters
-------

Roman letters can be used as is, capital or small. Greek letters could actually
also be written as unicode letters.  Yes, ``x = x_0 * np.exp(-δ*t) * np.cos(ω*t +
φ)`` is perfectly valid Python code.  However, not all of us are using decent
systems which allow you to type them easily.  That's why for Greek letters we
would spell them out like ``alpha``.  This does not work for ``lambda``, though
as it is a keyword in python.


Indices
-------

Indices should be separated with an underscore (``_``) from the symbol.
However, in some cases the underscore is not uses (see below.)


Variable names
==============

Stress values
-------------

* Stress tensor variables: ``S11``, ``S22``, ``S33``, ``S12``, ``S13``, ``S23``

* Cyclic stress variables:
  * ``amplitude``: stress or load amplitude,
  * ``meanstress``: meanstress,
  * ``R``: R-value


Coordinate values
-----------------

* Cartesian coordinates: ``x``, ``y``, ``z``


Displacement values
-------------------

* Displacements in a Cartesian coordinate system: ``dx``, ``dy``, ``dz``


Strength variables
------------------

* ``SD`` endurance limit in load direction,
  ``SD_xx`` for ``xx`` percent failure probability
* ``ND`` endurance limit in cycle direction,
  ``ND_xx`` for ``xx`` percent failure probability
* ``TS`` scatter in load direction (= ``SD_10/SD_90``)
* ``TN`` scatter in load direction (= ``ND_10/ND_90``)
* ``k`` slope of the Wöhler curve,
  ``k_1`` above the endurance limit, ``k_2`` below the endurance limit

***************************
Getting started with pyLife
***************************

This page gets you from a fresh installation to your first damage calculation
and then points you to the part of the documentation that matches what you want
to do next.

.. note::

   This guide assumes a basic familiarity with `pandas
   <https://pandas.pydata.org/>`_ and `numpy <https://numpy.org/>`_, since
   pyLife stores all its data in :class:`pandas.Series` and
   :class:`pandas.DataFrame` objects.


Install pyLife
==============

.. code-block:: console

   $ pip install pylife[extras]

See :doc:`INSTALLATION` for conda, development installs and the optional
dependencies.

You can also try pyLife without installing anything, by running the example
notebooks on `MyBinder
<https://mybinder.org/v2/gh/boschresearch/pylife/develop?labpath=demos%2Findex.ipynb>`_.


Your first damage calculation
=============================

A fatigue assessment in pyLife always combines two pieces of information:

*strength* — how much load the material tolerates, described by a Wöhler curve
(also called SN-curve), and *load* — how often which load level actually
occurs, described by a load collective.

Describe the material strength as a :class:`pandas.Series`:

.. doctest::

   >>> import pandas as pd
   >>> import pylife.strength.fatigue
   >>> import pylife.stress.collective

   >>> woehler_curve = pd.Series({
   ...     'k_1': 7.0,      # slope above the endurance limit
   ...     'ND': 2e6,       # endurance limit in cycle direction
   ...     'SD': 300.0,     # endurance limit in load direction, in MPa
   ...     'TN': 3.4,       # scatter in cycle direction
   ...     'TS': 1.234,     # scatter in load direction
   ... })

Ask it how many cycles the material survives at a stress amplitude of 400 MPa:

.. doctest::

   >>> float(woehler_curve.woehler.basquin_cycles(400.0).round(0))
   266968.0

Now describe the load as a collective, i.e. how many cycles occur in which
load range.  A histogram indexed by a :class:`pandas.IntervalIndex` is the
natural way to write that down:

.. doctest::

   >>> cycles = pd.Series(
   ...     [1e4, 1e3, 1e2],
   ...     index=pd.IntervalIndex.from_breaks(
   ...         [600.0, 800.0, 1000.0, 1200.0], name='range'
   ...     ),
   ...     name='cycles',
   ... )

Accumulate the damage of that collective on that material:

.. doctest::

   >>> damage = woehler_curve.fatigue.damage(cycles.load_collective)
   >>> damage
   range
   (600.0, 800.0]      0.014709
   (800.0, 1000.0]     0.008543
   (1000.0, 1200.0]    0.003481
   Name: damage, dtype: float64

   >>> float(damage.sum().round(4))
   0.0267

A damage sum of 0.0267 means that this collective consumes about 2.7 % of the
component's fatigue life.  Applying it roughly 37 times would reach a damage
sum of 1.0, which is the point where failure is expected.

Two things in this example are worth remembering, because they apply
everywhere in pyLife:

* **The data is plain pandas.**  ``woehler_curve`` and ``cycles`` are ordinary
  pandas objects.  You can slice, plot, save and load them with everything
  pandas offers.
* **The calculations are accessors.**  ``.woehler``, ``.fatigue`` and
  ``.load_collective`` are attributes that pyLife registers on pandas objects
  when you import the corresponding module.  They validate the data and add
  the fatigue specific methods.  This is the :doc:`Signal API <signal_api>`.


What pyLife can do for you
==========================

pyLife groups its functionality into the following tasks.

Analyse loads and stresses
--------------------------

Basic operations on time signals as well as more complex ones such as rainflow
counting.

* :mod:`pylife.stress.timesignal` — operations on time signals
* :mod:`pylife.stress.rainflow` — a versatile module for rainflow counting
* :mod:`pylife.stress.collective` — handling of load collectives
* :mod:`pylife.stress.equistress` — equivalent stresses from stress tensors

Fit material data
-----------------

Extract material parameters from experimental data.  As of now this is a
versatile set of classes to fit Wöhler curve parameters from experimental
fatigue data.

* :mod:`pylife.materialdata.woehler`

Predict material behaviour
--------------------------

Use material parameters — for example the ones fitted by the modules above —
to predict how the material responds.

* :class:`pylife.materiallaws.WoehlerCurve`
* :class:`pylife.materiallaws.RambergOsgood`
* :mod:`pylife.materiallaws.true_stress_strain` — true stress and true strain

Assess the lifetime of components
---------------------------------

Calculate lifetimes, failure probabilities and endurance limits of components
from load sequences and material data.

* :mod:`pylife.strength.fatigue` — damage accumulation
* :mod:`pylife.strength.meanstress` — mean stress transformation
* :mod:`pylife.strength.fkm_nonlinear.assessment_nonlinear_standard` — local
  strain concept of the nonlinear FKM guideline

Work with FE meshes
-------------------

* :mod:`pylife.mesh.meshsignal` — accessors for general mesh operations
* :class:`pylife.mesh.HotSpot` — hotspot detection
* :class:`pylife.mesh.Gradient` — gradients of scalar values along a mesh
* :class:`pylife.mesh.Meshmapper` — map a mesh onto another one of the same
  geometry by interpolation

Exchange FE results
-------------------

* :mod:`pylife.vmap` — import from and export to `VMAP
  <https://www.vmap.eu.com/>`_ files
* :doc:`tools/index` — read Abaqus ODB files

Utilities
---------

* :mod:`pylife.utils` — mathematical helpers used throughout the code base


Where to go next
================

The pyLife documentation is organised along what you want to do:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - If you want to ...
     - ... read this
   * - learn pyLife step by step
     - the :doc:`tutorials`, hands on notebooks that teach one concept at a
       time
   * - understand how pyLife thinks
     - the :doc:`user_guide`, which explains the data model and the signal API
   * - solve a concrete task
     - the :doc:`cookbook`, which shows complete workflows you can adapt
   * - look up a function or a class
     - the :doc:`reference`
   * - look up a symbol such as ``SD`` or ``k_1``
     - the :doc:`glossary`

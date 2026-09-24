************************
pyLife API reference
************************

This reference describes every public module, class and function of pyLife.

If you are looking for an introduction rather than a lookup, start with
:doc:`getting_started`.  The symbols used as keys in pyLife's data structures —
``SD``, ``ND``, ``k_1`` and friends — are explained in the :doc:`glossary`.

.. grid:: 1 2 3 3
   :gutter: 3
   :padding: 2 2 0 0

   .. grid-item-card:: Core
      :link: general/signal
      :link-type: doc
      :shadow: none

      The machinery every pyLife accessor is built on: signal validation and
      broadcasting.

   .. grid-item-card:: Stress
      :link: stress/index
      :link-type: doc
      :shadow: none

      Loads and stresses: time signals, rainflow counting, load collectives
      and equivalent stresses.

   .. grid-item-card:: Strength
      :link: reference-strength
      :link-type: ref
      :shadow: none

      Damage accumulation, mean stress transformation, failure probabilities
      and the FKM guidelines.

   .. grid-item-card:: Material laws
      :link: reference-materiallaws
      :link-type: ref
      :shadow: none

      Models that predict how a material responds: Hooke, Ramberg-Osgood,
      Wöhler curves and notch approximation.

   .. grid-item-card:: Material data
      :link: materialdata/woehler
      :link-type: doc
      :shadow: none

      Fit material parameters to experimental fatigue test data.

   .. grid-item-card:: Mesh
      :link: reference-mesh
      :link-type: ref
      :shadow: none

      Operations on FE meshes: gradients, hotspots and mesh mapping.

   .. grid-item-card:: VMAP
      :link: vmap/vmap
      :link-type: doc
      :shadow: none

      Read and write FE results in the vendor neutral VMAP format.

   .. grid-item-card:: Utils
      :link: reference-utils
      :link-type: ref
      :shadow: none

      Mathematical helpers used throughout the code base.

   .. grid-item-card:: Additional tools
      :link: tools/index
      :link-type: doc
      :shadow: none

      Companion packages, such as the Abaqus ODB client.


Core
====

The base classes that give pandas objects their pyLife behaviour.  See
:doc:`signal_api` for how to use and extend them.

.. toctree::
   :maxdepth: 1

   general/signal


Stress
======

Everything that describes what acts *on* the component.

.. toctree::
   :maxdepth: 1

   stress/index
   stress/timesignal
   stress/frequencysignal
   stress/rainflow
   stress/collective
   stress/equistress
   stress/stresssignal


.. _reference-strength:

Strength
========

Everything that describes what the component *tolerates*, and how the two are
brought together into a lifetime statement.

.. toctree::
   :maxdepth: 1

   strength/fatigue
   strength/miner
   strength/meanstress
   strength/failure_probability
   strength/damage_parameters
   strength/fkm_load_distribution

The FKM guideline, linear and nonlinear:

.. toctree::
   :maxdepth: 1

   strength/fkm_linear
   strength/fkm_nonlinear
   strength/fkm_nonlinear_parameter_calculations
   strength/fkm_nonlinear_damage_calculator
   strength/woehler_fkm_nonlinear


.. _reference-materiallaws:

Material laws
=============

Models that predict material behaviour from material parameters.

.. toctree::
   :maxdepth: 1

   materiallaws/hookeslaw
   materiallaws/rambgood
   materiallaws/woehlercurve
   materiallaws/true_stress_strain
   materiallaws/notch_approximation_laws


Material data
=============

Fitting material parameters to experimental data.

.. toctree::
   :maxdepth: 1

   materialdata/woehler


.. _reference-mesh:

Mesh
====

Operations on FE meshes.

.. toctree::
   :maxdepth: 1

   mesh/meshsignal
   mesh/hotspot
   mesh/gradient
   mesh/gradient3D
   mesh/meshmapping
   mesh/surface3D


VMAP interface
==============

Import and export of FE results in the VMAP format.

.. toctree::
   :maxdepth: 1

   vmap/vmap
   vmap/vmap_import
   vmap/vmap_export


.. _reference-utils:

Utils
=====

Mathematical helper functions used throughout pyLife.

.. toctree::
   :maxdepth: 1

   utils/functions
   utils/histogram
   utils/probability_data

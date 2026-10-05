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
      :link: strength/index
      :link-type: doc
      :shadow: none

      Damage accumulation, mean stress transformation, failure probabilities
      and the FKM guidelines.

   .. grid-item-card:: Material laws
      :link: materiallaws/index
      :link-type: doc
      :shadow: none

      Models that predict how a material responds: Hooke, Ramberg-Osgood,
      Wöhler curves and notch approximation.

   .. grid-item-card:: Material data
      :link: materialdata/woehler
      :link-type: doc
      :shadow: none

      Fit material parameters to experimental fatigue test data.

   .. grid-item-card:: Mesh
      :link: mesh/index
      :link-type: doc
      :shadow: none

      Operations on FE meshes: gradients, hotspots and mesh mapping.

   .. grid-item-card:: VMAP
      :link: vmap/index
      :link-type: doc
      :shadow: none

      Read and write FE results in the vendor neutral VMAP format.

   .. grid-item-card:: Utils
      :link: utils/index
      :link-type: doc
      :shadow: none

      Mathematical helpers used throughout the code base.

   .. grid-item-card:: Additional tools
      :link: tools/index
      :link-type: doc
      :shadow: none

      Companion packages, such as the Abaqus ODB client.


.. toctree::
   :maxdepth: 1
   :hidden:

   general/signal

.. toctree::
   :maxdepth: 1
   :hidden:

   stress/index

.. toctree::
   :maxdepth: 1
   :hidden:

   strength/index


.. toctree::
   :maxdepth: 1
   :hidden:

   materiallaws/index

.. toctree::
   :maxdepth: 1
   :hidden:

   materialdata/woehler

.. toctree::
   :maxdepth: 1
   :hidden:

   mesh/index

.. toctree::
   :maxdepth: 1
   :hidden:

   utils/index

.. toctree::
   :maxdepth: 1
   :hidden:

   vmap/index

.. toctree::
   :maxdepth: 1
   :hidden:

   tools/index

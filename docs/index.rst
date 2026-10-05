:html_theme.sidebar_secondary.remove:

******
pyLife
******

pyLife – an open source Python library for state of the art algorithms used in
the lifetime assessment of mechanical components subjected to fatigue.


What pyLife can do for you
==========================

.. grid:: 1 1 2 2
   :gutter: 3
   :padding: 2 2 0 0
   :class-container: pylife-add-card-grid

   .. grid-item-card:: Analyse load data
      :shadow: none

      .. image:: _static/images/rainflow-matrix-jet.png

      Rainflow counting, load collectives, equivalent stresses and time
      signal processing — see :mod:`pylife.stress`.

   .. grid-item-card:: Assess lifetime
      :shadow: none

      .. image:: _static/images/damage-calculation.png
         :class: only-light

      .. image:: _static/images/damage-calculation-dark.png
         :class: only-dark

      Damage accumulation, failure probabilities and the FKM guideline,
      linear and nonlinear — see :mod:`pylife.strength`.

   .. grid-item-card:: Work with FE meshes
      :shadow: none

      .. image:: _static/images/mesh.png

      Stress gradients, hotspot detection and mesh mapping on FE results —
      see :mod:`pylife.mesh`.

   .. grid-item-card:: Fit material data
      :shadow: none

      .. image:: _static/images/woehler_analyzer.png
         :class: only-light

      .. image:: _static/images/woehler_analyzer_dark.png
         :class: only-dark

      Derive Wöhler curve parameters from experimental fatigue test data with
      maximum likelihood or Bayesian methods — see
      :mod:`pylife.materialdata.woehler`.




.. grid:: 1 2 2 2
   :gutter: 4
   :padding: 2 2 0 0
   :class-container: pylife-chap-card-grid

   .. grid-item-card:: Getting started
      :shadow: md

      New to pyLife?  Install the package and run your first damage
      calculation.  This is the place to start.

      +++

      .. button-ref:: getting_started
         :ref-type: doc
         :color: primary
         :expand:

         Get started

   .. grid-item-card:: Tutorials
      :shadow: md

      Learning oriented, hands on notebooks that walk you through pyLife's
      building blocks: Wöhler curves, load collectives and the FKM nonlinear
      assessment.

      +++

      .. button-ref:: tutorials/index
         :ref-type: doc
         :color: primary
         :expand:

         To the tutorials

   .. grid-item-card:: User guide
      :shadow: md

      The concepts behind pyLife: how fatigue data is stored in pandas
      objects, how the signal API applies calculations to it and how
      broadcasting between load and strength works.

      +++

      .. button-ref:: user_guide
         :ref-type: doc
         :color: primary
         :expand:

         To the user guide

   .. grid-item-card:: Cookbook
      :shadow: md

      Task oriented recipes for real workflows — lifetime calculation,
      hotspot detection, stress gradients, FE result import and time series
      handling.

      +++

      .. button-ref:: cookbook
         :ref-type: doc
         :color: primary
         :expand:

         To the cookbook

   .. grid-item-card:: API reference
      :shadow: md

      The detailed description of every public module, class and function in
      pyLife, including parameters, return values and the underlying
      engineering standards.

      +++

      .. button-ref:: reference
         :ref-type: doc
         :color: primary
         :expand:

         To the reference

   .. grid-item-card:: Contributor guide
      :shadow: md

      pyLife is developed in the open and welcomes contributions from
      science, education and industry.  Find out how to report issues and
      submit improvements.

      +++

      .. button-ref:: CONTRIBUTING
         :ref-type: doc
         :color: primary
         :expand:

         To the contributor guide


Try it without installing
=========================

All notebooks in the tutorials and the cookbook can be run in the browser via
`MyBinder
<https://mybinder.org/v2/gh/boschresearch/pylife/develop?labpath=demos%2Findex.ipynb>`_
without installing anything on your computer.


.. toctree::
   :hidden:
   :caption: Getting started

   Getting started <getting_started>

.. toctree::
   :hidden:
   :caption: Learn

   Learn <learn>

.. toctree::
   :hidden:
   :caption: Reference

   Reference <reference>

.. toctree::
   :hidden:
   :caption: Contributing

   Contributing <contributing>

.. toctree::
   :hidden:
   :caption: About

   About <about>

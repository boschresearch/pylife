:html_theme.sidebar_secondary.remove:

********************
pyLife documentation
********************

**Version**: |release|

pyLife is an open source Python library for state of the art algorithms used in
the lifetime assessment of mechanical components subjected to fatigue.  It
brings rainflow counting, Wöhler curve (SN-curve) analysis, mean stress
transformation, damage accumulation and the FKM guidelines together in one
consistent, `pandas <https://pandas.pydata.org/>`_ based data model.

.. grid:: 1 2 2 2
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: pylife-card-grid

    .. grid-item-card:: Getting started
        :shadow: md

        New to pyLife?  Install the package and run your first lifetime
        assessment.  This is the place to start.

        +++

        .. button-ref:: INSTALLATION
            :ref-type: doc
            :color: primary
            :expand:

            Install pyLife

    .. grid-item-card:: Tutorials
        :shadow: md

        Learning oriented, hands on notebooks that walk you through pyLife's
        building blocks: Wöhler curves, load collectives and the FKM
        nonlinear assessment.

        +++

        .. button-ref:: tutorials
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
        hotspot detection, stress gradients, FE result import and time
        series handling.

        +++

        .. button-ref:: cookbook
            :ref-type: doc
            :color: primary
            :expand:

            To the cookbook

    .. grid-item-card:: API reference
        :shadow: md

        The detailed description of every public module, class and function
        in pyLife, including parameters, return values and the underlying
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


What pyLife can do for you
==========================

.. grid:: 1 2 3 3
    :gutter: 3
    :padding: 2 2 0 0

    .. grid-item-card:: Analyse load data
        :shadow: none

        Rainflow counting, load collectives, equivalent stresses and time
        signal processing — see :mod:`pylife.stress`.

    .. grid-item-card:: Fit material data
        :shadow: none

        Derive Wöhler curve parameters from experimental fatigue test data
        with maximum likelihood or Bayesian methods — see
        :mod:`pylife.materialdata.woehler`.

    .. grid-item-card:: Model material behaviour
        :shadow: none

        Ramberg-Osgood, Hooke's law, notch approximation laws and Wöhler
        curves — see :mod:`pylife.materiallaws`.

    .. grid-item-card:: Assess lifetime
        :shadow: none

        Damage accumulation, failure probabilities and the FKM guideline,
        linear and nonlinear — see :mod:`pylife.strength`.

    .. grid-item-card:: Work with FE meshes
        :shadow: none

        Stress gradients, hotspot detection and mesh mapping on FE results —
        see :mod:`pylife.mesh`.

    .. grid-item-card:: Exchange FE results
        :shadow: none

        Read and write `VMAP <https://www.vmap.eu.com/>`_ files and import
        Abaqus ODB data — see :mod:`pylife.vmap` and :doc:`tools/index`.


Try it without installing
=========================

All notebooks in the tutorials and the cookbook can be run in the browser via
`MyBinder <https://mybinder.org/v2/gh/boschresearch/pylife/develop?labpath=demos%2Findex.ipynb>`_
without installing anything on your computer.


.. toctree::
   :hidden:
   :caption: Getting started

   README
   INSTALLATION

.. toctree::
   :hidden:
   :caption: Learn

   tutorials
   user_guide
   cookbook

.. toctree::
   :hidden:
   :caption: Reference

   reference
   tools/index

.. toctree::
   :hidden:
   :caption: Development

   CONTRIBUTING
   CODINGSTYLE
   variable_names

.. toctree::
   :hidden:
   :caption: About

   NEWS-2.0
   CHANGELOG
   NOTICE
   LICENSE
   3rd-party-licenses

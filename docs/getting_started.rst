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

There are three ways you can install pyLife:

Project based
    When you want to use pyLife in your own code. Let's say you are writing
    your own small tool for your engineering and want to make use of pyLife's
    functionality.

Environment based
    You want to have a python environment with pyLife available. Let's say you
    are using `Jupyter <https://jupyter.org/>`_ or `Marimo <https://marimo.io/>`_
    notebooks to perform complex calculation and you want to use pyLife's
    functionality in them.

Install from the git repository
    That's only relevant if you actually want to develop and contribute to
    pyLife.


.. note::

   The installation instructions on this page assume that you have basic
   familiarity with UNIX command line or a command shell on
   Windows. Unfortunately we cannot cover the usage of command line tools in
   this documentation as it is way beyond our scope.

.. tab-set::

   .. tab-item:: Project based

      Project based tools such as `uv <https://docs.astral.sh/uv/>`_ or
      `pixi <https://pixi.sh/>`_ keep each project in its own folder with a
      manifest and a lockfile describing exact, reproducible dependencies,
      and manage the virtual environment for you. If unsure, use ``uv``.

      Once you installed ``uv`` you can setup a python project that uses pyLife
      like this:

      .. code-block:: console

         $ uv init my_project
         $ cd my_project
         $ uv add pylife

      Now you can use pyLife in python files inside this python project.

      We recommend checking out `uv <https://docs.astral.sh/uv/>`_ in more
      detail if you plan to write your own python packages or if you often work
      on python projects.

      Happy coding

   .. tab-item:: Environment based

      You will probably use some version of the ``conda`` tool to manage your
      python environments. It is commercially available from `Anaconda
      <https://www.anaconda.com/>`_ as well as a free community miniforge
      variant from `conda forge <https://conda-forge.org/download/>`_. If
      unsure, go with the miniforge variant.

      Once you installed your ``conda`` tool you can setup a python environment
      like this:

      .. code-block:: console

         $ conda create -n pylife-env python
         $ conda activate pylife-env
         $ pip install pylife

      If you want to use `Jupyter <https://jupyter.org/>`_ or
      `Marimo <https://marimo.io/>`_ notebooks for your work you can install
      the ``jupyter`` or the ``marimo`` package inside your environment.

      Deactivate the environment again with ``deactivate`` once you are
      done.

      Happy engineering

   .. tab-item:: From the git repository

      If you want to contribute to pyLife – read the :doc:`contributing guide
      <contributing>` for that – you can setup your pyLife working copy. First
      you need to install the `uv <https://docs.astral.sh/uv/>`_ tool. Once you
      have done that you can setup your pyLife working copy like this.

      .. code-block:: console

         $ git clone https://github.com/boschresearch/pylife.git
         $ cd pylife
         $ uv sync

      Now you should be able to run the test suite wit

      .. code-block:: console

         $ uv run pytest

      Happy coding




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



Where to go next
================

The pyLife documentation is organised along what you want to do:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - If you want to ...
     - ... read this
   * - learn pyLife step by step
     - the :doc:`tutorials/index`, hands on notebooks that teach one concept at a
       time
   * - understand how pyLife thinks
     - the :doc:`user_guide`, which explains the data model and the signal API
   * - solve a concrete task
     - the :doc:`cookbook`, which shows complete workflows you can adapt
   * - look up a function or a class
     - the :doc:`reference`
   * - look up a symbol such as ``SD`` or ``k_1``
     - the :doc:`glossary`

#################
pyLife User guide
#################

This document aims to briefly describe the overall design of pyLife and how to
use it inside your own applications, for example Jupyter Notebooks.


Overview
========

pyLife provides facilities to perform different kinds of tasks.  They can be
roughly grouped as follows


Fitting material data
---------------------

This is about extracting material parameters from experimental data.  As of now
this is a versatile set of classes to fit Wöhler curve (aka SN-curve)
parameters from experimental fatigue data.  Mid term we would like to see there
a module to fit tensile test data and things like that.

* :mod:`pylife.materialdata.woehler`


Predicting material behavior
----------------------------

These modules use material parameters, e.g. the ones fitted by the
corresponding module about data fitting, to predict material behavior.  As of
now these are

* :class:`pylife.materiallaws.RambergOsgood`
* :class:`pylife.materiallaws.WoehlerCurve`
* Functions to calculate the true stress and true strain, see
  :mod:`pylife.materiallaws.true_stress_strain`


Analyzing load collectives and stresses
---------------------------------------

These modules perform basic operations on time signals as well as more complex
things as rainflow counting.

* :mod:`pylife.stress.collective` – facilities to handle load collectives
* :mod:`pylife.stress.rainflow` – a versatile module for rainflow counting
* :mod:`pylife.stress.equistress` for equivalent stress calculations from
  stress tensors
* :mod:`pylife.stress.timesignal` for operations on time signals



Lifetime assessment of components
---------------------------------

Calculate lifetime, failure probabilities, endurance limits of
components based on load sequences and material data.

* :mod:`pylife.strength.fkm_nonlinear.assessment_nonlinear_standard` – Local strain concept / FKM guideline nonlinear

See also the tutorial about :doc:`FKM nonlinear <demos/fkm_nonlinear>`.

Mesh operations
---------------

For operations on FEM meshes

* :mod:`pylife.mesh.meshsignal` – accessor classes for general mesh operations
* :class:`pylife.mesh.HotSpot` for hotspot detection
* :class:`pylife.mesh.Gradient` to calculate gradients of scalar values along a mesh
* :class:`pylife.mesh.Meshmapper` to map one mesh to another of the same geometry by
  interpolating


VMAP interface
--------------

Import and export mesh data from/to VMAP files.


Utilities
---------

Some mathematical helper functions, that are useful throughout the code base.



General Concepts
================

pyLife aims to provide toolbox of calculation tools that can be plugged
together in order to perform complex operations.  We try to make the use of the
existing modules as well as writing custom ones as easy as possible while at
the same time performing well on larger amounts of data.  Moreover we try keep
data that belongs together in self explaining data structures, rather than in
individual variables.

In order to achieve all that we make extensive use of `pandas
<https://pandas.pydata.org/>`_ and `numpy <https://numpy.org/>`_. In this guide
we suppose that you have a basic understanding of these libraries and the data
structures and concepts they are using.

The page :doc:`Data Model <data_model>` describes the way how data should be stored in
pandas objects.

The page :doc:`Signal API <signal_api>` describes how mathematical operations are to be
performed on those pandas objects.

.. toctree::
   :maxdepth: 2

   data_model
   signal_api
   broadcaster

#################
pyLife User guide
#################

This document aims to briefly describe the overall design of pyLife and how to
use it inside your own applications, for example Jupyter Notebooks.

pyLife being a library for numerical data analysis and numerical data
processing makes extensive use of `pandas <https://pandas.pydata.org/>`_ and
`numpy <https://numpy.org/>`_. In this guide we suppose that you have a basic
understanding of these libraries and the data structures and concepts they are
using.


Overview
========

pyLife provides facilities to perform different kinds of tasks.  They can be
roughly grouped as follows


Fitting material data
---------------------

This is about extracting material parameters from experimental data.  As of now
this is a versatile set of classes to fit Wöhler corve (aka SN-curve)
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

These modules perform basic operations on time signals as well as more complpex
things as rainflow counting.

* :mod:`pylife.stress.rainflow` – a versatile module for rainflow counting
* :mod:`pylife.stress.equistress` for equivalent stress calculations from
  stress tensors
* :mod:`pylife.stress.timesignal` for operations on timesignals



Lifetime assessment of components
---------------------------------

Calculate lifetime, failure probabilities, nominal endurance limits of
components based on load collective and material data.


Mesh operations
---------------

For operations on FEM meshes

* :py:class:`pylife.mesh.Hotspot` for hotspot detection
* :py:class:`mesh.Gradient` to calculate gradients of scalar values along a mesh
* :class:`mesh.Meshmapping` to map one mesh to another of the same geometry by
  interpolating


VMAP interface
--------------

Import and export mesh data from/to VMAP files.

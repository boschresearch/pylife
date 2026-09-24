*****************
pyLife user guide
*****************

This guide explains the concepts that pyLife is built on.  It is the
background knowledge you need to combine pyLife's modules into your own
calculations, and to write new modules that fit in.

If you are looking for a first example instead, read :doc:`getting_started`.
If you are looking for a complete workflow, read the :doc:`cookbook`.


The idea behind pyLife
======================

pyLife provides a toolbox of calculation tools that can be plugged together in
order to perform complex operations.  We try to make the use of the existing
modules as well as the writing of custom ones as easy as possible, while at the
same time performing well on larger amounts of data.  Moreover we keep data
that belongs together in self explaining data structures, rather than in
individual variables.

To achieve that, pyLife makes extensive use of `pandas
<https://pandas.pydata.org/>`_ and `numpy <https://numpy.org/>`_.  This guide
supposes that you have a basic understanding of these libraries and of the data
structures they provide.

Three concepts follow from that, and each has its own page:

:doc:`data_model`
   How data is laid out in pandas objects — what belongs in row direction,
   what belongs in column direction, and how multidimensional quantities such
   as a rainflow matrix are represented.

:doc:`signal_api`
   How calculations are attached to those pandas objects as accessors, how
   they validate their input, and how you write your own accessor.

:doc:`broadcaster`
   How two pandas objects of different shape — for example one Wöhler curve
   per mesh node and one load collective per load case — are aligned so that
   they can be combined.

The symbols used as keys throughout these data structures are explained in the
:doc:`glossary`.

.. toctree::
   :maxdepth: 2

   data_model
   signal_api
   broadcaster
   glossary

***************
pyLife Cookbook
***************

The cookbook collects complete, runnable workflows that solve a concrete
task — calculate the lifetime of a component, detect hotspots in an FE mesh,
import a mesh from a VMAP file.  Take the recipe that comes closest to your
problem and adapt it.

The recipes assume that you already know the pyLife basics.  If you do not
yet, start with the :doc:`tutorials/index`, which teach the concepts one at a time.

Here you find the statically rendered HTML pages.  The notebook files are
available in the ``/demos`` `directory
<https://github.com/boschresearch/pylife/tree/develop/demos>`_ of
pyLife's codebase.

If you want to try out the notebooks without installing anything on your
computer, you can use `MyBinder
<https://mybinder.org/v2/gh/boschresearch/pylife/develop?labpath=demos%2Findex.ipynb>`_.

.. toctree::
   :maxdepth: 1
   :caption: Life time and reliability

   demos/lifetime_calc.nblink
   demos/fkm_nonlinear.nblink
   demos/fkm_nonlinear_full.nblink

.. toctree::
   :maxdepth: 1
   :caption: Material Laws

   demos/ramberg_osgood.nblink

.. toctree::
   :maxdepth: 1
   :caption: Material Data

   demos/woehler_analyzer.nblink

.. toctree::
   :maxdepth: 1
   :caption: FEM based methods

   demos/hotspot_beam.nblink
   demos/stress_gradient.nblink
   demos/local_stress_with_FE.nblink


.. toctree::
   :maxdepth: 1
   :caption: Tools

   demos/psd_optimizer.nblink
   demos/time_series_handling.nblink

.. toctree::
   :maxdepth: 1
   :caption: Load FEM meshes

   demos/import_mesh_vmap.nblink

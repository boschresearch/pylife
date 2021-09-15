The pyLife Signal API
=====================

The signal api is the higher level API of pyLife. It is the API that you
probably should be using. Some of the domain specific functions are also
available as pure numpy functions.  However, we highly recommend you to take a
closer look at pandas and consider to adapt your application to the pandas way
of doing things.


Motivation
----------

In pyLife's domain, we often deal with data structures that consist multiple
numerical values.  For example a Wöhler curve (see
:class:`~pylife.materiallaws.WoehlerCurve`) consists at least of the parameters
``k_1``, ``ND`` and ``SD``.  Optionally it can have the additional parameters
``k_2``, ``TN`` and ``TS``.  As experience teaches us, it is advisable to put
these kinds of values into one variable to avoid confusion.  For example a
function with more than five positional arguments often leads to hard to
debugable bugs due to wrong positioning.

Moreover some of these data structures can come in different dimensionalities.
For example data structures describing material behavior can come as one
dataset, describing one material.  They can also come as a map to a FEM mesh,
for example when you are dealing with case hardened components.  Then every
element of your FEM mesh can have a different associated Wöhler curve
dataset.  In pyLife we want to deal with these kinds of mappings easily without
much coding overhead for the programmer.

The pyLife signal API provides the class :class:`pylife.PylifeSignal` to
facilitate handling these kinds of data structures and broadcasting them to
another signal instance.

This page describes the basic concept of the pyLife signal API.  The next page
describes the broadcasting mechanism of a :class:`pylife.PylifeSignal`.


The basic concept
-----------------

The basic idea is to have all the data in a signal like data structure, that
can be piped through the individual calculation process steps. Each calculation
process step results in a new signal, that then can be handed over to the next
process step.

Signals can be for example

* stress tensors like from an FEM-solver

* load collectives like time signals or a rainflow matrix

* material data like Wöhler curve parameters

* ...


From a programmer's point of view, signals are objects of either
:class:`pandas.Series` or :class:`pandas.DataFrame`, depending if they are one
or two dimensional (see here about :doc:`dimensionality <data_model>`).

Functions that operate on a signal are usually written as methods of an
instance of as class derived from :class:`PylifeSignal`.  These classes are
usually decorated as Series or DataFrame accessor using
:func:`pandas.api.extensions.register_series_accessor()` resp.
:func:`pandas.api.extensions.register_dataframe_accessor()`.

Due to the decorators, signal accessor classes can be instantiated also as an
attribute of a :class:`pandas.Series` or :class:`pandas.DataFrame`. The
following two lines are equivalent.

Usual class instantiation:

::

   PlainMesh(df).coordinates

Or more convenient using the accessor decorator attribute:

::

   df.plain_mesh.coordinates


There is also the convenience function
:func:`~pylife.signal.PylifeSignal.from_parameters` to instantiate the signal
class from individual parameters.  So a
:class:`pylife.materialdata.WoehlerCurve` can be instantiated in three ways.

* directly with the class constructor

  .. code-block:: python

      data = pd.Series({
          'k_1': 7.0,
          'ND': 2e6,
          'SD': 320.
      })
      wc = WoehlerCurve(data)

* using the pandas accessor

  .. code-block:: python

      data = pd.Series({
          'k_1': 7.0,
          'ND': 2e6,
          'SD': 320.
      })
      wc = data.woehler

* from individual parameters

  .. code-block:: python

      wc = WoehlerCurve.from_parameters(k_1=7.0, ND=2e6, SD= 320.)



How to use predefined signal accessors
``````````````````````````````````````

There are too reasons to use a signal accessor:

* let it validate the accessed DataFrame
* use a method or access a property that the accessor defines

Example for validation
^^^^^^^^^^^^^^^^^^^^^^

In the following example we are validating a DataFrame that if it is a valid
plain mesh, i.e. if it has the columns `x` and `y`.

Import the modules. Note that the module with the signal accessors (here
:mod:`mesh`) needs to be imported explicitly.

.. jupyter-execute::

   import pandas as pd
   import pylife.mesh

Create a DataFrame and have it validated if it is a valid plain mesh, i.e. has
the columns `x` and `y`.

.. jupyter-execute::

   df = pd.DataFrame({'x': [1.0], 'y': [1.0]})
   df.plain_mesh


Now create a DataFrame which is not a valid plain mesh and try to have it
validated:

.. jupyter-execute::
   :raises:

   df = pd.DataFrame({'x': [1.0], 'a': [1.0]})
   df.plain_mesh


Example for accessing a property
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get the coordinates of a 2D plain mesh

.. jupyter-execute::

   df = pd.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [1.0, 2.0, 3.0]})
   df.plain_mesh.coordinates

Now a 3D mesh

.. jupyter-execute::

   df = pd.DataFrame({'x': [1.0], 'y': [1.0], 'z': [1.0], 'foo': [42.0], 'bar': [23.0]})
   df.plain_mesh.coordinates



Defining your own signal accessors
----------------------------------

If you want to write a processor for signals you need to put the processing
functionality in an accessor class that is derived from the signal accessor
base class like for example :class:`~.meshsignal.Mesh`. This class you
register as a pandas DataFrame accessor using a decorator

.. code-block:: python

    import pandas as pd
    import pylife.mesh

    @pd.api.extensions.register_dataframe_accessor('my_mesh_processor')
    class MyMesh(meshsignal.Mesh):
        def do_something(self):
	    # ... your code here
	    # the DataFrame is accessible by self._obj
	    # usually you would calculate a DataFrame df to return it.
	    df = ...
	    # you might want copy the index of self._obj to the returned
	    # DataFrame.
	    return df.set_index(self._obj.index)

As `MyMesh` is derived from :class:`~.meshsignal.Mesh` the
validation of `Mesh` is performed. So in the method `do_something()`
you can rely on that `self._obj` is a valid mesh DataFrame.

You then can use the class in the following way when the module is imported.


Performing additional validation
````````````````````````````````

Sometimes your signal accessor needs to perform an additional validation on the
accessed signal. For example you might need a mesh that needs to be
3D. Therefore you can reimplement `_validate()` to perform the additional
validation. Make sure to call `_validate()` of the accessor class you are
deriving from like in the following example.

.. jupyter-execute::
   :raises: AttributeError

   import pandas as pd
   import pylife.mesh

   @pd.api.extensions.register_dataframe_accessor('my_only_for_3D_mesh_processor')
   class MyOnlyFor3DMesh(pylife.mesh.PlainMesh):
       def _validate(self):
           super()._validate() # call PlainMesh._validate()
           self.fail_if_key_missing(['z'])

   df = pd.DataFrame({'x': [1.0], 'y': [1.0]})
   df.my_only_for_3D_mesh_processor


Defining your own signals
-------------------------

The same way the predefined pyLife signals are defined you can define your own
signals. Let's say, for example, that in your signal there needs to be the
columns `alpha`, `beta`, `gamma` all of which need to be positive.

You would put the signal class into a module file `my_signal_mod.py`

.. jupyter-execute::

    import pandas as pd
    from pylife import PylifeSignal

    @pd.api.extensions.register_dataframe_accessor('my_signal')
    class MySignal(PylifeSignal):
        def _validate(self):
            self.fail_if_key_missing(['alpha', 'beta', 'gamma'])
            for k in ['alpha', 'beta', 'gamma']:
                if (self._obj[k] < 0).any():
                    raise ValueError("All values of %s need to be positive. "
                                     "At least one is less than 0" % k)

	def some_method(self):
	    return self._obj[['alpha', 'beta', 'gamma']] * -3.0

You can then validate signals and/or call ``some_method()``.

Validation success.

.. jupyter-execute::

    df = pd.DataFrame({'alpha': [1.0, 2.0], 'beta': [1.0, 0.0], 'gamma': [1.0, 2.0]})
    df.my_signal.some_method()


Validation fails because of missing `gamma` column.

.. jupyter-execute::
   :raises: AttributeError

    df = pd.DataFrame({'alpha': [1.0, 2.0], 'beta': [1.0, -1.0]})
    df.my_signal.some_method()


Validation fail because one `beta` is negative.

.. jupyter-execute::
   :raises: ValueError

    df = pd.DataFrame({'alpha': [1.0, 2.0], 'beta': [1.0, -1.0], 'gamma': [1.0, 2.0]})
    df.my_signal.some_method()


Additional attributes in your own signals
`````````````````````````````````````````

If your accessor class needs to have attributes other than the accessed object
itself you can define default values in the `__init__()` of your accessor and
set these attributes with setter methods.

.. code-block:: python

    import pandas as pd
    from pylife import PylifeSignal

    @pd.api.extensions.register_dataframe_accessor('my_signal')
    class MySignal(PylifeSignal):
	def __init__(self, pandas_obj):
	    super(MySignal, self).__init__(pandas_obj)
	    self._my_attribute = 'the default value'

        def set_my_attribute(self, my_attribute):
	    self._my_attribute = my_attribute
	    return self

	def do_something(self, some_parameter):
	    # ... use some_parameter, self._my_attribute and self._obj


>>> df.my_signal.set_my_attribute('foo').do_something(2342)



Registering a method to an existing accessor class
--------------------------------------------------

.. note::
   This functionality might be dropped on the way to `pyLife-2.0` as it turns
   out that it is not that much used.

One drawback of the accessor class API is that you cannot extend accessors by
deriving from them. For example if you need a custom equivalent stress function
you cannot add it by deriving from :class:`~.equistress.StressTensorEquistress`,
and register it by the same accessor `equistress`.

The solution for that is :func:`register_method()` that lets you monkey patch a
new method to any class deriving from :class:`~.pylife.PylifeSignal`.

.. code-block:: python

    from pylife import equistress

    @pl.signal_register_method(equistress.StressTensorEquistress, 'my_equistress')
    def my_equistress_method(df)
	# your code here
	return ...

Then you can call the method on any `DataFrame` that is accessed by
`equistress`:

>>> df.equistress.my_equistress()


You can also have additional arguments in the registered method:

.. code-block:: python

    from pylife import equistress

    @pl.signal_register_method(equistress.StressTensorEquistress, 'my_equistress_with_arg')
    def my_equistress_method_with_arg(df, additional_arg)
	# your code here
	return ...


>>> df.equistress.my_equistress_with_arg(my_additional_arg)

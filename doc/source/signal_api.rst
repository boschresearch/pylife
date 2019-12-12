
The pyLife Signal API
=====================

The signal api is the higher level API of pyLife. It is the API that you
probably should be using. Most of the domain specific functions are also
available as pure numpy functions, if your application for some reason makes
the use of pandas harder. However, in such a case we highly recommend you to
take a closer look at pandas and consider to adapt your application to the
pandas way of doing things.


The basic concept
-----------------

The basic idea is to have all the data in a signal like data structure, that
can be piped through the individual calculation process steps. Each calculation
process step results in a new signal, that then can be handed over to the next
process step.

Signals can be for example

* stress tensors like from an FEM-solver

* load collectives

* cyclic load information over a component's geometry

* ...

Signals are usually kept in a :class:`pandas.DataFrame`. In cases, the
signal is only a small record of parameters, it is kept in a
:class:`pandas.Series`.

To implement signal processors we implement accessor classes using
:func:`pandas.api.extensions.register_dataframe_accessor()` resp.
:func:`pandas.api.extensions.register_series_accessor()`.

For predefined accessors that can be subclassed from see

* :mod:`meshsignal`
* :mod:`stresssignal`


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
:mod:`meshsignal`) needs to be imported explicitly.

>>> import pandas as pd
>>> import pylife.utils.meshsignal

Create a DataFrame and have it validated if it is a valid plain mesh, i.e. has
the columns `x` and `y`.

>>> df = pd.DataFrame({'x': [1.0], 'y': [1.0]})
>>> df.plain_mesh
<pylife.utils.meshsignal.PlainMeshAccessor object at 0x7f66da8d4d10>

Now create a DataFrame which is not a valid plain mesh and try to have it
validated:

>>> df = pd.DataFrame({'x': [1.0], 'a': [1.0]})
>>> df.plain_mesh
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jmu3si/Devel/pylife/_venv/lib/python3.7/site-packages/pandas/core/generic.py", line 5175, in __getattr__
    return object.__getattribute__(self, name)
  File "/home/jmu3si/Devel/pylife/_venv/lib/python3.7/site-packages/pandas/core/accessor.py", line 175, in __get__
    accessor_obj = self._accessor(obj)
  File "/home/jmu3si/Devel/pylife/pylife/utils/meshsignal.py", line 79, in __init__
    self._validate(pandas_obj)
  File "/home/jmu3si/Devel/pylife/pylife/utils/meshsignal.py", line 84, in _validate
    signal.fail_if_key_missing(obj, self._coord_keys)
  File "/home/jmu3si/Devel/pylife/pylife/core/signal.py", line 88, in fail_if_key_missing
    raise AttributeError(msg % (', '.join(keys_to_check), ', '.join(missing_keys)))
AttributeError: PlainMeshAccessor must have the items x, y. Missing y.


Example for accessing a property
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get the coordinates of a 2D plain mesh

>>> import pandas as pd
>>> import pylife.utils.meshsignal
>>> df = pd.DataFrame({'x': [1.0], 'y': [1.0], 'foo': [42.0], 'bar': [23.0]})
>>> df.plain_mesh.coordinates
     x    y
0  1.0  1.0

Now a 3D mesh

>>> df = pd.DataFrame({'x': [1.0], 'y': [1.0], 'z': [1.0], 'foo': [42.0], 'bar': [23.0]})
>>> df.plain_mesh.coordinates
     x    y    z
0  1.0  1.0  1.0


Defining your own signal accessors
----------------------------------

If you want to write a processor for signals you need to put the processing
functionality in an accessor class that is derived from the signal accessor
base class like for example :class:`~.meshsignal.MeshAccessor`. This class you
register as a pandas DataFrame accessor using a decorator

.. code-block:: python

    import pandas as pd
    import pylife.meshsignal

    @pd.api.extensions.register_dataframe_accessor('my_mesh_processor')
    class MyMeshAccessor(meshsignal.MeshAccessor):
        def do_something(self):
	    # ... your code here
	    # the DataFrame is accessible by self._obj
	    # usually you would calculate a DataFrame df to return it.
	    df = ...
	    # you might want copy the index of self._obj to the returned
	    # DataFrame.
	    return df.set_index(self._obj.index)

As `MyMeshAccessor` is derived from :class:`~.meshsignal.MeshAccessor` the
validation of `MeshAccessor` is performed. So in the method `do_something()`
you can rely on that `self._obj` is a valid mesh DataFrame.

You then can use the class in the following way when the module is imported.

>>> df = pd.read_hdf('demos/plate_with_hole.h5', '/node_data')
>>> result = df.my_mesh_processor.do_something()


Performing additional validation
````````````````````````````````

Sometimes your signal accessor needs to perform an additional validation on the
accessed signal. For example you might need a mesh that needs to be
3D. Therefore you can reimplement `_validate()` to perform the additional
validation. Make sure to call `_validate()` of the accessor class you are
deriving from like in the following example.

.. code-block:: python

    import pandas as pd
    import pylife.meshsignal
    from pylife import signal

    @pd.api.extensions.register_dataframe_accessor('my_only_for_3D_mesh_processor')
    class MyOnlyFor3DMeshAccessor(meshsignal.PlainMeshAccessor):
	def _validate(self, obj):
	    super(MyOnlyFor3DMeshAccessor, obj) # call PlainMeshAccessor._validate()
	    signal.fail_if_key_missing(['z'])



Defining your own signals
-------------------------

The same way the predefined pyLife signals are defined you can define your own
signals. Let's say, for example, that in your signal there needs to be the
columns `alpha`, `beta`, `gamma` all of which need to be positive.

You would put the signal class into a module file `my_signal_mod.py`

.. code-block:: python

    import pandas as pd
    from pylife import signal

    @pd.api.extensions.register_dataframe_accessor('my_signal')
    class MySignalAccessor(signal.PylifeSignal):
        def _validate(self, obj):
            signal.fail_if_key_missing(obj, ['alpha', 'beta', 'gamma'])
            for k in ['alpha', 'beta', 'gamma']:
                if (obj[k] < 0).any():
                    raise ValueError("All values of %s need to be positive. "
                                     "At least one is less than 0" % k)

	def some_method(self):
	    # some code

You can then validate signals and/or call `some_method()`.

Validation fails because of missing `gamma` column.

>>> import my_signal_mod
>>> df = pd.DataFrame({'alpha': [1.0, 2.0], 'beta': [1.0, -1.0]})
>>> df.my_signal
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jmu3si/Devel/pylife/_venv/lib/python3.7/site-packages/pandas/core/generic.py", line 5175, in __getattr__
    return object.__getattribute__(self, name)
  File "/home/jmu3si/Devel/pylife/_venv/lib/python3.7/site-packages/pandas/core/accessor.py", line 175, in __get__
    accessor_obj = self._accessor(obj)
  File "/home/jmu3si/Devel/pylife/signal_test.py", line 7, in __init__
    self._validate(pandas_obj)
  File "/home/jmu3si/Devel/pylife/signal_test.py", line 11, in _validate
    signal.fail_if_key_missing(obj, ['alpha', 'beta', 'gamma'])
  File "/home/jmu3si/Devel/pylife/pylife/core/signal.py", line 88, in fail_if_key_missing
    raise AttributeError(msg % (', '.join(keys_to_check), ', '.join(missing_keys)))
AttributeError: MySignalAccessor must have the items alpha, beta, gamma. Missing gamma.

Validation fail because one `beta` is negative.

>>> df = pd.DataFrame({'alpha': [1.0, 2.0], 'beta': [1.0, -1.0], 'gamma': [1.0, 2.0]})
>>> df.my_signal
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jmu3si/Devel/pylife/_venv/lib/python3.7/site-packages/pandas/core/accessor.py", line 175, in __get__
    accessor_obj = self._accessor(obj)
  File "/home/jmu3si/Devel/pylife/signal_test.py", line 7, in __init__
    self._validate(pandas_obj)
  File "/home/jmu3si/Devel/pylife/signal_test.py", line 15, in _validate
    "At least one is less than 0" % k)
ValueError: All values of beta need to be positive. At least one is less than 0

Validation success.

>>> df = pd.DataFrame({'alpha': [1.0, 2.0], 'beta': [1.0, 0.0], 'gamma': [1.0, 2.0]})
>>> df.my_signal
<signal_test.MySignalAccessor object at 0x7fb3268c4f50>

Call `some_method()`

>>> df = pd.DataFrame({'alpha': [1.0, 2.0], 'beta': [1.0, 0.0], 'gamma': [1.0, 2.0]})
>>> df.my_signal.some_method()


Additional attributes in your own signals
`````````````````````````````````````````

If your accessor class needs to have attributes other than the accessed object
itself you can define default values in the `__init__()` of your accessor and
set these attributes with setter methods.

.. code-block:: python

    import pandas as pd
    from pylife import signal

    @pd.api.extensions.register_dataframe_accessor('my_signal')
    class MySignalAccessor(signal.PylifeSignal):
	def __init__(self, pandas_obj):
	    super(MySignalAccessor, self).__init__(pandas_obj)
	    self._my_attribute = 'the default value'

        def set_my_attribute(self, my_attribute):
	    self._my_attribute = my_attribute
	    return self

	def do_something(self, some_parameter):
	    # ... use some_parameter, self._my_attribute and self._obj


>>> df.my_signal.set_my_attribute('foo').do_something(2342)



Registering a method to an existing accessor class
--------------------------------------------------

One drawback of the accessor class API is that you cannot extend accessors by
deriving from them. For example if you need a custom equivalent stress function
you cannot add it by deriving from :class:`~.equistress.StressTensorEquistressAccessor`,
and register it by the same accessor `equistress`.

The solution for that is :func:`register_method()` that lets you monkey patch a
new method to any class deriving from :class:`~.pylife.core.signal.PylifeSignal`.

.. code-block:: python

    from pylife import equistress

    @pl.signal_register_method(equistress.StressTensorEquistressAccessor, 'my_equistress')
    def my_equistress_method(df)
	# your code here
	return ...

Then you can call the method on any `DataFrame` that is accessed by
`equistress`:

>>> df.equistress.my_equistress()


You can also have additional arguments in the registered method:

.. code-block:: python

    from pylife import equistress

    @pl.signal_register_method(equistress.StressTensorEquistressAccessor, 'my_equistress_with_arg')
    def my_equistress_method_with_arg(df, additional_arg)
	# your code here
	return ...


>>> df.equistress.my_equistress_with_arg(my_additional_arg)

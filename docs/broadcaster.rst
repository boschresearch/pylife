The Signal Broadcaster
======================

Motivation
----------

pyLife tries to provide a flexible API for its functionality with respect to
sizes of the datasets involved.  No matter if you want to perform some
calculation on just a single value or on a whole FEM-mesh.  No matter if you
want to calculate the damage that a certain load amplitude does on a certain
material, or if you have a FEM-mesh with different materials associated with
to element and every node has its own rainflow matrix.


Example
~~~~~~~

Take for example the function
:func:`~pylife.materiallaws.WoehlerCurve.cycles`.  Imagine you have a
single Wöhler curve dataset like

.. jupyter-execute::

    import pandas as pd
    from pylife.materiallaws import WoehlerCurve

    woehler_curve_data = pd.Series({
        'k_1': 7.0,
        'ND': 2e5,
        'SD': 320.0,
        'TN': 2.3,
        'TS': 1.25
    })

    woehler_curve_data


Now you can calculate the cycles along the Basquin equation for a single load
value:


.. jupyter-execute::

    woehler_curve_data.woehler.cycles(load=350.)


Now let's say, you have different loads for each `element_id` if your FEM-mesh:

.. jupyter-execute::

    amplitude = pd.Series([320., 340., 330., 320.], index=pd.Index([1, 2, 3, 4], name='element_id'))
    amplitude


:func:`~pylife.materiallaws.WoehlerCurve.cycles` now gives you a result
for every `element_id`.

.. jupyter-execute::

    woehler_curve_data.woehler.cycles(load=amplitude)


In the next step, even the Wöhler curve data is different for every element,
like for example for a hardness gradient in your component:

.. jupyter-execute::

    woehler_curve_data = pd.DataFrame({
        'k_1': 7.0,
        'ND': 2e5,
        'SD': [370., 320., 280, 280],
        'TN': 2.3,
        'TS': 1.25
    }, index=pd.Index([1, 2, 3, 4], name='element_id'))

    woehler_curve_data

In this case the broadcaster determines from the identical index name
`element_id` that the two structures can be aligned, so every element is
associated with its load and with its Wöhler curve:

.. jupyter-execute::

   woehler_curve_data.woehler.cycles(load=amplitude)

In another case we assume that you have a Wöhler curve associated to every
element, and the loads are constant throughout the component but different for
different load scenarios.

.. jupyter-execute::

    amplitude_scenarios = pd.Series([320., 340., 330., 320.], index=pd.Index([1, 2, 3, 4], name='scenario'))
    amplitude_scenarios

In this case the broadcaster makes a cross product of load `scenario` and
`element_id`, i.e. for every `element_id` for every load `scenario` the
allowable cycles are calculated:

.. jupyter-execute::

    woehler_curve_data.woehler.cycles(load=amplitude_scenarios)

As is very uncommon that the load is constant all over the component like in
the previous example we now consider an even more complex one.  Let's say we
have a different load scenarios, which give us for every `element_id` multiple
load scenarios:

.. jupyter-execute::

    amplitude_scenarios = pd.Series(
        [320., 340., 330., 320, 220., 240., 230., 220, 420., 440., 430., 420],
        index=pd.MultiIndex.from_tuples([
            (1, 1), (1, 2), (1, 3), (1, 4),
            (2, 1), (2, 2), (2, 3), (2, 4),
            (3, 1), (3, 2), (3, 3), (3, 4)
        ], names=['scenario', 'element_id']))
    amplitude_scenarios

Now the broadcaster still aligns the `element_id`:

.. jupyter-execute::

    woehler_curve_data.woehler.cycles(load=amplitude_scenarios)

Note that in the above examples the call was always identical

.. code-block:: python

    woehler_curve_data.woehler.cycles(load=...)

That means that when you write a module for a certain functionality **you don't
need to know if your code later on receives a single value parameter or a whole
FEM-mesh**.  Your code will take both and handle them.


Usage
-----

As you might have seen, we did not call the :class:`pylife.Broadcaster` in the
above code snippets directly.  And that's the way it's meant to be.  When you
are on the level that you simply want to use pyLife's functionality to perform
calculations, you should not be required to think about how to broadcast your
datasets to one another.  It should simply happen automatically.  In our
example the the calls to the :class:`pylife.Broadcaster` are done inside
:func:`~pylife.materiallaws.WoehlerCurve.cycles`.

You do need to deal with the :class:`pylife.Broadcaster` when you implement new
calculation methods.  Let's go through an example.

.. todo::

   **Sorry**, this is still to be written.

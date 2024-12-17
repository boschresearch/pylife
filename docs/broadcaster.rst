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

Lets assume we have a collective of seasonal flood events on the river Vitava
in Prague.  This is an oversimplified damage model, which assumes that we
multiply the water level of a flood event with a sensitivity value of a bridge
to calculate the damage that the flood events causes to the bridge.

.. jupyter-execute::

    from pylife import Broadcaster

    flood_events = pd.Series(
        [10., 13., 9., 5.],
        name="water_level",
        index=pd.Index(
            ["spring", "summer", "autumn", "winter"],
            name="flood_event"
        )
    )
    flood_events

Lets assume some sensitivity value for the bridges of Prague.

.. jupyter-execute::

    sensitivities = pd.Series([2.3, 0.7, 2.7, 6.4, 3.9, 0.8],
        name="sensitivity",
        index=pd.Index(
            [
                "Palackého most",
                "Jiraskův most",
                "Most Legií",
                "Karlův most",
                "Mánesův most",
                "Centrův most"
            ],
            name="bridge"
        )
    )
    sensitivities


Now we want to multiply the water levels with the sensitivity value in order
to get a damage value:

.. jupyter-execute::

    damage = flood_events * sensitivities
    damage

As we can see, this multiplication failed, as the indices of our two series do
not match.  First we need to broadcast the two indices to a mapped hierarchical
index.

.. jupyter-execute::

    sens_mapped, flood_mapped = Broadcaster(flood_events).broadcast(sensitivities)

Now we have a mapped flood values

.. jupyter-execute::

    flood_mapped

and the mapped sensitivity values

.. jupyter-execute::

    sens_mapped

These mapped series we can multiply.

.. jupyter-execute::

    damage = flood_mapped * sens_mapped
    damage

Now we can see for every bridge for every flood event the expected damage to
every bridge. We can now reduce this map to get the total damage of every
bridge during all flood events:

.. jupyter-execute::

    damage.groupby("bridge").sum()


Now let's assume that we have for each bridge some kind of protection measure
that reduces the damage.

.. jupyter-execute::

    protection = pd.Series(
        [10.0, 15.0, 12.0, 25.0, 13.0, 17.0],
        name="dwell_time",
        index=pd.Index(
            [
                "Palackého most",
                "Jiraskův most",
                "Most Legií",
                "Karlův most",
                "Mánesův most",
                "Centrův most"
            ],
            name="bridge"
        )
    )
    protection

We also need divide the damage value by the protection value. Therefore we need
to broadcast the protection values to the damage values

.. jupyter-execute::

    protection_mapped, _ = Broadcaster(damage).broadcast(protection)
    protection_mapped

As you can see, the broadcaster recognized the common index name "bridge" and
did not spread it again.

Now we can easily multiply the mapped protection values to the damage.

.. jupyter-execute::

    damage_with_protection = damage / protection


And we can again easily calculate the damage for a certain bridge

.. jupyter-execute::

    damage_with_protection.groupby("bridge").sum()

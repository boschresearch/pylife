The concept of stress and strength
==================================

The fundamental principle of component lifetime and reliability design is to
calculate the superposition of *stress* and *strength*.  Sometimes you would also
say *load* and strength.  The basic assumption is that as soon as the stress
exceeds the strength the component fails.  Usually stress and strength are
statistically distributed.  In this tutorial we learn how to work with material
data and material laws to model the strength and then calculate the damage
using a given load.


Material laws
-------------

The material load that is used to model the strength for component fatigue is
the :class:`~pylife.materiallaws.WoehlerCurve`.


First we need to import :mod:`pandas` and the
:class:`~pylife.materiallaws.WoehlerCurve` class.

.. jupyter-execute::

   import pandas as pd
   from pylife.materiallaws import WoehlerCurve


The material data for a Wöhler curve is usually stored in a
:class:`pandas.Series`.  In the simplest form like this:

.. jupyter-execute::

   woehler_curve_data = pd.Series({
       'SD': 300.0,
       'ND': 1.5e6,
       'k_1': 6.2
   })


Using the :class:`~pylife.materiallaws.WoehlerCurve` class can do operations on
the data.  To instantiate the class we use the accessor string ``woehler``.
Then we can calculate the cycle number for a given load.

.. jupyter-execute::

   woehler_curve_data.woehler.cycles(350.0)


This basically means that a material of the given Wöhler curve will fail after
about 577k cycles when charged with a load of 350.  Note that we don't use any
units here.  They just have to be consistent.


Damage sums
-----------

Usually we don't have a single load amplitude but a collective.  We can
describe a collective using a python object that has an ``amplitude`` and a
``cycle`` attribute.  We can do that for example with a simple
:class:`pandas.DataFrame`:

.. jupyter-execute::

    load_collective = pd.DataFrame({
       'cycles': [2e5, 3e4, 5e3, 2e2, 7e1],
       'amplitude': [374.0, 355.0, 340.0, 320.0, 290.0]
    })

Using the :func:`~pylife.strength.fatigue.damage` function we can calculate the
damage of each block of the load collective.  Therefore we use the ``fatigue``
accessor to operate on the Wöhler data.

.. jupyter-execute::

   from pylife.strength import fatigue

   woehler_curve_data.fatigue.damage(load_collective)


Now we know the damage contribution of each block of the load collective.  Of
course we can also easily calculate the damage sum by just summing up:

.. jupyter-execute::

    woehler_curve_data.fatigue.damage(load_collective).sum()


Oftentimes we want to map a load collective to a whole FEM mesh to map a load
collective to every FEM node.  For those kinds of mappings pyLife provides the
:class:`~pylife.Broadcaster` facility.

In order to operate properly the ``Broadcaster`` needs to know the meanings of
the rows of a ``pandas.Series`` or a ``pandas.DataFrame``.  For that it uses
the index names.  Therefore we have to set the index names appropriately.

.. jupyter-execute::

    load_collective.index.name = 'load_block'


Then we setup simple node stress distribution and broadcast the load collective
to it.


.. jupyter-execute::

    node_stress = pd.Series(
        [1.0, 0.8, 1.3],
        index=pd.Index([1, 2, 3], name='node_id')
    )

    from pylife import Broadcaster
    load_collective, node_stress = Broadcaster(node_stress).broadcast(load_collective)


As you can see, the ``Broadcaster`` returns two objects.  The first is the
object that has been broadcasted, in our case the load collective:

.. jupyter-execute::

    load_collective


The second is the object that has been broadcasted to, in our case the node
stress distribution.

.. jupyter-execute::

    node_stress


As you can see, both have the same index, which is a cross product of the
indices of the two initial objects.  Now we can easily scale the load
collective to the node stress distribution.

.. jupyter-execute::

    load_collective['amplitude'] *= node_stress
    load_collective


Now we have for each ``load_block`` for each ``node_id`` the corresponding
amplitudes and cycle numbers.  Again we can use the ``damage`` function to
calculate the damage contribution of each load block on each node.

.. jupyter-execute::

    damage_contributions = woehler_curve_data.fatigue.damage(load_collective)
    damage_contributions


In order to calculate the damage sum for each node, we have to group the damage
contributions by the node and sum them up:

.. jupyter-execute::

    damage_contributions.groupby('node_id').sum()


As you can see the damage sum for node 3 is higher than 1, which means that the
stress exceeds the strength.  So we would expect failure at node 3.

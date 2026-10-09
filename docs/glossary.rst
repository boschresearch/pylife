********
Glossary
********

Fatigue assessment comes with a vocabulary of its own, and pyLife uses the
symbols of that vocabulary as key names in its pandas objects.  This page
explains what each of them means.

The same symbols are mandatory as variable names in pyLife's source code, see
:doc:`variable_names` for the naming rules that apply to contributors.

.. glossary::
   :sorted:

   amplitude
      Half of the difference between the maximum and the minimum load of a
      load cycle, i.e. ``amplitude = range / 2``.  Together with the
      :term:`meanstress` it fully describes a load cycle.

   collective
      See :term:`load collective`.

   damage
      The fraction of the fatigue life that a load consumes.  A damage of
      ``0.0`` means untouched, a damage of ``1.0`` means that failure is
      expected.  Damage contributions of the individual load levels are summed
      up according to :term:`Miner's rule`.  See :mod:`pylife.strength.fatigue`.

   damage parameter
      A scalar quantity that condenses a load cycle — including plasticity and
      mean stress effects — into a single number that can be looked up in a
      damage parameter Wöhler curve.  The nonlinear FKM guideline uses
      ``P_RAM`` and ``P_RAJ``.  See :mod:`pylife.strength.damage_parameter`.

   endurance limit
      The load level below which a material is assumed to endure an unlimited
      number of load cycles.  In pyLife it is given by the pair :term:`SD` and
      :term:`ND`.

   FKM guideline
      A German engineering guideline for the analytical strength assessment of
      mechanical components.  pyLife implements both the linear variant
      (:mod:`pylife.strength.fkm_linear`) and the nonlinear variant, i.e. the
      local strain concept (:mod:`pylife.strength.fkm_nonlinear`).

   hotspot
      A connected region of an FE mesh in which the stress exceeds a given
      threshold.  Hotspots identify the locations that need to be assessed.
      See :class:`pylife.mesh.HotSpot`.

   k
   k_1
   k_2
      The slope of the :term:`Wöhler curve` in the double logarithmic
      load-cycle diagram.  ``k_1`` is the slope above the :term:`endurance
      limit`, ``k_2`` the slope below it.  If ``k_2`` is missing it is assumed
      to be infinite, i.e. perfect endurance below the endurance limit.

   load collective
      A description of how often which load level occurs, usually the result
      of a :term:`rainflow counting`.  pyLife represents it either as a list
      of individual cycles or as a histogram indexed by a
      :class:`pandas.IntervalIndex`.  See :mod:`pylife.stress.collective`.

   meanstress
      The average of the maximum and the minimum load of a load cycle.  Since
      material strength depends on it, collectives are transformed to a common
      mean stress before they are assessed.  See
      :mod:`pylife.strength.meanstress`.

   Miner's rule
      The linear damage accumulation hypothesis: every load cycle consumes a
      fixed fraction of the fatigue life, and the fractions simply add up.
      pyLife implements the elementary, the modified and the consistent
      variant, see :mod:`pylife.strength.miner`.

   ND
      The cycle number of the :term:`endurance limit`, i.e. the abscissa of
      the knee point of the :term:`Wöhler curve`.  ``ND_xx`` denotes the value
      for a failure probability of ``xx`` percent.

   PylifeSignal
      The base class of all pyLife accessors.  It validates that a pandas
      object carries the keys a calculation needs and provides the fatigue
      specific methods.  See :doc:`signal_api`.

   R
      The ratio of the minimum to the maximum load of a load cycle,
      ``R = min / max``.  ``R = -1`` is a fully alternating load,
      ``R = 0`` a pulsating one.

   rainflow counting
      The standard procedure that decomposes an irregular load-time history
      into closed hysteresis loops, which then form a :term:`load collective`.
      See :mod:`pylife.stress.rainflow`.

   range
      The difference between the maximum and the minimum load of a load cycle,
      i.e. twice the :term:`amplitude`.

   SD
      The load level of the :term:`endurance limit`, i.e. the ordinate of the
      knee point of the :term:`Wöhler curve`.  ``SD_xx`` denotes the value for
      a failure probability of ``xx`` percent.

   signal
      A pandas object that carries a well defined set of keys and therefore
      can be processed by a pyLife accessor.  See :doc:`signal_api`.

   TN
      The scatter of the :term:`Wöhler curve` in cycle direction, defined as
      ``ND_10 / ND_90``.  A value of ``1.0`` means no scatter.

   TS
      The scatter of the :term:`Wöhler curve` in load direction, defined as
      ``SD_10 / SD_90``.  A value of ``1.0`` means no scatter.

   VMAP
      A vendor neutral standard for the exchange of FE simulation data.
      pyLife can read and write VMAP files, see :mod:`pylife.vmap`.

   Wöhler curve
      The curve that states how many load cycles a material survives at a
      given load amplitude.  Also known as SN-curve.  In pyLife it is a pandas
      object with the keys :term:`k_1`, :term:`ND` and :term:`SD`, optionally
      :term:`k_2`, :term:`TN` and :term:`TS`.  See
      :class:`pylife.materiallaws.WoehlerCurve`.

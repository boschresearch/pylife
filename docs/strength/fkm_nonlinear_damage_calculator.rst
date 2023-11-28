The classes for damage calculation
##################################

The following classes perform lifetime assessment according to the guideline FKM nonlinear. The objects are given a load collective and a damage parameter woehler curve. 
They calculate the damage parameters values for the collective and use the woehler curve to calculate a lifetime or infinite strength.
 
* ``DamageCalculatorPRAM``: uses the `P_RAM` damage parameter 
* ``DamageCalculatorPRAJ``: uses the `P_RAJ` damage parameter 
* ``DamageCalculatorPRAJMinerElementary`` uses the `P_RAJ` damage parameter but omits the horizontal finite strength line in the Woehler curve, corresponding to the *Miner Elementary* method.

.. autoclass:: pylife.strength.fkm_nonlinear.damage_calculator.DamageCalculatorPRAM
	:undoc-members:
	:members:

.. autoclass:: pylife.strength.fkm_nonlinear.damage_calculator.DamageCalculatorPRAJ
	:undoc-members:
	:members:

.. autoclass:: pylife.strength.fkm_nonlinear.damage_calculator_praj_miner.DamageCalculatorPRAJMinerElementary
	:undoc-members:
	:members:

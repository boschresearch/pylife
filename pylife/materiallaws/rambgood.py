import numpy as np


class RambergOsgood:

    def __init__(self, E, K, n):
        self._E = E
        self._K = K
        self._n = n

    def strain(self, stress):
        self._fail_if_negative(stress)
        return stress/self._E + self.plastic_strain(stress)

    def plastic_strain(self, stress):
        return np.power(stress/self._K, 1./self._n)

    def delta_strain(self, delta_stress):
        self._fail_if_negative(delta_stress)
        return delta_stress/self._E + 2.*np.power(delta_stress/(2.*self._K), 1./self._n)

    def lower_hysteresis(self, stress, max_stress):
        return self.strain(max_stress) - self.delta_strain(max_stress-stress)

    def _fail_if_negative(self, val):
        if (np.asarray(val) < 0).any():
            raise ValueError("Stress value in Ramberg-Osgood equation must not be negative.")

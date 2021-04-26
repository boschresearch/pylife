import numpy as np


class RambergOsgood:
    '''Simple implementation of the Ramberg-Osgood relation

    Parameters
    ----------

    E : float
        Young's Modulus
    K : float
        The strength coefficient
    n : float
        The strain hardening coefficient

    Notes
    -----
    The equation implemented is the one that `Wikipedia
    <https://en.wikipedia.org/wiki/Ramberg%E2%80%93Osgood_relationship#Alternative_Formulations>`_refers
    to as "Alternative Formulation". The parameters `n` and `k` in this are
    formulation are the Hollomon parameters.
    '''

    def __init__(self, E, K, n):
        self._E = E
        self._K = K
        self._n = n

    def strain(self, stress):
        '''Calculate the elastic plastic strain for a given stress

        Parameters
        ----------
        stress : array-like float
            The stress

        Returns
        -------
        strain : array-like float
            The resulting strain
        '''
        stress = np.asarray(stress)
        return stress/self._E + self.plastic_strain(stress)

    def plastic_strain(self, stress):
        '''Calculate the plastic strain for a given stress

        Parameters
        ----------
        stress : array-like float
            The stress

        Returns
        -------
        strain : array-like float
            The resulting plastic strain
        '''
        self._fail_if_negative(stress)
        return np.power(stress/self._K, 1./self._n)

    def delta_strain(self, delta_stress):
        '''Calculate the cyclic Masing strain span for a given stress span

        Parameters
        ----------
        delta_stress : array-like float
            The stress span

        Returns
        -------
        delta_strain : array-like float
            The corresponding stress span

        Notes
        -----
        A Masing like behavior is assumed for the material as described in
        `Kerbgrundkonzept<https://de.wikipedia.org/wiki/Kerbgrundkonzept#Masing-Verhalten_und_Werkstoffged%C3%A4chtnis>`
        '''
        self._fail_if_negative(delta_stress)
        return delta_stress/self._E + 2.*np.power(delta_stress/(2.*self._K), 1./self._n)

    def lower_hysteresis(self, stress, max_stress):
        '''Calculate the lower (relaxation to compression) hysteresis starting from a given maximum stress

        Parameters
        ----------
        stress : array-like float
            The stress (must be below the maximum stress)
        max_stress : float
            The maximum stress of the hysteresis look

        Returns
        -------
        lower_hysteresis : array-like float
            The lower hysteresis branch from `max_stress` all the way to `stress`
        '''
        return self.strain(max_stress) - self.delta_strain(max_stress-stress)

    def _fail_if_negative(self, val):
        if (np.asarray(val) < 0).any():
            raise ValueError("Stress value in Ramberg-Osgood equation must not be negative.")

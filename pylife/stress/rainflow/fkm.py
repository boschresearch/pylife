import cython
import numpy as np

from .general import AbstractRainflowDetector


class FKMRainflowDetector(AbstractRainflowDetector):
    '''Implements a rainflow counter as described in FKM non linear

    See the `here <subsection_FKM_>`_ in the demo for an example.

    The algorithm has been published by Clormann & Seeger 1985 and has
    been cited havily since.

    .. _subsection_FKM: ../demos/rainflow.ipynb#Algorithm-recommended-by-FKM-non-linear
    '''
    def __init__(self, recorder):
        super(FKMRainflowDetector, self).__init__(recorder)
        self._ir = 1
        self._residuals = []
        self._max_turn = 0.0

    @cython.locals(
        turns=cython.double[:],
        iz=cython.int, ir=cython.int,
        last0=cython.double, last1=cython.double,
        loop_assumed=cython.int,
        max_turn=cython.double)
    def process(self, samples):
        ''' Processes a sample chunk

        Parameters
        ----------
        samples : array_like, shape (N, )
            The samples to be processed

        Returns
        -------
        self : RainflowCounterFKM
            The ``self`` object so that processing can be chained

        Example
        -------
        >>> rfc = RainflowCounterFKM().process(samples)
        >>> rfc.get_rainflow_matrix_frame(128)
        '''
        ir = self._ir
        max_turn = self._max_turn
        turns = self._get_new_turns(samples)

        for current in turns:
            loop_assumed = True
            while loop_assumed:
                iz = len(self._residuals)
                loop_assumed = False
                if iz > ir:
                    last0 = self._residuals[-1]
                    last1 = self._residuals[-2]
                    if np.abs(current-last0) >= np.abs(last0-last1):
                        self._recorder.record(0, 0, last1, last0)
                        self._residuals.pop()
                        self._residuals.pop()
                        if np.abs(last0) < max_turn and np.abs(last1) < max_turn:
                            loop_assumed = True
                elif iz == ir:
                    if np.abs(current) > max_turn:
                        ir += 1
            max_turn = max(np.abs(current), max_turn)
            self._residuals.append(current)

            self._ir = ir
            self._max_turn = max_turn

        return self

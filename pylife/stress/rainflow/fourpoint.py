# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 09:02:14 2021

@author: PRV3SI
"""


import cython
import numpy as np

from .general import AbstractDetector


class FourPointDetector(AbstractDetector):
    ''' Implements 4 point rainflow counting algorithm

    We take four turning points into account to detect closed hysteresis loops.

    Consider four consecutive peak/valley points say, A, B, C, and D  If B and C are 
    contained within A and B, then a cycle is counted from B to C; otherwise no cycle is 
    counted. 

    i.e, If X ≥ Y AND Z ≥ Y then a cycle exsist FROM = B and TO = C
    where, ranges X = |D–C|, Y = |C–B|, and Z = |B–A|


        Load -----------------------------
        |        x B               F x
        --------/-\-----------------/-----
        |      /   \   x D         /
        ------/-----\-/-\---------/-------
        |    /     C x   \       /
        --\-/-------------\-----/---------
        |  x A             \   /
        --------------------\-/-----------
        |                    x E
        ----------------------------------
        |              Time

    So, if a cycle exsist from B to C then delete these peaks from the turns array 
    and perform next iteration by joining A&D else if no cylce exsists, then B would
    be the next strarting point. 
    '''

    def __init__(self, recorder):
        """Instantiate a FourPointDetector.

        Parameters
        ----------
        recorder : subclass of :class:`.AbstractRecorder`
            The recorder that the detector will report to.
        """
        super().__init__(recorder)
    def process(self, samples):
        '''
        Processes a sample chunk

        Parameters
        ----------
        samples : array_like, shape (N, )
            The samples to be processed

        Returns
        -------
        self : RainflowCounterThreePoint
            The ``self`` object so that processing can be chained

        Example
        -------
        >>> rfc = RainflowCounterFourPoint().process(samples)
        >>> rfc.get_rainflow_matrix_frame(128)
        '''

        if len(self._residuals) == 0:
            residuals = samples[:1]
            residual_index = [0, 1]
        else:
            residuals = self._residuals[:-1]
            residual_index = [*range(len(residuals))]

        turns_index, turns_values = self._new_turns(samples)

        turns_np = np.concatenate((residuals,turns_values, samples[-1:]))
        turns_index = np.concatenate((self._residual_index, turns_index))
        
        turns = turns_np
        i = 0
        while i+3 < len(turns):
            ds = np.abs(np.diff(turns)[i:i+3])
            if ds.min() == ds[1]:
                self._recorder.record_values(turns[i+1],turns[i+2])
                self._recorder.record_index(turns_index[i+1], turns_index[i+2])
                turns = np.delete(turns, i+1, 0)
                turns = np.delete(turns, i+1, 0)
                turns_index = np.delete(turns_index, i+1, 0)
                turns_index = np.delete(turns_index, i+1, 0)
                i = max(0, i-4)
            else:
                i += 1
        self._residuals = turns
        self._residual_index = turns_index[residual_index[:-1]]
        self._recorder.report_chunk(len(samples))

        return self
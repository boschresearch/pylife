
import cython
import numpy as np

from .general import AbstractRainflowDetector


class ThreePointDetector(AbstractRainflowDetector):
    '''Implements 3 point rainflow counting algorithm

    See the `here <subsection_TP_>`_ in the demo for an example.

    We take three turning points into account to detect closed hysteresis loops.

    * start: the point where the loop is starting from
    * front: the turning point after the start
    * back: the turning point after the front

    A loop is considered closed if following conditions are met:

    * the load difference between front and back is bigger than or
      equal the one between start and front. In other words: if the
      back goes beyond the starting point. For example (A-B-C) and
      (B-C-D) not closed, whereas (C-D-E) is.

    * the loop init has not been a loop front in a prior closed
      loop. For example F would close the loops (D-E-F) but D is
      already front of the closed loop (C-D-E).

    * the load level of the front has already been covered by a prior
      turning point. Otherwise it is considered part of the front
      residuum.

    When a loop is closed it is possible that the loop back also
    closes unclosed loops of the past by acting as loop back for an
    unclosed start/front pair. For example E closes the loop (C-D-E)
    and then also (A-B-E).

    ::

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

    .. _subsection_TP: ../demos/rainflow.ipynb#Classic-Three-Point-Counting
    '''
    def __init__(self, recorder):
        super().__init__(recorder)

    @cython.locals(
        start=cython.int, front=cython.int, back=cython.int,
        highest_front=cython.int, lowest_front=cython.int,
        start_val=cython.double, front_val=cython.double, back_val=cython.double,
        turns=cython.double[:])
    def process(self, samples):
        ''' Processes a sample chunk

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
        >>> rfc = RainflowCounterThreePoint().process(samples)
        >>> rfc.get_rainflow_matrix_frame(128)
        '''
        if len(self._residuals) == 0:
            residuals = samples[:1]
            residual_indeces = [0, 1]
        else:
            residuals = self._residuals[:-1]
            residual_indeces = [*range(len(residuals))]

        turns_np = np.concatenate((residuals, self._get_new_turns(samples), samples[-1:]))
        turns = turns_np

        highest_front = np.argmax(residuals)
        lowest_front = np.argmin(residuals)

        back = residual_indeces[-1] + 1
        while back < turns.shape[0]:
            if len(residual_indeces) >= 2:
                start = residual_indeces[-2]
                front = residual_indeces[-1]
                start_val, front_val, back_val = turns[start], turns[front], turns[back]

                if front_val > turns[highest_front]:
                    highest_front = front
                if front_val < turns[lowest_front]:
                    lowest_front = front

                if (start >= max(lowest_front, highest_front) and
                    np.abs(back_val - front_val) >= np.abs(front_val - start_val) and
                    front != highest_front and front != lowest_front):
                    self._recorder.record(0, 0, start_val, front_val)
                    residual_indeces.pop()
                    residual_indeces.pop()
                    continue

            residual_indeces.append(back)
            back += 1

        self._residuals = turns_np[residual_indeces]

        return self

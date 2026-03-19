"""

@author: Steve Wolff-Vorbeck wos2rng

Cythonize meanstress conversion

"""

import numpy
import sys

cimport cython
from libc.math cimport fabs


cdef double _compute_kak_factor(double M, double M2, double R):
    """Calculate mean stress sensitivity factor.

    Parameters
    ----------
    M: double
        the mean stress sensitivity between R=-inf and R=0.
    M2: double
        the mean stress sensitivity beyond R=0.
    R: double
        stress ratio

    Returns
    -------
    Kak: double
            mean stress influence factor
    """
    cdef double SmSa

    if R == 1.0:
        return 0.0

    SmSa = (1.0 + R) / (1.0 - R)

    if R > 1.0:
        return 1.0 / (1.0 - M)
    elif R >= -1.7976931348623157e+308 and R <= 0.0:  # -sys.float_info.max
        return 1.0 / (1.0 + M * SmSa)
    else:
        return (1.0 + M2) / ((1.0 + M) * (1.0 + M2 * SmSa))

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef transformed_amplitude_goodman(double[::1] amplitude, double[::1] mean, double M1, double M2, double Kak_factor):
    """
    Compute mean stress conversion according to FKM Goodman for R_goal > -inf

    Parameters
    ----------
    amplitude : 1D numpy array (dtype = double)
       array of amplitude values
    mean : 1D numpy array (dtype = double)
       array of mean values
    M1 : double
        meanstress sensitivity in the range -inf <= R <= 0
    M2 : double
        meanstress sensitivity in the range R>0
    Kak_factor : double
        mean stress influence factor

    Returns
    -------
    amplitude_corr : 1D numpy array (dtype = double)
        array of the values of the corrected amplitude
    """
    cdef Py_ssize_t i, amp_len
    cdef double R, rel_tol, abs_tol, sum, Kak_factor
    rel_tol = 1e-9
    abs_tol = 1e-12

    amp_len = len(amplitude)
    result = numpy.zeros(amp_len, dtype = numpy.float64)
    cdef double[::1] trans_amp = result

    # Compute KAK factor first
    Kak_factor = _compute_kak_factor(M1,M2,R_goal)

    for i in range(amp_len):

        # handle case where R = -inf
        sum = mean[i] + amplitude[i]
        if fabs(sum) <= fabs(rel_tol * amplitude[i]) or fabs(sum) <= abs_tol:
            trans_amp[i] = Kak_factor * (1+M1) / (1+M2) * (amplitude[i] + M2*mean[i])
        else:
            R = (mean[i] - amplitude[i]) / (sum)
            if R > 1:
                trans_amp[i] = Kak_factor * (amplitude[i] - M1 * amplitude[i])
            elif R <= 0:
                trans_amp[i] = Kak_factor * (amplitude[i] + M1*mean[i])
            else:
                trans_amp[i] = Kak_factor * (1+M1) / (1+M2) * (amplitude[i] + M2*mean[i])

    return result

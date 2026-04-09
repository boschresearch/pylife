# cython: language_level=3
# cython: cdivision=True

"""
Cythonized FKM Goodman meanstress transformation

@author: Steve Wolff-Vorbeck wos2rng
"""

import numpy
cimport cython
from libc.math cimport fabs, INFINITY

cdef inline double _compute_kak_factor(double M, double M2, double R) nogil:
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

    # Handle R = -inf case
    if R == -INFINITY:
        return 1.0 / (1.0 - M)

    SmSa = (1.0 + R) / (1.0 - R)

    if R > 1.0:
        return 1.0 / (1.0 - M)
    elif R <= 0.0:
        return 1.0 / (1.0 + M * SmSa)
    elif R > 0.0 and R < 0.5:
        return (1.0 + M2) / ((1.0 + M) * (1.0 + M2 * SmSa))
    else:
        return (1.0 + M2) / ((1.0 + M) * (1.0 + 3.0 * M2))

cpdef transformed_amplitude_goodman(double[::1] amplitude, double[::1] mean, double M, double M2, double R_goal):
    """
    Compute mean stress conversion according to FKM Goodman

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
    cdef double R, rel_tol, abs_tol, sum, kak_factor
    rel_tol = 1e-9
    abs_tol = 1e-12

    amp_len = len(amplitude)
    result = numpy.zeros(amp_len, dtype = numpy.float64)
    cdef double[::1] trans_amp = result

    # Comoute KAK-Factor for R_goal
    kak_factor = _compute_kak_factor(M, M2, R_goal)

    for i in range(amp_len):

        # handle case where R = -inf -> sum == 0.0
        sum = mean[i] + amplitude[i]
        if fabs(sum) <= fabs(rel_tol * amplitude[i]) or fabs(sum) <= abs_tol:
            trans_amp[i] = kak_factor * (amplitude[i] + M*mean[i])
        else:
            R = (mean[i] - amplitude[i]) / (sum)
            if R > 1:
                trans_amp[i] = kak_factor * (amplitude[i] - M * amplitude[i])
            elif R <= 0:
                trans_amp[i] = kak_factor * (amplitude[i] + M*mean[i])
            elif R > 0 and R < 0.5:
                trans_amp[i] = kak_factor * (1 + M) / ( 1 + M2) * (amplitude[i] + M2 * mean[i])
            else:
                trans_amp[i] = kak_factor * ((1+M) * (1 + 3.0 * M2)) / (1+M2) * amplitude[i]

    return result


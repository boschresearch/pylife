# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

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
    M : double
        The mean stress sensitivity between R=-inf and R=0.
    M2 : double
        The mean stress sensitivity beyond R=0.
    R : double
        Stress ratio

    Returns
    -------
    Kak : double
        Mean stress influence factor
    """
    cdef double SmSa

    if R == 1.0:
        return 0.0

    # Handle R = -inf case before computing SmSa to avoid NaN
    if R == -INFINITY:
        return 1.0 / (1.0 - M)

    SmSa = (1.0 + R) / (1.0 - R)

    if R > 1.0:
        return 1.0 / (1.0 - M)
    elif R <= 0.0:
        return 1.0 / (1.0 + M * SmSa)
    elif R < 0.5:
        return (1.0 + M2) / ((1.0 + M) * (1.0 + M2 * SmSa))
    else:  # R >= 0.5 and R < 1.0
        return (1.0 + M2) / ((1.0 + M) * (1.0 + 3.0 * M2))


cpdef transformed_amplitude_goodman(double[::1] amplitude, double[::1] mean, double M, double M2, double R_goal):
    """
    Compute mean stress conversion according to FKM Goodman

    Parameters
    ----------
    amplitude : double[::1]
        Array of amplitude values
    mean : double[::1]
        Array of mean stress values
    M : double
        Meanstress sensitivity in the range -inf <= R <= 0
    M2 : double
        Meanstress sensitivity in the range R > 0
    R_goal : double
        Target R-value for transformation

    Returns
    -------
     amplitude_corr : 1D numpy array (dtype = double)
        array of the values of the corrected amplitude

    Raises
    ------
    ValueError
        If amplitude and mean arrays have different lengths
    """
    cdef Py_ssize_t i, amp_len
    cdef double R, kak_factor
    cdef double sum_stress
    cdef double rel_tol = 1e-9
    cdef double abs_tol = 1e-12

    amp_len = amplitude.shape[0]

    # Check that arrays have the same length
    if amp_len != mean.shape[0]:
        raise ValueError("amplitude and mean arrays must have the same length")

    result = numpy.zeros(amp_len, dtype=numpy.float64)
    cdef double[::1] trans_amp = result

    # Compute KAK-Factor for R_goal
    kak_factor = _compute_kak_factor(M, M2, R_goal)

    for i in range(amp_len):
        # Handle case where R = -inf (i.e., when sum_stress ≈ 0)
        sum_stress = mean[i] + amplitude[i]

        if fabs(sum_stress) <= fabs(rel_tol * amplitude[i]) or fabs(sum_stress) <= abs_tol:
            # R = -inf case
            trans_amp[i] = kak_factor * (amplitude[i] + M * mean[i])
        else:
            R = (mean[i] - amplitude[i]) / sum_stress

            if R > 1.0:
                trans_amp[i] = kak_factor * amplitude[i] * (1.0 - M)
            elif R <= 0.0:
                trans_amp[i] = kak_factor * (amplitude[i] + M * mean[i])
            elif R < 0.5:  # 0.0 < R < 0.5
                trans_amp[i] = kak_factor * (1.0 + M) / (1.0 + M2) * (amplitude[i] + M2 * mean[i])
            else:  # 0.5 <= R <= 1.0
                trans_amp[i] = kak_factor * ((1.0 + M) * (1.0 + 3.0 * M2)) / (1.0 + M2) * amplitude[i]

    return result
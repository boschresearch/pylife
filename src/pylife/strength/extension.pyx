# meanstress_extension.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
@author: Steve Wolff-Vorbeck wos2rng

Cythonize meanstress conversion
"""

import numpy
cimport cython
from libc.math cimport fabs, INFINITY


cdef double _compute_kak_factor(double M, double M2, double R_goal):
    """Calculate mean stress sensitivity factor for target R.

    Parameters
    ----------
    M : double
        Mean stress sensitivity between R=-inf and R=0
    M2 : double
        Mean stress sensitivity beyond R=0
    R_goal : double
        Target stress ratio

    Returns
    -------
    Kak : double
        Mean stress influence factor
    """
    cdef double SmSa

    if R_goal == 1.0:
        return 0.0

    SmSa = (1.0 + R_goal) / (1.0 - R_goal)

    if R_goal > 1.0:
        return 1.0 / (1.0 - M)
    elif R_goal >= -1.7976931348623157e+308 and R_goal <= 0.0:  # -sys.float_info.max
        return 1.0 / (1.0 + M * SmSa)
    else:
        return (1.0 + M2) / ((1.0 + M) * (1.0 + M2 * SmSa))


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef fkm_goodman_transform(double[::1] amplitude, double[::1] mean, double M1, double M2, double R_goal):
    """
    Compute mean stress conversion according to FKM Goodman

    Parameters
    ----------
    amplitude : 1D numpy array (dtype = double)
        Array of amplitude values
    mean : 1D numpy array (dtype = double)
        Array of mean values
    M1 : double
        Meanstress sensitivity in the range -inf <= R <= 0
    M2 : double
        Meanstress sensitivity in the range R > 0
    R_goal : double
        Target R-value for transformation (can be -inf)

    Returns
    -------
    amplitude_corr : 1D numpy array (dtype = double)
        Array of the corrected amplitude values
    """
    cdef Py_ssize_t i, amp_len
    cdef double R, rel_tol, abs_tol, sum_val, Kak_factor

    rel_tol = 1e-9
    abs_tol = 1e-12

    amp_len = len(amplitude)
    result = numpy.zeros(amp_len, dtype=numpy.float64)
    cdef double[::1] trans_amp = result

    # Branch based on R_goal to avoid repeated checks
    if R_goal == -INFINITY:
        # Transform to fully reversed loading (R = -∞)
        for i in range(amp_len):
            sum_val = mean[i] + amplitude[i]

            # Handle case where R_current = -inf (sum ≈ 0)
            if fabs(sum_val) <= fabs(rel_tol * amplitude[i]) or fabs(sum_val) <= abs_tol:
                trans_amp[i] = amplitude[i]
            else:
                R = (mean[i] - amplitude[i]) / sum_val
                if R > 1.0:
                    trans_amp[i] = (amplitude[i] - M1 * amplitude[i]) / (1.0 - M1)
                elif R <= 0.0:
                    trans_amp[i] = (amplitude[i] + M1 * mean[i]) / (1.0 - M1)
                else:
                   trans_amp[i] = (1.0 + M1) / (1.0 + M2) * (amplitude[i] + M2 * mean[i]) / (1 - M1)

    else:
        # General transformation for R_goal ≠ -∞
        # Compute KAK factor once
        Kak_factor = _compute_kak_factor(M1, M2, R_goal)

        for i in range(amp_len):
            sum_val = mean[i] + amplitude[i]

            # Handle case where R_current = -inf (sum ≈ 0)
            if fabs(sum_val) <= fabs(rel_tol * amplitude[i]) or fabs(sum_val) <= abs_tol:
                trans_amp[i] = Kak_factor * (amplitude[i] - M1 * amplitude[i])
            else:
                R = (mean[i] - amplitude[i]) / sum_val

                if R > 1.0:
                    trans_amp[i] = Kak_factor * (amplitude[i] - M1 * amplitude[i])
                elif R <= 0.0:
                    trans_amp[i] = Kak_factor * (amplitude[i] + M1 * mean[i])
                else:  # 0 < R <= 1
                    trans_amp[i] = Kak_factor * (1.0 + M1) / (1.0 + M2) * (amplitude[i] + M2 * mean[i])

    return result

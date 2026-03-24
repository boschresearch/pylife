# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
Cythonized FKM Goodman meanstress transformation

@author: Steve Wolff-Vorbeck wos2rng
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs, INFINITY


cpdef cnp.ndarray[double, ndim=1] fkm_goodman_amplitude_transformation(
    double[::1] amplitude,
    double[::1] R_current,
    double M,
    double M2,
    double R_goal
):
    """
    Transform amplitudes according to FKM Goodman for cycles already at specific R values.

    This is the Cython equivalent of the transformed_amplitude() function in _SegmentTransformer.
    It performs a single transformation step in the progressive transformation process.

    Parameters
    ----------
    amplitude : memoryview of double
        Array of current amplitude values
    R_current : memoryview of double
        Array of current R values for each cycle
    M : double
        Meanstress sensitivity in the range -inf <= R <= 0
    M2 : double
        Meanstress sensitivity in the range R > 0
    R_goal : double
        Target R-value for this transformation step

    Returns
    -------
    transformed_amplitude : numpy array of double
        Array of transformed amplitude values
    """

    cdef:
        Py_ssize_t n = amplitude.shape[0]
        Py_ssize_t i
        double amp, R, mean, M_current
        double Sa_transformed
        double denominator
        double tolerance = 1e-10
        cnp.ndarray[double, ndim=1] out_amplitude
        double[::1] out_amplitude_view

    # Initialize output array
    out_amplitude = np.zeros(n, dtype=np.float64)
    out_amplitude_view = out_amplitude

    # Branch based on R_goal to avoid repeated checks in loop
    if R_goal == -INFINITY:
        # Transform to fully reversed loading (R = -∞)
        for i in range(n):
            amp = amplitude[i]
            R = R_current[i]

            # Calculate mean stress from amplitude and R
            # mean = amp * (1 + R) / (1 - R)
            denominator = 1.0 - R
            if R == -INFINITY or fabs(denominator) < tolerance:
                mean = -amp
            else:
                mean = amp * (1.0 + R) / denominator

            # Select M based on current R
            if R > 1.0:
                M_current = 0.0
            elif R <= 0.0:
                M_current = M
            else:  # 0 < R <= 1
                M_current = M2

            # Apply transformation formula for R_goal = -inf
            # trans_amp = (amp + M * mean) / (1 - M)
            denominator = 1.0 - M_current
            if fabs(denominator) < tolerance:
                Sa_transformed = amp
            else:
                Sa_transformed = (amp + M_current * mean) / denominator

            out_amplitude_view[i] = Sa_transformed

    else:
        # General transformation for R_goal != -inf
        # trans_amp = (1 - R_goal) * (amp + M * mean) / (1 - R_goal + M * (1 + R_goal))
        for i in range(n):
            amp = amplitude[i]
            R = R_current[i]

            # Calculate mean stress from amplitude and R
            denominator = 1.0 - R
            if R == -INFINITY or fabs(denominator) < tolerance:
                mean = -amp
            else:
                mean = amp * (1.0 + R) / denominator

            # Select M based on current R
            if R > 1.0:
                M_current = 0.0
            elif R <= 0.0:
                M_current = M
            else:  # 0 < R <= 1
                M_current = M2

            # Apply general transformation formula
            denominator = 1.0 - R_goal + M_current * (1.0 + R_goal)
            if fabs(denominator) < tolerance:
                Sa_transformed = amp
            else:
                Sa_transformed = ((1.0 - R_goal) * (amp + M_current * mean) /
                                denominator)

            out_amplitude_view[i] = Sa_transformed

    return out_amplitude
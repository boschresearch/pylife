

import numpy as np
cimport cython


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def fourpoint_loop(double [::1] turns, unsigned long [::1] turns_index):
    cdef Py_ssize_t len_turns = len(turns)

    from_vals = np.empty(len_turns//2, dtype=np.float64)
    to_vals = np.empty(len_turns//2, dtype=np.float64)
    from_index = np.empty(len_turns//2, dtype=np.int64)
    to_index = np.empty(len_turns//2, dtype=np.int64)

    cdef double [::1] from_vals_v = from_vals
    cdef double [::1] to_vals_v = to_vals
    cdef unsigned long [::1] from_index_v = from_index
    cdef unsigned long [::1] to_index_v = to_index

    residual_index = np.empty(len_turns, dtype=np.int64)

    cdef unsigned long [::1] residual_index_v = residual_index

    residual_index_v[0] = 0
    residual_index_v[1] = 1

    cdef unsigned long i = 2
    cdef unsigned long ii = 2
    cdef unsigned long t = 0

    cdef double a
    cdef double b
    cdef double c
    cdef double d
    cdef double ab
    cdef double bc
    cdef double cd

    while i < len_turns:
        if ii < 3:
            residual_index_v[ii] = i
            ii += 1
            i += 1
            continue

        a = turns[residual_index_v[ii-3]]
        b = turns[residual_index_v[ii-2]]
        c = turns[residual_index_v[ii-1]]
        d = turns[i]

        ab = np.abs(a - b)
        bc = np.abs(b - c)
        cd = np.abs(c - d)
        if bc <= ab and bc <= cd:
            from_vals_v[t] = b
            to_vals_v[t] = c

            ii -= 1
            to_index_v[t] = turns_index[residual_index_v[ii]]
            ii -= 1
            from_index_v[t] = turns_index[residual_index_v[ii]]
            t += 1
            continue

        residual_index_v[ii] = i
        ii += 1
        i += 1

    return from_vals[:t], to_vals[:t], from_index[:t], to_index[:t], residual_index[:ii]

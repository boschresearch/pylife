

cimport cython
import numpy as np
from libc.math cimport fabs


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def fourpoint_loop(double [::1] turns, size_t [::1] turns_index):
    cdef Py_ssize_t len_turns = len(turns)

    from_vals = np.empty(len_turns//2, dtype=np.float64)
    to_vals = np.empty(len_turns//2, dtype=np.float64)
    from_index = np.empty(len_turns//2, dtype=np.uintp)
    to_index = np.empty(len_turns//2, dtype=np.uintp)

    cdef double [::1] from_vals_v = from_vals
    cdef double [::1] to_vals_v = to_vals
    cdef size_t [::1] from_index_v = from_index
    cdef size_t [::1] to_index_v = to_index

    residual_index = np.empty(len_turns, dtype=np.uintp)
    cdef size_t [::1] residual_index_v = residual_index

    residual_index_v[0] = 0
    residual_index_v[1] = 1

    cdef size_t i = 2
    cdef size_t ri = 2
    cdef size_t t = 0

    cdef double a
    cdef double b
    cdef double c
    cdef double d
    cdef double ab
    cdef double bc
    cdef double cd

    while i < len_turns:
        if ri < 3:
            residual_index_v[ri] = i
            ri += 1
            i += 1
            continue

        a = turns[residual_index_v[ri-3]]
        b = turns[residual_index_v[ri-2]]
        c = turns[residual_index_v[ri-1]]
        d = turns[i]

        ab = fabs(a - b)
        bc = fabs(b - c)
        cd = fabs(c - d)
        if bc <= ab and bc <= cd:
            from_vals_v[t] = b
            to_vals_v[t] = c

            ri -= 1
            to_index_v[t] = turns_index[residual_index_v[ri]]
            ri -= 1
            from_index_v[t] = turns_index[residual_index_v[ri]]
            t += 1
            continue

        residual_index_v[ri] = i
        ri += 1
        i += 1

    return from_vals[:t], to_vals[:t], from_index[:t], to_index[:t], residual_index[:ri]


cpdef double _max(double a, double b):
    return a if a > b else b


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def threepoint_loop(double [::1] turns, size_t [::1] turns_index, size_t highest_front,
                    size_t lowest_front, size_t residual_length):
    cdef Py_ssize_t len_turns = len(turns)

    from_vals = np.empty(len_turns//2, dtype=np.float64)
    to_vals = np.empty(len_turns//2, dtype=np.float64)
    from_index = np.empty(len_turns//2, dtype=np.uintp)
    to_index = np.empty(len_turns//2, dtype=np.uintp)

    cdef double [::1] from_vals_v = from_vals
    cdef double [::1] to_vals_v = to_vals
    cdef size_t [::1] from_index_v = from_index
    cdef size_t [::1] to_index_v = to_index

    residual_index = np.empty(len_turns, dtype=np.uintp)
    residual_index[:residual_length] = np.arange(residual_length)
    cdef size_t [::1] residual_index_v = residual_index

    residual_index_v[0] = 0
    residual_index_v[1] = 1

    cdef size_t ri = 2
    cdef size_t t = 0

    cdef size_t back = residual_index_v[1] + 1
    cdef size_t front
    cdef size_t start

    cdef double start_val
    cdef double front_val
    cdef double back_valstar

    while back < len_turns:
        if ri >= 2:
            start = residual_index_v[ri-2]
            front = residual_index_v[ri-1]
            start_val, front_val, back_val = turns[start], turns[front], turns[back]

            if front_val > turns[highest_front]:
                highest_front = front
            elif front_val < turns[lowest_front]:
                lowest_front = front
            elif (start >= _max(lowest_front, highest_front) and
                  fabs(back_val - front_val) >= fabs(front_val - start_val)):
                from_vals_v[t] = start_val
                to_vals_v[t] = front_val

                from_index_v[t] = turns_index[start]
                to_index_v[t] = turns_index[front]

                t += 1
                ri -= 2
                continue

        residual_index[ri] = back
        ri += 1
        back += 1

    return from_vals[:t], to_vals[:t], from_index[:t], to_index[:t], residual_index[:ri]

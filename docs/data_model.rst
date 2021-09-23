The Data Model of pyLife
========================

pyLife stores data for calculations most often in `pandas` objects.  Modules
that want to operate on such pandas objects should use the `pandas` methods
wherever possible.


Dimensionality of data
----------------------

`pandas` comes with two data classes, :class:`pandas.Series` for one
dimensional data and :class:`pandas.DataFrame` for two dimensional data.

This dimensionality has nothing to do with mathematical or geometrical
dimensions, it only relates to the dimensionality from a data structure
perspective. There are two dimensions, we call them the one in *row direction*
and the one in *column direction*.

In row direction, all the values represent the same physical quantity, like a
stress, a length, a volume, a time.  It means that it is easily thinkable to
add infinitely more values in row direction.  Furthermore it mostly would make
sense to perform statistical operations in row direction, like the maximum
stress, the average volume, etc.

An one channel time signal is an example for a one dimensional data structure
in row direction.  You could add infinitely more samples, every sample
represents the same physical quantity, it makes sense to perform statistical
operations on the whole series.

::

    In [4]: row_direction
    Out[4]:
    time
    0     0.497646
    1     0.278503
    2     0.649374
    3     0.419474
    4     0.614923
    5     0.961856
    ... <infinitely more could be added>
    Name: Load, dtype: float64


In column direction, the values usually represent different physical
quantities.  You might be able to think of adding a few more values in column
direction, but not infinitely.  It almost never makes sense to perform
statistical operations in column direction.

A Wöhler curve is an example for a one dimensional example in column direction.
All the members (columns) represent different physical quantities, it is not
obvious to add more values to the data structure and it makes no sense to
perform statistical operations to it.

::

    In [8]: column_direction
    Out[8]:
    SD     3.000000e+02
    ND     2.300000e+06
    k_1    7.000000e+00
    TS     1.234363e+00
    TN     3.420000e+00
    dtype: float64


Two dimensional data structures have both dimensions. An example would be a
FEM-mesh.

::

    In [9]: vmap.make_mesh('1', 'STATE-2').join_coordinates().join_variable('STRESS_CAUCHY').join_variable('DISPLACEMENT').to_frame()
    Out[9]:
                                x         y    z        S11       S22  S33        S12  S13  S23        dx        dy   dz
    element_id node_id
    1          1734     14.897208  5.269875  0.0  27.080811  6.927080  0.0 -13.687358  0.0  0.0  0.005345  0.000015  0.0
               1582     14.555333  5.355806  0.0  28.319006  1.178649  0.0 -10.732705  0.0  0.0  0.005285  0.000003  0.0
               1596     14.630658  4.908741  0.0  47.701195  5.512213  0.0 -17.866833  0.0  0.0  0.005376  0.000019  0.0
               4923     14.726271  5.312840  0.0  27.699907  4.052865  0.0 -12.210032  0.0  0.0  0.005315  0.000009  0.0
               4924     14.592996  5.132274  0.0  38.010101  3.345431  0.0 -14.299768  0.0  0.0  0.005326  0.000013  0.0
    ...                       ...       ...  ...        ...       ...  ...        ...  ...  ...       ...       ...  ...
    4770       3812    -13.189782 -5.691876  0.0  36.527439  2.470588  0.0 -14.706686  0.0  0.0 -0.005300  0.000027  0.0
               12418   -13.560289 -5.278386  0.0  32.868889  3.320898  0.0 -14.260107  0.0  0.0 -0.005444  0.000002  0.0
               14446   -13.673285 -5.569107  0.0  34.291058  3.642457  0.0 -13.836027  0.0  0.0 -0.005404  0.000009  0.0
               14614   -13.389065 -5.709927  0.0  36.063541  2.828889  0.0 -13.774759  0.0  0.0 -0.005330  0.000022  0.0
               14534   -13.276068 -5.419206  0.0  33.804211  2.829817  0.0 -14.580153  0.0  0.0 -0.005371  0.000014  0.0

    [37884 rows x 12 columns]


Even though a two dimensional rainflow matrix is two dimensional from a
mathematical point of view, it is one dimensional from a data structure
perspective because every element represents the same physical quantity
(occurrence frequency of loops).  You could add infinitely more samples by a
finer bin and it can make sense to perform statistical operations on the whole
matrix, for example in order to normalize it to the maximum frequency value.

In order to represent mathematically multidimensional structures like a two
dimensional rainflow matrix we use :class:`pandas.MultiIndex`.  Example for a
rainflow matrix in a two dimensional :class:`pandas.IntervalIndex`.

::

    In [11]: rainflow_matrix
    Out[11]:
    from                                       to
    (-60.14656996518033, -49.911160871492996]  (-50.39826857402471, -40.36454994053457]      7.0
                                               (-40.36454994053457, -30.33083130704449]     10.0
                                               (-30.33083130704449, -20.29711267355441]      5.0
                                               (-20.29711267355441, -10.26339404006427]      4.0
                                               (-10.26339404006427, -0.2296754065741311]     6.0
                                                                                            ...
    (42.20752097169333, 52.44293006538072]     (9.804043226915951, 19.837761860406033]       9.0
                                               (19.837761860406033, 29.871480493896172]      6.0
                                               (29.871480493896172, 39.90519912738631]       5.0
                                               (39.90519912738631, 49.93891776087639]       11.0
                                               (49.93891776087639, 59.972636394366475]       5.0
    Name: frequency, Length: 121, dtype: float64

    In [12]: type(rainflow_matrix)
    Out[12]: pandas.core.series.Series

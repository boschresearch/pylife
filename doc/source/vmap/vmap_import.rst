Reading a VMAP file
===================

The most common use case is to get the element nodal stress tensor for
a certain geometry ``1`` and a certain load state ``STATE-2`` out of the
vmap file. The vmap interface provides you the nodal geometry (node
coordinates), the mesh connectivity index and the field variables.

You can retrieve a DataFrame of the mesh with the desired variables in just
one statement.

>>> (pylife.vmap.VMAPImport('demos/plate_with_hole.vmap')
     .make_mesh('1', 'STATE-2')
     .join_coordinates()
     .join_variable('STRESS_CAUCHY')
     .join_variable('E')
     .to_frame())
                            x         y    z        S11       S22  S33        S12  S13  S23       E11       E22  E33       E12  E13  E23
element_id node_id
1          1734     14.897208  5.269875  0.0  27.080811  6.927080  0.0 -13.687358  0.0  0.0  0.000119 -0.000006  0.0 -0.000169  0.0  0.0
           1582     14.555333  5.355806  0.0  28.319006  1.178649  0.0 -10.732705  0.0  0.0  0.000133 -0.000035  0.0 -0.000133  0.0  0.0
           1596     14.630658  4.908741  0.0  47.701195  5.512213  0.0 -17.866833  0.0  0.0  0.000219 -0.000042  0.0 -0.000221  0.0  0.0
           4923     14.726271  5.312840  0.0  27.699907  4.052865  0.0 -12.210032  0.0  0.0  0.000126 -0.000020  0.0 -0.000151  0.0  0.0
           4924     14.592996  5.132274  0.0  38.010101  3.345431  0.0 -14.299768  0.0  0.0  0.000176 -0.000038  0.0 -0.000177  0.0  0.0
...                       ...       ...  ...        ...       ...  ...        ...  ...  ...       ...       ...  ...       ...  ...  ...
4770       3812    -13.189782 -5.691876  0.0  36.527439  2.470588  0.0 -14.706686  0.0  0.0  0.000170 -0.000040  0.0 -0.000182  0.0  0.0
           12418   -13.560289 -5.278386  0.0  32.868889  3.320898  0.0 -14.260107  0.0  0.0  0.000152 -0.000031  0.0 -0.000177  0.0  0.0
           14446   -13.673285 -5.569107  0.0  34.291058  3.642457  0.0 -13.836027  0.0  0.0  0.000158 -0.000032  0.0 -0.000171  0.0  0.0
           14614   -13.389065 -5.709927  0.0  36.063541  2.828889  0.0 -13.774759  0.0  0.0  0.000168 -0.000038  0.0 -0.000171  0.0  0.0
           14534   -13.276068 -5.419206  0.0  33.804211  2.829817  0.0 -14.580153  0.0  0.0  0.000157 -0.000035  0.0 -0.000181  0.0  0.0

[37884 rows x 15 columns]


Supported features
------------------

So far the following data can be read from a vmap file

Geometry
........
* node positions
* node element index

Field variables
...............
Any field variables can be read and joined to the node element index
from the following locations:

* element
* node
* element nodal

In particular, field variables at integration point location *cannot*
cannot be read, as that would require extrapolating them to the node
positions. This functionality is not available in pyLife.


The VMAPImport Class
--------------------

.. autoclass:: pylife.vmap.VMAPImport
	:undoc-members:
	:members:
	:inherited-members:

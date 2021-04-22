# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:26:26 2020

@author: KRD2RNG
"""

import pyvista as pv
from pyvista import examples
import numpy as np

import pylife.vmap
import pylife.stress.equistress
#%%
filename = "../plate_with_hole.vmap"

vm_mesh = pylife.vmap.VMAPImport(filename)
pyLife_mesh = df = (vm_mesh.mesh_coords('1')
                    .join(vm_mesh.variable('1', 'STATE-2', 'STRESS_CAUCHY'))
                    .join(vm_mesh.variable('1', 'STATE-2', 'DISPLACEMENT')))
pyLife_mesh['mises'] = pyLife_mesh.equistress.mises()
pyLife_nodes = pyLife_mesh.groupby('node_id').mean()
mesh = pv.PolyData(pyLife_nodes[['x', 'y', 'z']].values)
mesh.plot(point_size=10, screenshot=False)
mesh.point_arrays["mises"] = pyLife_nodes["mises"].values
mesh.plot(scalars="mises")

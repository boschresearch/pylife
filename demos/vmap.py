# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

__author__ = "Daniel Kreuter"
__amintainer__ = __author__

import numpy as np
import pandas as pd
import pyvista as pv
import pylife.vmap

#%% Import 

vm_mesh = pylife.vmap.VMAP("plate_with_hole.vmap")
coords = vm_mesh.mesh_coords("1")
pyLife_mesh = (vm_mesh.mesh_coords('1')
                    .join(vm_mesh.variable('1', 'STATE-2', 'STRESS_CAUCHY'))
                    .join(vm_mesh.variable('1', 'STATE-2', 'DISPLACEMENT')))
pyLife_mesh.sample(10)


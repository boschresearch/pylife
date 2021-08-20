# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:42:12 2021

@author: KRD2RNG
"""

import numpy as np
import pandas as pd
import pylife.vmap
import pylife.mesh
import pylife.mesh.meshsignal
import pylife.stress.equistress
import pyvista as pv

pv.set_plot_theme('document')
pv.set_jupyter_backend('panel')






# ## Local stress approach ##
# #### FE based failure probability calculation

# #### FE Data 
# we are using VMAP data format and rst file formats. It is also possible to use odb data, but 
# this takes some more work (unbelievible but true: Abaqus supports python 2x only)

# In[ ]:
# VMAP
pyLife_mesh = (pylife.vmap.VMAPImport("plate_with_hole.vmap").make_mesh('1', 'STATE-2')
               .join_coordinates()
               .join_variable('STRESS_CAUCHY')
               .to_frame())

pyLife_mesh['mises'] = pyLife_mesh.equistress.mises()

grid = pv.UnstructuredGrid(*pyLife_mesh.mesh.vtk_data())
plotter = pv.Plotter(window_size=[1920, 1080])
plotter.add_mesh(grid, scalars=pyLife_mesh.groupby('element_id')['mises'].mean().to_numpy(),
                show_edges=True, cmap='jet')
plotter.add_scalar_bar()
plotter.show()
#%% Ansys (license is necessary)


# In[ ]:


mises = pyLife_mesh.groupby('element_id')['S11', 'S22', 'S33', 'S12', 'S13', 'S23'].mean().equistress.mises()
mises /= 200.0  # the nominal load level in the FEM analysis
#mises


# #### Damage Calculation ####

# In[ ]:


scaled_rainflow = transformed['total'].rainflow.scale(mises)
#scaled_rainflow.amplitude, scaled_rainflow.frequency


# In[ ]:


damage = mat.fatigue.damage(scaled_rainflow)
#damage


# In[ ]:


damage = damage.groupby(['element_id']).sum()
#damage


# In[ ]:


#pyLife_mesh = pyLife_mesh.join(damage)
#display(pyLife_mesh)


# In[ ]:


grid = pv.UnstructuredGrid(*pyLife_mesh.mesh.vtk_data())
plotter = pv.Plotter(window_size=[1920, 1080])
plotter.add_mesh(grid, scalars=damage.to_numpy(),
                show_edges=True, cmap='jet')
plotter.add_scalar_bar()
plotter.show()


# In[ ]:


print("Maximal damage sum: %f" % damage.max())


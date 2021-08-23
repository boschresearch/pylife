# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:42:12 2021

@author: KRD2RNG
"""

import numpy as np
import pandas as pd
import pickle
import pylife.vmap
import pylife.mesh
import pylife.mesh.meshsignal
import pylife.stress.equistress
import pylife.stress.rainflow
import pylife.strength.fatigue
import pyvista as pv

from ansys.dpf import post

pv.set_plot_theme('document')
pv.set_jupyter_backend('ipyvtklink')

import pylife.stress.histogram as psh




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

#%% Now we want to apply the collectives to the mesh
# In[ ]:
mises = pyLife_mesh.groupby('element_id')['S11', 'S22', 'S33', 'S12', 'S13', 'S23'].mean().equistress.mises()
mises /= mises.max()  # the nominal load level in the FEM analysis is set, that s_max = 1

# #### Damage Calculation ####
collectives = pickle.load(open("collectives.p", "rb"))
collectives = collectives.unstack().T.fillna(0)
collectives_sorted = psh.combine_hist([collectives[col] for col in collectives],
                                             method="sum", nbins=32, histtype="ranges")

scaled_collectives = collectives_sorted.rainflow.scale(mises)
#scaled_rainflow.amplitude, scaled_rainflow.frequency


# In[ ]:
mat = pd.Series({
    'k_1': 8.,
    'ND': 1.0e6,
    'SD': 200.0, # range
    'TN': 1./12.,
    'TS': 1./1.1
})
damage = mat.fatigue.miner_haibach().damage(scaled_collectives)
print(damage.sum())


# In[ ]:


damage = damage.groupby(['element_id']).sum()

grid = pv.UnstructuredGrid(*pyLife_mesh.mesh.vtk_data())
plotter = pv.Plotter(window_size=[1920, 1080])
plotter.add_mesh(grid, scalars=damage.to_numpy(),
                show_edges=True, cmap='jet')
plotter.add_scalar_bar()
plotter.show()
print("Max damage : %f" % damage.max())

#%% Ansys (license is necessary)
# For Ansys  *.rst files we are using pymapdl
# from ansys.mapdl import reader as pymapdl_reader
# # for more information please go to pymapdl
# # rst_input = post.load_solution("beam_3d.rst")
# # # pymapdl has some nice features
# # rst_input.plot_nodal_displacement(0)
# # rst_input.plot_nodal_stress(0,"X")
# ansys_mesh = pymapdl_reader.read_binary('beam_3d.rst')
# grid_ansys = ansys_mesh.grid
# plotter = pv.Plotter(window_size=[1920, 1080])
# _, volume, _  = ansys_mesh.element_solution_data(0,"ENG")
# volume = pd.DataFrame(volume)[1]

# nodes, ansys_mesh_mises = ansys_mesh.nodal_stress(0)
# ansys_mesh_mises = pd.DataFrame(data = ansys_mesh_mises,
#                                 columns=['S11', 'S22', 'S33', 'S12', 'S13', 'S23']).equistress.mises()


# test = pd.DataFrame(ansys_mesh.mesh.elem).iloc[:, 8:]
# #%%
# plotter.add_mesh(grid_ansys, scalars=ansys_mesh_mises,
#                 show_edges=True, cmap='jet')
# plotter.add_scalar_bar()
# plotter.show()
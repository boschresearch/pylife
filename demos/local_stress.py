# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:42:12 2021

@author: KRD2RNG
"""








# ## Local stress approach ##
# #### FE based failure probability calculation

# #### FE Data

# In[ ]:


vm_mesh = pylife.vmap.VMAPImport("plate_with_hole.vmap")
pyLife_mesh = (vm_mesh.make_mesh('1', 'STATE-2')
               .join_coordinates()
               .join_variable('STRESS_CAUCHY')
               .to_frame())


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


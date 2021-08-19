#!/usr/bin/env python
# coding: utf-8

# # Life time Calculation 
# 
# This Notebook shows a general calculation stream for a nominal and local stress reliability approach.
# 
# #### Stress derivation #####
# First we read in different time signals (coming from a test bench or a vehicle measurement e.g.).
# 
# 1. Import the time series into a pandas Data Frame
# 2. Resample the time series if necessary
# 3. Filter the time series with a bandpass filter if necessary
# 4. Edititing time series using Running Statistics methods
# 5. Rainflow Calculation
# 6. Mean stress correction
# 7. Multiplication with repeating factor of every manoveur
# 
# #### Damage Calculation ####
# 1. Select the damage calculation method (Miner elementary, Miner-Haibach, ...)
# 2. Calculate the damage for every load level and the damage sum
# 3. Calculate the failure probability with or w/o field scatter
# 
# #### Local stress approach ####
# 1. Load the FE mesh
# 2. Apply the load history to the FE mesh
# 3. Calculate the damage
# 

# In[ ]:


import numpy as np
import pandas as pd

import pylife.stress.histogram as psh
import pylife.stress.timesignal as ts
import pickle
import pyvista as pv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
import pylife.strength.meanstress
from pylife.strength.meanstress import FKM_goodman as FKM_G
from scipy import signal as sg

# mpl.style.use('seaborn')
# mpl.style.use('seaborn-notebook')
mpl.style.use('bmh')
get_ipython().run_line_magic('matplotlib', 'inline')

pv.set_plot_theme('document')
pv.set_jupyter_backend('panel')


#%% Load rf pickle file
rf_dict = pickle.load(open("rf_dict.p", "rb"))


# In[ ]:

# We define for every time signal a different set of mean stress parameters
meanstress_sensitivity = pd.DataFrame(data = np.array([[.3, .35, .4, .45], [.1, .15, .2, .25]]).T,
                                      columns = ["M", "M2"],
                                      index=rf_dict.keys())
meanstress_sensitivity = pd.Series({
    'M': 0.3,
    'M2': 0.2
})
R_goal=-1


# In[ ]:

# High level API
transformed_dict = {key: rf_act.meanstress_hist.FKM_goodman(meanstress_sensitivity, R_goal) for key, rf_act in rf_dict.items()}

# Low level API (here only for one signal)
rf_sine = rf_dict["sine"]
froms = rf_sine[rf_sine > 0].index.get_level_values('from').mid 
tos = rf_sine[rf_sine > 0].index.get_level_values('to').mid 
amplitude = np.abs(froms-tos)/2.
mean = (froms+tos)/2.

transformed_sine = FKM_G(amplitude, mean, 0.3, 0.1, -1) 

plt.plot(mean, amplitude, "x")
plt.plot(np.zeros_like(transformed_sine), transformed_sine, "ro")

#%% ## Repeating factor
### Let us assume that our component should be loaded 50 times with the sine load,
### 100 times with the noise signal and 25 times with the SoR load. 
# In[ ]:
repeating = {
    'wn': 50.0, 
    'sine': 100,
    'SoR' : 25
}
transformed_dict = {k:v for k,v in transformed_dict.items() if k in ["wn", "sine", "SoR"]}
repeating_dict = {k: v * repeating[k] for k,v in transformed_dict.items()}
# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 15))

for k, range_only in repeating_dict.items():
    # range_only.T.plot(drawstyle="steps", ax = ax)
    data = range_only.sort_index(ascending=False)
    ax.step(np.cumsum(data), data.index.mid, label=k)
    # ax.step(range_only, range_only.index.mid, label=k)
# for title, ai in zip(['Count', 'Cumulated'], ax):
#     ai.set_title(title)
#     ai.xaxis.grid(True)
#     ai.legend()
#     ai.set_xlabel('count')
#     ai.set_ylabel('amplitude')
#     ai.set_ylim((0,max(amplitude)))  


# ## Nominal stress approach ##

# ### Material parameters ###
# You can create your own material data from Woeler tests using the Notebook woehler_analyzer

# In[ ]:


mat = pd.Series({
    'k_1': 8.,
    'ND': 1.0e6,
    'SD': 300.0,
    'TN': 1./12.,
    'TS': 1./1.1
})
display(mat)


# ### Damage Calculation ###

# In[ ]:


damage_miner_original = mat.fatigue.damage(transformed['total'].rainflow)
damage_miner_elementary = mat.fatigue.miner_elementary().damage(transformed['total'].rainflow)
damage_miner_haibach = mat.fatigue.miner_haibach().damage(transformed['total'].rainflow)
damage_miner_original.sum(), damage_miner_elementary.sum(), damage_miner_haibach.sum()


# In[ ]:


wc = mat.woehler
cyc = pd.Series(np.logspace(1, 12, 200))
for pf, style in zip([0.1, 0.5, 0.9], ['--', '-', '--']):
    load = wc.basquin_load(cyc, failure_probability=pf)
    plt.plot(cyc, load, style)

plt.step(np.cumsum(cycles), transformed['total'].rainflow.amplitude[::-1])

plt.loglog()


# ## Failure Probaility ##

# #### Without field scatter ####

# In[ ]:


D50 = 0.05

damage = mat.fatigue.damage(transformed['total'].rainflow).sum()

di = np.logspace(np.log10(1e-2*damage), np.log10(1e4*damage), 1000)
std = pylife.utils.functions.scatteringRange2std(mat.TN)
failprob = fp.FailureProbability(D50, std).pf_simple_load(di)

fig, ax = plt.subplots()
ax.semilogx(di, failprob, label='cdf')

plt.xlabel("Damage")
plt.ylabel("cdf")
plt.title("Failure probability = %.2e" %fp.FailureProbability(D50,std).pf_simple_load(damage))  
plt.ylim(0,max(failprob))
plt.xlim(min(di), max(di))

fp.FailureProbability(D50, std).pf_simple_load(damage)


# #### With field scatter ####

# In[ ]:


field_std = 0.35
fig, ax = plt.subplots()
# plot pdf of material
mat_pdf = norm.pdf(np.log10(di), loc=np.log10(D50), scale=std)
ax.semilogx(di, mat_pdf, label='pdf_mat')
# plot pdf of load
field_pdf = norm.pdf(np.log10(di), loc=np.log10(damage), scale=field_std)
ax.semilogx(di, field_pdf, label='pdf_load',color = 'r')
plt.xlabel("Damage")
plt.ylabel("pdf")
plt.title("Failure probability = %.2e" %fp.FailureProbability(D50, std).pf_norm_load(damage, field_std))  
plt.legend()


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


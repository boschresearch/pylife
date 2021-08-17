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

# In[1]:


import numpy as np
import pandas as pd

from pylife.stress.histogram import *
import pylife.stress.timesignal as ts
from pylife.stress.rainflow import *
import pylife.stress.equistress

import pylife.stress.rainflow
import pylife.strength.meanstress
import pylife.strength.fatigue

import pylife.mesh.meshsignal

from pylife.strength import failure_probability as fp
import pylife.vmap

import pyvista as pv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl

from scipy.stats import norm


# mpl.style.use('seaborn')
# mpl.style.use('seaborn-notebook')
mpl.style.use('bmh')
get_ipython().run_line_magic('matplotlib', 'inline')

pv.set_plot_theme('document')
pv.set_jupyter_backend('panel')


# ### Time series signal ###
# import, filtering and so on. You can import your own signal with
# 
# * [pd.read_csv()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)
# * [pd.read_excel()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html)
# * [scipy.io.loadmat()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html) for matlab files 
# 
# and so on

# In[ ]:


np.random.seed(4711)
sample_frequency = 1024
t = np.linspace(0, 60, 60 * sample_frequency)

signals = {
    'wn': pd.DataFrame(index = t, columns = ['sensor_1'], data = 240*np.random.randn(len(t))),
    'sine': pd.DataFrame(index = t, columns = ['sensor_2'], data = 160*np.sin(2*np.pi*50*t))
}

signal_df = pd.DataFrame(data = np.array([80 * np.random.randn(len(t)), 160 * np.sin(2 * np.pi * 50 * t)]).T,
                         columns=["wn", "sine"],
                         index=t)
signal_df["SoR"] = signal_df["wn"] + signal_df["sine"]
signal_df.plot(subplots=True)

# In[ ]:

f_min = 5.0    # Hz
f_max = 100.0  # Hz

bandpass_df = ts.butter_bandpass(signal_df, f_min, f_max, sample_frequency)

# In[ ]:


# bandpass = {}
# for k, df_act in meas_resample.items():
#     bandpassDF = pd.DataFrame(index = df_act.index)
#     for col_act in df_act.columns:
#         bandpassDF[col_act] = ts.TimeSignalPrep(df_act[col_act]).butter_bandpass(f_min, f_max, resampling_freq, 5)
#     bandpass[k] = bandpassDF
    
# display(bandpassDF)


# ### Running statistics

# In[ ]:


statistics_method = 'rms'  # alternatively 'max', 'min', 'abs'

run_statt = 'window_length' # alternatively 'buffer_overlap', 'limit'

window_length = 800
buffer_overlap = 0.1
limit = 0.15


# In[ ]:


""" Running statistics to drop out zero values """
cleaned = {}
for k, df_act in bandpass.items():
    cleaned_df = ts.TimeSignalPrep(df_act).running_stats_filt(
                            col="sensor_1",
                            window_length=window_length,
                            buffer_overlap=buffer_overlap,
                            limit=limit,
                            method=statistics_method)
    cleaned[k] = cleaned_df


# In[ ]:


fig, ax = plt.subplots(len(meas_resample))
fig.suptitle('Cleaned input data')
for ax, (k, df_act) in zip(ax, cleaned.items()):
    ax.plot(df_act.index, df_act['sensor_1'])


# ### Rainflow ###

# In[ ]:


rainflow_bins = 64


# In[ ]:


rainflow = {}
for k, df_act in cleaned.items():
    rfc = RainflowCounterFKM().process(df_act['sensor_1'].values)
    rfm = rfc.get_rainflow_matrix_frame(rainflow_bins).loc[:, 0]
    rainflow[k] = rfm


# In[ ]:


colormap = cm.ScalarMappable()
cmap = cm.get_cmap('PuRd')
# fig, ax = plt.subplots(2,len(rainflow))
fig = plt.figure(figsize = (8,11))
fig.suptitle('Rainflow of Channel sensor_1')

for i, (k, rf_act) in enumerate(rainflow.items()):
    # 2D
    ax = fig.add_subplot(3,2,2*(i+1)-1)
    froms = rf_act.index.get_level_values('from').mid
    tos = rf_act.index.get_level_values('to').mid
    counts = np.flipud((rf_act.values.reshape(rf_act.index.levshape).T))#.ravel()
    ax.set_xlabel('From')
    ax.set_ylabel('To')
    ax.imshow(np.log10(counts), extent=[froms.min(), froms.max(), tos.min(), tos.max()])
    # 3D
    ax = fig.add_subplot(3,2,2*(i+1), projection='3d')
    bottom = np.zeros_like(counts.ravel())
    width = rf_act.index.get_level_values('from').length.min()
    depth = rf_act.index.get_level_values('to').length.min()
    max_height = np.max(counts.ravel())   # get range of colorbars
    min_height = np.min(counts.ravel())
    rgba = [cmap((k-min_height)/max_height) for k in counts.ravel()] 
    ax.set_xlabel('From')
    ax.set_ylabel('To')
    ax.set_zlabel('Count')
    ax.bar3d(froms.ravel(), tos.ravel(), bottom, width, depth, counts.ravel(), shade=True, color=rgba, zsort='average')


# ### Meanstress transformation ###

# In[ ]:


meanstress_sensitivity = pd.Series({
    'M': 0.3,
    'M2': 0.2
})


# In[ ]:


transformed = {k: rf_act.meanstress_hist.FKM_goodman(meanstress_sensitivity, R_goal=-1.) for k, rf_act in rainflow.items()}


# ## Repeating factor

# In[ ]:


repeating = {
    'wn': 50.0, 
    'sine': 25.0
}


# In[ ]:


transformed['total'] = combine_hist([transformed[k] * repeating[k] for k in ['wn', 'sine']], method="sum")


# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(10, 5))

for k, range_only in transformed.items():
    amplitude = range_only.rainflow.amplitude[::-1]
    cycles = range_only.values[::-1].ravel()
    ax[0].step(cycles, amplitude, label=k)
    ax[1].step(np.cumsum(cycles), amplitude, label=k)

for title, ai in zip(['Count', 'Cumulated'], ax):
    ai.set_title(title)
    ai.xaxis.grid(True)
    ai.legend()
    ai.set_xlabel('count')
    ai.set_ylabel('amplitude')
    ai.set_ylim((0,max(amplitude)))  


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


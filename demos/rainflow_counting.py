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

import pylife.stress.histogram as psh
import pylife.stress.timesignal as ts
import pylife.stress.rainflow as RF
import pylife.strength.meanstress
import pylife.stress.rainflow.recorders as RFR
import pickle


import pyvista as pv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl

from scipy import signal as sg

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

#%% 
def plot_rf(rf_series_dict):
    fig = plt.figure(figsize=(8, 16))
    cmap = cm.get_cmap('jet') # 
    for ii, key in enumerate(rf_series_dict.keys()):
        rf_series = rf_series_dict[key]
        ax = fig.add_subplot(1,len(rf_series_dict),ii + 1, projection='3d')
        
        froms = rf_series.index.get_level_values("from").mid
        tos = rf_series.index.get_level_values("to").mid
        
        width = rf_series.index.get_level_values('from').length.min()
        depth = rf_series.index.get_level_values('to').length.min()
        bottom = np.zeros_like(rf_series)
        
        max_height = np.max(rf_series) 
        min_height = np.min(rf_series)
        rgba = [cmap((k-min_height)/max_height) for k in rf_series] 
        ax.set_xlabel('From')
        ax.set_ylabel('To')
        ax.set_zlabel('Count')
        ax.bar3d(froms.ravel(), tos.ravel(), bottom, width, depth, rf_series, 
                 color=rgba, shade=True, zsort='average')
        ax.set_title(key)
    return fig

# In[ ]:


np.random.seed(4711)
sample_frequency = 1024
t = np.linspace(0, 60, 60 * sample_frequency)
signal_df = pd.DataFrame(data = np.array([80 * np.random.randn(len(t)), 160 * np.sin(2 * np.pi * 50 * t)]).T,
                         columns=["wn", "sine"],
                         index=t)
signal_df["SoR"] = signal_df["wn"] + signal_df["sine"]
signal_df.plot(subplots=True)
ts.psd_df(signal_df, NFFT = 512).plot(loglog=True)
# In[ ]:

f_min = 5.0    # Hz
f_max = 100.0  # Hz

bandpass_df = ts.butter_bandpass(signal_df, f_min, f_max)

df_psd = ts.psd_df(bandpass_df, NFFT = 512)
df_psd.plot(loglog=True)
# In[ ]: let us add some spike in to the signal and see, if we could filter it out
bandpass_df["spiky"] = bandpass_df["SoR"] + 1e3 * sg.unit_impulse(signal_df.shape[0], idx="mid")
cleaned_df = ts.clean_timeseries(bandpass_df, "spiky", window_size=200, overlap=20,
                     feature="abs_energy", n_gridpoints=3,
                     percentage_max=0.05, order=3).drop(["time"], axis=1)

ts.psd_df(cleaned_df, NFFT = 512).plot(loglog=True)
#%% Rainflow for a multiple time series
recorder_dict = {key: RFR.FullRecorder() for key in cleaned_df}
detector_dict = {key: RF.FKMDetector(recorder=recorder_dict[key]).process(cleaned_df[key]) for key in cleaned_df}
rf_series_dict = {key: detector_dict[key].recorder.matrix_series(10) for  key in detector_dict.keys()}

plot_rf(rf_series_dict)
    
#%% Now Combining different RFs to one
rf_series_dict["wn + sn"] = psh.combine_hist([rf_series_dict["wn"],rf_series_dict["sine"]],
                                             method="sum")
rf_series_dict.pop("spiky")
plot_rf(rf_series_dict)
#%%
df_psd["max"] =  df_psd[["sine", "wn"]].max(axis = 1)
df_psd.plot(loglog=True)
#%%
pickle.dump(rf_series_dict, open("rf_dict.p", "wb"))
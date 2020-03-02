#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np
import pandas as pd

from pylife.stress.rainflow import *
import pylife.strength.meanstress
import pylife.stress.timesignal as ts
from pylife.strength import miner
from pylife.strength.miner import MinerElementar, MinerHaibach
import matplotlib.pyplot as plt
from pylife.materialdata.woehler.diagram import *
from pylife.materialdata.woehler.widgets import *
from pylife.materialdata.woehler.analyzer import *
import pylife.utils.meshplot
import pylife.stress.equistress
from scipy.stats import norm
from pylife.strength import failure_probability as fp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import io
from IPython.display import HTML
import base64 
import matplotlib as mpl
# mpl.style.use('seaborn')
# mpl.style.use('seaborn-notebook')
mpl.style.use('bmh')


# ### Time series signal ###
# import, filtering and so on

# In[2]:


files = ["IIS0042_DAT_2017LK065HU004_000100_0206_ESP_FA_arf_0004163.dat.csv",
         "IIS0042_DAT_2017LK065HU004_000100_0206_ALC7_abf_0004587.dat.csv",
         "IIS0042_DAT_2017LK065HU004_000100_0205_hfc2_abf_0003880.dat.csv"]
data_loc = '\\\\fe00fs76.de.bosch.com\\SP17_Extern$\\SP17-010\\TSIPRAS\\MeasuredData\\Messdaten_ESP_Maneuver\\'


# In[3]:


input_data = []
for upload in files:
    data_akt = pd.read_csv(data_loc + upload, sep = ",")
    if len(data_akt.columns) == 1:
        print ('please use "," as seperator next time')
        data_akt = pd.read_csv(data_loc + upload, sep = ";")
        input_data.append(data_akt)
    print(upload + " imported succesfully")


# ### Resampling ###

# In[4]:


f_resample = widgets.FloatText(value = 40e3,min=1,max=100e3,step=10,
    description='Resampling frequency [Hz]',
    disabled=False,readout=True,readout_format='d')
display(f_resample)
# select time column
timeColumn = widgets.Dropdown(options = data_akt.columns)
display(timeColumn)


# In[5]:


meas_resample = []
for file_act in input_data:
    file_act = file_act.set_index(timeColumn.value)
    meas_resample.append(ts.TimeSignalPrep(file_act).resample_acc(f_resample.value))
display(file_act)


# In[6]:


print("select channel to plot")
plotChan = widgets.Dropdown(options = file_act.columns)
display(plotChan)


# In[77]:


get_ipython().run_line_magic('matplotlib', 'notebook')
fig, ax = plt.subplots(len(meas_resample))
fig.suptitle('Resampled input data')
ii = 0
for df_act in meas_resample:
    ax[ii].plot(df_act.index, df_act[plotChan.value])
    ii += 1


# ### Filtering 

# In[8]:


f_min = widgets.FloatText(value = 100,description='min frequency [Hz]',disabled=False)
f_max = widgets.FloatText(value = 5e3,description='max frequency [Hz]',disabled=False)
display(f_min)
display(f_max)


# In[9]:


bandpass = []
for df_act in meas_resample:
    bandpassDF = pd.DataFrame(index = df_act.index)
    for col_act in df_act.columns:
        bandpassDF[col_act] = ts.TimeSignalPrep(df_act[col_act]).butter_bandpass(f_min.value,f_max.value,f_resample.value,5)
    bandpass.append(bandpassDF) 
display(bandpassDF)


# ### Running statistics

# In[10]:


print("select channel to for running stats")
runChan = widgets.Dropdown(options = df_act.columns)
display(runChan)
print(" Running statistics method")
method_choice = widgets.Dropdown(options = ['rms','max','min','abs'])
display(method_choice)

paraRunStats = ['window_length', 'buffer_overlap', 'limit']
values = [800,0.1,0.1]
child = [widgets.FloatText(description=name) for name in paraRunStats]
tab = widgets.Tab()
tab.children = child
for i in range(len(child)):
    tab.set_title(i, paraRunStats[i])
    tab.children[i].value = values[i]

tab


# In[13]:


""" Running statistics to drop out zero values """
cleaned = []
for df_act in bandpass:
    cleaned_df = ts.TimeSignalPrep(df_act).running_stats_filt(
                            col = runChan.value,
                            window_length = int(tab.children[0].value),
                            buffer_overlap = int(tab.children[1].value),
                            limit = tab.children[2].value,
                            method = method_choice.value)
    cleaned.append(cleaned_df)
# display(cleaned_df)


# In[75]:


get_ipython().run_line_magic('matplotlib', 'notebook')
fig, ax = plt.subplots(len(meas_resample))
fig.suptitle('Cleaned input data')
ii = 0
for df_act in cleaned:
    ax[ii].plot(df_act.index, df_act[runChan.value])
    ii += 1    


# ### Rainflow ###

# In[21]:


rfcChan = widgets.Dropdown(options = df_act.columns)
display(rfcChan)
binwidget = widgets.IntSlider(value = 64, min=1, max=1024, step=1,description='Bins:')
display(binwidget)


# In[49]:


rainflow = []
for df_act in cleaned:
    rfc = RainflowCounterFKM().process(df_act[rfcChan.value].values)
    rfm = rfc.get_rainflow_matrix_frame(binwidget.value)
    rainflow.append(rfm)


# In[74]:


get_ipython().run_line_magic('matplotlib', 'notebook')
colormap = cm.ScalarMappable()
cmap = cm.get_cmap('PuRd')
# fig, ax = plt.subplots(2,len(rainflow))
fig = plt.figure(figsize = (8,11))
fig.suptitle('Rainflow of Channel ' + rfcChan.value)
ii = 1

for rf_act in rainflow:
    # 2D
    ax = fig.add_subplot(3,2,2*ii-1)
    froms = rf_act.index.get_level_values('from').mid
    tos = rf_act.index.get_level_values('to').mid
    counts = np.flipud((rf_act.values.reshape(rf_act.index.levshape).T))#.ravel()
    ax.set_xlabel('From')
    ax.set_ylabel('To')
    ax.imshow(np.log10(counts), extent=[froms.min(), froms.max(), tos.min(), tos.max()])
    # 3D
    ax = fig.add_subplot(3,2,2*ii, projection='3d')
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
    ii += 1


# ### Meanstress transformation ###

# In[78]:


meanstress_para = ['M', 'M2', 'R_Goal']
values = [0.3,0.2,-1]
child = [widgets.FloatText(description=name) for name in meanstress_para]
tab_mean = widgets.Tab()
tab_mean.children = child
for i in range(len(child)):
    tab_mean.set_title(i, meanstress_para[i])
    tab_mean.children[i].value = values[i]

tab_mean


# In[79]:


transformed = []
for rf_act in rainflow:
    transformed.append(rf_act.meanstress_hist.FKM_goodman(pd.Series({'M': tab_mean.children[0].value, 
                                                                     'M2': tab_mean.children[1].value})
                                                          , R_goal = tab_mean.children[2].value))


# ## Repeating factor

# In[80]:


child = [widgets.FloatText(description=name) for name in files]
tab_repeat = widgets.Tab()
tab_repeat.children = child
for i in range(len(child)):
    tab_repeat.set_title(i, files[i])
    tab_repeat.children[i].value = 1
tab_repeat


# In[87]:


for ii in range(len(files)): 
    transformed[ii] = transformed[ii]*tab_repeat.children[ii].value
    if ii == 0:
        transformed_total = transformed[ii]
    else:
        transformed_total = pd.concat([transformed_total,transformed[ii]]).sort_index()
display(transformed_total)    
#%%
index_min =  transformed_total.index.left.min()
index_max =  transformed_total.index.right.max()
bins = 24
new_index = pd.interval_range(index_min,index_max,periods= bins)
test = transformed_total.groupby(pd.cut(transformed_total.index.mid.values,
                             np.linspace(index_min,index_max,bins))).sum()
test.index.name = 'range'
# In[83]:


get_ipython().run_line_magic('matplotlib', 'notebook')
for range_only in transformed:
    amplitude = range_only.index.get_level_values('range').mid.values[::-1]/2
    cycles = range_only.values[::-1].ravel()

    # fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(10, 5))
    # ax[0].barh(amplitude,np.cumsum(cycles), height = range_only.index.get_level_values('range').length.min())
    # ax[1].barh(amplitude,cycles, height = range_only.index.get_level_values('range').length.min())
    # ax[0].set_title('Cumulated sum count')
    # ax[1].set_title('Count')
    # for ai in ax:
    #     ai.xaxis.grid(True)
    # #     ai.set_xticks([y+1 for y in range(len(all_data))], )
    #     ai.set_xlabel('count')
    #     ai.set_ylabel('amplitude of ' + colName)
    #     ai.set_ylim((0,max(amplitude)))  


# ## Nominal stress approach ##

# ### Material parameters ###

# In[ ]:


mat = WL_param()
display(mat)


# In[ ]:


mat_para, _ = WL_param_display(mat)
sn_curve_parameters =  {key:mat_para[key] for key in mat_para.keys() - {'TN_inv', 'TS_inv'}} 
mat_para['TS_inv'] = mat_para['TN_inv']**(1/mat_para['k_1'])
print(mat_para)
print(sn_curve_parameters)


# ### Damage Calculation ###

# In[ ]:


SNmethod = widgets.Dropdown(options = ['Miner Elementar','Miner Haibach','Miner original'])
display(SNmethod)


# In[ ]:


damage_calc = MinerElementar(**sn_curve_parameters)
data_act = transformed[0]
range_mid = data_act.index.get_level_values('range').mid.values
data_finite = data_act[range_mid > sn_curve_parameters['SD_50']][::-1]
stress_finite = data_finite.index.get_level_values('range').mid.values
N = damage_calc.sn_curve.calc_N(stress_finite)
d = np.sum(data_act[range_mid > sn_curve_parameters['SD_50']].values/N)
print("\033[1m  Total Damage: %.2f  \033[0m" %d) 


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
class wc_data:
    def __init__(self, k, TN,loads_max):
        self.k = k
        self.TN = TN
        self.loads_max = loads_max
SRI = mat_para['SD_50']*(mat_para['ND_50']**(1/mat_para['k_1']))
wc = wc_data(mat_para['k_1'],mat_para['TN_inv'],SRI) 
_ = PlotWoehlerCurve.final_curve_plot(wc, mat_para['SD_50'], mat_para['ND_50'],mat_para['TS_inv'],0, 
                                      None,"Amplitude","Load", " ", (1,1e8), (1e1,1e3), 0)
plt.barh(stress_finite,np.cumsum(data_finite.values), height = data_finite.index.get_level_values('range').length.min())


# ## Failure Probaility ##

# #### Without field scatter ####

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
D50 = 0.3
mat_std = np.log10(mat_para['TN_inv'])/2.5631031311
dmin = d/10
di = np.linspace(dmin,50*d,1000)
failprob = fp.FailureProbability(D50,mat_std).pf_simple_load(di)
#print(failprob)
fig, ax = plt.subplots()
ax.semilogx(di, failprob, label='cdf')
ax.vlines(d, 0, fp.FailureProbability(D50,mat_std).pf_simple_load(d))
# 
ix = np.linspace(dmin, d)
ax.fill_between(ix, fp.FailureProbability(D50,mat_std).pf_simple_load(ix), y2=0,color = 'y')
plt.xlabel("Damage")
plt.ylabel("cdf")
plt.title("Failure probability = %.2f" %fp.FailureProbability(D50,mat_std).pf_simple_load(d))  
plt.ylim(0,1)


# #### With field scatter ####

# In[ ]:


field_std = 0.35
fig, ax = plt.subplots()
# plot pdf of material
mat_pdf = norm.pdf(np.log10(di), loc=np.log10(D50), scale=mat_std)
ax.semilogx(di, mat_pdf, label='pdf_mat')
# plot pdf of load
field_pdf = norm.pdf(np.log10(di), loc=np.log10(d), scale=field_std)
ax.semilogx(di, field_pdf, label='pdf_load',color = 'r')
# area_1
area = np.minimum(mat_pdf, field_pdf)
ax.fill_between(di, area, y2=0,color = 'y')
plt.xlabel("Damage")
plt.ylabel("pdf")
plt.title("Failure probability = %.2f" %fp.FailureProbability(D50,mat_std).pf_norm_load(d,field_std))  
plt.legend()


# ## Local stress approach ##
# #### FE based failure probability calculation

# #### FE Data

# In[ ]:


filename = 'plate_with_hole.h5'

stress = pd.read_hdf(filename, 'node_data')
stress['S13'] = np.zeros_like(stress['S11'])
stress['S23'] = np.zeros_like(stress['S11'])
""" Equivalent stress """
s_vm = stress.groupby('element_id').mean().equistress.mises().rename(columns={'mises': 'sigma_a'})
s_vm = 2*s_vm/s_vm.max()
""" Scale with """
ampl = pd.DataFrame(data = data_finite.index.get_level_values('range').mid.values, columns = ["ampl"] ,index = data_finite["frequency"].values)
s_vm_scaled = pd.DataFrame(data = ampl.values*s_vm.transpose().values,index = ampl.index,columns = s_vm.index)
#display(s_vm_scaled)
#data_finite = data_act[range_mid > sn_curve_parameters["sigma_ak"]][::-1]


# #### Damage Calculation ####

# In[ ]:


s_vm_scaled[s_vm_scaled < mat_para['SD_50']] = 0
N = damage_calc.sn_curve.calc_N(s_vm_scaled,ignore_limits = True)
d_mesh_cycle =  1/(N.div(N.index.values, axis = 'index'))
#np.sum(data_act[range_mid > sn_curve_parameters["sigma_ak"]].values/N)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
d_mesh = d_mesh_cycle.sum()
fig, ax = plt.subplots()
stress.join(pd.DataFrame(data = d_mesh,columns = ['d'])).meshplot.plot(ax, 'd', cmap='jet_r')
plt.show()
plt.title("Damage per element")


# In[ ]:


print(d_mesh.max())


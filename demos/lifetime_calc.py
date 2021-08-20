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

import pickle
import pyvista as pv

import matplotlib.pyplot as plt
import matplotlib as mpl
import pylife.strength.meanstress
from pylife.strength.meanstress import FKM_goodman as FKM_G
import pylife.strength.fatigue
from pylife.strength import failure_probability as fp

from scipy.stats import norm
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
    'wn': 20, 
    'sine': 5,
    'SoR' : 2
}
transformed_dict = {k:v for k,v in transformed_dict.items() if k in ["wn", "sine", "SoR"]}
repeating_dict = {k: v * repeating[k] for k,v in transformed_dict.items()}
# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10, 10))

for k, range_only in repeating_dict.items():
    # range_only.T.plot(drawstyle="steps", ax = ax)
    data = range_only.sort_index(ascending=False)
    ax.step(np.cumsum(data), data.index.mid / 2, label=k)
    # ax.step(range_only, range_only.index.mid, label=k)
# for title, ai in zip(['Count', 'Cumulated'], ax):
#     ai.set_title(title)
#     ai.xaxis.grid(True)
    ax.legend()
    ax.set_xlabel('count')
    ax.set_ylabel('amplitude')
#     ai.set_ylim((0,max(amplitude)))  


# ## Nominal stress approach ##

# ### Material parameters ###
# You can create your own material data from Woeler tests using the Notebook woehler_analyzer
# In[ ]:


mat = pd.Series({
    'k_1': 8.,
    'ND': 1.0e6,
    'SD': 125.0, # range
    'TN': 1./12.,
    'TS': 1./1.1
})

wc = mat.woehler
cyc = pd.Series(np.logspace(1, 12, 200))
curves = pd.DataFrame({"pf = " + str(pf): wc.basquin_load(cyc, failure_probability=pf) for pf in [0.1, 0.5, 0.9]}).set_index(cyc)
curves.plot(loglog=True, ax=ax)
# ### Damage Calculation ###
# First option: We calculate the damage of every channel and load collective.
# Now we are using miner original and we will compare this with the Miner-Haibach approach
# In[ ]:
damage_miner_original = {k : mat.fatigue.damage(v.rainflow) for k,v in repeating_dict.items()}
damage_sum = pd.DataFrame({k : v.sum() for k,v in damage_miner_original.items()},
                                         index = ["original"])
damage_miner_haibach = {k : mat.fatigue.miner_haibach().damage(v.rainflow) for k,v in repeating_dict.items()}
damage_sum = damage_sum.append(pd.DataFrame({k : v.sum() for k,v in damage_miner_haibach.items()},
                                         index = ["haibach"]))

damage_sum["compare_collectives_combined"] = damage_sum.sum(axis=1)
# Now we combine first all load collectives together and will compare the results with the previous
collectives_combined = pd.concat(repeating_dict)
damage_sum["collectives_combined"] = [mat.fatigue.damage(collectives_combined.rainflow).sum(),
                          mat.fatigue.miner_haibach().damage(collectives_combined.rainflow).sum()]
# In[ ]:

# ## Failure Probaility ##

# #### Without field scatter ####

# In[ ]:


D50 = 5

di = np.logspace(-2, np.log10(1e3*damage_sum.max().max()), 1000)

mat_std = pylife.utils.functions.scatteringRange2std(mat.TN)
failprob = pd.DataFrame(fp.FailureProbability(D50, mat_std).pf_simple_load(di),
                        index = di)

fig, ax = plt.subplots()
failprob.plot(ax=ax, logx=True)
pf_collectives_combined = fp.FailureProbability(D50, mat_std).pf_simple_load(damage_sum.collectives_combined.max())
# ax.semilogx(di, failprob, label='cdf')
ax.vlines(damage_sum.collectives_combined.max(), failprob.min(), pf_collectives_combined)
plt.title("Failure probability = %.2e" %pf_collectives_combined)  

# #### With field scatter ####

# In[ ]:

field_std = 0.35
fig, ax = plt.subplots()
# pdf = pd.DataFrame(data=[norm.pdf(np.log10(di), loc=np.log10(D50), scale=std),
fp_field = [fp.FailureProbability(D50, mat_std).pf_norm_load(k, field_std) for k  in damage_sum.loc["haibach"]]
pf_pdf = pd.DataFrame({k + ", fp=%.2e" %fp_field[i] : norm.pdf(np.log10(di), loc=damage_sum.loc["haibach"][k],
                          scale=field_std) for i,k in enumerate(damage_sum.loc["haibach"].index)},
                         index=pd.Index(di, name="damage"))
pf_pdf["material"] = norm.pdf(np.log10(di), loc=np.log10(D50), scale=mat_std)
pf_pdf.plot(logx=True, title="Failure probability for different load collectives")

#%% Saving the combined collectives for the local stress approach (which yu can find here)
pickle.dump(collectives_combined, open("collectives.p", "wb"))
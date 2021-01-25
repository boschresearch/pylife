# Copyright (c) 2019-2021 - for information on the respective copyright owner
# see the NOTICE file and/or the repository
# https://github.com/boschresearch/pylife
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import numpy as np
import pandas as pd
from scipy import signal as sg
from pylife.stress import timesignal as tsig
import matplotlib.pyplot as plt
#%%
def test_resample_acc():
    # sine
    omega = 10*2*np.pi # Hz
    ts_sin = pd.DataFrame(np.sin(
        omega*np.arange(0,2,1/1024)),index = np.arange(0,2,1/1024))
    expected_sin = ts_sin.describe().drop( ['count','mean','50%','25%','75%'])
    test_sin = tsig.TimeSignalPrep(ts_sin).resample_acc(int(12*omega)).describe().drop(
        ['count','mean','50%','25%','75%'])
    pd.testing.assert_frame_equal(test_sin,expected_sin, check_less_precise = 2)
    # white noise
    ts_wn = pd.DataFrame(np.random.randn(129),index = np.linspace(0,1,129))
    expected_wn = ts_wn.describe()
    test_wn = tsig.TimeSignalPrep(ts_wn).resample_acc(128).describe()
    pd.testing.assert_frame_equal(test_wn,expected_wn,check_exact=True)
    # SoR
    t = np.arange(0,20,1/4096)
    f =  np.logspace(np.log10(1),np.log10(10),num = len(t))
    df = pd.DataFrame(data = np.multiply(
                np.interp(f,np.array([1,10]),np.array([0.5,5]),left = 0, right = 0),
                sg.chirp(t,1,20,10,method = "logarithmic",phi = -90)),
                index = t,columns = ["Sweep"])
    df['sor'] = df['Sweep'] + 0.1*np.random.randn(len(df))
    expected_sor = df.describe().drop(['count','mean','50%','25%','75%'])
    test_sor = tsig.TimeSignalPrep(df).resample_acc(2048).describe().drop(['count','mean','50%','25%','75%'])
    pd.testing.assert_frame_equal(test_sor,expected_sor, check_less_precise = 1)
#%%
#Test timeseries 1
print("doing timeseries 1")
size_df = 10
ts_inp = pd.DataFrame(index=np.arange(size_df)+1,
              data=np.arange(size_df)**2)
ts_inp_test =ts_inp.copy()
ts_test1 = tsig.TimeSignalPrep(ts_inp)   

    

def test_prepare_rolling():
    
    exact_res = pd.DataFrame()
    exact_res.loc[:,0]=np.arange(size_df)**2
    exact_res['id']=0
    exact_res['time']=np.int64(np.arange(size_df))
    exact_res.index=exact_res["time"]
    
    ts_prep_rolling = ts_test1.prepare_rolling()
    
    pd.testing.assert_frame_equal(exact_res, ts_prep_rolling)
    return ts_prep_rolling


  
def test_roll_dataset():
    
    ts_test1.roll_dataset(timeshift=3, rolling_direction = 2)
    exact_res = pd.DataFrame()
    x = np.array([0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8])
    df=pd.DataFrame()
    
    df["id"]=np.int64(np.zeros(len(x),dtype=int))
    df["max_time"]= [2,2,2,4,4,4,6,6,6,8,8,8]
    
    
    exact_res.loc[:,0]=x**2
    
    exact_res.index=np.arange(len(x))
    exact_res["id"]=pd.MultiIndex.from_frame(df)
    exact_res['time']=np.int64(x)
    
    pd.testing.assert_frame_equal(exact_res, ts_test1.df_rolled)
    return ts_test1.df_rolled


def test_extract_features_df():
    ts_test1.extract_features_df(feature="maximum")
    exact_res = pd.DataFrame()
    exact_res["0__maximum"]= [4.,16.,36.,64.]
    pd.testing.assert_frame_equal(exact_res, ts_test1.extracted_features)
    return ts_test1.extracted_features



def test_select_relevant_windows():
    
    ts_prep_selected = ts_test1.prep_roll.copy()
    ts_test1.select_relevant_windows(percentage_max=0.24, timeshift=3, rolling_direction=2)
    ts_prep_selected.iloc[0:3, 0] =None  
    exact_res = ts_prep_selected
    pd.testing.assert_frame_equal(exact_res, ts_test1.relevant_windows) 
    return ts_test1.relevant_windows



def test_create_gridpoints1():
    
    ts_selected_test = ts_test1.relevant_windows.copy()
    ts_test1.create_gridpoints(n_gridpoints=2)
    exact_res = ts_selected_test.drop(0, axis=0)
    pd.testing.assert_frame_equal(exact_res, ts_test1.grid_points)
    return ts_test1.grid_points


#%%
def test_polyfit_gridpoints1():
    
    ts_grid_test= ts_test1.grid_points.copy()
    ts_test1.polyfit_gridpoints(order=1, verbose=False, n_gridpoints=2)  
    x= [-1,0,3,4]
    y= [0, 0, 9, 16]    
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    top_row= ts_grid_test.iloc[0,:]
    top_row.name=0.
    top_row= pd.DataFrame(top_row)
    top_row=top_row.transpose()
    top_row.iloc[0,:]=0
    
    ts_grid_test=pd.concat([top_row, ts_grid_test])
    ts_grid_test.iloc[1,0]=p(1)
    ts_grid_test.iloc[2,0]=p(2)
    exact_res= ts_grid_test
    exact_res["time"]=np.int64(exact_res["time"])
    exact_res.index= exact_res["time"]
    
    pd.testing.assert_frame_equal(exact_res, ts_test1.poly_gridpoints)
    return ts_test1.poly_gridpoints
  
#%%                 

def test_clean_dataset():

    ts_cleaned = tsig.TimeSignalPrep(ts_inp_test).clean_dataset(timeshift = 3, rolling_direction = 2, feature="maximum", n_gridpoints=2,percentage_max=0.24,order=1)    
    ts_test1.poly_gridpoints.pop("id")
    
    pd.testing.assert_frame_equal(ts_test1.poly_gridpoints, ts_cleaned)
    return ts_cleaned, 

    
#%% Test timeseries 2   
print("doing timeseries 2")
t = np.linspace(0, np.pi,20)
y1 = np.sin(4*t)+1
y2 = np.sin(t)+1
    
ts_gaps = pd.DataFrame(index=t,
          data=np.array([y1,y2]).T)
ts_gaps_test=ts_gaps.copy()

ts_test2 = tsig.TimeSignalPrep(ts_gaps)
ts_test2.prepare_rolling()
ts_test2.roll_dataset(timeshift= 5, rolling_direction = 3)
ts_test2.extract_features_df(feature="maximum")


def test_select_relevant_windows2():
    ts_gaps_selected = ts_test2.prep_roll.copy()
    ts_test2.select_relevant_windows(percentage_max=0.91, timeshift=5, rolling_direction=3)
    ts_gaps_selected.iloc[0:5,0:2] =np.nan
    ts_gaps_selected.iloc[3*2:3*2+5,0:2] =np.nan
    
    pd.testing.assert_frame_equal(ts_gaps_selected, ts_test2.relevant_windows) 
    return ts_test2.relevant_windows

#%%
def test_create_gridpoints2(): 
   
    ts_selected_gaps = ts_test2.relevant_windows.copy()
    ts_test2.create_gridpoints(n_gridpoints=3)
    list=[ts_selected_gaps.index[0],ts_selected_gaps.index[1],ts_selected_gaps.index[6],ts_selected_gaps.index[7]]
    exact_res = ts_selected_gaps.drop(list, axis=0)
    pd.testing.assert_frame_equal(exact_res, ts_test2.grid_points)
    return ts_test2.grid_points

#%%
def test_polyfit_gridpoints2():
    ts_test2.polyfit_gridpoints(order=3, verbose=False, n_gridpoints=3)
    
    ts_g= ts_test2.grid_points.copy()
    ts_time = np.linspace(0, np.pi+np.pi/19,21)
    y1_time = ts_time
    
    
    ts_time = pd.DataFrame(index=ts_time, data=np.array([y1_time]).T)
    ts_time= tsig.TimeSignalPrep(ts_time).prepare_rolling()

    top_row= ts_g.iloc[0,:]
    top_row.name=0.
    top_row= pd.DataFrame(top_row)
    top_row=top_row.transpose()
    top_row.iloc[0,:]=0
    
    ts_g=pd.concat([top_row, ts_g])
    
    ts_time = ts_time.head(len(ts_g))
    ts_g["time"]=ts_time["time"].values
    ts_g.index=ts_g["time"]

    #4 Polynome
    x1 = [ts_g.index[0]-ts_g.index[1] ,ts_g.index[0],ts_g.index[4], ts_g.index[5]]
    y1 = [0.,0.,ts_g.iloc[4,0],ts_g.iloc[4,0]]
    y2 = [0.,0.,ts_g.iloc[4,1],ts_g.iloc[4,1]]
    
    z1 = np.polyfit(x1, y1, 3)
    p1 = np.poly1d(z1)
    z2 = np.polyfit(x1, y2, 3)
    p2 = np.poly1d(z2)
    
    ts_g.iloc[1,0]=p1(ts_g.index[1])
    ts_g.iloc[2,0]=p1(ts_g.index[2])
    ts_g.iloc[3,0]=p1(ts_g.index[3])
    ts_g.iloc[1,1]=p2(ts_g.index[1])
    ts_g.iloc[2,1]=p2(ts_g.index[2])
    ts_g.iloc[3,1]=p2(ts_g.index[3])

    x3 = [ts_g.index[3],ts_g.index[4],ts_g.index[8],ts_g.index[9]]
    y3 = [ts_g.iloc[3,0], ts_g.iloc[4,0], ts_g.iloc[8,0], ts_g.iloc[9,0]]
    y4 = [ts_g.iloc[3,1], ts_g.iloc[4,1], ts_g.iloc[8,1], ts_g.iloc[9,1]]
     
    z3 = np.polyfit(x3, y3, 3)
    p3 = np.poly1d(z3)
    z4 = np.polyfit(x3, y4, 3)
    p4 = np.poly1d(z4)
    
    ts_g.iloc[5,0]=p3(ts_g.index[5])
    ts_g.iloc[6,0]=p3(ts_g.index[6])
    ts_g.iloc[7,0]=p3(ts_g.index[7])
    ts_g.iloc[5,1]=p4(ts_g.index[5])
    ts_g.iloc[6,1]=p4(ts_g.index[6])
    ts_g.iloc[7,1]=p4(ts_g.index[7])

    exact_res= ts_g
    exact_res.index= exact_res["time"]
    
    pd.testing.assert_frame_equal(exact_res, ts_test2.poly_gridpoints)
    return ts_test2.poly_gridpoints

#%%
def test_clean_dataset2():
    ts_test2.poly_gridpoints.pop("id")
    exact_res = ts_test2.poly_gridpoints
    ts_cleaned = tsig.TimeSignalPrep(ts_gaps_test).clean_dataset(timeshift = 5, rolling_direction = 3, feature="maximum", n_gridpoints=3,percentage_max=0.91,order=3)    
    pd.testing.assert_frame_equal(exact_res, ts_cleaned)
    return ts_cleaned


#%% Test timeseries 3  
print("doing timeseries 3") 
t1 = np.linspace(0, 2*np.pi,1000,endpoint=False)
t2 = np.linspace(2*np.pi, 4*np.pi,1000)
y1 = 100 * np.cos(8*t1)
y2 = 0.1* np.sin(8*t2)
t = np.append(t1, t2)
y = np.append(y1, y2)
ts_sin = pd.DataFrame(index=t,
          data=y)


#%%
def test_clean_dataset3():
    
    
    
    ts_sin_test=ts_sin.copy()
    ts_cleaned = tsig.TimeSignalPrep(ts_sin_test).clean_dataset(timeshift = 100, rolling_direction = 1, feature="maximum", n_gridpoints=5,percentage_max=0.1,order=3)
    #ts_cleaned.plot(subplots=True, sharex=True, figsize=(10,10))
    #plt.show()
    ts_time=tsig.TimeSignalPrep(ts_sin).prepare_rolling()
    
    exact_res= pd.DataFrame(index=t1, data=y1)
    exact_res = tsig.TimeSignalPrep(exact_res).prepare_rolling()
    
    top_row= exact_res.iloc[0,:]
    top_row.name=0.
    top_row= pd.DataFrame(top_row)
    
    top_row=top_row.transpose()
    top_row.iloc[0,:]=0
    
    exact_res=pd.concat([top_row, exact_res])
    
    ts_time = ts_time.head(len(exact_res))
    exact_res["time"]=ts_time["time"].values
    exact_res.index=exact_res["time"]
    exact_res.pop("id")
    pd.testing.assert_frame_equal(exact_res, ts_cleaned)
    return ts_cleaned
  


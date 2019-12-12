# Copyright (c) 2019 - for information on the respective copyright owner
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

__author__ = "Mustapha Kassem"
__maintainer__ = "Johannes Mueller"

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display
from os import path
import io
import numpy as np

def excel_upload():
    file_name = widgets.FileUpload(
                accept='.xlsx',  # Accepted file extension
                multiple=False  # True to accept multiple files upload else False
            )
    return file_name


def data_head_tail():
    w = widgets.RadioButtons(
        options=['Head of the data', 'Details of the data'],
        description='Visualization:',
        disabled=False,
        style={'description_width': 'initial'}
        )

    return w

def method_mali_probit():
    w2 = widgets.RadioButtons(
        options=['Mali', 'Probit'],
        description='Visualization',
        disabled=False,
        style={'description_width': 'initial'}
        )

    return w2


def k_1_def():
    w3 = widgets.RadioButtons(
        options=[('Fractures', "fractures"), ('Finite-life zone', "zone_fin")],
        description='Data points',
        disabled=False,
        style={'description_width': 'initial'}
    )

    return w3


def WL_param():

    tab_contents = ['k_1', '1/TN', 'SD_50', '1/TS', 'ND_50']
    items_a = ['Mali k_1', 'Mali 1/TN', 'Mali SD_50','Mali 1/TS', 'Mali ND_50']
    children = [widgets.Text(description=name) for name in tab_contents]
    tab = widgets.Tab()
    tab.children = children
    for i in range(len(children)):
        tab.set_title(i, items_a[i])


    return tab


def WL_param_display(tab):

    print('Maximum Likelihood Method:\n')
    print('Estimated Parameters:')

    for i in range(len(tab.children)):
        if tab.children[i].value=='':
            print(tab.children[i].description)

    fixed_param ={}
    estim_param={}
    print('\nFixed Parameters:')
    for i in range(len(tab.children)):
        if tab.children[i].value!='':
            fixed_param[tab.children[i].description] = float(tab.children[i].value)
            print(tab.children[i].description + ' = ' + tab.children[i].value)
        else:
            estim_param[tab.children[i].description] = tab.children[i].value

    return fixed_param, estim_param

 #+ '\033[91m' +'\033[1m' +
def results_mali_probit(WC_data, fixed_param):

    print('Maximum Likelihood %d Param method:\n'% len(WC_data.p_opt))

    if 'SD_50' in fixed_param:
        print ('Endurance SD50 = ', np.round(WC_data.Mali_5p_result['SD_50'],decimals=1))
    else:
        print ('\033[91m' +'\033[1m' + 'Endurance SD50 = ' + '\033[1;34m'+ str(np.round(WC_data.Mali_5p_result['SD_50'],decimals=1)))
    if 'ND_50' in fixed_param:
        print ('\033[0;0m' + 'Endurance load-cycle ND50 = ' + str('{:1.2e}'.format(WC_data.Mali_5p_result['ND_50'])))
    else:
        print ('\033[91m' +'\033[1m' +'Endurance load-cycle ND50 = ' + '\033[1;34m'+  str('{:1.2e}'.format(WC_data.Mali_5p_result['ND_50'])))
    if '1/TS' in fixed_param:
        print ('\033[0;0m' + 'Deviation in load direction 1/TS = ' + str(np.round(WC_data.Mali_5p_result['1/TS'],decimals=2)))
    else:
        print ('\033[91m' +'\033[1m' +'Deviation in load direction 1/TS = '+ '\033[1;34m'+  str(np.round(WC_data.Mali_5p_result['1/TS'],decimals=2)))
    if 'k_1' in fixed_param:
        print ('\033[0;0m' + 'Slope k = ' + str(np.round(WC_data.Mali_5p_result['k_1'],decimals=2)))
    else:
        print ('\033[91m' +'\033[1m' + 'Slope k = '+ '\033[1;34m'+ str(np.round(WC_data.Mali_5p_result['k_1'],decimals=2)))
    if '1/TN' in fixed_param:
        print ('\033[0;0m' + 'Deviation in load-cycle direction 1/TN = ' + str(np.round(WC_data.Mali_5p_result['1/TN'],decimals=2)))
    else:
        print ('\033[91m' +'\033[1m' + 'Deviation in load-cycle direction 1/TN = '+ '\033[1;34m'+  str(np.round(WC_data.Mali_5p_result['1/TN'],decimals=2)))

    print ('\033[0;0m' + '\n------ Results Maximum Likelihood 2 Param method -------')
    print ('Endurance SD50 =', np.round(WC_data.Mali_2p_result['SD_50'],decimals=1))
    print ('Endurance load-cycle ND50 =', '{:1.2e}'.format(WC_data.Mali_2p_result['ND_50']))
    print ('Deviation in load direction 1/TS_mali =', np.round(WC_data.Mali_2p_result['1/TS'],decimals=2))

    print('\033[0;0m' + '\n------ Slope using linear regression -------')
    print('Slope K_1 = '+str(np.round(WC_data.k, decimals=2)))

    print('\n------ Deviation 1/TN using pearl-chain method -------')
    print('Deviation in load-cycle direction 1/TN =', np.round(WC_data.TN, decimals=2))
    print('Deviation in load direction (Mali köder) 1/TS* =', np.round(WC_data.TS, decimals=2))

    print('\n------ Results Probit-Method -------')
    if len(WC_data.ld_lvls_inf[0])<2:
        print("Not enough load levels in the infinite zone for the probit method")
    else:
        print('Endurance SD50 =', np.round(WC_data.Probit_result['SD_50'], decimals=1))
        print('Endurance load-cycle ND50 =', '{:1.2e}'.format(WC_data.Probit_result['ND_50']))
        print('Deviation in load direction 1/TS =', np.round(WC_data.Probit_result['1/TS'], decimals=2))

#    if not len(WC_data.ld_lvls_inf[0])<2 and WC_data.Probit_result['1/TS']<10:
#        print('\n------ Results Probit-Method -------')
#        print('Endurance SD50 =', np.round(WC_data.Probit_result['SD_50'], decimals=1))
#        print('Endurance load-cycle ND50 =', '{:1.2e}'.format(WC_data.Probit_result['ND_50']))
#        print('Deviation in load direction 1/TS =', np.round(WC_data.Probit_result['1/TS'], decimals=2))


def results_visual():
    w4 = widgets.RadioButtons(
        options=[('Initial data', 'Initial data'), ('Slope', 'Slope'),
                 ('Pearl chain method', 'Pearl chain method'),
                 ('Probability plot of the finite zone','Probability plot of the finite zone'),
                 ('Deviation in load-cycle direction','Deviation TN'),
                 ('Probability plot of the infinite zone', 'Probability plot of the infinite zone')],
        value= 'Initial data',
        description='Plot Type',
        disabled=False,
        style={'description_width': 'initial'}
        )

    return w4

def inf_plot(WC_data, k_1):
    w5 = widgets.RadioButtons(
        options=[('k_2 = 0', 0), ('k_2 = k_1', k_1), ('k_2 = 2 k_1 - 1', 2*k_1-1)],
        value= 0,
        description='runout-zone plot',
        disabled=False,
        style={'description_width': 'initial'}
    )

    return w5

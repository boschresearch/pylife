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
from pylife.materialdata.woehler.probability_plot_creator import ProbabilityPlotCreator

class WoehlerWidget:
    
    @staticmethod
    def excel_upload():
        file_name = widgets.FileUpload(
                    accept='.xlsx',  # Accepted file extension
                    multiple=False  # True to accept multiple files upload else False
                )
        return file_name

    @staticmethod
    def data_head_tail():
        w = widgets.RadioButtons(
            options=['Head of the data', 'Details of the data'],
            description='Visualization:',
            disabled=False,
            style={'description_width': 'initial'}
            )

        return w

    @staticmethod
    def method_mali_probit():
        w2 = widgets.RadioButtons(
            options=['Mali', 'Probit'],
            description='Visualization',
            disabled=False,
            style={'description_width': 'initial'})

        return w2
    
    @staticmethod
    def k_1_def(): 
        w3 = widgets.RadioButtons(
            options=[('Fractures', "fractures"), ('Finite-life zone', "zone_fin")],
            description='Data points',
            disabled=False,
            style={'description_width': 'initial'})

        return w3

    @staticmethod
    def WL_param():

        tab_contents = ['k_1', '1/TN', 'SD_50', '1/TS', 'ND_50']
        items_a = ['Mali k_1', 'Mali 1/TN', 'Mali SD_50','Mali 1/TS', 'Mali ND_50']
        children = [widgets.Text(description=name) for name in tab_contents]
        tab = widgets.Tab()
        tab.children = children
        for i in range(len(children)):
            tab.set_title(i, items_a[i])


        return tab

    @staticmethod
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

    @staticmethod
    def print_mali_5p_result(woehler_curve, fixed_param):
        print('Maximum Likelihood %d Param method:\n'% len(woehler_curve.p_opt))

        if 'SD_50' in fixed_param:
            print ('Endurance SD50 = ', np.round(woehler_curve.curve_parameters['SD_50'],decimals=1))
        else:
            print ('\033[91m' +'\033[1m' + 'Endurance SD50 = ' + '\033[1;34m'+ str(np.round(woehler_curve.curve_parameters['SD_50'],decimals=1)))
            
        if 'ND_50' in fixed_param:
            print ('\033[0;0m' + 'Endurance load-cycle ND50 = ' + str('{:1.2e}'.format(woehler_curve.curve_parameters['ND_50'])))
        else:
            print ('\033[91m' +'\033[1m' +'Endurance load-cycle ND50 = ' + '\033[1;34m'+  str('{:1.2e}'.format(woehler_curve.curve_parameters['ND_50'])))
            
        if '1/TS' in fixed_param:
            print ('\033[0;0m' + 'Deviation in load direction 1/TS = ' + str(np.round(woehler_curve.curve_parameters['1/TS'],decimals=2)))
        else:
            print ('\033[91m' +'\033[1m' +'Deviation in load direction 1/TS = '+ '\033[1;34m'+  str(np.round(woehler_curve.curve_parameters['1/TS'],decimals=2)))
            
        if 'k_1' in fixed_param:
            print ('\033[0;0m' + 'Slope k = ' + str(np.round(woehler_curve.curve_parameters['k_1'],decimals=2)))
        else:
            print ('\033[91m' +'\033[1m' + 'Slope k = '+ '\033[1;34m'+ str(np.round(woehler_curve.curve_parameters['k_1'],decimals=2)))
            
        if '1/TN' in fixed_param:
            print ('\033[0;0m' + 'Deviation in load-cycle direction 1/TN = ' + str(np.round(woehler_curve.curve_parameters['1/TN'],decimals=2)))
        else:
            print ('\033[91m' +'\033[1m' + 'Deviation in load-cycle direction 1/TN = '+ '\033[1;34m'+  str(np.round(woehler_curve.curve_parameters['1/TN'],decimals=2)))

    
    @staticmethod
    def print_mali_2p_result(woehler_curve):
        print ('\033[0;0m' + '\n------ Results Maximum Likelihood 2 Param method -------')
        print ('Endurance SD50 =', np.round(woehler_curve.Mali_2p_result['SD_50'],decimals=1))
        print ('Endurance load-cycle ND50 =', '{:1.2e}'.format(woehler_curve.curve_parameters['ND_50']))
        print ('Deviation in load direction 1/TS_mali =', np.round(woehler_curve.curve_parameters['1/TS'],decimals=2))    
        
    @staticmethod
    def print_slope(woehler_curve): 
        print('\033[0;0m' + '\n------ Slope using linear regression -------')
        print('Slope K_1 = '+str(np.round(woehler_curve.fatigue_data.k, decimals=2)))
        
    @staticmethod
    def print_deviation_results(woehler_curve):        
        print('\n------ Deviation 1/TN using pearl-chain method -------')
        print('Deviation in load-cycle direction 1/TN =', np.round(woehler_curve.TN, decimals=2))
        print('Deviation in load direction (Mali köder) 1/TS* =', np.round(woehler_curve.TS, decimals=2))
    
    @staticmethod
    def print_probit_results(woehler_curve):
        print('\n------ Results Probit-Method -------')
        if len(woehler_curve.ld_lvls_inf[0])<2:
            print("Not enough load levels in the infinite zone for the probit method")
        else:
            print('Endurance SD50 =', np.round(woehler_curve.Probit_result['SD_50'], decimals=1))
            print('Endurance load-cycle ND50 =', '{:1.2e}'.format(woehler_curve.Probit_result['ND_50']))
            print('Deviation in load direction 1/TS =', np.round(woehler_curve.Probit_result['1/TS'], decimals=2))

    @staticmethod
    def results_visual_woehler_curve():
        w4 = widgets.RadioButtons(
            options=[('Initial data', 'Initial data'), 
                     ('Slope', 'Slope'),
                     ('Pearl chain method', 'Pearl chain method'),
                     ('Deviation in load-cycle direction','Deviation TN')],
            value= 'Initial data',
            description='Plot Type',
            disabled=False,
            style={'description_width': 'initial'})

        return w4
              
    @staticmethod     
    def on_results_visual_probability_plot_selection_changed(change):
        switcher = {options[0][0]: ProbabilityPlotCreator.probability_plot_finite,
                    options[1][0]: ProbabilityPlotCreator.probability_plot_inifinite}
        
        if change['new'] == 'Probability plot of the finite zone':
            ProbabilityPlotCreator.probability_plot_finite()
        else:
            ProbabilityPlotCreator.probability_plot_inifinite()
        out = widgets.Output()
        with out:
            if change['type'] == 'change' and change['name'] == 'value':
                print("changed to %s" % change['new'])
                       
    @staticmethod
    def results_visual_probability_plot():
        w4 = widgets.RadioButtons(
            options=[('Probability plot of the finite zone','Probability plot of the finite zone'),
                     ('Probability plot of the infinite zone', 'Probability plot of the infinite zone')],
            value= 'Probability plot of the finite zone',
            description='Plot Type',
            disabled=False,
            style={'description_width': 'initial'})
        #w4.observe(WoehlerWidget.on_results_visual_probability_plot_selection_changed, names = 'value')
        return w4   

    @staticmethod
    def inf_plot(woehler_curve, k_1):
        w5 = widgets.RadioButtons(
            options=[('k_2 = 0', 0), ('k_2 = k_1', k_1), ('k_2 = 2 k_1 - 1', 2*k_1-1)],
            value= 0,
            description='runout-zone plot',
            disabled=False,
            style={'description_width': 'initial'}
        )

        return w5












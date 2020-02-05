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

import ipywidgets as widgets
from IPython.display import display, clear_output
import sys, os


sys.path.insert(0, os.path.abspath('..\\pylife'))

from pylife.materialdata.woehler.plot_woehler_curve_data import PlotWoehlerCurveData

from pylife.materialdata.woehler.fatigue_data import FatigueData
from pylife.materialdata.woehler.woehler_curve_creator import WoehlerCurveCreator
from pylife.materialdata.woehler.probability_plot_creator import ProbabilityPlotCreator
from pylife.materialdata.woehler.plot_woehler_curve_data import PlotWoehlerCurveData
from pylife.materialdata.woehler.plot_probability_plot_finite import PlotProbabilityPlotFinite
from pylife.materialdata.woehler.plot_probability_plot_infinite import PlotProbabilityPlotInfinite
from pylife.materialdata.woehler.woehler_curve_widgets import WoehlerCurveWidgets
from pylife.materialdata.woehler.radio_button_file_display import RadioButtonFileDisplay

class WoehlerCurveDataPlotter:
    def __init__(self, woehler_curve):
        self.woehler_curve = woehler_curve
        self.plot_woehler_curve_data = PlotWoehlerCurveData(self.woehler_curve)
        self.plot_woehler_curve_data.plot_initial_data()
        self.radio_button = widgets.RadioButtons(options=['Initial data', 'Slope', 'Pearl chain method', 'Deviation in load-cycle direction'], 
                                                 description='Plot Type',  style={'description_width': 'initial'})

        self.radio_button.observe(self.selection_changed_handler, names = 'value')
        display(self.radio_button)
        
    def selection_changed_handler(self, change):
        clear_output()
        display(self.radio_button)
        if change['new'] == change.owner.options[0]:
            self.plot_woehler_curve_data.plot_initial_data()
        elif change['new'] == change.owner.options[1]:
            self.plot_woehler_curve_data.plot_slope()
        elif change['new'] == change.owner.options[2]:
            self.plot_woehler_curve_data.plot_pearl_chain_method()
        elif change['new'] == change.owner.options[3]:
            self.plot_woehler_curve_data.plot_deviation()
        else:
            raise AttributeError('Unexpected selection')

# if __name__ == "__main__":
#     import pandas as pd
#     file_name = 'woehler-test-data.csv'
#     data = pd.read_csv(file_name, sep='\t')
#     data.columns=['loads', 'cycles']
#     ld_cyc_lim = data['cycles'].max()
#     fatigue_data = FatigueData(data, ld_cyc_lim)
#     woehler_curve_creator = WoehlerCurveCreator(fatigue_data)
#     fixed_param = []
#     woehler_curve = woehler_curve_creator.maximum_like_procedure(fixed_param)
#     woehler_curve_data_plotter = WoehlerCurveDataPlotter(woehler_curve)



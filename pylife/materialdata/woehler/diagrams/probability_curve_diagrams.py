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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pylife.materialdata.woehler.curves.probability_curve import ProbabilityCurve
from abc import ABC, abstractmethod
from scipy import stats

class ProbabilityCurveDiagrams(ABC):
    def __init__(self, probability_curve):
        self.probability_curve = probability_curve
        #self.ax = self.__default_ax_config() if ax is None else ax

    @abstractmethod
    def get_scatter_label(self):
        pass
    
    @abstractmethod
    def get_xlabel(self):
        pass

    @abstractmethod   
    def get_title(self):
        pass
        
    def __default_ax_config(self):
        a = self.probability_curve.a
        b = self.probability_curve.b
        T = self.probability_curve.T
        xlabel = self.get_xlabel()
        scatter = self.get_scatter_label()
        title = self.get_title()

        fig, ax = plt.subplots()
        
        yticks = [1, 5, 10, 20, 50, 80, 90, 95, 99]
        ax.set_yticks([stats.norm.ppf(i/100.) for i in yticks])
        ax.set_yticklabels([str(i)+' %' for i in yticks])

        ax.set_xscale('log')    
        ax.grid(True)    
        
        ax.set_xlabel(xlabel)		
        ax.set_ylabel('Failure probability')

        ax.set_title(title)
        ax.text(0.15, 0.03, 'N($P_{A,10}$)='+'{:1.1e}'.format(10**((stats.norm.ppf(0.1)-b)/a), decimals=1),
            verticalalignment='bottom',horizontalalignment='left', transform=ax.transAxes,
            bbox={'facecolor':'grey', 'alpha':0.1, 'pad':10}, fontsize=11)
        ax.text(0.5, 0.88, scatter + str(np.round(T,decimals=2)),
            verticalalignment='bottom',horizontalalignment='center', transform=ax.transAxes,
            bbox={'facecolor':'grey', 'alpha':0.2, 'pad':10}, fontsize=11)
        ax.text(0.9, 0.03, 'N($P_{A,90}$)='+'{:1.1e}'.format(10**((stats.norm.ppf(0.9)-b)/a),decimals=1),
                 verticalalignment='bottom',horizontalalignment='right', transform=ax.transAxes,
                 bbox={'facecolor':'grey', 'alpha':0.1, 'pad':10}, fontsize=11)

        fig.tight_layout()
        return ax

    
    def plot_probability_curve_diagram(self, ax = None):
        ax_local = self.__default_ax_config() if ax is None else ax
        ax_local.plot(self.probability_curve.X, self.probability_curve.Y, 'ro')
        ax_local.plot([10**((i-self.probability_curve.b)/self.probability_curve.a) for i in np.arange(-2.5, 2.5, 0.1)], np.arange(-2.5, 2.5, 0.1), 'r')

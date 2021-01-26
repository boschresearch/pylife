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

__author__ = "Mustapha Kassem"
__maintainer__ = "Johannes Mueller"

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats


class ProbabilityDataDiagram:
    def __init__(self, probability_data, **kw):
        self._prob_data = probability_data
        self._occurrences_name = 'Occurrences' if 'occurrences_name' not in kw.keys() else kw['occurrences_name']
        self._probability_name = 'Probability' if 'probability_name' not in kw.keys() else kw['probability_name']
        self._title = 'Probability plot' if 'title' not in kw.keys() else kw['title']

    @property
    def occurrences_name(self):
        return self._occurrences_name

    @occurrences_name.setter
    def occurrences_name(self, name):
        self._occurrences_name = name

    @property
    def probability_name(self):
        return self._probability_name

    @probability_name.setter
    def probability_name(self, name):
        self._probability_name = name

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, name):
        self._title = name

    def __default_ax_config(self):
        fig, ax = plt.subplots()

        yticks = [1, 5, 10, 20, 50, 80, 90, 95, 99]
        ax.set_yticks([stats.norm.ppf(i/100.) for i in yticks])
        ax.set_yticklabels([str(i)+' %' for i in yticks])

        ax.set_xscale('log')
        ax.grid(True)

        ax.set_xlabel(self._occurrences_name)
        ax.set_ylabel(self._probability_name)

        ax.set_title(self._title)
        return ax

    def plot(self, ax=None):
        ax_local = self.__default_ax_config() if ax is None else ax
        ax_local.plot(self._prob_data.occurrences, self._prob_data.percentiles, 'ro')
        ax_local.plot([10**((i-self._prob_data.intercept)/self._prob_data.slope) for i in np.arange(-2.5, 2.5, 0.1)], np.arange(-2.5, 2.5, 0.1), 'r')

        return ax_local

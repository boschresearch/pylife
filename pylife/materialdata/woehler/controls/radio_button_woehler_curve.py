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

__author__ = "Gyöngyvér Kiss"
__maintainer__ = __author__

from abc import ABC, abstractmethod
from IPython.display import display, clear_output
import ipywidgets as widgets


class RadioButtonWoehlerCurve(ABC):
    def __init__(self, options, description):
        self.radio_button = widgets.RadioButtons(options=options, description=description, disabled=False, style={'description_width': 'initial'})
        self.radio_button.observe(self.selection_changed_handler, names = 'value')
        display(self.radio_button)

    @abstractmethod
    def selection_changed_handler(self, change):
        pass

    def clear_selection_change_output(self):
        clear_output()
        display(self.radio_button)

    


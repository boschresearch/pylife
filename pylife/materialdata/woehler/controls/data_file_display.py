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

from IPython.display import display

from pylife.materialdata.woehler.controls.radio_button_woehler_curve import RadioButtonWoehlerCurve

class DataFileDisplay(RadioButtonWoehlerCurve):
    def __init__(self, data):
        super().__init__(['Head of the data', 'Details of the data'], 'File display selection')
        self.data = data
        display(self.data.head())

    def selection_changed_handler(self, change):
        self.clear_selection_change_output()

        if change['new'] == change.owner.options[0]:
            display(self.data.head())
        elif change['new'] == change.owner.options[1]:
            display(self.data.describe())
        else:
            raise AttributeError('Unexpected selection')


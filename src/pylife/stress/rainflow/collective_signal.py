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

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import pandas as pd
import numpy as np

from pylife import PylifeSignal

@pd.api.extensions.register_dataframe_accessor('rainflow')
class RainflowCollective(PylifeSignal):
    """A Rainflow collective.

    The usual use of this signal is to process hysteresis loop data from a
    rainflow recording.  Usually the keys ``from`` and ``to`` are used to
    describe the hysteresis loops.  Alternatively also the keys ``range`` and
    ``mean`` can be given.  In that case the frame is internally converted to
    ``from`` and ``to`` where the ``from`` values are the lower ones.
    """

    def _validate(self):
        if 'from' in self.keys() and 'to' in self.keys():
            return
        if 'range' in self.keys() and 'mean' in self.keys():
            fr = self._obj['mean'] - self._obj['range'] / 2.
            to = self._obj['mean'] + self._obj['range'] / 2.
            self._obj = pd.DataFrame({
                'from': fr,
                'to': to
            }, index=self._obj.index)
            return
        raise AttributeError("Rainflow needs either 'range'/'mean' or 'from'/'to' in column names.")

    @property
    def amplitude(self):
        """Calculate the amplitudes of the load collective.

        Returns
        -------
        amplitude : pd.Series
            The amplitudes of the load collective
        """
        fr = self._obj['from']
        to = self._obj['to']
        rng = np.abs(fr-to)

        return pd.Series(rng/2., name='amplitude', index=self._obj.index)

    @property
    def meanstress(self):
        """Calculate the mean load values of the load collective.

        Returns
        -------
        mean : pd.Series
            The mean load values of the load collective
        """
        fr = self._obj['from']
        to = self._obj['to']
        return pd.Series((fr+to)/2., name='meanstress')

    @property
    def upper(self):
        """Calculate the upper load values of the load collective.

        Returns
        -------
        upper : pd.Series
            The upper load values of the load collective
        """
        res = self._obj.max(axis=1)
        res.name = 'upper'
        return res

    @property
    def lower(self):
        """Calculate the lower load values of the load collective.

        Returns
        -------
        lower : pd.Series
            The lower load values of the load collective
        """
        res = self._obj.min(axis=1)
        res.name = 'lower'
        return res

    @property
    def cycles(self):
        """The cycles of each member of the collective is 1.0.

        This is for compatibility with :class:`pylife.stress.rainflow.RainflowMatrix`
        """
        return pd.Series(1.0, name='cycles', index=self._obj.index)

    def scale(self, factors):
        """Scale the collective.

        Parameters
        ----------
        factors : scalar or :class:`pandas.Series`
            The factor(s) to scale the collective with.
        """
        factors, obj = self.broadcast(factors)
        return obj.multiply(factors, axis=0).rainflow

    def shift(self, diffs):
        """Shift the collective.

        Parameters
        ----------
        diffs : scalar or :class:`pandas.Series`
            The diff(s) to shift the collective by.
        """
        diffs, obj = self.broadcast(diffs)
        return obj.add(diffs, axis=0).rainflow

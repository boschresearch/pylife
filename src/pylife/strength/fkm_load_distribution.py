# Copyright (c) 2019-2023 - for information on the respective copyright owner
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

r"""Scale a load sequence for FKM nonlinear load distribution safety.

The accessors in this module multiply a stress or load sequence by the FKM
nonlinear load factor :math:`\gamma_L`.  The factor accounts for the assumed
load distribution, the assessment failure probability :math:`P_A`, and the
load probability :math:`P_L` of either ``2.5`` percent or ``50`` percent.

The FKM nonlinear guideline defines three possible methods to consider the
statistical distribution of the load:

* Normal distribution with standard deviation :math:`s_L`.
* Lognormal distribution with logarithmic standard deviation :math:`LSD_s`.
* Unknown distribution, using :math:`\gamma_L = 1.1` for
  :math:`P_L = 2.5\%`.

The corresponding accessors are ``fkm_safety_normal_from_stddev``,
``fkm_safety_lognormal_from_stddev``, and ``fkm_safety_blanket``.

The resulting scaling factor can be retrieved with
``.gamma_L(input_parameters)``, the scaled load series can be obtained with
``.scaled_load_sequence(input_parameters)``.

Notes
-----
The statistical load factors implement FKM nonlinear guideline clause 2.3.2.

Examples
--------
>>> input_parameters = pd.Series({"P_A": 1e-5, "P_L": 50, "s_L": 10, "LSD_s": 1e-2,})
>>> load_sequence = pd.Series([100.0, 150.0, 200.0], name="load")
>>> # uses input_parameters.s_L, input_parameters.P_L, input_parameters.P_A
>>> load_sequence.fkm_safety_normal_from_stddev.scaled_load_sequence(input_parameters)
0    114.9450
1    172.4175
2    229.8900
Name: load, dtype: float64

>>> # uses input_parameters.s_L, input_parameters.P_L, input_parameters.P_A
>>> load_sequence.fkm_safety_lognormal_from_stddev.scaled_load_sequence(input_parameters)
0    107.124794
1    160.687191
2    214.249588
Name: load, dtype: float64

>>> # uses input_parameters.P_L
>>> load_sequence.fkm_safety_blanket.scaled_load_sequence(input_parameters)
0    100.0
1    150.0
2    200.0
Name: load, dtype: float64
"""

__author__ = "Benjamin Maier"
__maintainer__ = __author__

import numpy as np
import pandas as pd
from pylife import PylifeSignal

@pd.api.extensions.register_dataframe_accessor("fkm_load_sequence")
@pd.api.extensions.register_series_accessor("fkm_load_sequence")
class FKMLoadSequence(PylifeSignal):
    r"""Scale load data by a constant FKM load factor.

    This accessor is available as ``.fkm_load_sequence`` on
    :class:`pandas.Series` and :class:`pandas.DataFrame` objects.  It is the
    shared base for the FKM statistical load distribution accessors and can also
    be used directly when a factor :math:`\gamma_L` is already known.

    Signal contract:

    * A :class:`pandas.Series` represents one scalar load value per load step.
    * A :class:`pandas.DataFrame` with one column represents one scalar load
      value per index row.
    * A :class:`pandas.DataFrame` with multiple columns stores the load in the
      first column; additional columns, such as a stress gradient, are copied
      unchanged by :meth:`scaled_by_constant`.
    * The object must not be empty.

    Parameters
    ----------
    pandas_obj : pandas.Series or pandas.DataFrame
        Load sequence, stress sequence, or mesh-indexed load table to scale.
        Load or stress values are interpreted in the units used by the
        subsequent assessment, typically MPa for stresses.

    Notes
    -----
    Mesh-like data are commonly indexed by ``load_step`` and ``node_id``.  The
    scaling methods preserve the original index and all non-load columns.
    """


    def scaled_by_constant(self, gamma_L):
        """Scale the load sequence by the given constant ``gamma_L``.

        This method basically computes ``gamma_L * self._obj``. The data in
        ``self._obj`` is either a :class:`pandas.Series`, a
        :class:`pandas.DataFrame` with a single column or a pandas.DataFrame
        with multiple columns (usually two for stress and stress gradient).  In
        the case of a Series or only one column, it simply scales all values by
        the factor ``gamma_L``.  In the case of a DataFrame with multiple columns,
        it only scales the first column by the factor gamma_L and keeps the
        other columns unchanged.

        Returns a scaled copy of the data.

        Parameters
        ----------
        gamma_L : float
            Scaling factor for the load sequence, dimensionless.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Scaled copy of the load sequence.  For a multi-column
            :class:`pandas.DataFrame`, only the first column is multiplied by
            ``gamma_L``.
        """

        # `self._obj` can be either a pd.Series or a pd.DataFrame. A gradient can only be included if we have a pd.DataFrame
        if isinstance(self._obj, pd.DataFrame):

            # if the number of columns in the DataFrame is at least two (for the stress and the gradient)
            if len(self._obj.columns) >= 2:

                # only scale the first column
                result = self._obj.copy().astype(np.float64)
                result.iloc[:, 0] = result.iloc[:, 0] * gamma_L
                return result

        return self._obj * gamma_L

    def maximum_absolute_load(self, max_load_independently_for_nodes=False):
        """Get the maximum absolute load over all nodes and load steps.

        This is implemented for pd.Series (where the index is just the index of
        the load step), pd.DataFrame with one column (where the index is a
        MultiIndex of load_step and node_id), and pd.DataFrame with multiple
        columns with the load is given in the first column.

        Parameters
        ----------
        max_load_independently_for_nodes : bool, optional
            Flag indicating whether to compute the maximum absolute load
            separately for every node.  If set to ``False``, a single maximum
            value is computed over all nodes.  Default is ``False``.

        Returns
        -------
        float
            Maximum absolute load.  The unit is the same as the load sequence,
            typically MPa for stress values.
        """

        # if the load sequence is a pd.Series
        if len(self._obj.index.names) == 1:
            L_max = max(abs(self._obj))

        # if the load sequence is a pd.DataFrame
        else:
            # if we have a multi-indexed DataFrame with (load_step, node_id)
            if "node_id" in self._obj.index.names:
                L_max = self._obj.abs().groupby("node_id").max()
            else:
                levels = list(range(self._obj.index.nlevels - 1))  # all but the last level
                L_max = self._obj.abs().groupby(level=levels).max()

            # if there are multiple columns, select the first one
            if isinstance(L_max, pd.DataFrame) and len(L_max.columns) > 1:
                L_max = L_max.iloc[:,0]

            if max_load_independently_for_nodes:
                return L_max

                # take maximum over all load steps
            L_max = L_max.max()

        if isinstance(L_max, pd.DataFrame) or isinstance(L_max, pd.Series):
            L_max = L_max.squeeze()

        return float(L_max)

    def _validate(self):
        if len(self._obj) == 0:
            raise AttributeError("Load series is empty.")

    def _validate_parameters(self, input_parameters, required_parameters):

        for required_parameter in required_parameters:
            if required_parameter not in input_parameters:
                raise ValueError(f"Given parameters have to include \"{required_parameter}\".")

    def _get_beta(self, input_parameters):
        r"""Compute the reliability index for the assessment failure probability.

        For details, refer to the FKM nonlinear document.

        The beta factors are also described in "A. Fischer. Bestimmung
        modifizierter Teilsicherheitsbeiwerte zur semiprobabilistischen
        Bemessung von Stahlbetonkonstruktionen im Bestand. TU Kaiserslautern,
        2010"

        Parameters
        ----------
        input_parameters : pd.Series
            Assessment parameters.  The series must contain ``P_A``, the
            assessment failure probability as a dimensionless value.  Supported
            values are ``1e-7``, ``1e-6``, ``1e-5``, ``7.2e-5``, ``1e-3``,
            ``2.3e-1``, and ``0.5``.

        Returns
        -------
        float
            Reliability index :math:`\beta`, dimensionless.

        Raises
        ------
        ValueError
            If ``P_A`` is not one of the tabulated probabilities.
        """

        # list of predefined P_A and beta values
        P_A_beta_list = [(1e-7, 5.20), (1e-6, 4.75), (1e-5, 4.27), (7.2e-5, 3.8), (1e-3, 3.09), (2.3e-1, 0.739), (0.5, 0)]

        # check if given P_A values is close to any of the tabulated predefined values
        for P_A, beta in P_A_beta_list:
            if np.isclose(input_parameters.P_A, P_A):
                return beta

        # raise error if the given P_A value is not among the known ones
        P_A_list = [str(P_A) for P_A,_ in P_A_beta_list]
        raise ValueError(f"P_A={input_parameters.P_A} has to be one of "+"{"+", ".join(P_A_list)+"}.")


@pd.api.extensions.register_dataframe_accessor("fkm_safety_normal_from_stddev")
@pd.api.extensions.register_series_accessor("fkm_safety_normal_from_stddev")
class FKMLoadDistributionNormal(FKMLoadSequence):
    r"""Scale a load sequence assuming normally distributed loads.

    Use this accessor when the load values are normally distributed and the
    standard deviation :math:`s_L` is known in the same unit as the load
    sequence.  It converts a load sequence from the reference load probability
    to the requested FKM load probability and assessment failure probability.

    Signal contract:

    * The accessor accepts the same non-empty :class:`pandas.Series` or
      :class:`pandas.DataFrame` objects as :class:`FKMLoadSequence`.
    * In a multi-column :class:`pandas.DataFrame`, the first column contains the
      load or stress values to scale; further columns are copied unchanged.

    Parameters
    ----------
    pandas_obj : pandas.Series or pandas.DataFrame
        Load sequence or stress sequence.  Stress values are typically given in
        MPa.

    See Also
    --------
    FKMLoadDistributionLognormal : Scale loads with a lognormal distribution.
    FKMLoadDistributionBlanket : Scale loads when the distribution is unknown.

    Notes
    -----
    Implement FKM nonlinear guideline clause 2.3.2.1.  Prefer this accessor
    over :class:`FKMLoadDistributionLognormal` when additive scatter in load or
    stress is the appropriate model.
    """

    def gamma_L(self, input_parameters):
        r"""Compute the scaling factor :math:`\gamma_L = (L_\text{max} + \alpha_L) / L_\text{max}`.

        Note that for load sequences on multiple
        nodes (i.e. on a full mesh), :math:`L_\text{max}` is the maximum load
        over all nodes and load steps, not different for different nodes.

        Parameters
        ----------
        input_parameters : pandas Series
            The parameters to specify the upscaling method.

            * ``input_parameters.s_L``: standard deviation of the normal distribution
            * ``input_parameters.P_L``: probability in [%] of the load for which to do the assessment, one of {2.5, 50}
            * ``input_parameters.P_A``: probability in [%], one of {1e-7, 1e-6, 1e-5, 7.2e-5, 1e-3, 2.3e-1, 0.5}
              (de: Ausfallwahrscheinlichkeit)
            * ``input_parameters.max_load_independently_for_nodes``: optional, whether the scaling should be performed
              independently at every node (True), or uniformly over all nodes (False). The default value is False.

        Returns
        -------
        float
            Load scaling factor :math:`\gamma_L`, dimensionless.

        Raises
        ------
        ValueError
            If a required parameter is missing or ``P_A`` is not supported.
        """

        self._validate_parameters(input_parameters, required_parameters=["P_L", "s_L", "P_A"])
        beta = self._get_beta(input_parameters)

        # eq. 2.3-4
        if np.isclose(input_parameters.P_L, 2.5):
            alpha_L = (0.7 * beta - 2) * input_parameters.s_L
        else:
            alpha_L = 0.7 * beta * input_parameters.s_L

        # eq. 2.3-5
        if "max_load_independently_for_nodes" not in input_parameters:
            input_parameters["max_load_independently_for_nodes"] = False

        L_max = self.maximum_absolute_load(input_parameters.max_load_independently_for_nodes)

        gamma_L = (L_max + alpha_L) / L_max

        return gamma_L

    def scaled_load_sequence(self, input_parameters):
        r"""Scale the load sequence with the normal-distribution factor.

        The following parameters are used: s_L, P_L, P_A.

        Parameters
        ----------
        input_parameters : pandas Series
            The parameters to specify the upscaling method.

            * ``input_parameters.s_L``: standard deviation of the normal distribution
            * ``input_parameters.P_L``: probability in [%] of the load for which to do the assessment, one of {2.5, 50}
            * ``input_parameters.P_A``: probability in [%], one of {1e-7, 1e-6, 1e-5, 7.2e-5, 1e-3, 2.3e-1, 0.5}
              (de: Ausfallwahrscheinlichkeit)

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Scaled copy of the input object.  The load column is multiplied by
            :math:`\gamma_L` according to FKM nonlinear guideline clause
            2.3.2.1.

        Raises
        ------
        ValueError
            If a required parameter is missing or ``P_A`` is not supported.
        """
        gamma_L = self.gamma_L(input_parameters)

        return self.scaled_by_constant(gamma_L)


@pd.api.extensions.register_dataframe_accessor("fkm_safety_lognormal_from_stddev")
@pd.api.extensions.register_series_accessor("fkm_safety_lognormal_from_stddev")
class FKMLoadDistributionLognormal(FKMLoadSequence):
    r"""Scale a load sequence assuming lognormally distributed loads.

    Use this accessor when the logarithm of the load values is normally
    distributed and the logarithmic standard deviation :math:`LSD_s` is known.
    It is suited to multiplicative scatter, where the safety factor is
    independent of the absolute load level.

    Signal contract:

    * The accessor accepts the same non-empty :class:`pandas.Series` or
      :class:`pandas.DataFrame` objects as :class:`FKMLoadSequence`.
    * In a multi-column :class:`pandas.DataFrame`, the first column contains the
      load or stress values to scale; further columns are copied unchanged.

    Parameters
    ----------
    pandas_obj : pandas.Series or pandas.DataFrame
        Load sequence or stress sequence.  Stress values are typically given in
        MPa.

    See Also
    --------
    FKMLoadDistributionNormal : Scale loads with a normal distribution.
    FKMLoadDistributionBlanket : Scale loads when the distribution is unknown.

    Notes
    -----
    Implement FKM nonlinear guideline clause 2.3.2.2.  Prefer this accessor
    over :class:`FKMLoadDistributionNormal` when scatter acts as a relative
    factor on the load level.
    """

    def gamma_L(self, input_parameters):
        r"""Compute the scaling factor :math:`\gamma_L`.

        Parameters
        ----------
        input_parameters : pandas Series
            The parameters to specify the upscaling method.

            * ``input_parameters.LSD_s``: standard deviation of the lognormal distribution
            * ``input_parameters.P_L``: probability in [%] of the load for which to do the assessment, one of {2.5, 50}
            * ``input_parameters.P_A``: probability in [%], one of {1e-7, 1e-6, 1e-5, 7.2e-5, 1e-3, 2.3e-1, 0.5}
              (de: Ausfallwahrscheinlichkeit)

        Returns
        -------
        float
            Load scaling factor :math:`\gamma_L`, dimensionless.

        Raises
        ------
        ValueError
            If a required parameter is missing or ``P_A`` is not supported.
        """

        self._validate_parameters(input_parameters, required_parameters=["P_L", "LSD_s", "P_A"])
        beta = self._get_beta(input_parameters)

        # eq. 2.3-6
        if np.isclose(input_parameters.P_L, 2.5):
            alpha_LSD = (0.7 * beta - 2) * input_parameters.LSD_s
        else:
            alpha_LSD = 0.7 * beta * input_parameters.LSD_s

        # eq. 2.3-7
        gamma_L = max(1, 10 ** alpha_LSD)

        return gamma_L

    def scaled_load_sequence(self, input_parameters):
        r"""Scale the load sequence with the lognormal-distribution factor.

        The following parameters are used: LSD_s, P_L, P_A.

        Parameters
        ----------
        input_parameters : pandas Series
            The parameters to specify the upscaling method.

            * ``input_parameters.LSD_s``: standard deviation of the lognormal distribution
            * ``input_parameters.P_L``: probability in [%] of the load for which to do the assessment, one of {2.5, 50}
            * ``input_parameters.P_A``: probability in [%], one of {1e-7, 1e-6, 1e-5, 7.2e-5, 1e-3, 2.3e-1, 0.5}
              (de: Ausfallwahrscheinlichkeit)

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Scaled copy of the input object.  The load column is multiplied by
            :math:`\gamma_L` according to FKM nonlinear guideline clause
            2.3.2.2.

        Raises
        ------
        ValueError
            If a required parameter is missing or ``P_A`` is not supported.
        """
        gamma_L = self.gamma_L(input_parameters)

        return self.scaled_by_constant(gamma_L)


@pd.api.extensions.register_dataframe_accessor("fkm_safety_blanket")
@pd.api.extensions.register_series_accessor("fkm_safety_blanket")
class FKMLoadDistributionBlanket(FKMLoadSequence):
    r"""Scale a load sequence with the FKM blanket load factor.

    Use this accessor when the statistical load distribution is unknown.  For
    :math:`P_L = 2.5\%` it applies the constant blanket factor
    :math:`\gamma_L = 1.1`; for :math:`P_L = 50\%` it leaves the load unchanged.

    Signal contract:

    * The accessor accepts the same non-empty :class:`pandas.Series` or
      :class:`pandas.DataFrame` objects as :class:`FKMLoadSequence`.
    * In a multi-column :class:`pandas.DataFrame`, the first column contains the
      load or stress values to scale; further columns are copied unchanged.

    Parameters
    ----------
    pandas_obj : pandas.Series or pandas.DataFrame
        Load sequence or stress sequence.  Stress values are typically given in
        MPa.

    See Also
    --------
    FKMLoadDistributionNormal : Scale loads with a normal distribution.
    FKMLoadDistributionLognormal : Scale loads with a lognormal distribution.

    Notes
    -----
    Implement FKM nonlinear guideline clause 2.3.2.3.  Prefer this accessor
    when no reliable statistical distribution parameters are available.
    """

    def gamma_L(self, input_parameters):
        r"""Compute the scaling factor :math:`\gamma_L`.

        Parameters
        ----------
        input_parameters : pandas Series
            The parameters to specify the upscaling method.

            * ``input_parameters.P_L``: probability in [%] of the load for which to do the assessment,
              has to be on of {2.5, 50}. (de: Ausfallwahrscheinlichkeit)

        Returns
        -------
        float
            Load scaling factor :math:`\gamma_L`, dimensionless.

        Raises
        ------
        ValueError
            If ``P_L`` is missing or is neither ``2.5`` nor ``50`` percent.
        """

        self._validate_parameters(input_parameters, required_parameters=["P_L"])

        if np.isclose(input_parameters.P_L, 2.5):
            # eq. 2.3-8
            gamma_L = 1.1
        elif np.isclose(input_parameters.P_L, 50):
            gamma_L = 1.0
        else:
            raise ValueError(
                "fkm_safety_blanket is only possible for P_L=2.5 % "
                f"or P_L=50 %, not P_L={input_parameters.P_L} %"
            )

        return gamma_L

    def scaled_load_sequence(self, input_parameters):
        r"""Scale the load sequence with the blanket load factor.

        The only required input parameter is ``P_L``.

        Parameters
        ----------
        input_parameters : pandas Series
            The parameters to specify the upscaling method.

            * ``input_parameters.P_L``: probability in [%] of the load
              for which to do the assessment, one of {2.5, 50}

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Scaled copy of the input object.  The load column is multiplied by
            :math:`\gamma_L` according to FKM nonlinear guideline clause
            2.3.2.3.

        Raises
        ------
        ValueError
            If ``P_L`` is missing or is neither ``2.5`` nor ``50`` percent.
        """
        gamma_L = self.gamma_L(input_parameters)

        # eq. 2.3-5
        return self.scaled_by_constant(gamma_L)

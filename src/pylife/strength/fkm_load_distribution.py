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

r"""Scale up a load sequence to incorporate safety factors for FKM
nonlinear lifetime assessment.

Given a pandas Series of load values, return a scaled version where the safety
has been incorporated.  The series is scaled by a constant value
:math:`\gamma_L`, which models the distribution of the load, the severity of
the failure (modeled by :math:`P_A`) and the considered load probability of
either :math:`P_L=2.5 \%` or :math:`P_L=50 \%`.

The FKM nonlinear guideline defines three possible methods to consider the
statistical distribution of the load:

    * a normal distribution with given standard deviation, :math:`s_L`
    * a logarithmic-normal distribution with given standard deviation :math:`LSD_s`
    * an unknown distribution, use the constant factor :math:`\gamma_L=1.1` for
      :math:`P_L = 2.5\%`

For these three methods, there exist the three accessors
`fkm_safety_normal_from_stddev`, `fkm_safety_lognormal_from_stddev`, and
`fkm_safety_blanket`.

The resulting scaling factor can be retrieved with
``.gamma_L(input_parameters)``, the scaled load series can be obtained with
``.scaled_load_sequence(input_parameters)``.

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
    """Base class used by the safety scaling method. It is used to compute the beta parameter
    and to scale the load sequence by a constant ``gamma_L``.

    This class can be used from user code to scale a load sequence, potentially
    on a mesh with other fields set for every node.

    In such a case, the other fields are not modified.

    Example
    -------

    .. jupyter-execute::


        import pandas as pd
        import pylife.strength.fkm_load_distribution

        # create an example load sequence with stress (S_v) and an arbitrary other column (col2)
        mesh = pd.DataFrame(
            index=pd.MultiIndex.from_product([range(2), range(4)], names=["load_step", "node_id"]),
            data={
                "S_v": [10, 20, -10, -20, 30, 60, 40, 80],
                "col2":  [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            })
        print(mesh)

        # scale the load sequence by the factor 2, note that col2 is not scaled
        mesh.fkm_load_sequence.scaled_by_constant(2)
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
            scaling factor for the load sequence.

        Returns
        -------
        The scaled load sequence.
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
            If the maximum absolute should be computed separately for every node.

            If set to False, a single maximum value is computed over all nodes.
            The default is False, which means the whole mesh will be assessed
            by the same maximum load. This, however, means that calculating the
            FKMnonlinear assessment for the whole mesh at once yields a
            different result than calculating the assessment for every single
            node one after each other. (When doing it all at once, the maximum
            absolute load used for the failure probability is the maximum of
            the loads at every node, when doing it only for a single node, the
            maximum absolute load value may be lower.)

        Returns
        -------
        L_max : float
            The maximum absolute load.

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
        """Compute a scaling factor for assessing a load sequence for a given failure probability.

        For details, refer to the FKM nonlinear document.

        The beta factors are also described in "A. Fischer. Bestimmung
        modifizierter Teilsicherheitsbeiwerte zur semiprobabilistischen
        Bemessung von Stahlbetonkonstruktionen im Bestand. TU Kaiserslautern,
        2010"

        Parameters
        ----------
        input_parameters : pd.Series
            The set of assessment parameters, has to contain the assessment failure probability ``input_parameters.P_A``.
            This variable has to be one of {1e-7, 1e-6, 1e-5, 7.2e-5, 1e-3, 2.3e-1, 0.5}.

        Raises
        ------
        ValueError
            If the given failure probability has an invalid value.

        Returns
        -------
        float
            The value of the beta parameter.

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
    r"""Series accessor to get a scaled up load series.

    A load series is a list of load values with included load safety, as used
    in FKM nonlinear lifetime assessments.

    The loads are assumed to follow a **normal distribution** with standard
    deviation :math:`s_L`.  To incorporate safety, reduce the values of the
    load series from :math:`P_L = 50\%` up to the given load probability
    :math:`P_L` and the given failure probability :math:`P_A`.

    For more information, see 2.3.2.1 of the FKM nonlinear guideline.

    See Also
    --------
    :class:`AbstractFKMLoadDistribution`: accesses meshes with connectivity information
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

        Raises
        ------
        ValueError
            If not all parameters that are required were given.

        Returns
        -------
        gamma_L : float
            The resulting scaling factor.

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
        r"""The scaled load sequence according to the given parameters.

        The following parameters are used: s_L, P_L, P_A.

        Parameters
        ----------
        input_parameters : pandas Series
            The parameters to specify the upscaling method.

            * ``input_parameters.s_L``: standard deviation of the normal distribution
            * ``input_parameters.P_L``: probability in [%] of the load for which to do the assessment, one of {2.5, 50}
            * ``input_parameters.P_A``: probability in [%], one of {1e-7, 1e-6, 1e-5, 7.2e-5, 1e-3, 2.3e-1, 0.5}
              (de: Ausfallwahrscheinlichkeit)

        Raises
        ------
        ValueError
            If not all parameters that are required were given.

        Returns
        -------
        pandas Series
            The input series where all values have been scaled by :math:`\gamma_L`,
            see 2.3.2.1 of the FKM nonlinear guideline.

        """
        gamma_L = self.gamma_L(input_parameters)

        return self.scaled_by_constant(gamma_L)


@pd.api.extensions.register_dataframe_accessor("fkm_safety_lognormal_from_stddev")
@pd.api.extensions.register_series_accessor("fkm_safety_lognormal_from_stddev")
class FKMLoadDistributionLognormal(FKMLoadSequence):
    r"""Series accessor to get a scaled up load series.

    A load series is a list of load values with included load safety, as used
    in FKM nonlinear lifetime assessments.

    The loads are assumed to follow a **lognormal distribution** with standard
    deviation :math:`LSD_s`.  To incorporate safety, reduce the values of the
    load series from :math:`P_L = 50\%` up to the given load probability
    :math:`P_L` and the given failure probability :math:`P_A`.

    For more information, see 2.3.2.2 of the FKM nonlinear guideline.

    See Also
    --------
    :class:`AbstractFKMLoadDistribution`: accesses meshes with connectivity information

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

        Raises
        ------
        ValueError
            If not all parameters that are required were given.

        Returns
        -------
        gamma_L : float
            The resulting scaling factor.

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
        r"""The scaled load sequence according to the given parameters.

        The following parameters are used: LSD_s, P_L, P_A.

        Parameters
        ----------
        input_parameters : pandas Series
            The parameters to specify the upscaling method.

            * ``input_parameters.LSD_s``: standard deviation of the lognormal distribution
            * ``input_parameters.P_L``: probability in [%] of the load for which to do the assessment, one of {2.5, 50}
            * ``input_parameters.P_A``: probability in [%], one of {1e-7, 1e-6, 1e-5, 7.2e-5, 1e-3, 2.3e-1, 0.5}
              (de: Ausfallwahrscheinlichkeit)

        Raises
        ------
        ValueError
            If not all parameters which are required were given.

        Returns
        -------
        pandas Series
            The input series where all values have been scaled by :math:`\gamma_L`,
            see 2.3.2.2 of the FKM nonlinear guideline.

        """
        gamma_L = self.gamma_L(input_parameters)

        return self.scaled_by_constant(gamma_L)


@pd.api.extensions.register_dataframe_accessor("fkm_safety_blanket")
@pd.api.extensions.register_series_accessor("fkm_safety_blanket")
class FKMLoadDistributionBlanket(FKMLoadSequence):
    r"""Series accessor to get a scaled up load series, i.e., a list of load values with included load safety,
      as used in FKM nonlinear lifetime assessments.

      The distribution of loads is unknown, therefore a scaling factor of :math:`\gamma_L` = 1.1 is assumed.
      This is only used for :math:`P_L = 2.5\%`.
      As an alternative, we can use no scaling for the load distribution at all,
      corresponding to :math:`\gamma_L` = 1 and :math:`P_L = 50\%`

      For more information, see 2.3.2.3 of the FKM nonlinear guideline.

    See also
    --------
    :class:`AbstractFKMLoadDistribution`: accesses meshes with connectivity information
    """

    def gamma_L(self, input_parameters):
        r"""Compute the scaling factor :math:`\gamma_L`.

        Parameters
        ----------
        input_parameters : pandas Series
            The parameters to specify the upscaling method.

            * ``input_parameters.P_L``: probability in [%] of the load for which to do the assessment,
              has to be on of {2.5, 50}. (de: Ausfallwahrscheinlichkeit)

        Raises
        ------
        ValueError
            If not all parameters which are required were given.

        Returns
        -------
        gamma_L : float
            The resulting scaling factor.

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
        r"""The scaled load sequence according to the given parameters.

        The following parameters are used: LSD_s, P_L, P_A.

        Parameters
        ----------
        input_parameters : pandas Series
            The parameters to specify the upscaling method.

            * ``input_parameters.P_L``: probability in [%] of the load
              for which to do the assessment, one of {2.5, 50}

        Raises
        ------
        ValueError
            If not all parameters which are required were given.

        Returns
        -------
        pandas Series
            The input series where all values have been scaled by :math:`\gamma_L`,
            see 2.3.2.2 of the FKM nonlinear guideline.

        """
        gamma_L = self.gamma_L(input_parameters)

        # eq. 2.3-5
        return self.scaled_by_constant(gamma_L)

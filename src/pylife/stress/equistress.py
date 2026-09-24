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

"""Calculate principal and equivalent stresses from stress tensors.

The module provides NumPy functions for component arrays and a pandas
``DataFrame`` accessor for stress tensors stored in Voigt notation
``S11, S22, S33, S12, S13, S23``.  Use the plain functions when the stress
components are already available as arrays.  Use the
:class:`StressTensorEquistress` accessor when stresses are stored as a
validated pyLife stress signal.

Examples
--------
>>> round(float(mises(100.0, 0.0, 0.0, 0.0, 0.0, 0.0)), 6)
100.0
"""

__author__ = "Johannes Mueller, Vivien Le Baube et. al."
__maintainer__ = "Johannes Mueller"

import numpy as np
import pandas as pd
from pylife.stress import stresssignal


def eigenval(s11, s22, s33, s12, s13, s23):
    r"""Calculate principal stresses of a symmetric stress tensor.

    Parameters
    ----------
    s11 : array_like
        Normal stress component in direction 1 in MPa.
    s22 : array_like
        Normal stress component in direction 2 in MPa.
    s33 : array_like
        Normal stress component in direction 3 in MPa.
    s12 : array_like
        Shear stress component in the 1-2 plane in MPa.
    s13 : array_like
        Shear stress component in the 1-3 plane in MPa.
    s23 : array_like
        Shear stress component in the 2-3 plane in MPa.

    Returns
    -------
    numpy.ndarray
        Principal stresses in MPa, sorted in ascending order along the last
        axis.

    Notes
    -----
    The input components are interpreted as the symmetric stress tensor in
    Voigt notation ``S11, S22, S33, S12, S13, S23``:

    .. math::

        \sigma =
        \begin{pmatrix}
        S11 & S12 & S13\\
        S12 & S22 & S23\\
        S13 & S23 & S33
        \end{pmatrix}.

    The returned values are the eigenvalues
    :math:`\sigma_1 \leq \sigma_2 \leq \sigma_3` of this tensor.
    """
    a = np.array([[s11, s12, s13],
                  [s12, s22, s23],
                  [s13, s23, s33]]).T
    return np.linalg.eigvalsh(a)


def _sign_trace(s11, s22, s33):
    r"""Calculate the sign of the first stress invariant.

    Parameters
    ----------
    s11 : array_like
        Normal stress component in direction 1 in MPa.
    s22 : array_like
        Normal stress component in direction 2 in MPa.
    s33 : array_like
        Normal stress component in direction 3 in MPa.

    Returns
    -------
    numpy.ndarray
        Sign of the trace with the same shape as the input components.

    Notes
    -----
    The sign is calculated from the trace of the stress tensor,

    .. math::

        \operatorname{sign}(S11 + S22 + S33).

    A zero trace is treated as positive and therefore returns ``1``.
    """
    s11 = np.array(s11)
    s22 = np.array(s22)
    s33 = np.array(s33)
    assert (s11.shape == s22.shape and
            s11.shape == s33.shape), "Components' shape is not consistent."
    sgn = np.sign(s11 + s22 + s33)  # calculate sign of trace, careful: sign of 0 is 0
    if sgn.ndim == 0:
        if sgn == 0:
            sgn = np.array(1)
    else:
        sgn[sgn == 0] = 1
    return sgn


def _sign_abs_max_principal(s11, s22, s33, s12, s13, s23):
    r"""Calculate the sign of the absolute maximum principal stress.

    Parameters
    ----------
    s11 : array_like
        Normal stress component in direction 1 in MPa.
    s22 : array_like
        Normal stress component in direction 2 in MPa.
    s33 : array_like
        Normal stress component in direction 3 in MPa.
    s12 : array_like
        Shear stress component in the 1-2 plane in MPa.
    s13 : array_like
        Shear stress component in the 1-3 plane in MPa.
    s23 : array_like
        Shear stress component in the 2-3 plane in MPa.

    Returns
    -------
    numpy.ndarray
        Sign of the principal stress with the largest absolute value.

    Notes
    -----
    With principal stresses
    :math:`\sigma_1 \leq \sigma_2 \leq \sigma_3`, the sign is

    .. math::

        \operatorname{sign}(\sigma_1 + \sigma_3).

    This is positive when the tensile maximum principal stress dominates and
    negative when the compressive minimum principal stress dominates.  A tie is
    treated as positive and therefore returns ``1``.
    """
    w = eigenval(s11, s22, s33, s12, s13, s23).T
    w_max = np.amax(w, axis=0)
    w_min = np.amin(w, axis=0)
    sgn = np.sign(w_max + w_min)
    # sign of 0 is 0, replace it with 1
    zero_sign_bool = np.array(sgn == 0, dtype=int)
    sgn = sgn + zero_sign_bool  # works for all dimensions, no need to differentiate value from array.
    return sgn


def tresca(s11, s22, s33, s12, s13, s23):
    r"""Calculate the Tresca equivalent stress.

    Parameters
    ----------
    s11 : array_like
        Normal stress component in direction 1 in MPa.
    s22 : array_like
        Normal stress component in direction 2 in MPa.
    s33 : array_like
        Normal stress component in direction 3 in MPa.
    s12 : array_like
        Shear stress component in the 1-2 plane in MPa.
    s13 : array_like
        Shear stress component in the 1-3 plane in MPa.
    s23 : array_like
        Shear stress component in the 2-3 plane in MPa.

    Returns
    -------
    numpy.ndarray
        Tresca equivalent stress in MPa.

    Notes
    -----
    With principal stresses
    :math:`\sigma_1 \leq \sigma_2 \leq \sigma_3`, pyLife uses the maximum
    principal-stress difference as the Tresca equivalent stress:

    .. math::

        \sigma_\mathrm{Tresca}
        = \max\left(|\sigma_1-\sigma_2|,
                   |\sigma_1-\sigma_3|,
                   |\sigma_2-\sigma_3|\right)
        = \sigma_3 - \sigma_1.
    """
    w = eigenval(s11, s22, s33, s12, s13, s23).T
    w_diff = np.zeros(w.shape)
    w_diff[0] = np.fabs(w[0] - w[1])
    w_diff[1] = np.fabs(w[0] - w[2])
    w_diff[2] = np.fabs(w[1] - w[2])
    return np.amax(w_diff, axis=0)


def signed_tresca_trace(s11, s22, s33, s12, s13, s23):
    r"""Calculate the trace-signed Tresca equivalent stress.

    Parameters
    ----------
    s11 : array_like
        Normal stress component in direction 1 in MPa.
    s22 : array_like
        Normal stress component in direction 2 in MPa.
    s33 : array_like
        Normal stress component in direction 3 in MPa.
    s12 : array_like
        Shear stress component in the 1-2 plane in MPa.
    s13 : array_like
        Shear stress component in the 1-3 plane in MPa.
    s23 : array_like
        Shear stress component in the 2-3 plane in MPa.

    Returns
    -------
    numpy.ndarray
        Tresca equivalent stress in MPa, signed by the stress trace.

    Notes
    -----
    The signed value is

    .. math::

        \sigma_\mathrm{Tresca,trace}
        = \operatorname{sign}(S11 + S22 + S33)\,
          \sigma_\mathrm{Tresca}.

    A zero trace is treated as positive.
    """
    return _sign_trace(s11, s22, s33) * tresca(s11, s22, s33, s12, s13, s23)


def signed_tresca_abs_max_principal(s11, s22, s33, s12, s13, s23):
    r"""Calculate the principal-stress-signed Tresca equivalent stress.

    Parameters
    ----------
    s11 : array_like
        Normal stress component in direction 1 in MPa.
    s22 : array_like
        Normal stress component in direction 2 in MPa.
    s33 : array_like
        Normal stress component in direction 3 in MPa.
    s12 : array_like
        Shear stress component in the 1-2 plane in MPa.
    s13 : array_like
        Shear stress component in the 1-3 plane in MPa.
    s23 : array_like
        Shear stress component in the 2-3 plane in MPa.

    Returns
    -------
    numpy.ndarray
        Tresca equivalent stress in MPa, signed by the absolute maximum
        principal stress.

    Notes
    -----
    With principal stresses
    :math:`\sigma_1 \leq \sigma_2 \leq \sigma_3`, the signed value is

    .. math::

        \sigma_\mathrm{Tresca,absmax}
        = \operatorname{sign}(\sigma_1 + \sigma_3)\,
          \sigma_\mathrm{Tresca}.

    A tie between tensile and compressive absolute principal stress is treated
    as positive.
    """
    return _sign_abs_max_principal(s11, s22, s33, s12, s13, s23) * tresca(s11, s22, s33, s12, s13, s23)


def abs_max_principal(s11, s22, s33, s12, s13, s23):
    r"""Calculate the signed absolute maximum principal stress.

    Parameters
    ----------
    s11 : array_like
        Normal stress component in direction 1 in MPa.
    s22 : array_like
        Normal stress component in direction 2 in MPa.
    s33 : array_like
        Normal stress component in direction 3 in MPa.
    s12 : array_like
        Shear stress component in the 1-2 plane in MPa.
    s13 : array_like
        Shear stress component in the 1-3 plane in MPa.
    s23 : array_like
        Shear stress component in the 2-3 plane in MPa.

    Returns
    -------
    numpy.ndarray
        Principal stress in MPa with the largest absolute value and its
        original sign.

    Notes
    -----
    With principal stresses
    :math:`\sigma_1 \leq \sigma_2 \leq \sigma_3`, the result is

    .. math::

        \sigma_\mathrm{absmax} =
        \begin{cases}
        \sigma_3, & |\sigma_3| \geq |\sigma_1|,\\
        \sigma_1, & |\sigma_3| < |\sigma_1|.
        \end{cases}

    A tie is treated as tensile and returns :math:`\sigma_3`.
    """
    w = eigenval(s11, s22, s33, s12, s13, s23).T
    w_max = np.amax(w, axis=0)
    w_min = np.amin(w, axis=0)
    sign = _sign_abs_max_principal(s11, s22, s33, s12, s13, s23)
    positive_sign_bool = np.array(sign >= 0)
    return w_max * positive_sign_bool + w_min * np.invert(positive_sign_bool)


def principals(s11, s22, s33, s12, s13, s23):
    r"""Calculate all principal stress components.

    Parameters
    ----------
    s11 : array_like
        Normal stress component in direction 1 in MPa.
    s22 : array_like
        Normal stress component in direction 2 in MPa.
    s33 : array_like
        Normal stress component in direction 3 in MPa.
    s12 : array_like
        Shear stress component in the 1-2 plane in MPa.
    s13 : array_like
        Shear stress component in the 1-3 plane in MPa.
    s23 : array_like
        Shear stress component in the 2-3 plane in MPa.

    Returns
    -------
    numpy.ndarray
        Principal stresses in MPa, sorted in ascending order along the last
        axis.

    Notes
    -----
    The principal stresses are the eigenvalues of the symmetric stress tensor
    in Voigt notation ``S11, S22, S33, S12, S13, S23``:

    .. math::

        \det(\sigma - \lambda I) = 0.
    """
    return eigenval(s11, s22, s33, s12, s13, s23)


def max_principal(s11, s22, s33, s12, s13, s23):
    r"""Calculate the maximum principal stress.

    Parameters
    ----------
    s11 : array_like
        Normal stress component in direction 1 in MPa.
    s22 : array_like
        Normal stress component in direction 2 in MPa.
    s33 : array_like
        Normal stress component in direction 3 in MPa.
    s12 : array_like
        Shear stress component in the 1-2 plane in MPa.
    s13 : array_like
        Shear stress component in the 1-3 plane in MPa.
    s23 : array_like
        Shear stress component in the 2-3 plane in MPa.

    Returns
    -------
    numpy.ndarray
        Largest principal stress in MPa.

    Notes
    -----
    With principal stresses
    :math:`\sigma_1 \leq \sigma_2 \leq \sigma_3`, the result is

    .. math::

        \sigma_\mathrm{max} = \sigma_3.
    """
    w = eigenval(s11, s22, s33, s12, s13, s23).T
    return np.amax(w, axis=0)


def min_principal(s11, s22, s33, s12, s13, s23):
    r"""Calculate the minimum principal stress.

    Parameters
    ----------
    s11 : array_like
        Normal stress component in direction 1 in MPa.
    s22 : array_like
        Normal stress component in direction 2 in MPa.
    s33 : array_like
        Normal stress component in direction 3 in MPa.
    s12 : array_like
        Shear stress component in the 1-2 plane in MPa.
    s13 : array_like
        Shear stress component in the 1-3 plane in MPa.
    s23 : array_like
        Shear stress component in the 2-3 plane in MPa.

    Returns
    -------
    numpy.ndarray
        Smallest principal stress in MPa.

    Notes
    -----
    With principal stresses
    :math:`\sigma_1 \leq \sigma_2 \leq \sigma_3`, the result is

    .. math::

        \sigma_\mathrm{min} = \sigma_1.
    """
    w = eigenval(s11, s22, s33, s12, s13, s23).T
    return np.amin(w, axis=0)


def mises(s11, s22, s33, s12, s13, s23):
    r"""Calculate the von Mises equivalent stress.

    Parameters
    ----------
    s11 : array_like
        Normal stress component in direction 1 in MPa.
    s22 : array_like
        Normal stress component in direction 2 in MPa.
    s33 : array_like
        Normal stress component in direction 3 in MPa.
    s12 : array_like
        Shear stress component in the 1-2 plane in MPa.
    s13 : array_like
        Shear stress component in the 1-3 plane in MPa.
    s23 : array_like
        Shear stress component in the 2-3 plane in MPa.

    Returns
    -------
    numpy.ndarray
        Von Mises equivalent stress in MPa.

    Raises
    ------
    AssertionError
        Raised if the component arrays do not have identical shapes.

    Notes
    -----
    pyLife calculates the scalar distortion-energy equivalent stress from the
    Voigt components:

    .. math::

        \sigma_\mathrm{vM} =
        \sqrt{S11^2 + S22^2 + S33^2
        - S11\,S22 - S11\,S33 - S22\,S33
        + 3\,(S12^2 + S13^2 + S23^2)}.
    """
    s11 = np.array(s11)
    s22 = np.array(s22)
    s33 = np.array(s33)
    s12 = np.array(s12)
    s13 = np.array(s13)
    s23 = np.array(s23)

    assert (s11.shape == s22.shape and
            s11.shape == s33.shape and
            s11.shape == s12.shape and
            s11.shape == s13.shape and
            s11.shape == s23.shape), "Components' shape is not consistent."

    mises_stress = np.sqrt(s11 ** 2 + s22 ** 2 + s33 ** 2
                           - s11 * s22 - s11 * s33 - s22 * s33
                           + 3 * (s12 ** 2 + s13 ** 2 + s23 ** 2))
    return mises_stress


def signed_mises_trace(s11, s22, s33, s12, s13, s23):
    r"""Calculate the trace-signed von Mises equivalent stress.

    Parameters
    ----------
    s11 : array_like
        Normal stress component in direction 1 in MPa.
    s22 : array_like
        Normal stress component in direction 2 in MPa.
    s33 : array_like
        Normal stress component in direction 3 in MPa.
    s12 : array_like
        Shear stress component in the 1-2 plane in MPa.
    s13 : array_like
        Shear stress component in the 1-3 plane in MPa.
    s23 : array_like
        Shear stress component in the 2-3 plane in MPa.

    Returns
    -------
    numpy.ndarray
        Von Mises equivalent stress in MPa, signed by the stress trace.

    Notes
    -----
    The signed value is

    .. math::

        \sigma_\mathrm{vM,trace}
        = \operatorname{sign}(S11 + S22 + S33)\,
          \sigma_\mathrm{vM}.

    A zero trace is treated as positive.
    """
    return _sign_trace(s11, s22, s33) * mises(s11, s22, s33, s12, s13, s23)


def signed_mises_abs_max_principal(s11, s22, s33, s12, s13, s23):
    r"""Calculate the principal-stress-signed von Mises equivalent stress.

    Parameters
    ----------
    s11 : array_like
        Normal stress component in direction 1 in MPa.
    s22 : array_like
        Normal stress component in direction 2 in MPa.
    s33 : array_like
        Normal stress component in direction 3 in MPa.
    s12 : array_like
        Shear stress component in the 1-2 plane in MPa.
    s13 : array_like
        Shear stress component in the 1-3 plane in MPa.
    s23 : array_like
        Shear stress component in the 2-3 plane in MPa.

    Returns
    -------
    numpy.ndarray
        Von Mises equivalent stress in MPa, signed by the absolute maximum
        principal stress.

    Notes
    -----
    With principal stresses
    :math:`\sigma_1 \leq \sigma_2 \leq \sigma_3`, the signed value is

    .. math::

        \sigma_\mathrm{vM,absmax}
        = \operatorname{sign}(\sigma_1 + \sigma_3)\,
          \sigma_\mathrm{vM}.

    A tie between tensile and compressive absolute principal stress is treated
    as positive.
    """
    return _sign_abs_max_principal(s11, s22, s33, s12, s13, s23) * mises(s11, s22, s33, s12, s13, s23)


@pd.api.extensions.register_dataframe_accessor("equistress")
class StressTensorEquistress(stresssignal.StressTensorVoigt):
    """Calculate equivalent stresses from a Voigt stress tensor signal.

    The accessor is available as ``df.equistress`` for pandas DataFrames that
    satisfy the :class:`~pylife.stress.stresssignal.StressTensorVoigt`
    contract.  The mandatory columns define one symmetric stress tensor per
    row in Voigt notation ``S11, S22, S33, S12, S13, S23``:

    * ``S11``: Normal stress component in direction 1 in MPa.
    * ``S22``: Normal stress component in direction 2 in MPa.
    * ``S33``: Normal stress component in direction 3 in MPa.
    * ``S12``: Shear stress component in the 1-2 plane in MPa.
    * ``S13``: Shear stress component in the 1-3 plane in MPa.
    * ``S23``: Shear stress component in the 2-3 plane in MPa.

    The accessor methods preserve the input index and return pandas objects.
    Use the module-level functions when working directly with NumPy arrays.

    Parameters
    ----------
    pandas_obj : pandas.DataFrame
        DataFrame containing the mandatory Voigt stress tensor component
        columns.

    See Also
    --------
    pylife.stress.stresssignal.StressTensorVoigt : Validate the underlying
        Voigt stress tensor signal.
    pylife.stress.equistress.mises : Calculate von Mises stress from arrays.
    pylife.stress.equistress.tresca : Calculate Tresca stress from arrays.
    """
    def tresca(self):
        """Calculate the Tresca equivalent stress for each row.

        Returns
        -------
        pandas.Series
            Tresca equivalent stress in MPa, indexed like the input DataFrame
            and named ``'tresca'``.

        See Also
        --------
        pylife.stress.equistress.tresca : Calculate the same quantity from
            component arrays.
        """
        return pd.Series(tresca(s11=self._obj['S11'].to_numpy(),
                                s22=self._obj['S22'].to_numpy(),
                                s33=self._obj['S33'].to_numpy(),
                                s12=self._obj['S12'].to_numpy(),
                                s13=self._obj['S13'].to_numpy(),
                                s23=self._obj['S23'].to_numpy()),
                         name='tresca', index=self._obj.index)

    def signed_tresca_trace(self):
        """Calculate trace-signed Tresca stress for each row.

        Returns
        -------
        pandas.Series
            Tresca equivalent stress in MPa, signed by ``S11 + S22 + S33``,
            indexed like the input DataFrame and named
            ``'signed_tresca_trace'``.

        See Also
        --------
        pylife.stress.equistress.signed_tresca_trace : Calculate the same
            quantity from component arrays.
        """
        return pd.Series(signed_tresca_trace(s11=self._obj['S11'].to_numpy(),
                                             s22=self._obj['S22'].to_numpy(),
                                             s33=self._obj['S33'].to_numpy(),
                                             s12=self._obj['S12'].to_numpy(),
                                             s13=self._obj['S13'].to_numpy(),
                                             s23=self._obj['S23'].to_numpy()),
                         name='signed_tresca_trace', index=self._obj.index)

    def signed_tresca_abs_max_principal(self):
        """Calculate principal-stress-signed Tresca stress for each row.

        Returns
        -------
        pandas.Series
            Tresca equivalent stress in MPa, signed by the absolute maximum
            principal stress, indexed like the input DataFrame and named
            ``'signed_tresca_abs_max_principal'``.

        See Also
        --------
        pylife.stress.equistress.signed_tresca_abs_max_principal : Calculate
            the same quantity from component arrays.
        """
        return pd.Series(signed_tresca_abs_max_principal(s11=self._obj['S11'].to_numpy(),
                                                         s22=self._obj['S22'].to_numpy(),
                                                         s33=self._obj['S33'].to_numpy(),
                                                         s12=self._obj['S12'].to_numpy(),
                                                         s13=self._obj['S13'].to_numpy(),
                                                         s23=self._obj['S23'].to_numpy()),
                         name='signed_tresca_abs_max_principal', index=self._obj.index)

    def principals(self):
        """Calculate all principal stresses for each row.

        Returns
        -------
        pandas.DataFrame
            Principal stresses in MPa with columns ``'min_principal'``,
            ``'med_principal'``, and ``'max_principal'``, indexed like the
            input DataFrame.

        See Also
        --------
        pylife.stress.equistress.principals : Calculate principal stresses
            from component arrays.
        """
        all_princ = eigenval(s11=self._obj['S11'].to_numpy(),   # ascending order (numpy.eigvalsh)
                             s22=self._obj['S22'].to_numpy(),
                             s33=self._obj['S33'].to_numpy(),
                             s12=self._obj['S12'].to_numpy(),
                             s13=self._obj['S13'].to_numpy(),
                             s23=self._obj['S23'].to_numpy())
        return pd.DataFrame({'min_principal': all_princ[...,0],
                             'med_principal': all_princ[...,1],
                             'max_principal': all_princ[...,2]},
                             index=self._obj.index)

    def abs_max_principal(self):
        """Calculate the signed absolute maximum principal stress per row.

        Returns
        -------
        pandas.Series
            Principal stress in MPa with the largest absolute value and its
            original sign, indexed like the input DataFrame and named
            ``'abs_max_principal'``.

        See Also
        --------
        pylife.stress.equistress.abs_max_principal : Calculate the same
            quantity from component arrays.
        """
        return pd.Series(abs_max_principal(s11=self._obj['S11'].to_numpy(),
                                           s22=self._obj['S22'].to_numpy(),
                                           s33=self._obj['S33'].to_numpy(),
                                           s12=self._obj['S12'].to_numpy(),
                                           s13=self._obj['S13'].to_numpy(),
                                           s23=self._obj['S23'].to_numpy()),
                         name='abs_max_principal', index=self._obj.index)

    def max_principal(self):
        """Calculate the maximum principal stress for each row.

        Returns
        -------
        pandas.Series
            Largest principal stress in MPa, indexed like the input DataFrame
            and named ``'max_principal'``.

        See Also
        --------
        pylife.stress.equistress.max_principal : Calculate the same quantity
            from component arrays.
        """
        return pd.Series(max_principal(s11=self._obj['S11'].to_numpy(),
                                       s22=self._obj['S22'].to_numpy(),
                                       s33=self._obj['S33'].to_numpy(),
                                       s12=self._obj['S12'].to_numpy(),
                                       s13=self._obj['S13'].to_numpy(),
                                       s23=self._obj['S23'].to_numpy()),
                         name='max_principal', index=self._obj.index)

    def min_principal(self):
        """Calculate the minimum principal stress for each row.

        Returns
        -------
        pandas.Series
            Smallest principal stress in MPa, indexed like the input DataFrame
            and named ``'min_principal'``.

        See Also
        --------
        pylife.stress.equistress.min_principal : Calculate the same quantity
            from component arrays.
        """
        return pd.Series(min_principal(s11=self._obj['S11'].to_numpy(),
                                       s22=self._obj['S22'].to_numpy(),
                                       s33=self._obj['S33'].to_numpy(),
                                       s12=self._obj['S12'].to_numpy(),
                                       s13=self._obj['S13'].to_numpy(),
                                       s23=self._obj['S23'].to_numpy()),
                         name='min_principal', index=self._obj.index)

    def mises(self):
        """Calculate the von Mises equivalent stress for each row.

        Returns
        -------
        pandas.Series
            Von Mises equivalent stress in MPa, indexed like the input
            DataFrame and named ``'mises'``.

        See Also
        --------
        pylife.stress.equistress.mises : Calculate the same quantity from
            component arrays.
        """
        return pd.Series(mises(s11=self._obj['S11'].to_numpy(),
                               s22=self._obj['S22'].to_numpy(),
                               s33=self._obj['S33'].to_numpy(),
                               s12=self._obj['S12'].to_numpy(),
                               s13=self._obj['S13'].to_numpy(),
                               s23=self._obj['S23'].to_numpy()),
                         name='mises', index=self._obj.index)

    def signed_mises_trace(self):
        """Calculate trace-signed von Mises stress for each row.

        Returns
        -------
        pandas.Series
            Von Mises equivalent stress in MPa, signed by
            ``S11 + S22 + S33``, indexed like the input DataFrame and named
            ``'signed_mises_trace'``.

        See Also
        --------
        pylife.stress.equistress.signed_mises_trace : Calculate the same
            quantity from component arrays.
        """
        return pd.Series(signed_mises_trace(s11=self._obj['S11'].to_numpy(),
                                            s22=self._obj['S22'].to_numpy(),
                                            s33=self._obj['S33'].to_numpy(),
                                            s12=self._obj['S12'].to_numpy(),
                                            s13=self._obj['S13'].to_numpy(),
                                            s23=self._obj['S23'].to_numpy()),
                         name='signed_mises_trace', index=self._obj.index)

    def signed_mises_abs_max_principal(self):
        """Calculate principal-stress-signed von Mises stress for each row.

        Returns
        -------
        pandas.Series
            Von Mises equivalent stress in MPa, signed by the absolute maximum
            principal stress, indexed like the input DataFrame and named
            ``'signed_mises_abs_max_principal'``.

        See Also
        --------
        pylife.stress.equistress.signed_mises_abs_max_principal : Calculate
            the same quantity from component arrays.
        """
        return pd.Series(signed_mises_abs_max_principal(s11=self._obj['S11'].to_numpy(),
                                                        s22=self._obj['S22'].to_numpy(),
                                                        s33=self._obj['S33'].to_numpy(),
                                                        s12=self._obj['S12'].to_numpy(),
                                                        s13=self._obj['S13'].to_numpy(),
                                                        s23=self._obj['S23'].to_numpy()),
                         name='signed_mises_abs_max_principal', index=self._obj.index)

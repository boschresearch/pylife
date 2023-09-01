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

"""

Equivalent Stresses
===================

Library to calculate the equivalent stress values of a FEM stress tensor.

By now the following calculation methods are implemented:

* Principal stresses
* Maximum principal stress
* Minimum principal stress
* Absolute maximum principal stress
* Von Mises
* Signed von Mises, sign from trace
* Signed von Mises, sign from absolute maximum principal stress
* Tresca
* Signed Tresca, sign from trace
* Signed Tresca, sign from absolute maximum principal stress

"""

__author__ = "Johannes Mueller, Vivien Le Baube et. al."
__maintainer__ = "Johannes Mueller"

import numpy as np
import pandas as pd
from pylife.stress import stresssignal


def eigenval(s11, s22, s33, s12, s13, s23):
    """Calculate eigenvalues of a symmetric 3D tensor.

    Parameters
    ----------
    s11: array_like
        Component 11 of 3D tensor.
    s22: array_like
        Component 22 of 3D tensor.
    s33: array_like
        Component 33 of 3D tensor.
    s12: array_like
        Component 12 of 3D tensor.
    s13: array_like
        Component 13 of 3D tensor.
    s23: array_like
        Component 23 of 3D tensor.

    Returns
    -------
    numpy.ndarray:
        Array containing eigenvalues sorted in ascending order.
        Shape is (length of components, 3) or simply 3 if components are single
        values.
    """
    a = np.array([[s11, s12, s13],
                  [s12, s22, s23],
                  [s13, s23, s33]]).T
    return np.linalg.eigvalsh(a)


def _sign_trace(s11, s22, s33):
    """Calculate sign of trace. Sign of 0 is set to 1.

    Parameters
    ----------
    s11: array_like
        Component 11 of 3D tensor.
    s22: array_like
        Component 22 of 3D tensor.
    s33: array_like
        Component 33 of 3D tensor.

    Returns
    -------
    numpy.ndarray:
        Array containing sign of trace. Shape is the same as the components.
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
    """Calculate sign of absolute maximum principal stress. Sign of 0 is set to
     1.

    Parameters
    ----------
    s11: array_like
        Component 11 of 3D tensor.
    s22: array_like
        Component 22 of 3D tensor.
    s33: array_like
        Component 33 of 3D tensor.
    s12: array_like
        Component 12 of 3D tensor.
    s13: array_like
        Component 13 of 3D tensor.
    s23: array_like
        Component 23 of 3D tensor.

    Returns
    -------
    numpy.ndarray:
        Array containing sign of absolute maximum principal stress. Shape is the
        same as the components.
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
    """Calculate equivalent stress according to Tresca.

    Parameters
    ----------
    s11: array_like
        Component 11 of 3D tensor.
    s22: array_like
        Component 22 of 3D tensor.
    s33: array_like
        Component 33 of 3D tensor.
    s12: array_like
        Component 12 of 3D tensor.
    s13: array_like
        Component 13 of 3D tensor.
    s23: array_like
        Component 23 of 3D tensor.

    Returns
    -------
    numpy.ndarray:
        Equivalent Tresca stress. Shape is the same as the components.
    """
    w = eigenval(s11, s22, s33, s12, s13, s23).T
    w_diff = np.zeros(w.shape)
    w_diff[0] = np.fabs(w[0] - w[1])
    w_diff[1] = np.fabs(w[0] - w[2])
    w_diff[2] = np.fabs(w[1] - w[2])
    return np.amax(w_diff, axis=0)


def signed_tresca_trace(s11, s22, s33, s12, s13, s23):
    """Calculate equivalent stress according to Tresca, signed with the sign
    of the trace (i.e s11 + s22 + s33).

    Parameters
    ----------
    s11: array_like
        Component 11 of 3D tensor.
    s22: array_like
        Component 22 of 3D tensor.
    s33: array_like
        Component 33 of 3D tensor.
    s12: array_like
        Component 12 of 3D tensor.
    s13: array_like
        Component 13 of 3D tensor.
    s23: array_like
        Component 23 of 3D tensor.

    Returns
    -------
    numpy.ndarray:
        Signed Tresca equivalent stress. Shape is the same as the components.
    """
    return _sign_trace(s11, s22, s33) * tresca(s11, s22, s33, s12, s13, s23)


def signed_tresca_abs_max_principal(s11, s22, s33, s12, s13, s23):
    """Calculate equivalent stress according to Tresca, signed with the sign
    of the absolute maximum principal stress.

    Parameters
    ----------
    s11: array_like
        Component 11 of 3D tensor.
    s22: array_like
        Component 22 of 3D tensor.
    s33: array_like
        Component 33 of 3D tensor.
    s12: array_like
        Component 12 of 3D tensor.
    s13: array_like
        Component 13 of 3D tensor.
    s23: array_like
        Component 23 of 3D tensor.

    Returns
    -------
    numpy.ndarray:
        Signed Tresca equivalent stress. Shape is the same as the components.
    """
    return _sign_abs_max_principal(s11, s22, s33, s12, s13, s23) * tresca(s11, s22, s33, s12, s13, s23)


def abs_max_principal(s11, s22, s33, s12, s13, s23):
    """Calculate absolute maximum principal stress (maximum of absolute
    eigenvalues with corresponding sign).

    Parameters
    ----------
    s11: array_like
        Component 11 of 3D tensor.
    s22: array_like
        Component 22 of 3D tensor.
    s33: array_like
        Component 33 of 3D tensor.
    s12: array_like
        Component 12 of 3D tensor.
    s13: array_like
        Component 13 of 3D tensor.
    s23: array_like
        Component 23 of 3D tensor.

    Returns
    -------
    numpy.ndarray:
        Absolute maximum principal stress. Shape is the same as the components.
    """
    w = eigenval(s11, s22, s33, s12, s13, s23).T
    w_max = np.amax(w, axis=0)
    w_min = np.amin(w, axis=0)
    sign = _sign_abs_max_principal(s11, s22, s33, s12, s13, s23)
    positive_sign_bool = np.array(sign >= 0)
    return w_max * positive_sign_bool + w_min * np.invert(positive_sign_bool)


def principals(s11, s22, s33, s12, s13, s23):
    """Calculate all principal stress components (eigenvalues).

    Parameters
    ----------
    s11: array_like
        Component 11 of 3D tensor.
    s22: array_like
        Component 22 of 3D tensor.
    s33: array_like
        Component 33 of 3D tensor.
    s12: array_like
        Component 12 of 3D tensor.
    s13: array_like
        Component 13 of 3D tensor.
    s23: array_like
        Component 23 of 3D tensor.

    Returns
    -------
    numpy.ndarray:
        All principal stresses. Shape `(..., 3)`.
    """
    return eigenval(s11, s22, s33, s12, s13, s23)


def max_principal(s11, s22, s33, s12, s13, s23):
    """Calculate maximum principal stress (maximum of eigenvalues).

    Parameters
    ----------
    s11: array_like
        Component 11 of 3D tensor.
    s22: array_like
        Component 22 of 3D tensor.
    s33: array_like
        Component 33 of 3D tensor.
    s12: array_like
        Component 12 of 3D tensor.
    s13: array_like
        Component 13 of 3D tensor.
    s23: array_like
        Component 23 of 3D tensor.

    Returns
    -------
    numpy.ndarray:
        Maximum principal stress. Shape is the same as the components.
    """
    w = eigenval(s11, s22, s33, s12, s13, s23).T
    return np.amax(w, axis=0)


def min_principal(s11, s22, s33, s12, s13, s23):
    """Calculate minimum principal stress (minimum of eigenvalues).

    Parameters
    ----------
    s11: array_like
        Component 11 of 3D tensor.
    s22: array_like
        Component 22 of 3D tensor.
    s33: array_like
        Component 33 of 3D tensor.
    s12: array_like
        Component 12 of 3D tensor.
    s13: array_like
        Component 13 of 3D tensor.
    s23: array_like
        Component 23 of 3D tensor.

    Returns
    -------
    numpy.ndarray:
        Minimum principal stress. Shape is the same as the components.
    """
    w = eigenval(s11, s22, s33, s12, s13, s23).T
    return np.amin(w, axis=0)


def mises(s11, s22, s33, s12, s13, s23):
    """Calculate equivalent stress according to von Mises.

    Parameters
    ----------
    s11: array_like
        Component 11 of 3D tensor.
    s22: array_like
        Component 22 of 3D tensor.
    s33: array_like
        Component 33 of 3D tensor.
    s12: array_like
        Component 12 of 3D tensor.
    s13: array_like
        Component 13 of 3D tensor.
    s23: array_like
        Component 23 of 3D tensor.

    Returns
    -------
    numpy.ndarray:
        Von Mises equivalent stress. Shape is the same as the components.

    Raises
    ------
    AssertionError
        Components' shape is not consistent.
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
    """Calculate equivalent stress according to von Mises, signed with the sign
    of the trace (i.e s11 + s22 + s33).

    Parameters
    ----------
    s11: array_like
        Component 11 of 3D tensor.
    s22: array_like
        Component 22 of 3D tensor.
    s33: array_like
        Component 33 of 3D tensor.
    s12: array_like
        Component 12 of 3D tensor.
    s13: array_like
        Component 13 of 3D tensor.
    s23: array_like
        Component 23 of 3D tensor.

    Returns
    -------
    numpy.ndarray:
        Signed von Mises equivalent stress. Shape is the same as the components.
    """
    return _sign_trace(s11, s22, s33) * mises(s11, s22, s33, s12, s13, s23)


def signed_mises_abs_max_principal(s11, s22, s33, s12, s13, s23):
    """Calculate equivalent stress according to von Mises, signed with the sign
    of the absolute maximum principal stress.

    Parameters
    ----------
    s11: array_like
        Component 11 of 3D tensor.
    s22: array_like
        Component 22 of 3D tensor.
    s33: array_like
        Component 33 of 3D tensor.
    s12: array_like
        Component 12 of 3D tensor.
    s13: array_like
        Component 13 of 3D tensor.
    s23: array_like
        Component 23 of 3D tensor.

    Returns
    -------
    numpy.ndarray:
        Signed von Mises equivalent stress. Shape is the same as the components.
    """
    return _sign_abs_max_principal(s11, s22, s33, s12, s13, s23) * mises(s11, s22, s33, s12, s13, s23)


@pd.api.extensions.register_dataframe_accessor("equistress")
class StressTensorEquistress(stresssignal.StressTensorVoigt):
    def tresca(self):
        return pd.Series(tresca(s11=self._obj['S11'].to_numpy(),
                                s22=self._obj['S22'].to_numpy(),
                                s33=self._obj['S33'].to_numpy(),
                                s12=self._obj['S12'].to_numpy(),
                                s13=self._obj['S13'].to_numpy(),
                                s23=self._obj['S23'].to_numpy()),
                         name='tresca', index=self._obj.index)

    def signed_tresca_trace(self):
        return pd.Series(signed_tresca_trace(s11=self._obj['S11'].to_numpy(),
                                             s22=self._obj['S22'].to_numpy(),
                                             s33=self._obj['S33'].to_numpy(),
                                             s12=self._obj['S12'].to_numpy(),
                                             s13=self._obj['S13'].to_numpy(),
                                             s23=self._obj['S23'].to_numpy()),
                         name='signed_tresca_trace', index=self._obj.index)

    def signed_tresca_abs_max_principal(self):
        return pd.Series(signed_tresca_abs_max_principal(s11=self._obj['S11'].to_numpy(),
                                                         s22=self._obj['S22'].to_numpy(),
                                                         s33=self._obj['S33'].to_numpy(),
                                                         s12=self._obj['S12'].to_numpy(),
                                                         s13=self._obj['S13'].to_numpy(),
                                                         s23=self._obj['S23'].to_numpy()),
                         name='signed_tresca_abs_max_principal', index=self._obj.index)

    def principals(self):
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
        return pd.Series(abs_max_principal(s11=self._obj['S11'].to_numpy(),
                                           s22=self._obj['S22'].to_numpy(),
                                           s33=self._obj['S33'].to_numpy(),
                                           s12=self._obj['S12'].to_numpy(),
                                           s13=self._obj['S13'].to_numpy(),
                                           s23=self._obj['S23'].to_numpy()),
                         name='abs_max_principal', index=self._obj.index)

    def max_principal(self):
        return pd.Series(max_principal(s11=self._obj['S11'].to_numpy(),
                                       s22=self._obj['S22'].to_numpy(),
                                       s33=self._obj['S33'].to_numpy(),
                                       s12=self._obj['S12'].to_numpy(),
                                       s13=self._obj['S13'].to_numpy(),
                                       s23=self._obj['S23'].to_numpy()),
                         name='max_principal', index=self._obj.index)

    def min_principal(self):
        return pd.Series(min_principal(s11=self._obj['S11'].to_numpy(),
                                       s22=self._obj['S22'].to_numpy(),
                                       s33=self._obj['S33'].to_numpy(),
                                       s12=self._obj['S12'].to_numpy(),
                                       s13=self._obj['S13'].to_numpy(),
                                       s23=self._obj['S23'].to_numpy()),
                         name='min_principal', index=self._obj.index)

    def mises(self):
        return pd.Series(mises(s11=self._obj['S11'].to_numpy(),
                               s22=self._obj['S22'].to_numpy(),
                               s33=self._obj['S33'].to_numpy(),
                               s12=self._obj['S12'].to_numpy(),
                               s13=self._obj['S13'].to_numpy(),
                               s23=self._obj['S23'].to_numpy()),
                         name='mises', index=self._obj.index)

    def signed_mises_trace(self):
        return pd.Series(signed_mises_trace(s11=self._obj['S11'].to_numpy(),
                                            s22=self._obj['S22'].to_numpy(),
                                            s33=self._obj['S33'].to_numpy(),
                                            s12=self._obj['S12'].to_numpy(),
                                            s13=self._obj['S13'].to_numpy(),
                                            s23=self._obj['S23'].to_numpy()),
                         name='signed_mises_trace', index=self._obj.index)

    def signed_mises_abs_max_principal(self):
        return pd.Series(signed_mises_abs_max_principal(s11=self._obj['S11'].to_numpy(),
                                                        s22=self._obj['S22'].to_numpy(),
                                                        s33=self._obj['S33'].to_numpy(),
                                                        s12=self._obj['S12'].to_numpy(),
                                                        s13=self._obj['S13'].to_numpy(),
                                                        s23=self._obj['S23'].to_numpy()),
                         name='signed_mises_abs_max_principal', index=self._obj.index)

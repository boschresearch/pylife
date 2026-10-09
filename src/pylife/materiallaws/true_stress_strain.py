# Copyright (c) 2019-2026 - for information on the respective copyright owner
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

"""Convert technical tensile-test quantities to true stress-strain values.

The helpers in this module implement the logarithmic strain and area-corrected
stress quantities commonly used in the FKM nonlinear static assessment.
"""

__author__ = "Simone Schreijäg"
__maintainer__ = "Johannes Mueller"

import numpy as np


def true_strain(tech_strain):
    r"""Calculate true strain from technical strain.

    Parameters
    ----------
    tech_strain : array_like
        Technical strain (engineering strain), dimensionless.

    Returns
    -------
    array_like
        True logarithmic strain, dimensionless.

    Notes
    -----
    The calculation follows the standard conversion used for tensile-test data
    before necking:

    .. math::

        \varepsilon_\mathrm{true} = \ln(1 + \varepsilon_\mathrm{tech})

    Examples
    --------
    >>> from pylife.materiallaws.true_stress_strain import true_strain
    >>> round(float(true_strain(0.1)), 6)
    0.09531
    """
    return np.log(1. + tech_strain)


def true_stress(tech_stress, tech_strain):
    r"""Calculate true stress from technical stress and strain.

    Parameters
    ----------
    tech_stress : array_like
        Technical stress (engineering stress) in MPa or another consistent
        force-per-area unit.
    tech_strain : array_like
        Technical strain (engineering strain), dimensionless.

    Returns
    -------
    array_like
        True stress in the same unit as ``tech_stress``.

    Notes
    -----
    The conversion assumes volume constancy and uniform elongation:

    .. math::

        \sigma_\mathrm{true} = \sigma_\mathrm{tech}
        (1 + \varepsilon_\mathrm{tech})
    """
    return tech_stress * (1. + tech_strain)


def true_fracture_strain(reduction_area_fracture):
    r"""Calculate true fracture strain from reduction of area.

    Parameters
    ----------
    reduction_area_fracture : float
        Relative reduction of cross-sectional area at fracture, dimensionless.
        Use a value between ``0.0`` and ``1.0``.

    Returns
    -------
    float
        True fracture strain, dimensionless.

    Notes
    -----
    This quantity is used by the FKM nonlinear static assessment and is based
    on the measured reduction of area after fracture:

    .. math::

        \varepsilon_\mathrm{f,true} = \ln \left(\frac{1}{1 - Z}\right)
    """
    return np.log(1./(1. - reduction_area_fracture))


def true_fracture_stress(fracture_force, initial_cross_section, reduction_area_fracture):
    r"""Calculate true fracture stress from force and reduced area.

    Parameters
    ----------
    fracture_force : float
        Force at fracture in N or another consistent force unit.
    initial_cross_section : float
        Initial cross-sectional area of the tensile specimen in mm² or another
        consistent area unit.
    reduction_area_fracture : float
        Relative reduction of cross-sectional area at fracture, dimensionless.
        Use a value between ``0.0`` and ``1.0``.

    Returns
    -------
    float
        True fracture stress in the force-per-area unit implied by
        ``fracture_force`` and ``initial_cross_section``.

    Notes
    -----
    The FKM nonlinear static assessment uses the fracture force divided by the
    remaining area at fracture:

    .. math::

        \sigma_\mathrm{f,true} = \frac{F_\mathrm{f}}{S_0 (1 - Z)}
    """
    return fracture_force/(initial_cross_section * (1. - reduction_area_fracture))

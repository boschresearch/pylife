#!/usr/bin/env python3
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

"""Plot a fitted Wöhler curve next to its fatigue data as an eye-catcher PNG.

Reads a two column fatigue test data CSV file (load, cycles), fits a Wöhler
curve with :class:`pylife.materialdata.woehler.MaxLikeFull` -- the same
"Maximum Likelihood Full" method shown in ``demos/woehler_analyzer.ipynb`` --
and plots the fractures and runouts together with the fitted curve's scatter
band. There are no legends, axis labels or grid lines, aimed purely at visual
appeal (e.g. for a website eye-catcher).

Example
-------
::

    python scripts/woehler_analyzer.py --background dark --output woehler_analyzer.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle

import pylife.materialdata.woehler as woehler
from pylife.materiallaws import WoehlerCurve  # noqa: F401  (registers the .woehler accessor)

REPO_ROOT = Path(__file__).resolve().parent.parent

# The two greens of the pyLife logo.
DARK_GREEN = "#006249"
LIGHT_GREEN = "#78be20"
DARK_BACKGROUND = "#1b1f23"\


BACKGROUND_SETTINGS = {
    "light": {
        "background_color": "#ffffff",
        "data_color": DARK_GREEN,
        "band_colors": ["#ffffff", "#008ecf"],
    },
    "dark": {
        "background_color": DARK_BACKGROUND,
        "data_color": LIGHT_GREEN,
        "band_colors": [DARK_BACKGROUND, "#008ecf"]
    },
}


def read_fatigue_data(input_path, load_cycle_limit):
    """Read a two column fatigue test data CSV file into a ``fatigue_data`` accessor.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Path of the tab separated CSV file containing a ``load`` and a
        ``cycles`` column.
    load_cycle_limit : float or None
        Cycle number above which a test point is guessed to be a runout, see
        :func:`pylife.materialdata.woehler.determine_fractures`. If ``None``
        the default of that function is used.

    Returns
    -------
    pylife fatigue_data accessor
        The validated fatigue data.
    """
    df = pd.read_csv(input_path, sep='\t')
    df.columns = ['load', 'cycles']
    df = woehler.determine_fractures(df, load_cycle_limit)
    return df.fatigue_data


def fit_woehler_curve(fatigue_data):
    """Fit a Wöhler curve with the Maximum Likelihood Full method.

    Parameters
    ----------
    fatigue_data : pylife fatigue_data accessor
        The fatigue test data to fit the curve to.

    Returns
    -------
    pylife.materiallaws.WoehlerCurve
        The fitted Wöhler curve accessor.
    """
    result = woehler.MaxLikeFull(fatigue_data).analyze()
    return result.woehler


def plot(fatigue_data, wc, background):
    """Plot the fatigue data and the fitted Wöhler curve as an eye-catcher.

    Parameters
    ----------
    fatigue_data : pylife fatigue_data accessor
        The fatigue test data to scatter plot.
    wc : pylife.materiallaws.WoehlerCurve
        The fitted Wöhler curve accessor to plot.
    background : str
        ``"light"`` or ``"dark"``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    settings = BACKGROUND_SETTINGS[background]

    cycles_min = min(fatigue_data.cycles.min(), wc.ND / 10.0)
    cycles_max = max(fatigue_data.cycles.max(), wc.ND * 10.0)
    cycles = np.geomspace(cycles_min, cycles_max, 500)

    load_10 = wc.basquin_load(cycles, failure_probability=0.1)
    load_90 = wc.basquin_load(cycles, failure_probability=0.9)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(settings["background_color"])
    ax.set_facecolor(settings["background_color"])

    _fill_band_gradient(ax, cycles, load_10, load_90, settings["band_colors"], zorder=1)

    dashed_color = settings["band_colors"][-1]
    ax.plot(cycles, load_10, '--', color=dashed_color, linewidth=1.0, alpha=0.8, zorder=2)
    ax.plot(cycles, load_90, '--', color=dashed_color, linewidth=1.0, alpha=0.8, zorder=2)

    data_color = settings["data_color"]
    fractures = fatigue_data.fractures
    runouts = fatigue_data.runouts
    ax.scatter(fractures.cycles, fractures.load, marker='o', s=40,
               facecolors=data_color, edgecolors=data_color, zorder=4)
    ax.scatter(runouts.cycles, runouts.load, marker='o', s=40,
               facecolors='none', edgecolors=data_color, linewidths=1.5, zorder=4)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(cycles_min, cycles_max)
    ax.set_ylim(min(load_10.min(), fatigue_data.load.min()) * 0.9,
                max(load_90.max(), fatigue_data.load.max()) * 1.1)
    ax.axis('off')

    frame_color = 'white' if background == 'dark' else 'black'
    ax.add_patch(Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes, fill=False,
        edgecolor=frame_color, linewidth=0.8, zorder=10, clip_on=False,
    ))

    return fig


def _fill_band_gradient(ax, cycles, load_10, load_90, colors, zorder, center_alpha=0.75, edge_alpha=0.75):
    """Fill the scatter band with a gradient perpendicular to the curve.

    For every cycle count the color is most intense at the 50 % line (the
    geometric mean of ``load_10`` and ``load_90`` in log space) and fades
    out towards the 10 % and 90 % boundary lines, giving the band a glow
    that follows its local thickness rather than the plot's absolute
    vertical axis. Identical to the helper of the same name in
    ``woehler_collective.py``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    cycles : numpy.ndarray
        Cycle counts the band is evaluated at.
    load_10 : numpy.ndarray
        Lower (10 %) boundary loads matching ``cycles``.
    load_90 : numpy.ndarray
        Upper (90 %) boundary loads matching ``cycles``.
    colors : list of str
        Two color stops (hex strings): the edge color the band fades from
        and the center color it fades to at the 50 % line.
    zorder : float
        Drawing order of the filled gradient.
    center_alpha : float
        Opacity at the center (50 % line); fades to ``edge_alpha`` at the
        band edges.
    edge_alpha : float
        Opacity of the solid edge color at the band boundaries.
    """
    n_rows = 60
    edge_color = to_rgb(colors[0])
    center_color = to_rgb(colors[-1])
    cmap = LinearSegmentedColormap.from_list(
        'band_perp',
        [(*center_color, center_alpha), (*edge_color, edge_alpha)],
    )

    distance_from_center = np.abs(np.linspace(-1.0, 1.0, n_rows))

    x_grid = np.tile(cycles, (n_rows, 1))
    log_lower, log_upper = np.log(load_10), np.log(load_90)
    row_fraction = np.linspace(0.0, 1.0, n_rows).reshape(-1, 1)
    y_grid = np.exp(log_lower + row_fraction * (log_upper - log_lower))
    color_values = np.tile(distance_from_center.reshape(-1, 1), (1, len(cycles)))

    ax.pcolormesh(
        x_grid, y_grid, color_values, cmap=cmap, vmin=0.0, vmax=1.0,
        shading='gouraud', zorder=zorder, rasterized=True,
    )


def parse_args():
    """Parse the command line arguments.

    Returns
    -------
    argparse.Namespace
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input",
        type=Path, default=REPO_ROOT / "demos" / "data" / "woehler" / "fatigue-data-plain.csv",
        help="the tab separated CSV file containing 'load' and 'cycles' columns (default: %(default)s)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path, default=Path("woehler_analyzer.png"),
        help="the PNG file to write the plot to (default: %(default)s)",
    )
    parser.add_argument(
        "--load-cycle-limit",
        type=float, default=None,
        help="cycle number above which a test point is guessed to be a runout "
             "(default: use pylife's own default)",
    )
    parser.add_argument(
        "-b", "--background",
        choices=sorted(BACKGROUND_SETTINGS), default="light",
        help="the background color of the plot (default: %(default)s)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    fatigue_data = read_fatigue_data(args.input, args.load_cycle_limit)
    wc = fit_woehler_curve(fatigue_data)

    fig = plot(fatigue_data, wc, args.background)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.1,
                facecolor=fig.get_facecolor())

    print(f"Wöhler curve and fatigue data plot written to {args.output}")


if __name__ == "__main__":
    main()

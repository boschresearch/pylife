#!/usr/bin/env python3
# Copyright (c) 2019-2025 - for information on the respective copyright owner
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

"""Plot a Wöhler curve next to a load collective as an eye-catcher PNG.

Reads a one column time signal CSV file, performs a rainflow count with
:class:`pylife.stress.rainflow.FourPointDetector` to derive a cumulative load
collective, scales it to a plausible safety distance below a Wöhler curve and
plots both without any axis labels, aimed purely at visual appeal (e.g. for a
website eye-catcher).

Example
-------
::

    python scripts/woehler_collective.py --input load_signal.csv --output woehler_collective.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import pylife.stress.collective  # noqa: F401  (registers the .load_collective accessor)
import pylife.stress.rainflow as RF
from pylife.materiallaws import WoehlerCurve  # noqa: F401  (registers the .woehler accessor)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Gradients taken from the pyLife logo: dark to light green under the
# collective, white to blue across the Wöhler scatter band. Both outlines
# use the lighter color of their respective gradient.
GRADIENT_COLLECTIVE = ["#006249", "#78be20"]
GRADIENT_WOEHLER_BAND = ["#ffffff", "#008ecf"]

COLOR_COLLECTIVE = GRADIENT_COLLECTIVE[-1]
COLOR_WOEHLER = GRADIENT_WOEHLER_BAND[0]

BACKGROUND_COLORS = {
    "transparent": None,
    "light": "#ffffff",
    "dark": "#1b1f23",
}


def read_signal(input_path):
    """Read a one column time signal CSV file into a numpy array.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Path of the CSV file containing one load sample per line.

    Returns
    -------
    numpy.ndarray
        The load signal as a 1D array.
    """
    return np.loadtxt(input_path)


def rainflow_collective(signal, bins):
    """Perform a rainflow count and derive a cumulative load collective.

    Parameters
    ----------
    signal : array-like
        The load time signal.
    bins : int
        The number of amplitude classes used for the histogram.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray)
        ``(amplitude, cumulated_cycles)`` sorted by descending amplitude,
        with zero-count classes removed.
    """
    detector = RF.FourPointDetector(recorder=RF.LoopValueRecorder())
    detector.process(signal)
    histogram = detector.recorder.histogram(bins)

    collective = histogram.load_collective
    amplitude = np.asarray(collective.amplitude)
    cycles = np.asarray(collective.cycles)

    order = np.argsort(amplitude)[::-1]
    amplitude = amplitude[order]
    cycles = cycles[order]

    nonzero = cycles > 0
    amplitude = amplitude[nonzero]
    cumulated_cycles = np.cumsum(cycles[nonzero])

    return amplitude, cumulated_cycles


def truncate_collective(amplitude, cumulated_cycles, cutoff):
    """Drop the low-amplitude tail of a load collective.

    Parameters
    ----------
    amplitude : numpy.ndarray
        Amplitudes sorted in descending order.
    cumulated_cycles : numpy.ndarray
        Cumulated cycle counts matching ``amplitude``.
    cutoff : float
        Classes with an amplitude below ``cutoff`` times the maximum
        amplitude are dropped.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray)
        The truncated ``(amplitude, cumulated_cycles)``.
    """
    keep = amplitude >= cutoff * amplitude[0]
    return amplitude[keep], cumulated_cycles[keep]


def stretch_collective_cycles(cumulated_cycles, ND, span_fraction):
    """Rescale the collective's cycle axis relative to the curve's life range.

    The collective is drawn purely for visual appeal, so its cycle axis is
    stretched (keeping its shape) until its total number of cycles equals
    ``span_fraction`` times ``ND``. This makes the collective span a cycle
    range that visually overlaps the inclined, finite-life part of the
    Wöhler curve instead of being squeezed into a tiny corner of the plot.

    Parameters
    ----------
    cumulated_cycles : numpy.ndarray
        Cumulated cycle counts sorted in ascending order.
    ND : float
        Number of cycles at the Wöhler curve knee point.
    span_fraction : float
        Target ratio between the collective's total cycles and ``ND``.

    Returns
    -------
    numpy.ndarray
        The rescaled cumulated cycle counts.
    """
    scale = (span_fraction * ND) / cumulated_cycles[-1]
    return cumulated_cycles * scale


def woehler_curve(k_1, ND, SD, TS):
    """Build a Wöhler curve accessor with an equally wide scatter band.

    Parameters
    ----------
    k_1 : float
        Finite-life Wöhler slope.
    ND : float
        Number of cycles at the knee point.
    SD : float
        Load amplitude at the knee point.
    TS : float
        Scatter range in load direction. The scatter range in cycle
        direction ``TN`` is derived as ``TS ** k_1`` by the accessor so
        both scatter bands appear equally wide on a log-log plot.

    Returns
    -------
    pylife.materiallaws.WoehlerCurve
        The Wöhler curve accessor.
    """
    return pd.Series({'k_1': k_1, 'ND': ND, 'SD': SD, 'TS': TS}).woehler


def scale_collective(amplitude, cumulated_cycles, wc, safety_factor):
    """Scale a load collective to a given safety factor below the curve.

    The collective is scaled uniformly in load direction so that, at its
    tightest point, it stays ``safety_factor`` below the 10 % failure
    probability curve. This keeps the whole collective below the curve's
    scatter band rather than just its peak class.

    Parameters
    ----------
    amplitude : numpy.ndarray
        Amplitudes sorted in descending order.
    cumulated_cycles : numpy.ndarray
        Cumulated cycle counts matching ``amplitude``.
    wc : pylife.materiallaws.WoehlerCurve
        The Wöhler curve accessor to scale against.
    safety_factor : float
        Target ratio between the 10 % curve and the scaled collective at
        their closest point.

    Returns
    -------
    numpy.ndarray
        The scaled amplitudes.
    """
    load_at_10_percent = np.asarray(wc.basquin_load(cumulated_cycles, failure_probability=0.1))
    scale = np.min(load_at_10_percent / amplitude) / safety_factor
    return amplitude * scale


def plot(amplitude, cumulated_cycles, wc, background, collective_gradient):
    """Plot the Wöhler curve and the load collective as an eye-catcher.

    Parameters
    ----------
    amplitude : numpy.ndarray
        Scaled collective amplitudes sorted in descending order.
    cumulated_cycles : numpy.ndarray
        Cumulated cycle counts matching ``amplitude``.
    wc : pylife.materiallaws.WoehlerCurve
        The Wöhler curve accessor to plot.
    background : str
        One of ``"transparent"``, ``"light"`` or ``"dark"``.
    collective_gradient : str
        One of ``"slope"`` or ``"level"``, see
        :func:`_fill_collective_gradient`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    cycles_min = min(cumulated_cycles[0], wc.ND / 50.0)
    cycles_max = max(cumulated_cycles[-1], wc.ND * 50.0)
    cycles = np.geomspace(cycles_min, cycles_max, 500)

    load_10 = wc.basquin_load(cycles, failure_probability=0.1)
    load_90 = wc.basquin_load(cycles, failure_probability=0.9)

    face_color = BACKGROUND_COLORS[background]
    fig, ax = plt.subplots(figsize=(8, 5))
    if face_color is not None:
        fig.patch.set_facecolor(face_color)
        ax.set_facecolor(face_color)

    # On a dark background the gradient is inverted so the band still reads
    # as a glow: blue at the edges fading to white at the 50 % line.
    band_colors = list(reversed(GRADIENT_WOEHLER_BAND)) if background == 'dark' else GRADIENT_WOEHLER_BAND
    _fill_band_gradient(ax, cycles, load_10, load_90, band_colors, zorder=1)

    dashed_color = band_colors[0]
    ax.plot(cycles, load_10, '--', color=dashed_color, linewidth=1.0, alpha=0.8, zorder=2)
    ax.plot(cycles, load_90, '--', color=dashed_color, linewidth=1.0, alpha=0.8, zorder=2)

    collective_baseline = min(amplitude[-1], load_10.min()) * 0.8
    collective_polygon = np.vstack([
        np.column_stack([cumulated_cycles, amplitude]),
        [[cumulated_cycles[-1], collective_baseline], [cumulated_cycles[0], collective_baseline]],
    ])
    _fill_collective_gradient(
        ax, collective_polygon, cumulated_cycles, amplitude, GRADIENT_COLLECTIVE, collective_gradient,
        wc.k_1, alpha=0.65, zorder=4,
    )
    collective_outline = np.vstack([collective_polygon, collective_polygon[0]])
    ax.plot(collective_outline[:, 0], collective_outline[:, 1], '-', color=COLOR_COLLECTIVE, linewidth=1.5, zorder=5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(cycles_min, cycles_max)
    ax.set_ylim(collective_baseline, max(load_90.max(), amplitude[0]) * 1.1)
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
    vertical axis.

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
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

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


def _fill_collective_gradient(
    ax, polygon_points, curve_cycles, curve_amplitude, colors, mode, k_1, alpha, zorder,
):
    """Fill the area under the load collective with a directional gradient.

    Two gradient directions are supported, both computed in
    ``(log N, log load)`` space so they stay consistent across the log-log
    plot:

    * ``"slope"`` : the gradient direction is ``(1, -1 / k_1)``, i.e. the
      direction of the Wöhler curve itself (a straight line of slope
      ``-1 / k_1`` in log-log space, Basquin's equation). The multi-stop
      ``colors`` are interpolated along that direction, so the gradient's
      bands run parallel to the curve above the collective.
    * ``"level"`` : the gradient runs left to right and follows the load
      level of the collective curve itself: every vertical column of the
      fill is colored uniformly according to the curve's own height
      (``curve_amplitude`` interpolated at that cycle count), brightest
      where the collective is at its highest load level and fading to
      fully transparent where it is at its lowest, independent of the
      mesh's absolute vertical position.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    polygon_points : numpy.ndarray
        Array of shape ``(n, 2)`` with the ``(x, y)`` points outlining the
        polygon to fill, in data coordinates.
    curve_cycles : numpy.ndarray
        Cumulated cycle counts of the collective curve (ascending),
        used by the ``"level"`` gradient to look up the local load level.
    curve_amplitude : numpy.ndarray
        Collective amplitudes matching ``curve_cycles``.
    colors : list of str
        Color stops (hex strings) for the ``"slope"`` gradient; only the
        last (brightest) stop is used as the base color for the ``"level"``
        gradient.
    mode : str
        ``"slope"`` or ``"level"``, see above.
    k_1 : float
        Finite-life Wöhler slope used to derive the ``"slope"`` gradient
        direction.
    alpha : float
        Opacity of the filled gradient at its most intense point.
    zorder : float
        Drawing order of the filled gradient.
    """
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    from matplotlib.patches import Polygon

    log_x, log_y = np.log10(polygon_points[:, 0]), np.log10(polygon_points[:, 1])

    n_grid = 120
    grid_log_x = np.linspace(log_x.min(), log_x.max(), n_grid)
    grid_log_y = np.linspace(log_y.min(), log_y.max(), n_grid)
    mesh_log_x, mesh_log_y = np.meshgrid(grid_log_x, grid_log_y)

    if mode == 'slope':
        direction = np.array([1.0, -1.0 / k_1])
        direction /= np.linalg.norm(direction)
        rgba_colors = [(*to_rgb(c), alpha) for c in colors]
        cmap = LinearSegmentedColormap.from_list('gradient', rgba_colors)
        projection = mesh_log_x * direction[0] + mesh_log_y * direction[1]
        vertex_projection = log_x * direction[0] + log_y * direction[1]
        vmin, vmax = vertex_projection.min(), vertex_projection.max()
    elif mode == 'level':
        base_rgb = to_rgb(colors[-1])
        cmap = LinearSegmentedColormap.from_list('gradient', [(*base_rgb, 0.0), (*base_rgb, alpha)])
        order = np.argsort(curve_cycles)
        log_curve_x = np.log10(curve_cycles)[order]
        log_curve_y = np.log10(curve_amplitude)[order]
        column_level = np.interp(grid_log_x, log_curve_x, log_curve_y)
        projection = np.tile(column_level, (n_grid, 1))
        vmin, vmax = log_curve_y.min(), log_curve_y.max()
    else:
        raise ValueError(f"Unknown collective gradient mode {mode!r}, expected 'slope' or 'level'")

    mesh = ax.pcolormesh(
        10 ** mesh_log_x, 10 ** mesh_log_y, projection,
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading='nearest', antialiased=False, zorder=zorder,
    )

    clip_path = Polygon(polygon_points, closed=True, transform=ax.transData)
    mesh.set_clip_path(clip_path)


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
        type=Path, default=REPO_ROOT / "load_signal.csv",
        help="the CSV file containing the load signal (default: %(default)s)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path, default=Path("woehler_collective.png"),
        help="the PNG file to write the plot to (default: %(default)s)",
    )
    parser.add_argument(
        "-b", "--bins",
        type=int, default=128,
        help="the number of amplitude classes for the load collective (default: %(default)s)",
    )
    parser.add_argument(
        "--collective-cutoff",
        type=float, default=0.1,
        help="drop collective classes below this fraction of the maximum amplitude (default: %(default)s)",
    )
    parser.add_argument(
        "--k1",
        type=float, default=5.0,
        help="the Wöhler curve finite-life slope k_1 (default: %(default)s)",
    )
    parser.add_argument(
        "--ND",
        type=float, default=1e6,
        help="the number of cycles at the Wöhler curve knee point (default: %(default)s)",
    )
    parser.add_argument(
        "--SD",
        type=float, default=200.0,
        help="the load amplitude at the Wöhler curve knee point (default: %(default)s)",
    )
    parser.add_argument(
        "--TS",
        type=float, default=1.24,
        help="the Wöhler curve scatter range in load direction (default: %(default)s)",
    )
    parser.add_argument(
        "--safety-factor",
        type=float, default=1.08,
        help="target ratio between the 10%% curve and the collective at their closest point (default: %(default)s)",
    )
    parser.add_argument(
        "--collective-span-fraction",
        type=float, default=0.85,
        help="target ratio between the collective's total cycles and ND (default: %(default)s)",
    )
    parser.add_argument(
        "--collective-gradient",
        choices=("slope", "level"), default="slope",
        help="gradient direction under the collective: 'slope' aligns it with the Wöhler curve's "
             "slope, 'level' fades it from the highest to the lowest load level (default: %(default)s)",
    )
    parser.add_argument(
        "--background",
        choices=sorted(BACKGROUND_COLORS), default="transparent",
        help="the figure background (default: %(default)s)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    signal = read_signal(args.input)
    amplitude, cumulated_cycles = rainflow_collective(signal, args.bins)
    amplitude, cumulated_cycles = truncate_collective(amplitude, cumulated_cycles, args.collective_cutoff)

    wc = woehler_curve(args.k1, args.ND, args.SD, args.TS)
    cumulated_cycles = stretch_collective_cycles(cumulated_cycles, wc.ND, args.collective_span_fraction)
    amplitude = scale_collective(amplitude, cumulated_cycles, wc, args.safety_factor)

    fig = plot(amplitude, cumulated_cycles, wc, args.background, args.collective_gradient)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.1,
                transparent=(args.background == "transparent"))

    print(f"Wöhler curve and load collective plot written to {args.output}")


if __name__ == "__main__":
    main()

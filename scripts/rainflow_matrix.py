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

"""Plot a rainflow matrix of a load signal as a PNG image.

Reads a one column time signal CSV file, performs a rainflow count with
:class:`pylife.stress.rainflow.FourPointDetector` and plots the resulting
from/to histogram (the "rainflow matrix") as a PNG image.

Example
-------
::

    python scripts/rainflow_matrix.py --input load_signal.csv --output rainflow_matrix.png \
        --bins 64 --colormap plasma
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import pylife.stress.rainflow as RF

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def rainflow_histogram(signal, bins):
    """Perform a rainflow count and return the from/to histogram.

    Parameters
    ----------
    signal : array-like
        The load time signal.
    bins : int
        The number of bins used for the ``from`` and ``to`` axes.

    Returns
    -------
    pandas.Series
        The histogram of cycle counts indexed by the ``from`` and ``to``
        :class:`pandas.IntervalIndex` levels.
    """
    detector = RF.FourPointDetector(recorder=RF.LoopValueRecorder())
    detector.process(signal)

    return detector.recorder.histogram(bins)


def plot_matrix(histogram, colormap="jet"):
    """Plot a rainflow histogram as a bare pseudocolor matrix.

    The resulting figure only shows the matrix image itself, without axes,
    labels, title or colorbar.

    Parameters
    ----------
    histogram : pandas.Series
        A from/to histogram as returned by :func:`rainflow_histogram`.
    colormap : str, optional
        The name of the matplotlib colormap used to color the matrix
        (default ``"jet"``).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    from_levels, to_levels = histogram.index.levels
    matrix = histogram.to_numpy().reshape(len(from_levels), len(to_levels))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.pcolormesh(matrix.T, cmap=colormap)
    ax.set_aspect("auto")
    ax.axis("off")

    return fig


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
        type=Path, default=Path("rainflow_matrix.png"),
        help="the PNG file to write the rainflow matrix plot to (default: %(default)s)",
    )
    parser.add_argument(
        "-b", "--bins",
        type=int, default=64,
        help="the number of bins for the from/to histogram (default: %(default)s)",
    )
    parser.add_argument(
        "-c", "--colormap",
        type=str, default="jet", metavar="NAME", choices=sorted(matplotlib.colormaps),
        help="the matplotlib colormap used to color the plot (default: %(default)s)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    signal = read_signal(args.input)
    histogram = rainflow_histogram(signal, args.bins)
    fig = plot_matrix(histogram, args.colormap)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0)

    print(f"Rainflow matrix written to {args.output}")


if __name__ == "__main__":
    main()

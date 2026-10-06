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

"""Combine the four "What pyLife can do for you" images into a single row.

Reads the four screenshot PNGs shown in ``docs/index.rst`` (rainflow matrix,
damage calculation, FE mesh and Wöhler analyzer) and arranges them side by
side in a single image, once with a light and once with a dark background,
for use in ``README.md``.

Example
-------
::

    python scripts/overview_image.py \
        --light-output docs/_static/images/overview.png \
        --dark-output docs/_static/images/overview-dark.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = REPO_ROOT / "docs" / "_static" / "images"

LIGHT_BACKGROUND = "white"
DARK_BACKGROUND = "#0d1117"

LIGHT_IMAGES = [
    IMAGES_DIR / "rainflow-matrix-jet.png",
    IMAGES_DIR / "damage-calculation.png",
    IMAGES_DIR / "mesh.png",
    IMAGES_DIR / "woehler_analyzer.png",
]

DARK_IMAGES = [
    IMAGES_DIR / "rainflow-matrix-jet.png",
    IMAGES_DIR / "damage-calculation-dark.png",
    IMAGES_DIR / "mesh.png",
    IMAGES_DIR / "woehler_analyzer_dark.png",
]


def combine_row(image_paths, background, height=2.5, gap=0.15, dpi=150):
    """Arrange a list of images side by side in a single row.

    Each image keeps its own aspect ratio and is scaled to a common height.
    The images are separated by a small gap and drawn on top of a solid
    background color.

    Parameters
    ----------
    image_paths : list of str or pathlib.Path
        The PNG files to combine, from left to right.
    background : str
        The matplotlib color used as background of the combined image.
    height : float, optional
        The height of the combined image in inches (default ``2.5``).
    gap : float, optional
        The width of the gap between two images in inches (default ``0.15``).
    dpi : int, optional
        The resolution of the combined image in dots per inch
        (default ``240``).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the combined row of images.
    """
    images = [plt.imread(path) for path in image_paths]
    aspect_ratios = [image.shape[1] / image.shape[0] for image in images]
    widths = [height * aspect_ratio for aspect_ratio in aspect_ratios]
    total_width = sum(widths) + gap * (len(images) - 1)

    fig = plt.figure(figsize=(total_width, height), dpi=dpi)
    fig.patch.set_facecolor(background)

    left = 0.0
    for image, width in zip(images, widths):
        ax = fig.add_axes((left / total_width, 0, width / total_width, 1))
        ax.set_facecolor(background)
        ax.imshow(image)
        ax.axis("off")
        left += width + gap

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
        "--light-output",
        type=Path, default=IMAGES_DIR / "overview.png",
        help="the PNG file to write the light mode overview to (default: %(default)s)",
    )
    parser.add_argument(
        "--dark-output",
        type=Path, default=IMAGES_DIR / "overview-dark.png",
        help="the PNG file to write the dark mode overview to (default: %(default)s)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    fig = combine_row(LIGHT_IMAGES, LIGHT_BACKGROUND)
    fig.savefig(args.light_output, facecolor=fig.get_facecolor(), pil_kwargs={"optimize": True})
    print(f"Light mode overview written to {args.light_output}")

    fig = combine_row(DARK_IMAGES, DARK_BACKGROUND)
    fig.savefig(args.dark_output, facecolor=fig.get_facecolor(), pil_kwargs={"optimize": True})
    print(f"Dark mode overview written to {args.dark_output}")


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Union

import cairosvg
import matplotlib.pyplot as plt
import skunk
from matplotlib.axes import Axes


def label_axes(
    axs: Axes, fontsize: int = 30, label_pos: tuple[float, float] = (0.0, 1.0)
) -> None:
    for label, ax in axs.items():
        ax.text(
            *label_pos,
            label + "",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=fontsize,
        )
        ax.set(xticks=[], yticks=[])


def format_axes(axs: Axes, panel_borders: bool = False) -> None:
    for label, ax in axs.items():
        if not panel_borders:
            ax.axis("off")


def panel_mosaic(
    mosaic: Union[str, list[list]],
    panel_mapping: dict[str, str],
    figsize: tuple = (10, 8),
    fontsize: int = 30,
    panel_borders: bool = False,
    constrained_layout: bool = True,
    label_pos: tuple = (0, 1),
) -> str:
    fig, axs = plt.subplot_mosaic(
        mosaic=mosaic,
        figsize=figsize,
        constrained_layout=constrained_layout,
        gridspec_kw=dict(hspace=0.0, wspace=0.0),
    )

    label_axes(axs, fontsize=fontsize, label_pos=label_pos)
    format_axes(axs, panel_borders=panel_borders)

    png_panel_mapping = {}
    for label, path in panel_mapping.items():
        if path.endswith(".png"):
            png_panel_mapping[label] = path

    for label, path in png_panel_mapping.items():
        with open(path, "rb") as f:
            img = plt.imread(f)
            # plot inside a box in matplotlib so as to not modify the current axes
            new_ax = axs[label].inset_axes([0.01, 0.01, 0.98, 0.98], zorder=-1)
            new_ax.imshow(img, interpolation="none", aspect=None)
            new_ax.axis("off")

    svg_panel_mapping = {}
    for label, path in panel_mapping.items():
        if path.endswith(".svg"):
            svg_panel_mapping[label] = path

    for label in svg_panel_mapping.keys():
        skunk.connect(axs[label], label)

    svg = skunk.insert(svg_panel_mapping)

    # only necessary to close the figure so that it doesn't display in the notebook
    # maybe not everyone would want this
    plt.close()
    return svg


def write_svg(
    svg: str, out_path: Union[str, Path], formats: tuple = ("svg", "pdf"), show=True
) -> None:
    if show:
        skunk.display(svg)

    out_path = Path(out_path)

    if "svg" in formats:
        with open(str(out_path) + ".svg", "w") as f:
            f.write(svg)

    if "pdf" in formats:
        cairosvg.svg2pdf(bytestring=svg, write_to=str(out_path) + ".pdf")

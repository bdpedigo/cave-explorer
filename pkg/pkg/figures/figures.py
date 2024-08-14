from pathlib import Path

import cairosvg
import matplotlib.pyplot as plt
import skunk


def label_axes(axs, fontsize=30, label_pos=(0, 1)):
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


def format_axes(axs, panel_borders=False):
    for label, ax in axs.items():
        if not panel_borders:
            ax.axis("off")


def panel_mosaic(
    mosaic,
    panel_mapping,
    figsize=(10, 8),
    fontsize=30,
    panel_borders=False,
    constrained_layout=True,
    label_pos=(0, 1),
):
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

            # box = skunk.ImageBox(label, img)
            # axs[label].imshow(img, interpolation="none", aspect=None)

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

    plt.close()
    return svg


def write_svg(svg: str, out_path: Path, formats: tuple = ("svg", "pdf"), show=True):
    if show:
        skunk.display(svg)

    out_path = Path(out_path)
    out_name = out_path.stem

    if "svg" in formats:
        with open(str(out_path) + ".svg", "w") as f:
            f.write(svg)

    if "pdf" in formats:
        cairosvg.svg2pdf(bytestring=svg, write_to=str(out_path) + ".pdf")

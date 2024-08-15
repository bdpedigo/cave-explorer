from pathlib import Path
from typing import Union

import cairosvg
import matplotlib.pyplot as plt
import skunk


class PanelMosaic:
    def __init__(
        self,
        mosaic,
        figsize=(10, 8),
        layout="tight",
        gridspec_kw=None,
    ):
        self.mosaic = mosaic
        self.figsize = figsize
        self.layout = layout

        if gridspec_kw is None:
            gridspec_kw = dict(hspace=0.0, wspace=0.0)

        # ioff/ion is to avoid displaying the matplotlib figure in notebooks, which
        # will just look like a bunch of blue boxes
        plt.ioff()
        self.fig, self.axs = plt.subplot_mosaic(
            mosaic=self.mosaic,
            figsize=self.figsize,
            layout=self.layout,
            gridspec_kw=gridspec_kw,
        )
        plt.ion()

        self.panel_mapping = None

        # self.svg = skunk.pltsvg(self.fig)

    def label_axes(
        self, fontsize: int = 30, label_pos: tuple[float, float] = (0.0, 1.0)
    ) -> None:
        axs = self.axs
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

    def format_axes(self, panel_borders: bool = False) -> None:
        for _, ax in self.axs.items():
            if not panel_borders:
                ax.axis("off")

    def map(self, panel_mapping: dict):
        self.panel_mapping = panel_mapping

        png_panel_mapping = {}
        for label, path in panel_mapping.items():
            if path.endswith(".png"):
                png_panel_mapping[label] = path
        self.png_panel_mapping = png_panel_mapping

        for label, path in png_panel_mapping.items():
            with open(path, "rb") as f:
                img = plt.imread(f)
                new_ax = self.axs[label].inset_axes([0.01, 0.01, 0.98, 0.98], zorder=-1)
                new_ax.imshow(img, interpolation="none", aspect=None)
                new_ax.axis("off")

        svg_panel_mapping = {}
        for label, path in panel_mapping.items():
            if path.endswith(".svg"):
                svg_panel_mapping[label] = path
        self.svg_panel_mapping = svg_panel_mapping

        for label in svg_panel_mapping.keys():
            skunk.connect(self.axs[label], label)

        self.svg = skunk.insert(svg_panel_mapping)

    def __repr__(self) -> str:
        rep = ""
        rep += PanelMosaic.__name__ + "(\n"
        rep += f"    fig={self.fig.__repr__()},\n"
        rep += f"    axs={self.axs.__repr__()},\n"
        rep += f"    panel_mapping={self.panel_mapping.__repr__()},\n"
        rep += ")"
        return rep

    def _repr_pretty_(self, p, cycle):
        # simply show the plot
        """A convenience function to dispaly SVG string in Jupyter Notebook"""
        import base64

        import IPython.display as display

        data = base64.b64encode(self.svg.encode("utf8"))
        display.display(
            display.HTML("<img src=data:image/svg+xml;base64," + data.decode() + ">")
        )

    def show(self) -> None:
        skunk.display(self.svg)

    def write(self, out_path: Union[str, Path], formats=("svg", "pdf")) -> None:
        if "svg" in formats:
            self.write_svg(out_path)
        if "pdf" in formats:
            self.write_pdf(out_path)

    def write_svg(self, out_path: Union[str, Path]) -> None:
        with open(out_path + ".svg", "w") as f:
            f.write(self.svg)

    def write_pdf(self, out_path: Union[str, Path]) -> None:
        cairosvg.svg2pdf(bytestring=self.svg, write_to=str(out_path) + ".pdf")


def panel_mosaic(
    mosaic: Union[str, list[list]],
    panel_mapping: dict[str, str],
    figsize: tuple = (10, 8),
    fontsize: int = 30,
    panel_borders: bool = False,
    layout: str = "tight",
    label_pos: tuple = (0, 1),
) -> str:
    pm = PanelMosaic(
        mosaic=mosaic,
        figsize=figsize,
        layout=layout,
    )
    pm.label_axes(fontsize=fontsize, label_pos=label_pos)
    pm.format_axes(panel_borders=panel_borders)
    pm.map(panel_mapping)
    return pm


# def panel_mosaic(
#     mosaic: Union[str, list[list]],
#     panel_mapping: dict[str, str],
#     figsize: tuple = (10, 8),
#     fontsize: int = 30,
#     panel_borders: bool = False,
#     constrained_layout: bool = True,
#     label_pos: tuple = (0, 1),
# ) -> str:
#     fig, axs = plt.subplot_mosaic(
#         mosaic=mosaic,
#         figsize=figsize,
#         constrained_layout=constrained_layout,
#         gridspec_kw=dict(hspace=0.0, wspace=0.0),
#     )

#     label_axes(axs, fontsize=fontsize, label_pos=label_pos)
#     format_axes(axs, panel_borders=panel_borders)

#     png_panel_mapping = {}
#     for label, path in panel_mapping.items():
#         if path.endswith(".png"):
#             png_panel_mapping[label] = path

#     for label, path in png_panel_mapping.items():
#         with open(path, "rb") as f:
#             img = plt.imread(f)
#             # plot inside a box in matplotlib so as to not modify the current axes
#             new_ax = axs[label].inset_axes([0.01, 0.01, 0.98, 0.98], zorder=-1)
#             new_ax.imshow(img, interpolation="none", aspect=None)
#             new_ax.axis("off")

#     svg_panel_mapping = {}
#     for label, path in panel_mapping.items():
#         if path.endswith(".svg"):
#             svg_panel_mapping[label] = path

#     for label in svg_panel_mapping.keys():
#         skunk.connect(axs[label], label)

#     svg = skunk.insert(svg_panel_mapping)

#     # only necessary to close the figure so that it doesn't display in the notebook
#     # maybe not everyone would want this
#     plt.close()
#     return svg

# %%

from pathlib import Path

import cairosvg
import matplotlib.pyplot as plt
import skunk

mosaic = """
AABDDD
AACDDD
EEE...
EEE...
"""
figsize = (10, 8)
fontsize = 30
panel_borders = False
constrained_layout = True

fig, axs = plt.subplot_mosaic(
    mosaic=mosaic,
    figsize=figsize,
    constrained_layout=constrained_layout,
    gridspec_kw=dict(hspace=0.0, wspace=0.0),
)
for label, ax in axs.items():
    ax.text(
        0,
        1,
        label + "",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=fontsize,
    )
    ax.set(xticks=[], yticks=[])
    if not panel_borders:
        ax.axis("off")
    skunk.connect(axs[label], label)

panel_mapping = {
    "A": "docs/result_images/show_neuron_edits/whole_neuron.svg",
    "B": "docs/result_images/show_neuron_edits/split_example.svg",
    "C": "docs/result_images/show_neuron_edits/merge_example.svg",
    "D": "docs/result_images/show_neuron_edits/neuron_gallery.svg",
    "E": "docs/result_images/simple_stats/edit_count_histogram.svg",
}

svg = skunk.insert(panel_mapping)

skunk.display(svg)

out_path = Path("docs/figures/figure1")
out_name = "figure1"

with open(str(out_path / f"{out_name}.svg"), "w") as f:
    f.write(svg)

cairosvg.svg2pdf(bytestring=svg, write_to=str(out_path / f"{out_name}.pdf"))

plt.close()

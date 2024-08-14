# %%

from pathlib import Path

import cairosvg
import matplotlib.pyplot as plt
import skunk

mosaic = """
AAB
AAC
"""
figsize = (5, 4)
fontsize = 30

fig, axs = plt.subplot_mosaic(
    mosaic=mosaic,
    figsize=figsize,
    constrained_layout=True,
    gridspec_kw=dict(hspace=0.0, wspace=0.0),
)
for label, ax in axs.items():
    ax.text(
        0,
        1,
        label + ")",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=fontsize,
    )
    ax.axis("off")
    skunk.connect(axs[label], label)


panel_mapping = {
    "A": "docs/result_images/show_neuron_edits/whole_neuron.svg",
    "B": "docs/result_images/show_neuron_edits/split_example.svg",
    "C": "docs/result_images/show_neuron_edits/merge_example.svg",
}

svg = skunk.insert(panel_mapping)

skunk.display(svg)

out_path = Path("docs/figures/figure1")
out_name = "figure1"

with open(str(out_path / f"{out_name}.svg"), "w") as f:
    f.write(svg)

cairosvg.svg2pdf(bytestring=svg, write_to=str(out_path / f"{out_name}.pdf"))

plt.close()

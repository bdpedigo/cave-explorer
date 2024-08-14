# %%
from pkg.figures import panel_mosaic, write_svg

mosaic = """
AABBBCCCCC
"""

panel_mapping = {
    "A": "docs/images/output_count_diagram.png",
    "B": "docs/images/output_proportion_diagram.png",
    "C": "docs/images/distances_diagram.png",
}
figsize = (10, 2)
svg = panel_mosaic(
    mosaic, panel_mapping, constrained_layout=True, panel_borders=False, figsize=figsize
)
write_svg(svg, "docs/figures/figure2/figure2")

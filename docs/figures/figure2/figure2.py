# %%
from pkg.figures import panel_mosaic

mosaic = """
AAABBBCCC
.........

"""

panel_mapping = {
    "A": "docs/images/connectivity_feature_space.png",
    "B": "docs/images/output_proportion_diagram.png",
    # "A": "docs/images/output_count_diagram.png",
    # "C": "docs/images/distances_diagram.png",
}
figsize = (10, 6)
pm = panel_mosaic(
    mosaic, panel_mapping, layout="tight", panel_borders=False, figsize=figsize
)
pm.write("docs/figures/figure2/figure2")
pm

# %%
pm.show_dummies()

# %%

# %%


from pkg.figures import panel_mosaic
from pkg.utils import load_manifest

manifest = load_manifest()
index = manifest.query("is_example").set_index("target_id").index

figsize = (10, 8)
fontsize = 20
panel_borders = False
mosaic = """
ABCDE
FGHIJ
KLMNO
PQRST
"""
letters = "ABCDEFGHIJKLMNOPQRST"


panel_mapping = {
    letter: f"docs/result_images/precision_recall/precision_recall_target={target}_scheme=lumped-time.svg"
    for letter, target in zip(letters, index)
}


pm = panel_mosaic(
    mosaic,
    panel_mapping,
    figsize=figsize,
    fontsize=fontsize,
    panel_borders=panel_borders,
    layout="tight",
    label_pos=None,
    # label_pos=(-0.05, 1.05)
)
pm
#%%
pm.write(
    "docs/figures/figure_precision_recall_gallery/figure_precision_recall_gallery",
    formats=("svg", "pdf"),
)
pm


# %%
pm.show_dummies()

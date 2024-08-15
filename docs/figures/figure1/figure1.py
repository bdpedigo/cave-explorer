# %%


from pkg.figures import panel_mosaic

figsize = (10, 8)
fontsize = 20
panel_borders = True
constrained_layout = False
mosaic = """
AABDDD
AACDDD
EEFGGG
EEFGGG
"""

panel_mapping = {
    "A": "docs/result_images/show_neuron_edits/whole_neuron.svg",
    "B": "docs/result_images/show_neuron_edits/split_example.svg",
    "C": "docs/result_images/show_neuron_edits/merge_example.svg",
    "D": "docs/result_images/show_neuron_edits/neuron_gallery.svg",
    "E": "docs/result_images/simple_stats/edit_count_histogram.svg",
    "F": "docs/result_images/simple_stats/edit_count_by_compartment.svg",
    "G": "docs/result_images/simple_stats/error_rate_vs_radius_axon_dendrite.svg",
}
pm = panel_mosaic(
    mosaic,
    panel_mapping,
    figsize=figsize,
    fontsize=fontsize,
    panel_borders=panel_borders,
)
pm.write("docs/figures/figure1/figure1", formats=("svg", "pdf"))
pm


# %%
# write_svg(svg, "docs/figures/figure1/figure1")

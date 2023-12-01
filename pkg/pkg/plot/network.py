import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection


def networkplot(
    nodes,
    edges,
    x,
    y,
    node_palette=None,
    node_hue=None,
    node_hue_norm=None,
    node_color="grey",
    node_size=20,
    node_zorder=1,
    edge_palette=None,
    edge_hue=None,
    edge_color="grey",
    edge_linewidth=0.5,
    edge_alpha=1,
    edge_zorder=0,
    ax=None,
    figsize=(10, 10),
    clean_axis=True,
    scatterplot_kws={},
    linecollection_kws={},
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    nodes = nodes.copy()
    edges = edges.copy()

    # map the x and y coordinates to the edges
    edges["source_x"] = edges["source"].map(nodes[x])
    edges["source_y"] = edges["source"].map(nodes[y])
    edges["target_x"] = edges["target"].map(nodes[x])
    edges["target_y"] = edges["target"].map(nodes[y])

    if node_hue is None:
        scatterplot_kws["color"] = node_color

    sns.scatterplot(
        data=nodes,
        x=x,
        y=y,
        hue=node_hue,
        hue_norm=node_hue_norm,
        palette=node_palette,
        linewidth=0,
        s=node_size,
        ax=ax,
        zorder=node_zorder,
        **scatterplot_kws,
    )

    source_locs = list(zip(edges["source_x"], edges["source_y"]))
    target_locs = list(zip(edges["target_x"], edges["target_y"]))
    segments = list(zip(source_locs, target_locs))

    if edge_palette is not None:
        edge_colors = edges[edge_hue].map(edge_palette)
    else:
        edge_colors = edge_color

    lc = LineCollection(
        segments,
        linewidths=edge_linewidth,
        alpha=edge_alpha,
        color=edge_colors,
        zorder=edge_zorder,
        **linecollection_kws,
    )
    ax.add_collection(lc)

    if clean_axis:
        ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    return ax

import numpy as np
import pandas as pd
from anytree import PreOrderIter
from .network import networkplot


def set_span_position(node):
    # set some starting positions for the leaf nodes to anchor everything else
    if node.is_root:
        i = 0
        for _node in PreOrderIter(node):
            if _node.is_leaf:
                _node._span_position = i
                i += 1

    if node.is_leaf:
        return node._span_position
    else:
        child_positions = [set_span_position(child) for child in node.children]
        min_position = np.min(child_positions)
        max_position = np.max(child_positions)
        node._span_position = (min_position + max_position) / 2
        return node._span_position


def treeplot(
    root,
    node_palette=None,
    node_hue=None,
    node_hue_norm=None,
    node_color="black",
    node_size=20,
    node_zorder=1,
    edge_palette=None,
    edge_hue=None,
    edge_color="black",
    edge_linewidth=1,
    edge_alpha=1,
    edge_zorder=0,
    orient="h",
    ax=None,
    figsize=(6, 6),
    clean_axis=True,
    scatterplot_kws={},
    linecollection_kws={},
):
    set_span_position(root)

    node_df = {}
    for node in PreOrderIter(root):
        node_df[node.name] = {
            "span_position": node._span_position,
            "depth": node.depth,
            node_hue: node.__getattribute__(node_hue),
        }
    node_df = pd.DataFrame(node_df).T

    edge_df = []
    for node in PreOrderIter(root):
        if node.parent is not None:
            edge_df.append(
                {
                    "source": node.parent.name,
                    "target": node.name,
                }
            )
    edge_df = pd.DataFrame(edge_df)

    if orient == "h":
        x = "span_position"
        y = "depth"
    elif orient == "v":
        x = "depth"
        y = "span_position"

    ax = networkplot(
        nodes=node_df,
        edges=edge_df,
        x=x,
        y=y,
        node_palette=node_palette,
        node_hue=node_hue,
        node_hue_norm=node_hue_norm,
        node_color=node_color,
        node_size=node_size,
        node_zorder=node_zorder,
        edge_palette=edge_palette,
        edge_hue=edge_hue,
        edge_color=edge_color,
        edge_linewidth=edge_linewidth,
        edge_alpha=edge_alpha,
        edge_zorder=edge_zorder,
        ax=ax,
        figsize=figsize,
        clean_axis=clean_axis,
        scatterplot_kws=scatterplot_kws,
        linecollection_kws=linecollection_kws,
    )

    return ax

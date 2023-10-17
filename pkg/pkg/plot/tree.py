from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anytree import PreOrderIter, Walker


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
    ax=None,
    hue=None,
    figsize=(6, 6),
    palette="Set1",
    node_size=20,
    node_color="black",
    edge_linewidth=1.5,
    clean_axis=True,
    orient="h",
    scatterplot_kws={},
):
    set_span_position(root)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    # node_color_map = {}
    # if hue is None:
    #     node_color_map = {node.name: node_color for node in PreOrderIter(root)}
    # else:
    #     if isinstance(palette, str):
    #         hue_values = np.unique(
    #             [node.__getattribute__(hue) for node in PreOrderIter(root)]
    #         )
    #         colors = sns.color_palette(palette, n_colors=len(hue_values))
    #         palette = dict(zip(hue_values, colors))
    #     node_color_map = {
    #         node.name: palette[node.__getattribute__(hue)]
    #         for node in PreOrderIter(root)
    #     }

    df = {}
    # plot dots for the nodes themselves, colored by hue
    for node in PreOrderIter(root):
        # color = node_color_map[node.name]
        # ax.scatter(node._span_position, node.depth, color=color, s=node_size, zorder=1)
        df[node.name] = {
            "span_position": node._span_position,
            "depth": node.depth,
            hue: node.__getattribute__(hue),
        }
    df = pd.DataFrame(df).T
    sns.scatterplot(
        data=df,
        x="span_position",
        y="depth",
        hue=hue,
        ax=ax,
        s=node_size,
        palette=palette,
        **scatterplot_kws,
    )

    # plot lines to denote the edit structure, colored by the process (merge/split)
    visited = set()
    for leaf in root.leaves:
        walker = Walker()
        path, _, _ = walker.walk(leaf, root)
        path = list(path) + [root]
        for source, target in pairwise(path):
            if (source, target) not in visited:
                # color = palette[len(target.children)]
                color = "black"
                ax.plot(
                    [source._span_position, target._span_position],
                    [source.depth, target.depth],
                    color=color,
                    linewidth=edge_linewidth,
                    zorder=0,
                )
                visited.add((source, target))

    if clean_axis:
        ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    return ax
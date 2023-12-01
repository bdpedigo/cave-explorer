# %%
from itertools import pairwise

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anytree import Node, Walker
from anytree.iterators import PreOrderIter
from requests import HTTPError
from tqdm.autonotebook import tqdm
import pcg_skel
import skeleton_plot

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

# %%
root_id = 864691135867734294

# %%


def reroot(root_id):
    new_roots = client.chunkedgraph.get_latest_roots(root_id)
    if len(new_roots) > 1:
        raise ValueError("More than one current root found!")

    root_id = new_roots[0]
    return root_id


root_id = reroot(root_id)


# %%


def create_lineage_tree(root_id, order="edits"):
    lineage = client.chunkedgraph.get_lineage_graph(root_id)
    links = lineage["links"]

    nodes = {}
    for link in links:
        source = link["source"]
        target = link["target"]
        if source not in nodes:
            nodes[source] = Node(source)
        if target not in nodes:
            nodes[target] = Node(target)

        # if nodes[source].parent is not None:
        #     raise ValueError(f"Node {source} has multiple parents!")

        nodes[source].parent = nodes[target]

    root = nodes[target].root

    for node in PreOrderIter(root):
        # make the left-hand side of the tree be whatever side has the most children
        if order == "edits":
            node._order_value = len(node.descendants)
        elif order == "l2_size":
            node._order_value = len(
                client.chunkedgraph.get_leaves(node.name, stop_layer=2)
            )
        else:
            raise ValueError(f"Unknown order: {order}")

    # reorder the children of each node by the values
    for node in PreOrderIter(root):
        children_order_value = np.array([child._order_value for child in node.children])
        inds = np.argsort(-children_order_value)
        node.children = [node.children[i] for i in inds]

    return root


root = create_lineage_tree(root_id, order="l2_size")

# %%


def compute_skeleton(name):
    try:
        skeleton = pcg_skel.coord_space_skeleton(name, client)
        return skeleton
    except HTTPError:
        print("HTTPError on node", name)
        return None


def compute_lineage_skeletons(root):
    # get skeletons for each node in the lineage
    pbar = tqdm(total=len(root.descendants) + 1)

    skeletons = {}
    for i, node in enumerate(PreOrderIter(root)):
        skeletons[node.name] = compute_skeleton(node.name)
        pbar.update(1)
    pbar.close()
    return skeletons


# skeletons = compute_lineage_skeletons(root)

# %%


# recursively set the span (non-depth position) of each node
# visually it looks ok to have this just be the mean of the children's span positions


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


def plot_edit_lineage(
    root,
    skeletons=None,
    ax=None,
    figsize=(6, 6),
    palette="Set1",
    node_size=20,
    edge_linewidth=1.5,
    skeleton_linewidth=0.25,
    clean_axis=True,
):
    set_span_position(root)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    if isinstance(palette, str):
        colors = sns.color_palette(palette)
        palette = dict(zip([0, 1, 2], colors))
    # else, assumed to be a dict mapping 0,1,2 to colors

    # plot dots for the nodes themselves, colored by what process spawned them
    for node in PreOrderIter(root):
        color = palette[len(node.children)]
        ax.scatter(node._span_position, node.depth, color=color, s=node_size, zorder=1)

    # plot lines to denote the edit structure, colored by the process (merge/split)
    visited = set()
    for leaf in root.leaves:
        walker = Walker()
        path, _, _ = walker.walk(leaf, root)
        path = list(path) + [root]
        for source, target in pairwise(path):
            if (source, target) not in visited:
                color = palette[len(target.children)]
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

    if skeletons is not None:

        def plot_skeleton(node, position="above"):
            if isinstance(skeletons, dict):
                if node.name in skeletons:
                    skeleton = skeletons[node.name]
            else:
                try:
                    skeleton = compute_skeleton(node.name)
                except AssertionError:
                    print(node.name, "failed")

            if skeleton is not None:
                if position == "above":
                    inset_ax = ax.inset_axes(
                        [node._span_position - 0.5, node.depth + 0.5, 1, 2],
                        transform=ax.transData,
                    )
                elif position == "below":
                    inset_ax = ax.inset_axes(
                        [node._span_position - 0.5, node.depth - 2.5, 1, 2],
                        transform=ax.transData,
                    )
                skeleton_plot.plot_tools.plot_skel(
                    skeleton,
                    ax=inset_ax,
                    plot_soma=False,
                    invert_y=True,
                    line_width=skeleton_linewidth,
                    color="black",
                )
                inset_ax.axis("off")

        for node in root.leaves:
            plot_skeleton(node, position="above")

        for node in PreOrderIter(root):
            # find split nodes which are then merged
            if node.parent is not None:
                if (len(node.children) == 1) and (len(node.parent.children) == 2):
                    plot_skeleton(node, position="below")

        plot_skeleton(root, position="below")

    return ax


# %%
plot_edit_lineage(root, skeletons=True)

from pathlib import Path

save_path = Path("cave-explorer/results/figs")
plt.savefig(save_path / f"lineage_tree_root={root.name}.png", dpi=300)

# %%

meta = client.materialize.query_table("allen_v1_column_types_slanted_ref")
nuc = client.materialize.query_table("nucleus_detection_v0").set_index("id")

# %%
target_id = meta.iloc[26]["target_id"]
root_id = nuc.loc[target_id]["pt_root_id"]
latest = client.chunkedgraph.get_latest_roots(root_id)
assert len(latest) == 1
root_id = latest[0]

root = create_lineage_tree(root_id, order="edits")
plot_edit_lineage(root, skeletons=True, node_size=5, edge_linewidth=0.2)
plt.savefig(save_path / f"lineage_tree_root={root.name}.png", dpi=300)

# %%
pcg_skel.coord_space_skeleton(864691134876998589, client)

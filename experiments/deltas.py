# %%
import caveclient as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib_venn import venn2
from tqdm.autonotebook import tqdm
import pcg_skel
from meshparty import trimesh_vtk

# %%

root_id = 864691135867734294

client = cc.CAVEclient("minnie65_phase3_v1")

new_roots = client.chunkedgraph.get_latest_roots(root_id)
root_id = new_roots[0]

# %%

changelog = client.chunkedgraph.get_change_log(root_id)
changelog

# %%
lineage = client.chunkedgraph.get_lineage_graph(root_id)

# %%

links = lineage["links"]

from anytree import Node

nodes = {}
for link in links:
    source = link["source"]
    target = link["target"]
    if source not in nodes:
        nodes[source] = Node(source)
    if target not in nodes:
        nodes[target] = Node(target)

    if nodes[source].parent is not None:
        print("broke!")
    nodes[source].parent = nodes[target]

root = nodes[target].root

query_node = nodes[root_id]

# %%
from anytree import RenderTree

print(RenderTree(root))

# %%

from anytree.iterators import PreOrderIter

double_count = 0
for node in PreOrderIter(nodes[root_id]):
    if len(node.children) == 2:
        double_count += 1
print(double_count)
# matches the number of mergers!!!

zero_count = 0
for node in PreOrderIter(nodes[root_id]):
    if len(node.children) == 0:
        zero_count += 1
print(zero_count)
# must be the number of original connected components which are part of current root_id

one_count = 0
for node in PreOrderIter(nodes[root_id]):
    if len(node.children) == 1:
        one_count += 1
print(one_count)

# matches the number of splits!!!

# # %%

# skeleton = pcg_skel.coord_space_skeleton(root_id, client, return_mesh=False)

# # %%

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))


# skeleton_plot.plot_tools.plot_skel(
#     skeleton,
#     ax=ax,
#     # pull_radius=True,
#     # pull_compartment_colors=True,
#     plot_soma=False,
#     invert_y=True,
#     line_width=1,
# )
# ax.axis("off")

# %%

from anytree import LevelOrderIter

from requests import HTTPError

skeletons = {}
for i, node in enumerate(LevelOrderIter(root)):
    print(i, node.name)
    try:
        skeleton = pcg_skel.coord_space_skeleton(node.name, client)
        skeletons[node.name] = skeleton
    except HTTPError:
        print("HTTPError on node", node.name)


# %%

import numpy as np
from anytree import Walker
from itertools import pairwise
import skeleton_plot


# make the left-hand side of the tree be whatever side has the most children
for node in PreOrderIter(root):
    n_descendants = len(node.descendants)
    node.n_descendants = n_descendants

for node in PreOrderIter(root):
    children_n_descendants = np.array([child.n_descendants for child in node.children])
    inds = np.argsort(-children_n_descendants)
    node.children = [node.children[i] for i in inds]

# set some starting positions for the leaf nodes to anchor everything else
i = 0
for node in PreOrderIter(root):
    if node.is_leaf:
        node.span_position = i
        i += 1

# recursively set the span (non-depth position) of each node
# visually it looks ok to have this just be the mean of the children's span positions


def set_span_position(node):
    if node.is_leaf:
        return node.span_position
    else:
        child_positions = [set_span_position(child) for child in node.children]
        min_position = np.min(child_positions)
        max_position = np.max(child_positions)
        node.span_position = (min_position + max_position) / 2
        return node.span_position


set_span_position(root)


fig, ax = plt.subplots(1, 1, figsize=(5, 6))

colors = sns.color_palette("Set1")

palette = dict(zip([0, 1, 2], colors))

# plot dots for the nodes themselves, colored by what process spawned them
for node in PreOrderIter(root):
    color = palette[len(node.children)]
    ax.scatter(node.span_position, node.depth, color=color, s=20, zorder=1)

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
                [source.span_position, target.span_position],
                [source.depth, target.depth],
                color=color,
                linewidth=1.5,
                zorder=0,
            )
            visited.add((source, target))

ax.axis("off")


def plot_skeleton(node, position="above"):
    if node.name in skeletons:
        skeleton = skeletons[node.name]
        if position == "above":
            inset_ax = ax.inset_axes(
                [node.span_position - 0.5, node.depth + 0.5, 1, 2],
                transform=ax.transData,
            )
        elif position == "below":
            inset_ax = ax.inset_axes(
                [node.span_position - 0.5, node.depth - 2.5, 1, 2],
                transform=ax.transData,
            )
        skeleton_plot.plot_tools.plot_skel(
            skeleton,
            ax=inset_ax,
            plot_soma=False,
            invert_y=True,
            line_width=0.25,
            color="black",
        )
        inset_ax.axis("off")


draw_skeletons = True
if draw_skeletons:
    for node in root.leaves:
        plot_skeleton(node, position="above")

    for node in PreOrderIter(root):
        # find split nodes which are then merged
        if (len(node.children) == 1) and (len(node.parent.children) == 2):
            plot_skeleton(node, position="below")

    plot_skeleton(root, position="below")

from pathlib import Path

save_path = Path("cave-explorer/results/figs")
plt.savefig(save_path / f"lineage_tree_root={root.name}.png", dpi=300)

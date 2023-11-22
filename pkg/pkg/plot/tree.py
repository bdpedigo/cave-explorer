import math
import random

import networkx as nx
import numpy as np
import pandas as pd
from anytree import PreOrderIter

from .network import networkplot


def initialize_leaf_spans(node):
    # set some starting positions for the leaf nodes to anchor everything else
    i = 0
    for _node in PreOrderIter(node):
        if _node.is_leaf:
            _node._span_position = i
            i += 1


def set_spans(node):
    if node.is_leaf:
        return node._span_position
    else:
        child_positions = [set_spans(child) for child in node.children]
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
    initialize_leaf_spans(root)
    set_spans(root)

    node_df = {}
    for node in PreOrderIter(root):
        node_df[node.name] = {
            "span_position": node._span_position,
            "depth": node.depth,
        }
        if node_hue is not None:
            node_df[node.name][node_hue] = node.__getattribute__(node_hue)
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


def radial_hierarchy_pos(g):
    pos = hierarchy_pos(g, width=2 * math.pi)
    new_pos = {
        u: (r * math.cos(theta), r * math.sin(theta)) for u, (theta, r) in pos.items()
    }
    return new_pos


def hierarchy_pos(
    G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, leaf_vs_root_factor=0.5
):
    """
    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.

    We include this because it may be useful for plotting transmission trees,
    and there is currently no networkx equivalent (though it may be coming soon).

    There are two basic approaches we think of to allocate the horizontal
    location of a node.

    - Top down: we allocate horizontal space to a node.  Then its ``k``
      descendants split up that horizontal space equally.  This tends to result
      in overlapping nodes when some have many descendants.
    - Bottom up: we allocate horizontal space to each leaf node.  A node at a
      higher level gets the entire space allocated to its descendant leaves.
      Based on this, leaf nodes at higher levels get the same space as leaf
      nodes very deep in the tree.

    We use use both of these approaches simultaneously with ``leaf_vs_root_factor``
    determining how much of the horizontal space is based on the bottom up
    or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    down.


    :Arguments:

    **G** the graph (must be a tree)

    **root** the root node of the tree
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be
      just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.

    **width** horizontal space allocated for this branch - avoids overlap with other branches

    **vert_gap** gap between levels of hierarchy

    **vert_loc** vertical location of root

    **leaf_vs_root_factor**

    xcenter: horizontal location of root
    """

    # REF: https://epidemicsonnetworks.readthedocs.io/en/latest/_modules/EoN/auxiliary.html#hierarchy_pos
    # stole this code from the above

    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(
                iter(nx.topological_sort(G))
            )  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(
        G,
        root,
        leftmost,
        width,
        leafdx=0.2,
        vert_gap=0.2,
        vert_loc=0,
        xcenter=0.5,
        rootpos=None,
        leafpos=None,
        parent=None,
    ):
        """
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        """

        if rootpos is None:
            rootpos = {root: (xcenter, vert_loc)}
        else:
            rootpos[root] = (xcenter, vert_loc)
        if leafpos is None:
            leafpos = {}
        children = list(G.neighbors(root))
        leaf_count = 0
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            rootdx = width / len(children)
            nextx = xcenter - width / 2 - rootdx / 2
            for child in children:
                nextx += rootdx
                rootpos, leafpos, newleaves = _hierarchy_pos(
                    G,
                    child,
                    leftmost + leaf_count * leafdx,
                    width=rootdx,
                    leafdx=leafdx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    rootpos=rootpos,
                    leafpos=leafpos,
                    parent=root,
                )
                leaf_count += newleaves

            leftmostchild = min((x for x, y in [leafpos[child] for child in children]))
            rightmostchild = max((x for x, y in [leafpos[child] for child in children]))
            leafpos[root] = ((leftmostchild + rightmostchild) / 2, vert_loc)
        else:
            leaf_count = 1
            leafpos[root] = (leftmost, vert_loc)
        #        pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
        #        print(leaf_count)
        return rootpos, leafpos, leaf_count

    xcenter = width / 2.0
    if isinstance(G, nx.DiGraph):
        leafcount = len(
            [node for node in nx.descendants(G, root) if G.out_degree(node) == 0]
        )
    elif isinstance(G, nx.Graph):
        leafcount = len(
            [
                node
                for node in nx.node_connected_component(G, root)
                if G.degree(node) == 1 and node != root
            ]
        )
    rootpos, leafpos, leaf_count = _hierarchy_pos(
        G,
        root,
        0,
        width,
        leafdx=width * 1.0 / leafcount,
        vert_gap=vert_gap,
        vert_loc=vert_loc,
        xcenter=xcenter,
    )
    pos = {}
    for node in rootpos:
        pos[node] = (
            leaf_vs_root_factor * leafpos[node][0]
            + (1 - leaf_vs_root_factor) * rootpos[node][0],
            leafpos[node][1],
        )
    #    pos = {node:(leaf_vs_root_factor*x1+(1-leaf_vs_root_factor)*x2, y1) for ((x1,y1), (x2,y2)) in (leafpos[node], rootpos[node]) for node in rootpos}
    xmax = max(x for x, y in pos.values())
    for node in pos:
        pos[node] = (pos[node][0] * width / xmax, pos[node][1])
    return pos

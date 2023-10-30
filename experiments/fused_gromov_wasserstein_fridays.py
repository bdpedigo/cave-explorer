# %%

import time

t0 = time.time()

import caveclient as cc
import navis
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from cloudfiles import CloudFiles
from meshparty import skeletonize, trimesh_io
from pkg.edits import (
    NetworkDelta,
    find_supervoxel_component,
    get_initial_network,
    get_network_edits,
    get_network_metaedits,
)
from pkg.paths import OUT_PATH, FIG_PATH
from plotly.subplots import make_subplots
from tqdm.autonotebook import tqdm
from pkg.plot import networkplot
import matplotlib.pyplot as plt
import pcg_skel.skel_utils as sk_utils
from pcg_skel.chunk_tools import build_spatial_graph
from neuropull.graph import NetworkFrame
from pkg.utils import get_positions
from sklearn.neighbors import NearestNeighbors

# %%
recompute = False
cloud = False

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

cg = client.chunkedgraph

# %%
meta = client.materialize.query_table("aibs_metamodel_mtypes_v661_v2")
meta = meta.sample(100, random_state=0)
nuc = client.materialize.query_table("nucleus_detection_v0").set_index("id")


skel_path = "https://storage.googleapis.com/allen-minnie-phase3/minniephase3-emily-pcg-skeletons/minnie_all/BIL_neurons/file_groups/"


# %%
from skeleton_plot.skel_io import read_skeleton

skeletons = {}
for i in tqdm(range(100)):
    row = meta.iloc[i]
    target_id = meta.iloc[i]["target_id"]
    root_id = meta.iloc[i]["pt_root_id"]
    # root_id = nuc.loc[target_id]["pt_root_id"]
    try:
        skel = read_skeleton(
            skel_path + f"{root_id}_{target_id}", f"{root_id}_{target_id}.swc"
        )
        skeletons[root_id] = skel
    except:
        pass

print(len(skeletons))

#%%

meta.set_index('pt_root_id').loc[skeletons.keys()]['cell_type']

# %%

from neuropull.graph import NetworkFrame
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import pairwise_distances

# def compute_skeleton_costs(skeleton):


def skeleton_to_networkframe(skeleton):
    nodes = pd.DataFrame(skeleton.vertices, columns=["x", "y", "z"])
    edges = pd.DataFrame(skeleton.edges, columns=["source", "target"])

    source_locs = nodes.loc[edges["source"]]
    target_locs = nodes.loc[edges["target"]]

    edge_lengths = np.linalg.norm(source_locs.values - target_locs.values, axis=1)

    edges["distance"] = edge_lengths

    nf = NetworkFrame(nodes=nodes, edges=edges)
    return nf


from ot.gromov import (
    entropic_fused_gromov_wasserstein,
    entropic_fused_gromov_wasserstein2,
)
from ot.bregman import sinkhorn, empirical_sinkhorn
from skeleton_plot.plot_tools import plot_skel


def plot_transport_map(
    transport_map, node_positions1, node_positions2, ax, threshold=1e-4
):
    for i in range(transport_map.shape[0]):
        for j in range(transport_map.shape[1]):
            if transport_map[i, j] > threshold:
                ax.plot(
                    [node_positions1[i, 0], node_positions2[j, 0]],
                    [node_positions1[i, 1], node_positions2[j, 1]],
                    color="black",
                    linewidth=0.05,
                    alpha=0.1,
                )


root_ids = list(skeletons.keys())[:10]

n_total = len(root_ids) * (len(root_ids) - 1) // 2
pbar = tqdm(total=n_total, desc="Computing skeleton costs")

verbose = False
pairwise_skeleton_costs = pd.DataFrame(index=root_ids, columns=root_ids).fillna(0)

for i, root_id1 in enumerate(root_ids):
    skeleton1 = skeletons[root_id1]
    nf1 = skeleton_to_networkframe(skeleton1)

    # within-skeleton cost
    adjacency = nf1.to_sparse_adjacency(weight_col="distance")
    path_distances1 = shortest_path(adjacency, directed=False, unweighted=False)

    # for the between skeleton cost
    node_positions1 = nf1.nodes[["x", "y", "z"]].values
    node_positions1 -= node_positions1[0, :]
    skeleton1.vertices -= skeleton1.vertices[0, :]

    for j, root_id2 in enumerate(root_ids[i + 1 :]):
        skeleton2 = skeletons[root_id2]
        nf2 = skeleton_to_networkframe(skeleton2)

        if verbose:
            print(root_id1, root_id2)
            print(len(nf1.nodes), len(nf2.nodes))

        # within-skeleton cost
        adjacency = nf2.to_sparse_adjacency(weight_col="distance")
        path_distances2 = shortest_path(adjacency, directed=False, unweighted=False)

        # for the between skeleton cost
        node_positions2 = nf2.nodes[["x", "y", "z"]].values
        node_positions2 -= node_positions2[0, :]
        skeleton2.vertices -= skeleton2.vertices[0, :]

        # compute the between skeleton cost, which is just the euclidean distance
        # between nodes in skel 1 and those in skel 2
        node_distances = pairwise_distances(node_positions1, node_positions2)

        # precondition by solving the entropic regularized transport just for the node
        # distances
        pre_transport_map = empirical_sinkhorn(
            node_positions1, node_positions2, reg=10, metric="euclidean"
        )

        # compute the fused gromov wasserstein distance and coupling map between
        # these skeletons
        cost = entropic_fused_gromov_wasserstein2(
            node_distances,
            path_distances1,
            path_distances2,
            epsilon=100,
            alpha=0.5,
            G0=pre_transport_map,
            verbose=verbose,
        )
        pairwise_skeleton_costs.loc[root_id1, root_id2] = cost
        pairwise_skeleton_costs.loc[root_id2, root_id1] = cost

        if verbose:
            transport_map = entropic_fused_gromov_wasserstein(
                node_distances,
                path_distances1,
                path_distances2,
                epsilon=100,
                alpha=0.5,
                G0=pre_transport_map,
                verbose=False,
            )

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            plot_skel(skeleton1, color="red", ax=ax)
            plot_skel(skeleton2, color="blue", ax=ax)

            plot_transport_map(
                transport_map, node_positions1, node_positions2, ax, threshold=5e-5
            )
            ax.autoscale()
            print()

        pbar.update(1)

pbar.close()

# %%

import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

labels = meta.set_index("pt_root_id").loc[root_ids]["cell_type"]
colors = sns.color_palette("tab10", n_colors=len(labels.unique()))
palette = dict(zip(labels.unique(), colors))
color_labels = [palette[x] for x in labels]
costs = pairwise_skeleton_costs.values

Z = linkage(squareform(costs), method="average")

sns.clustermap(
    pairwise_skeleton_costs,
    row_colors=color_labels,
    col_colors=color_labels,
    row_linkage=Z,
    col_linkage=Z,
)

# %%


root_id1 = root_ids[0]
root_id2 = root_ids[1]
skeleton1 = skeletons[root_id1]
skeleton2 = skeletons[root_id2]
nf1 = skeleton_to_networkframe(skeleton1)
nf2 = skeleton_to_networkframe(skeleton2)
node_positions1 = nf1.nodes[["x", "y", "z"]].values
node_positions1 -= node_positions1[0, :]
node_positions2 = nf2.nodes[["x", "y", "z"]].values
node_positions2 -= node_positions2[0, :]
skeleton1.vertices -= skeleton1.vertices[0, :]
skeleton2.vertices -= skeleton2.vertices[0, :]

plot_skel(skeleton1, color="red", ax=ax)
plot_skel(skeleton2, color="blue", ax=ax)

pre_transport_map = empirical_sinkhorn(
    node_positions1, node_positions2, reg=0.1, metric="euclidean"
)
# threshold = 0.01
# pre_transport_map[pre_transport_map < threshold] = 0
pre_transport_map.max()


# %%
sns.histplot(pre_transport_map.flatten())

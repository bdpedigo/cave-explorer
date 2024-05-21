# %%

from typing import Callable, Optional, Union

import caveclient as cc
import pandas as pd
import pcg_skel
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances_argmin
from tqdm.auto import tqdm

from pkg.neuronframe import NeuronFrame, load_neuronframe
from pkg.plot import set_context
from pkg.utils import get_nucleus_point_nm, load_manifest, load_mtypes


def apply_to_synapses_by_sample(
    func: Callable,
    synapses_df: pd.DataFrame,
    resolved_synapses: Union[dict, pd.Series],
    output="series",
    name: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Apply a function which takes in a DataFrame of synapses and returns a DataFrame
    of results.

    Parameters
    ----------
    func
        A function which takes in a DataFrame of synapses and returns a DataFrame
        of results.
    which
        Whether to apply the function to the pre- or post-synaptic synapses.
    kwargs
        Additional keyword arguments to pass to `func`.
    """

    results_by_sample = []
    for i, key in enumerate(resolved_synapses.keys()):
        sample_resolved_synapses = resolved_synapses[key]

        input = synapses_df.loc[sample_resolved_synapses]
        result = func(input, **kwargs)
        if output == "dataframe":
            result["edit"] = key
        elif output == "series":
            result.name = key
        elif output == "scalar":
            pass

        results_by_sample.append(result)

    if output == "dataframe":
        results_df = pd.concat(results_by_sample, axis=0)
        return results_df
    elif output == "series":
        results_df = pd.concat(results_by_sample, axis=1).T
        results_df.index.name = "edit"
        return results_df
    else:
        results_df = pd.Series(
            results_by_sample, index=resolved_synapses.keys()
        ).to_frame()
        results_df.index.name = "edit"
        if name is not None:
            results_df.columns = [name]
        return results_df


def compute_diffs_to_final(sequence_df):
    # the final rows the one with Nan index
    final_row_idx = -1
    final_row = sequence_df.loc[final_row_idx].fillna(0).values.reshape(1, -1)

    X = sequence_df.drop(index=final_row_idx).fillna(0)

    sample_wise_metrics = []
    for metric in ["euclidean", "cityblock", "jensenshannon", "cosine"]:
        distances = cdist(X.values, final_row, metric=metric)
        distances = pd.Series(
            distances.flatten(),
            name=metric,
            index=X.index,
        )
        sample_wise_metrics.append(distances)
    sample_wise_metrics = pd.concat(sample_wise_metrics, axis=1)

    return sample_wise_metrics


# %%

set_context()

client = cc.CAVEclient("minnie65_phase3_v1")

MTYPES = load_mtypes(client)

manifest = load_manifest()
manifest = manifest.query("is_example")

# %%
root_id = manifest.index[0]
root_point = get_nucleus_point_nm(root_id, client=client)

# %%

import networkx as nx

sequence_features_by_neuron = []
feature_diffs_by_neuron = []

root_id = manifest.index[0]
root_point = get_nucleus_point_nm(root_id, client=client)

# set the neuronframe to the final state
level2_nf = load_neuronframe(root_id, client=client)
final_nf = level2_nf.set_edits(level2_nf.edits.index)
final_nf.select_nucleus_component(inplace=True)
final_nf.remove_unused_nodes(inplace=True)
final_nf.remove_unused_synapses(inplace=True)

# generate a skeleton and map it back to the level2 nodes
meshwork = pcg_skel.coord_space_meshwork(
    root_id, client=client, root_point=root_point, root_point_resolution=[1, 1, 1]
)

level2_nodes = meshwork.anno.lvl2_ids.df.copy()
level2_nodes.set_index("mesh_ind_filt", inplace=True)
level2_nodes["skeleton_index"] = meshwork.anno.lvl2_ids.mesh_index.to_skel_index_padded
level2_nodes = level2_nodes.rename(columns={"lvl2_id": "level2_id"}).drop(
    columns="mesh_ind"
)
skeleton_to_level2 = level2_nodes.groupby("skeleton_index")["level2_id"].unique()
# level2_to_skeleton = level2_nodes.set_index
# %%
# skeleton networkframe
skeleton_nodes = pd.DataFrame(meshwork.skeleton.vertices, columns=["x", "y", "z"])
# note the convention reversal here
skeleton_edges = pd.DataFrame(meshwork.skeleton.edges, columns=["source", "target"])
skel_nuc_id = pairwise_distances_argmin(
    skeleton_nodes[["x", "y", "z"]], [root_point], axis=0
)[0]
skeleton_nf = NeuronFrame(skeleton_nodes, skeleton_edges, nucleus_id=skel_nuc_id)
g = skeleton_nf.to_networkx(create_using=nx.DiGraph)
sorted_nodes = list(nx.topological_sort(g))
sorted_nodes_map = {node: i for i, node in enumerate(sorted_nodes)}
skeleton_nf.nodes["topological_order"] = skeleton_nf.nodes.index.map(sorted_nodes_map)
skeleton_nf.apply_node_features("topological_order", inplace=True)
skeleton_nf.edges.sort_values("source_topological_order", inplace=True)

skeleton_nf.edges["source_indexer"] = skeleton_nf.nodes.index.get_indexer_for(
    skeleton_nf.edges["source"]
)
skeleton_nf.edges["target_indexer"] = skeleton_nf.nodes.index.get_indexer_for(
    skeleton_nf.edges["target"]
)
# remove each edge one by one
# HACK:there is a much much smarter way to do this starting from tips and removing edges
# one by one. could do this in a way that avoids recomputing the adjacency matrix
# every time, also.

# def drop_edge(edge_id, skeleton_nf):

# return dropped_level2_nf.pre_synapses.index

dropped_skeleton_nf = NeuronFrame(
    skeleton_nf.nodes.copy(),
    skeleton_nf.edges.copy(),
    nucleus_id=skeleton_nf.nucleus_id,
)
n_edges = len(skeleton_nf.edges)
pre_synapse_ids_by_edge = {}

lil_adjacency = skeleton_nf.to_sparse_adjacency(format="lil")
# for edge_id in tqdm(skeleton_nf.edges.index, total=n_edges):

# dropped_skeleton_nf.edges.drop(index=edge_id, inplace=True)
# dropped_skeleton_nf.select_nucleus_component(inplace=True)
# dropped_skeleton_nf.remove_unused_nodes(inplace=True)
# used_level2_nodes = (
#     skeleton_to_level2[dropped_skeleton_nf.nodes.index].explode().values.astype(int)
# )
# # map the new skeleton to the level2 nodes
# dropped_level2_nf = final_nf.reindex_nodes(used_level2_nodes)
# dropped_level2_nf.select_nucleus_component(inplace=True)
# dropped_level2_nf.remove_unused_nodes(inplace=True)
# dropped_level2_nf.remove_unused_synapses(inplace=True)


import numpy as np
from scipy.sparse.csgraph import dijkstra
from itertools import chain

soma_index = skeleton_nf.nodes["topological_order"].idxmax()

skeleton_to_level2_long = level2_nodes.set_index("skeleton_index")["level2_id"]

reachable_level2_nodes_by_drop = {}
for edge_id, row in tqdm(skeleton_nf.edges.iterrows(), total=n_edges):
    # drop the edge
    i = row["source_indexer"]
    j = row["target_indexer"]
    lil_adjacency[i, j] = 0

    mask = dijkstra(lil_adjacency, directed=False, indices=soma_index)

    reachable_nodes_at_iter = skeleton_nf.nodes.index[np.isfinite(mask)]
    # reachable_nodes[edge_id] = reachable_nodes_at_iter

    # used_level2_nodes = skeleton_to_level2_long[reachable_nodes_at_iter].values
    used_level2_nodes = skeleton_to_level2[reachable_nodes_at_iter]
    used_level2_nodes = list(chain.from_iterable(used_level2_nodes))
    # reachable_level2_nodes_by_drop[edge_id] = used_level2_nodes

    # replace
    lil_adjacency[i, j] = 1

# %%

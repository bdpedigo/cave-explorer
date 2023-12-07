# %%
import os
import pickle

import caveclient as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from neuropull.graph import NetworkFrame
from sklearn.metrics import pairwise_distances_argmin
from tqdm.autonotebook import tqdm

from pkg.edits import (
    collate_edit_info,
    find_supervoxel_component,
    get_operation_metaoperation_map,
    lazy_load_network_edits,
    reverse_edit,
)
from pkg.neuroglancer import (
    add_level2_edits,
    finalize_link,
    generate_neuron_base_builders,
)
from pkg.plot import radial_hierarchy_pos
from pkg.utils import get_level2_nodes_edges, get_nucleus_point_nm

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v507")
query_neurons.sort_values("id", inplace=True)

nuc = client.materialize.query_table(
    "nucleus_detection_v0",  # select_columns=["pt_supervoxel_id", "pt_root_id"]
).set_index("pt_root_id")

# %%

os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"

# %%

# getting a table of additional metadata for each operation
# TODO something weird with 3, 6

root_id = query_neurons["pt_root_id"].values[13]

(
    networkdeltas_by_operation,
    networkdeltas_by_metaoperation,
) = lazy_load_network_edits(root_id, client=client)

# %%

operation_to_metaoperation = get_operation_metaoperation_map(
    networkdeltas_by_metaoperation
)

# %%

edit_stats, modified_level2_nodes = collate_edit_info(
    networkdeltas_by_operation, operation_to_metaoperation, root_id, client
)

# %%

so.Plot(
    edit_stats.query("is_merge & is_filtered"),
    x="n_modified_nodes",
    y="centroid_distance_to_nuc_um",
).add(so.Dots(pointsize=8, alpha=0.9))


# %%

# because for this analysis we are just interested in rolling back merges from the
# most proofread version of this neuron, we can safely just use the synapses from the
# latest version

# TODO this needs to be redone with the query for all synapses ever potentially on a
# neuron, not just the latest version

# pre_synapses = client.materialize.query_table(
#     "synapses_pni_2", filter_equal_dict={"pre_pt_root_id": root_id}
# )
# pre_synapses.query("pre_pt_root_id != post_pt_root_id", inplace=True)
# pre_synapses.set_index("id", inplace=True)
# pre_synapses["pre_pt_current_level2_id"] = client.chunkedgraph.get_roots(
#     pre_synapses["pre_pt_supervoxel_id"], stop_layer=2
# )

# post_synapses = client.materialize.query_table(
#     "synapses_pni_2", filter_equal_dict={"post_pt_root_id": root_id}
# )
# post_synapses.query("pre_pt_root_id != post_pt_root_id", inplace=True)
# post_synapses.set_index("id", inplace=True)
# post_synapses["post_pt_current_level2_id"] = client.chunkedgraph.get_roots(
#     post_synapses["post_pt_supervoxel_id"], stop_layer=2
# )

# %%

from pkg.edits import get_initial_network

# get the initial network
initial_nf = get_initial_network(root_id, client, positions=True)
# %%
if False:
    # create a networkframe for the final network and add annotations for synapses
    nodes, edges = get_level2_nodes_edges(root_id, client, positions=True)
    nf = NetworkFrame(nodes, edges)
# %%

from pkg.edits import apply_additions

nf = initial_nf.copy()

# %%
for networkdelta in tqdm(networkdeltas_by_operation.values()):
    apply_additions(nf, networkdelta)

# %%
# apply positions to ALL nodes

from pkg.utils import get_positions

node_positions = get_positions(list(nf.nodes.index), client)

# %%
nf.nodes[["rep_coord_nm", "x", "y", "z"]] = node_positions[
    ["rep_coord_nm", "x", "y", "z"]
]

# %%

from pkg.edits import get_level2_lineage_components
from pkg.morphology import get_pre_post_synapses, map_synapses

verbose = True

level2_lineage_component_map = get_level2_lineage_components(networkdeltas_by_operation)

# TODO why are so many not current here?
pre_synapses, post_synapses = get_pre_post_synapses(root_id, client, verbose=True)

pre_synapses, post_synapses = map_synapses(
    pre_synapses,
    post_synapses,
    networkdeltas_by_operation,
    nf.nodes.index,
    client,
    verbose=True,
)


# for side, synapses in zip(["pre", "post"], [pre_synapses, post_synapses]):
#     map_synapse_level2_ids(
#         synapses,
#         level2_lineage_component_map,
#         nf.nodes.index,
#         side,
#         client,
#         verbose=verbose,
#     )
# synapses[f"{side}_pt_current_level2_id"] = client.chunkedgraph.get_roots(
#     synapses[f"{side}_pt_supervoxel_id"], stop_layer=2
# )
# is_current = ~synapses[f"{side}_pt_current_level2_id"].isin(nf.nodes.index)

# if verbose:
#     print(f"{len(is_current) - is_current.sum()} synapses are not current")

# %%

nf.nodes["synapses"] = [[] for _ in range(len(nf.nodes))]
nf.nodes["pre_synapses"] = [[] for _ in range(len(nf.nodes))]
nf.nodes["post_synapses"] = [[] for _ in range(len(nf.nodes))]

for idx, synapse in pre_synapses.iterrows():
    nf.nodes.loc[synapse["pre_pt_current_level2_id"], "synapses"].append(idx)
    nf.nodes.loc[synapse["pre_pt_current_level2_id"], "pre_synapses"].append(idx)

for idx, synapse in post_synapses.iterrows():
    nf.nodes.loc[synapse["post_pt_current_level2_id"], "synapses"].append(idx)
    nf.nodes.loc[synapse["post_pt_current_level2_id"], "post_synapses"].append(idx)

# annotate a point on the level2 graph as the nucleus; whatever is closest
nuc_pt_nm = get_nucleus_point_nm(root_id, client, method="table")

pos_nodes = nf.nodes[["x", "y", "z"]]
pos_nodes = pos_nodes[pos_nodes.notna().all(axis=1)]

ind = pairwise_distances_argmin(nuc_pt_nm.reshape(1, -1), pos_nodes)[0]
nuc_level2_id = pos_nodes.index[ind]
nf.nodes["nucleus"] = False
nf.nodes.loc[nuc_level2_id, "nucleus"] = True

# add annotations for metaoperations to the networkframe
nf.nodes["metaoperation_id"] = np.nan
nf.nodes["metaoperation_id"] = nf.nodes["metaoperation_id"].astype("Int64")

for metaoperation_id, networkdelta in networkdeltas_by_metaoperation.items():
    added_nodes = networkdelta.added_nodes
    # TODO more robustly deal with add->removes in the metaedits
    # TODO maybe just take the intersection with the final network as a shortcut
    #      Since some of the metaedits might not have merges which are relevant so added
    #      nodes get removed later or something...
    net_added_node_ids = networkdelta.added_nodes.index.difference(
        networkdelta.removed_nodes.index
    )

    net_added_node_ids = networkdelta.added_nodes.index.intersection(nf.nodes.index)

    if len(net_added_node_ids) == 0:
        print(networkdelta.metadata["is_merges"])
        for operation_id in networkdelta.metadata["operation_ids"]:
            print(edit_stats.loc[operation_id, "is_filtered"])
        print()

    if not nf.nodes.loc[net_added_node_ids, "metaoperation_id"].isna().all():
        raise AssertionError("Some nodes already exist")
    else:
        nf.nodes.loc[net_added_node_ids, "metaoperation_id"] = metaoperation_id

assert nf.nodes["pre_synapses"].apply(len).sum() == len(pre_synapses)
assert nf.nodes["post_synapses"].apply(len).sum() == len(post_synapses)

# %%

# plot this neuron with its merge edits, grouped by metaoperation

from pkg.edits import get_initial_node_ids
from pkg.neuroglancer import generate_neurons_base_builders

# TODO add something to make this center on the final neuron soma we care about

original_node_ids = list(get_initial_node_ids(root_id, client))

state_builders, dataframes = generate_neurons_base_builders(original_node_ids, client)

state_builders, dataframes = add_level2_edits(
    state_builders, dataframes, modified_level2_nodes.reset_index(), client, by=None
)

link = finalize_link(state_builders, dataframes, client)
link


# %%

nuc_dist_threshold = 10
n_modified_nodes_threshold = 10
soma_nuc_merge_metaoperation = None

for metaoperation_id, networkdelta in networkdeltas_by_metaoperation.items():
    metadata = networkdelta.metadata
    operation_ids = metadata["operation_ids"]

    has_filtered = False
    has_soma_nuc = False
    for operation_id in operation_ids:
        # check if any of the operations in this metaoperation are filtered
        is_filtered = edit_stats.loc[operation_id, "is_filtered"]
        if is_filtered:
            has_filtered = True

        # check if any of the operations in this metaoperation are a soma/nucleus merge
        is_merge = edit_stats.loc[operation_id, "is_merge"]
        if is_merge:
            dist_um = edit_stats.loc[operation_id, "centroid_distance_to_nuc_um"]
            n_modified_nodes = edit_stats.loc[operation_id, "n_modified_nodes"]
            if (
                dist_um < nuc_dist_threshold
                and n_modified_nodes >= n_modified_nodes_threshold
            ):
                has_soma_nuc = True
                soma_nuc_merge_metaoperation = metaoperation_id
                print("Found soma/nucleus merge operation: ", operation_id)
                print("Found soma/nucleus merge metaoperation: ", metaoperation_id)
                break

# %%

metaoperations_to_query = edit_stats.query("is_filtered & is_merge")[
    "metaoperation_id"
].unique()
# metaoperations_to_query = list(edit_df["metaoperation_id"].unique())
final_nf_keypoints = nf.nodes.query(
    "metaoperation_id.isin(@metaoperations_to_query) | nucleus", engine="python"
)

# %%

# add distance information to edges
nf = nf.apply_node_features(["x", "y", "z"], axis="both")
nf.edges["distance_nm"] = np.sqrt(
    (nf.edges["source_x"] - nf.edges["target_x"]) ** 2
    + (nf.edges["source_y"] - nf.edges["target_z"]) ** 2
    + (nf.edges["source_z"] - nf.edges["target_z"]) ** 2
)
nf.edges["distance_um"] = nf.edges["distance_nm"] / 1000

# %%


g = nf.to_networkx().to_undirected()

predecessors_by_metaoperation = {}

missing_paths = []
missing_metaoperations = []
for metaoperation_id, metaoperation_points in tqdm(
    final_nf_keypoints.groupby("metaoperation_id", dropna=True)
):
    if metaoperation_id == soma_nuc_merge_metaoperation:
        continue
    min_path_length = np.inf
    min_path = None
    for level2_id in metaoperation_points.index:
        # TODO rather than choosing the one with shortest path length, might be best to
        # choose the one with minimal dependency set, somehow...
        try:
            path = nx.shortest_path(
                g, source=nuc_level2_id, target=level2_id, weight="distance_um"
            )
            path_length = nx.shortest_path_length(
                g, source=nuc_level2_id, target=level2_id, weight="distance_um"
            )
            if path_length < min_path_length:
                min_path_length = path_length
                min_path = path
        except nx.NetworkXNoPath:
            missing_paths.append(level2_id)

    if min_path is None:
        missing_metaoperations.append(metaoperation_id)

    min_path = pd.Index(min_path)

    keypoint_predecessors = min_path.intersection(final_nf_keypoints.index)

    metaoperation_predecessors = final_nf_keypoints.loc[
        keypoint_predecessors, "metaoperation_id"
    ]
    metaoperation_predecessors = metaoperation_predecessors[
        ~metaoperation_predecessors.isna()
    ]
    metaoperation_predecessors = metaoperation_predecessors[
        metaoperation_predecessors != metaoperation_id
    ]
    predecessors_by_metaoperation[metaoperation_id] = list(
        metaoperation_predecessors.unique()
    )

# TODO dont quite understand how some L2 nodes could be disconnected here, given that I
# am adding all edges from any operation. Maybe missing some operations on filtered out
# chunks?
print(
    f"Number of L2 nodes missing path: {len(missing_paths)}/{len(final_nf_keypoints)}"
)
print(
    f"Number of metaoperations missing path: {len(missing_metaoperations)}/{final_nf_keypoints['metaoperation_id'].nunique()}"
)

# %%
# decent amount of redundant computation here but seems fast enough...

# TODO redo this more efficiently to deal with the fact that some of the paths are going
# to be redundant/make use of the same MST
# scipy version does have a case for doing only lookup from some nodes, but I couldn't
# tell if one could then reconstruct the entire path along the way when using that
# lookup.


nf.nodes["iloc"] = np.arange(len(nf.nodes))

adjacency = nf.to_sparse_adjacency(weight_col="distance_um")
adjacency = adjacency + adjacency.T

nf.nodes["has_synapses"] = nf.nodes["synapses"].apply(len) > 0
keypoints = nf.nodes.query(
    "metaoperation_id.notna() | nucleus | has_synapses", engine="python"
)
keypoints = keypoints.sort_values("nucleus", ascending=False)

# %%
print("test")

# %%
import time

from scipy.sparse.csgraph import shortest_path

currtime = time.time()

dists, predecessors = shortest_path(
    adjacency,
    directed=False,
    indices=keypoints["iloc"],
    return_predecessors=True,
    unweighted=True,
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")


# %%

# undirected graph looks like:
# 0 -> 1 -> 2 -> 3
A = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])

shortest_path(A, directed=False, indices=[0], return_predecessors=True)

# %%


def reconstruct_soma_paths(predecessors, keypoints):
    target = keypoints[keypoints["nucleus"]]["iloc"].values[0]

    paths = []
    for i in tqdm(range(len(keypoints))):
        query = keypoints["iloc"].values[i]
        query_predecessors = predecessors[i]

        path = [target]
        current = target

        while current != query and current != -9999:
            current = query_predecessors[current]
            path.append(current)

        paths.append(path)

    paths = pd.Series(paths, index=keypoints.index)
    return paths


iloc_paths = reconstruct_soma_paths(predecessors, keypoints)


def remap_path(path, index):
    new_path = []
    for p in path:
        new_path.append(index[p])
    return new_path


# %%%
# paths = []
metaoperation_dependencies = {}

metaoperation_points = keypoints.query("metaoperation_id.notna()")

for path, (l2_id, metaoperation) in tqdm(
    zip(iloc_paths, keypoints["metaoperation_id"].items()), total=len(keypoints)
):
    path = nf.nodes.index[path]
    path = path.intersection(metaoperation_points.index)
    metaoperation_path = metaoperation_points.loc[path, "metaoperation_id"]
    # metaoperation_path = metaoperation_path[~metaoperation_path.isna()]
    if not pd.isna(metaoperation):
        metaoperation_path = metaoperation_path[metaoperation_path != metaoperation]
    metaoperation_path = metaoperation_path.unique().tolist()
    metaoperation_dependencies[l2_id] = metaoperation_path

keypoints["metaoperation_dependencies"] = pd.Series(metaoperation_dependencies)
keypoints

# %%

predecessors_by_synapse = {}
path_lengths_by_synapse = {}
for synapse_id, row in tqdm(pre_synapses.iterrows(), total=len(pre_synapses)):
    level2_id = row["pre_pt_current_level2_id"]
    path = nx.shortest_path(
        g, source=nuc_level2_id, target=level2_id, weight="distance_um"
    )
    # path_length = nx.shortest_path_length(
    #     g, source=nuc_level2_id, target=level2_id, weight="distance_um"
    # )

    starts = nf.nodes.loc[path[:-1], ["x", "y", "z"]]
    ends = nf.nodes.loc[path[1:], ["x", "y", "z"]]
    path_length_nm = np.sqrt(((starts.values - ends.values) ** 2).sum(axis=1)).sum()

    path = pd.Index(path)

    keypoint_predecessors = path.intersection(final_nf_keypoints.index)

    metaoperation_predecessors = final_nf_keypoints.loc[
        keypoint_predecessors, "metaoperation_id"
    ]
    metaoperation_predecessors = metaoperation_predecessors[
        ~metaoperation_predecessors.isna()
    ]
    metaoperation_predecessors = metaoperation_predecessors[
        metaoperation_predecessors != metaoperation_id
    ]
    predecessors_by_synapse[synapse_id] = list(metaoperation_predecessors.unique())
    path_lengths_by_synapse[synapse_id] = path_length_nm

# %%

synapse_metaoperation_dependencies = pd.Series(predecessors_by_synapse)
pre_synapses["metaoperation_dependencies"] = synapse_metaoperation_dependencies
pre_synapses["n_metaoperation_dependencies"] = pre_synapses[
    "metaoperation_dependencies"
].apply(len)

synapse_path_lengths = pd.Series(path_lengths_by_synapse)
pre_synapses["path_length_nm"] = synapse_path_lengths
pre_synapses["path_length_um"] = pre_synapses["path_length_nm"] / 1000


# %%

plot = (
    so.Plot(data=pre_synapses, x="path_length_um", y="n_metaoperation_dependencies")
    .add(so.Dots(alpha=0.1, pointsize=3), so.Jitter(x=0, y=0.4))
    .label(
        title=f"Root ID {root_id}, Found soma/nuc merge = {soma_nuc_merge_metaoperation is not None}",
        x="Path length (um)",
        y="# of metaoperation dependencies",
    )
)
plot.save(f"path_length_vs_n_dependencies-root={root_id}")

# %%
plot = (
    so.Plot(data=pre_synapses, x="n_metaoperation_dependencies")
    .add(so.Bar(width=1), so.Hist(discrete=True, stat="proportion"))
    .label(
        x="# of metaoperation dependencies",
        y="Proportion of synapses",
        title=f"Root ID {root_id}, Found soma/nuc merge = {soma_nuc_merge_metaoperation is not None}",
    )
)
plot.save(f"n_dependencies_hist-root={root_id}")

# %%

metaoperation_synapse_dependents = pre_synapses.explode("metaoperation_dependencies")[
    "metaoperation_dependencies"
].value_counts()
metaoperation_synapse_dependents = metaoperation_synapse_dependents.reindex(
    metaoperations_to_query
).fillna(0)


# %%


state_builders, dataframes = generate_neuron_base_builders(root_id, client)

edits_to_show = edit_stats.query("is_merge & is_filtered")

level2_nodes_to_show = modified_level2_nodes.query(
    "is_merge & is_filtered"
).reset_index()

state_builders, dataframes = add_level2_edits(
    state_builders, dataframes, level2_nodes_to_show, nf, client
)

# for path, pathnames in zip([path_489, path_485, path_484], ["489", "485", "484"]):
#     level2_graph_mapper = PointMapper(
#         point_column="rep_coord_nm",
#         description_column="l2_id",
#         split_positions=False,
#         gather_linked_segmentations=False,
#         set_position=False,
#         collapse_groups=True,
#     )
#     level2_graph_layer = AnnotationLayerConfig(
#         f"path-{pathnames}",
#         data_resolution=[1, 1, 1],
#         color=(0.6, 0.6, 0.6),
#         mapping_rules=level2_graph_mapper,
#     )
#     level2_graph_statebuilder = StateBuilder(
#         layers=[level2_graph_layer],
#         client=client,  # view_kws=view_kws
#     )
#     state_builders.append(level2_graph_statebuilder)
#     path = pd.Index(path, name="l2_id")
#     dataframes.append(final_nf.nodes.loc[path].reset_index())


link = finalize_link(state_builders, dataframes, client)
link


# %%
metaoperation_dependencies = nx.DiGraph()

for metaoperation, predecessors in predecessors_by_metaoperation.items():
    path = predecessors + [metaoperation]
    if soma_nuc_merge_metaoperation is None:
        path = [-1] + path
    nx.add_path(metaoperation_dependencies, path)

is_tree = nx.is_tree(metaoperation_dependencies)

print("Merge dependency graph is a tree: ", is_tree)


# %%

nontrivial_metaoperation_dependencies = metaoperation_dependencies.copy()
out_degrees = metaoperation_dependencies.out_degree()

in_degrees = list(metaoperation_dependencies.in_degree())
in_degrees = list(zip(*in_degrees))
in_degrees = pd.Series(index=in_degrees[0], data=in_degrees[1])
max_degree_node = in_degrees.idxmax()

# %%
for node_id in metaoperation_dependencies.nodes():
    in_edges = metaoperation_dependencies.in_edges(node_id)
    n_edges = len(in_edges)
    if n_edges == 0:
        continue
    first_edge = next(iter(in_edges))
    if n_edges == 1 and first_edge == (max_degree_node, node_id):
        nontrivial_metaoperation_dependencies.remove_node(node_id)

# %%


fig, ax = plt.subplots(1, 1, figsize=(15, 15))
pos = radial_hierarchy_pos(metaoperation_dependencies)

nodelist = metaoperation_synapse_dependents.index.to_list()
if soma_nuc_merge_metaoperation is None:
    nodelist = [-1] + nodelist

node_sizes = list(metaoperation_synapse_dependents.values)
if soma_nuc_merge_metaoperation is None:
    node_sizes = [max(node_sizes) + 1] + node_sizes

nx.draw_networkx(
    metaoperation_dependencies,
    nodelist=nodelist,
    pos=pos,
    with_labels=True,
    ax=ax,
    node_size=node_sizes,
    font_size=6,
)
ax.axis("equal")
ax.axis("off")
plt.savefig(f"merge_dependency_tree_root={root_id}.png", bbox_inches="tight", dpi=300)


# %%

nuc_supervoxel = nuc.loc[root_id, "pt_supervoxel_id"]
p_merge_rollbacks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
n_samples = 25

nfs_by_sample = {}
for p_merge_rollback in p_merge_rollbacks:
    if p_merge_rollback == 0.0 or p_merge_rollback == 1.0:
        _n_samples = 1
    else:
        _n_samples = n_samples
    for i in tqdm(range(_n_samples)):
        sampled_metaedits = candidate_metaedits.sample(frac=p_merge_rollback)

        nf = nf.copy()
        for metaedit in sampled_metaedits:
            networkdelta = networkdeltas_by_metaoperation[metaedit]
            reverse_edit(nf, networkdelta)

        nuc_nf = find_supervoxel_component(nuc_supervoxel, nf, client)
        if p_merge_rollback == 0.0:
            assert nuc_nf == nf
        nfs_by_sample[(np.round(1 - p_merge_rollback, 1), i)] = nuc_nf

# %%
synapses_by_sample = {}
for key, nf in nfs_by_sample.items():
    all_synapses = []
    for idx, node in nf.nodes.iterrows():
        # TODO I think can safely ignore nodes w/o "synapses" column here since they
        # were added, but could check this
        if isinstance(node["synapses"], list):
            all_synapses += node["synapses"]
    synapses_by_sample[key] = all_synapses

# %%
all_postsynaptic_targets = np.unique(pre_synapses["post_pt_root_id"])
connectivity_df = pd.DataFrame(
    columns=all_postsynaptic_targets, index=synapses_by_sample.keys()
).fillna(0)
connectivity_df.index.names = ["p_merge", "sample"]

for key, synapses in synapses_by_sample.items():
    for synapse in synapses:
        post_root_id = pre_synapses.loc[synapse, "post_pt_root_id"]
        connectivity_df.loc[key, post_root_id] += 1


# %%
p_found = connectivity_df.sum(axis=0)
p_found = p_found.sort_values(ascending=False)

connectivity_df = connectivity_df[p_found.index]
# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 9))
sns.heatmap(connectivity_df, ax=ax, cmap="Blues", xticklabels=False)

# %%
mtypes = client.materialize.query_table("aibs_metamodel_mtypes_v661_v2")

# %%
mtypes["target_id"].isin(nuc["id"]).mean()

# %%
new_root_ids = mtypes["target_id"].map(nuc.reset_index().set_index("id")["pt_root_id"])
mtypes["root_id"] = new_root_ids

root_id_counts = mtypes["root_id"].value_counts().sort_values(ascending=False)

root_id_dups = root_id_counts[root_id_counts > 1].index

# %%
mtypes = mtypes.query("~root_id.isin(@root_id_dups)")

# %%
mtypes.set_index("root_id", inplace=True)

# %%
connectivity_df.columns.isin(mtypes.index).mean()

# %%

connectivity_df.groupby(by=mtypes["cell_type"], axis=1).sum()

# %%
p_connectivity_df = connectivity_df / connectivity_df.sum(axis=1).values[:, None]

p_connectivity_df.sum(axis=1)

# %%
group_connectivity_df = connectivity_df.groupby(by=mtypes["cell_type"], axis=1).sum()
group_connectivity_df = group_connectivity_df.reindex(
    labels=np.unique(mtypes["cell_type"]), axis=1
).fillna(0)

group_p_connectivity_df = (
    group_connectivity_df / group_connectivity_df.sum(axis=1).values[:, None]
)

# %%
exc_group_p_connectivity_df = group_p_connectivity_df.drop(
    ["DTC", "ITC", "PTC", "STC"], axis=1
)
exc_group_p_connectivity_df = exc_group_p_connectivity_df.sort_index(axis=1)

exc_group_connectivity_df = group_connectivity_df.drop(
    ["DTC", "ITC", "PTC", "STC"], axis=1
)
exc_group_connectivity_df = exc_group_connectivity_df.sort_index(axis=1)
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.heatmap(exc_group_p_connectivity_df, cmap="Blues", ax=ax)

# %%

palette_file = "data/ctype_hues.pkl"

with open(palette_file, "rb") as f:
    ctype_hues = pickle.load(f)

ctype_hues = {ctype: tuple(ctype_hues[ctype]) for ctype in ctype_hues.keys()}


# %%
exc_group_connectivity_tidy = pd.melt(
    exc_group_connectivity_df.reset_index(),
    id_vars=["p_merge", "sample"],
    value_name="n_synapses",
)
exc_group_p_connectivity_tidy = pd.melt(
    exc_group_p_connectivity_df.reset_index(),
    id_vars=["p_merge", "sample"],
    value_name="p_synapses",
)


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle(
    f"Root ID {root_id}, Motif group: {query_neurons.set_index('pt_root_id').loc[root_id, 'cell_type']}"
)
plot1 = (
    so.Plot(
        exc_group_connectivity_tidy,
        x="p_merge",
        y="n_synapses",
        color="cell_type",
    )
    .add(so.Dots(pointsize=3, alpha=0.5), so.Jitter())
    .add(so.Line(), so.Agg())
    .add(so.Band(), so.Est())
    .label(
        x="Proportion of filtered merges used",
        y="Number of synapses",
        # title=f"Root ID {root_id}, Motif group: {query_neurons.set_index('pt_root_id').loc[root_id, 'cell_type']}",
        color="Target M-type",
    )
    .layout(engine="tight")
    .scale(color=ctype_hues)
    .on(axs[0])
    # .show()
    .save(f"exc_group_connectivity_root={root_id}.png", bbox_inches="tight")
)
plot2 = (
    so.Plot(
        exc_group_p_connectivity_tidy, x="p_merge", y="p_synapses", color="cell_type"
    )
    .add(so.Dots(pointsize=3, alpha=0.5), so.Jitter())
    .add(so.Line(), so.Agg())
    .add(so.Band(), so.Est())
    .label(
        x="Proportion of filtered merges used",
        y="Proportion of known outputs",
        # title=f"Root ID {root_id}, Motif group: {query_neurons.set_index('pt_root_id').loc[root_id, 'cell_type']}",
        color="Target M-type",
    )
    .layout(engine="tight")
    .scale(color=so.Nominal(ctype_hues))
    .on(axs[1])
    # .show()
    .save(f"exc_group_connectivity_root={root_id}.png", bbox_inches="tight")
)

# %%

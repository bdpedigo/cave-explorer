# %%
import os
import pickle
import time

import caveclient as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from scipy.sparse.csgraph import shortest_path
from tqdm.auto import tqdm

from pkg.edits import (
    apply_additions,
    apply_metaoperation_info,
    apply_synapses,
    collate_edit_info,
    find_soma_nuc_merge_metaoperation,
    find_supervoxel_component,
    get_initial_network,
    get_initial_node_ids,
    get_operation_metaoperation_map,
    lazy_load_network_edits,
    reverse_edit,
)
from pkg.morphology import (
    apply_nucleus,
)
from pkg.neuroglancer import (
    add_level2_edits,
    finalize_link,
    generate_neurons_base_builders,
)
from pkg.plot import radial_hierarchy_pos
from pkg.utils import get_positions

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

# get the initial network
initial_nf = get_initial_network(root_id, client, positions=False)

# %%

nf = initial_nf.copy()

for networkdelta in tqdm(networkdeltas_by_operation.values()):
    apply_additions(nf, networkdelta)

# %%

node_positions = get_positions(list(nf.nodes.index), client)

nf.nodes[["rep_coord_nm", "x", "y", "z"]] = node_positions[
    ["rep_coord_nm", "x", "y", "z"]
]

# %%

pre_synapses, post_synapses = apply_synapses(
    nf,
    networkdeltas_by_operation,
    root_id,
    client,
)

apply_nucleus(nf, root_id, client)

# %%

# add annotations for metaoperations to the networkframe

apply_metaoperation_info(nf, networkdeltas_by_metaoperation, edit_stats)

assert nf.nodes["pre_synapses"].apply(len).sum() == len(pre_synapses)
assert nf.nodes["post_synapses"].apply(len).sum() == len(post_synapses)

# %%

# TODO add something to make this center on the final neuron soma we care about

original_node_ids = list(get_initial_node_ids(root_id, client))

state_builders, dataframes = generate_neurons_base_builders(original_node_ids, client)

state_builders, dataframes = add_level2_edits(
    state_builders, dataframes, modified_level2_nodes.reset_index(), client, by=None
)

link = finalize_link(state_builders, dataframes, client)
link

# %%
soma_nuc_merge_metaoperation = find_soma_nuc_merge_metaoperation(
    networkdeltas_by_metaoperation, edit_stats
)

# %%

# add distance information to edges
nf = nf.apply_node_features(["x", "y", "z"], axis="both")
nf.edges["distance_nm"] = np.sqrt(
    (nf.edges["source_x"] - nf.edges["target_x"]) ** 2
    + (nf.edges["source_y"] - nf.edges["target_y"]) ** 2
    + (nf.edges["source_z"] - nf.edges["target_z"]) ** 2
)
nf.edges["distance_um"] = nf.edges["distance_nm"] / 1000

# %%


def reconstruct_soma_paths(predecessors, keypoints, nodelist, verbose=True):
    target = keypoints[keypoints["nucleus"]]["iloc"].values[0]

    paths = []
    for i in tqdm(
        range(len(keypoints)), desc="Reconstructing paths", disable=not verbose
    ):
        query = keypoints["iloc"].values[i]
        query_predecessors = predecessors[i]

        path = [target]
        current = target

        while current != query and current != -9999:
            current = query_predecessors[current]
            path.append(current)

        path = tuple(nodelist[path])
        paths.append(path)

    paths = pd.Series(paths, index=keypoints.index)

    # paths = reconstruct_soma_paths(predecessors, keypoints, nf.nodes.index)

    # target = keypoints[keypoints["nucleus"]]["iloc"].values[0]

    soma_path_dists = dists[:, target]

    keypoints["nuc_path_length_um"] = soma_path_dists

    return soma_path_dists, paths


def compute_keypoint_soma_paths(nf, keypoints, verbose=True):
    assert "iloc" in keypoints.columns

    adjacency = nf.to_sparse_adjacency(weight_col="distance_um")
    # make undirected for the purposes of shortest path lookup
    adjacency = adjacency + adjacency.T

    if verbose:
        currtime = time.time()
        print("Querying shortest paths...")

    dists, predecessors = shortest_path(
        adjacency,
        directed=False,
        indices=keypoints["iloc"],
        return_predecessors=True,
        unweighted=False,
    )

    if verbose:
        print(f"{time.time() - currtime:.3f} seconds elapsed.")
        print()
    return dists, predecessors


# select points that have operations, or are the nucleus, or have synapses
nf.nodes["iloc"] = np.arange(len(nf.nodes))
keypoints = nf.nodes.query("has_operation | nucleus | has_synapses").copy()
dists, predecessors = compute_keypoint_soma_paths(nf, keypoints)
soma_path_dists, paths = reconstruct_soma_paths(predecessors, keypoints, nf.nodes.index)


# %%

adjacency = nf.to_sparse_adjacency()
degree = adjacency.sum(axis=0) + adjacency.sum(axis=1)

# seems like there are many "tips"
# more than the number of keypoints that I'm looking up...
len(degree[degree == 1])

# %%


# %%

keypoints.groupby("metaoperation_dependencies").size().sort_values(ascending=False)

# %%%

metaoperation_dependencies = {}

metaoperation_points = keypoints.query("metaoperation_id.notna()")

for path, (l2_id, metaoperation) in tqdm(
    zip(paths, keypoints["metaoperation_id"].items()),
    total=len(keypoints),
    desc="Finding metaoperation dependencies",
):
    path = pd.Index(path)
    path = path.intersection(metaoperation_points.index)
    metaoperation_path = metaoperation_points.loc[path, "metaoperation_id"]
    if not pd.isna(metaoperation):
        metaoperation_path = metaoperation_path[metaoperation_path != metaoperation]
    metaoperation_path = tuple(metaoperation_path.unique().tolist())
    metaoperation_dependencies[l2_id] = metaoperation_path

keypoints["metaoperation_dependencies"] = pd.Series(metaoperation_dependencies)
keypoints["n_metaoperation_dependencies"] = keypoints[
    "metaoperation_dependencies"
].apply(len)


# %%

synapses = keypoints.query("has_synapses")

# %%

metaoperation_keypoints = keypoints.query("has_operation")

predecessors_by_metaoperation = {}

for metaoperation_id, metaoperation_points in metaoperation_keypoints.groupby(
    "metaoperation_id", dropna=True
):
    if metaoperation_id == soma_nuc_merge_metaoperation:
        continue

    unique_dependency_sets = metaoperation_points["metaoperation_dependencies"].unique()
    if len(unique_dependency_sets) > 1:
        print(metaoperation_id)
        print(unique_dependency_sets)
        # print(metaoperation_points)
        print()
# %%
keypoints.index.name = "level2_node_id"

viz_index = keypoints.query("metaoperation_id == 151", engine="python").index

query_keypoints = keypoints.loc[viz_index]
query_keypoints.index.name = "level2_node_id"

viz_metaoperations = set()

for node_id, row in query_keypoints.iterrows():
    for item in row["metaoperation_dependencies"]:
        viz_metaoperations.add(item)

viz_metaoperations = list(viz_metaoperations)

viz_keypoints = keypoints.query(
    "metaoperation_id in @viz_metaoperations", engine="python"
)
viz_keypoints.index.name = "level2_node_id"

# %%
from nglui.statebuilder import (
    AnnotationLayerConfig,
    LineMapper,
    StateBuilder,
)

state_builders, dataframes = generate_neurons_base_builders(original_node_ids, client)


for i, (path, query_l2_node) in enumerate(zip(paths[viz_index], viz_index)):
    path_df = nf.nodes.loc[list(path)].copy()
    path_df.index.name = "level2_node_id"
    path_df.reset_index(inplace=True)

    # edit_point_mapper = PointMapper(
    #     point_column="rep_coord_nm",
    #     description_column="level2_node_id",
    #     split_positions=False,
    #     gather_linked_segmentations=False,
    #     set_position=False,
    # )
    # edit_point_layer = AnnotationLayerConfig(
    #     f"level2-path-nodes-{i}",
    #     data_resolution=[1, 1, 1],
    #     color=(0.4, 0.4, 0.4),
    #     mapping_rules=edit_point_mapper,
    # )
    # sb_edits = StateBuilder([edit_point_layer], client=client)  # view_kws=view_kws)
    # state_builders.append(sb_edits)
    # dataframes.append(path_df)

    path_line_df = path_df.iloc[:-1].join(
        path_df.iloc[1:].reset_index(drop=True), rsuffix="_target", lsuffix="_source"
    )

    edit_line_mapper = LineMapper(
        point_column_a="rep_coord_nm_source",
        point_column_b="rep_coord_nm_target",
        description_column="level2_node_id_source",
        set_position=False,
    )
    edit_line_layer = AnnotationLayerConfig(
        f"level2-path-lines-{query_l2_node}",
        data_resolution=[1, 1, 1],
        color=(0.4, 0.4, 0.4),
        mapping_rules=edit_line_mapper,
    )
    sb_edit_lines = StateBuilder([edit_line_layer], client=client)
    state_builders.append(sb_edit_lines)
    dataframes.append(path_line_df)

state_builders, dataframes = add_level2_edits(
    state_builders, dataframes, query_keypoints.reset_index(), client, by=None
)

state_builders, dataframes = add_level2_edits(
    state_builders,
    dataframes,
    viz_keypoints.reset_index(),
    client,
    by="metaoperation_id",
)

link = finalize_link(state_builders, dataframes, client)
link

# %%

nf.nodes.loc[list(paths[159896547725673406])]["metaoperation_id"].unique()


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

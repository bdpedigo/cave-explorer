# %%
import os

import caveclient as cc
import numpy as np
import pandas as pd
import seaborn.objects as so

from pkg.edits import (
    lazy_load_network_edits,
)
from pkg.utils import pt_to_xyz

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v507")

nuc = client.materialize.query_table(
    "nucleus_detection_v0",  # select_columns=["pt_supervoxel_id", "pt_root_id"]
).set_index("pt_root_id")

# %%

os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"


# %%

# getting a table of additional metadata for each operation

root_id = query_neurons["pt_root_id"].values[7]


(
    networkdeltas_by_operation,
    networkdeltas_by_metaoperation,
) = lazy_load_network_edits(root_id, client=client)

# %%

operation_to_metaoperation = {}
for metaoperation_id, networkdelta in networkdeltas_by_metaoperation.items():
    metadata = networkdelta.metadata
    for operation_id in metadata["operation_ids"]:
        operation_to_metaoperation[operation_id] = metaoperation_id

# %%

nuc_supervoxel = nuc.loc[root_id, "pt_supervoxel_id"]
current_nuc_level2 = client.chunkedgraph.get_roots([nuc_supervoxel], stop_layer=2)[0]
nuc_pt_nm = client.l2cache.get_l2data(
    [current_nuc_level2], attributes=["rep_coord_nm"]
)[str(current_nuc_level2)]["rep_coord_nm"]

raw_modified_nodes = []
rows = []
for operation_id, networkdelta in networkdeltas_by_operation.items():
    if "user_id" in networkdelta.metadata:
        info = {
            **networkdelta.metadata,
            "nuc_pt_nm": nuc_pt_nm,
            "nuc_x": nuc_pt_nm[0],
            "nuc_y": nuc_pt_nm[1],
            "nuc_z": nuc_pt_nm[2],
            "metaoperation_id": operation_to_metaoperation[operation_id],
        }
        rows.append(info)

        modified_nodes = pd.concat(
            (networkdelta.added_nodes, networkdelta.removed_nodes)
        )
        modified_nodes.index.name = "level2_node_id"
        modified_nodes["root_id"] = root_id
        modified_nodes["operation_id"] = operation_id
        modified_nodes["is_merge"] = info["is_merge"]
        raw_modified_nodes.append(modified_nodes)

# %%
edit_stats = pd.DataFrame(rows)
all_modified_nodes = pd.concat(raw_modified_nodes)

raw_node_coords = client.l2cache.get_l2data(
    np.unique(all_modified_nodes.index.to_list()), attributes=["rep_coord_nm"]
)

node_coords = pd.DataFrame(raw_node_coords).T
node_coords[node_coords["rep_coord_nm"].isna()]

# %%

node_coords[["x", "y", "z"]] = pt_to_xyz(node_coords["rep_coord_nm"])
node_coords.index = node_coords.index.astype(int)
node_coords.index.name = "level2_node_id"

all_modified_nodes = all_modified_nodes.join(node_coords, validate="many_to_one")

centroids = all_modified_nodes.groupby(["root_id", "operation_id"])[
    ["x", "y", "z"]
].mean()

centroids.columns = ["centroid_x", "centroid_y", "centroid_z"]

edit_stats = edit_stats.set_index(["root_id", "operation_id"]).join(centroids)

edit_stats["centroid_distance_to_nuc"] = (
    (edit_stats["centroid_x"] - edit_stats["nuc_x"]) ** 2
    + (edit_stats["centroid_y"] - edit_stats["nuc_y"]) ** 2
    + (edit_stats["centroid_z"] - edit_stats["nuc_z"]) ** 2
) ** 0.5

edit_stats["was_forrest"] = edit_stats["user_name"].str.contains("Forrest")

edit_stats["centroid_distance_to_nuc_um"] = (
    edit_stats["centroid_distance_to_nuc"] / 1000
)

# %%

so.Plot(
    edit_stats.query("is_merge & is_filtered"),
    x="n_modified_nodes",
    y="centroid_distance_to_nuc_um",
    color="was_forrest",
).add(so.Dots(pointsize=3, alpha=0.5))

# %%

edit_df = all_modified_nodes.query("root_id == @root_id").copy()
edit_df = edit_df.reset_index().set_index(["root_id", "operation_id"])
edit_df["is_filtered"] = edit_df.index.map(edit_stats["is_filtered"])
edit_df["metaoperation_id"] = edit_df.index.map(edit_stats["metaoperation_id"])
edit_df["is_relevant"] = edit_df.index.map(edit_stats["is_relevant"])
edit_df = edit_df.reset_index().copy()
edit_df = edit_df.query("is_filtered & is_merge")

import seaborn as sns

root_ids = [root_id]
df1 = pd.DataFrame({"root_id": root_ids})
dataframes = [df1]
data_resolution_pre = None
data_resolution_post = None

from nglui.statebuilder.helpers import package_state

DEFAULT_POSTSYN_COLOR = (0.25098039, 0.87843137, 0.81568627)  # CSS3 color turquise
DEFAULT_PRESYN_COLOR = (1.0, 0.38823529, 0.27843137)  # CSS3 color tomato

return_as = "html"
shorten = "always"
show_inputs = False
show_outputs = False
sort_inputs = True
sort_outputs = True
sort_ascending = False
input_color = DEFAULT_POSTSYN_COLOR
output_color = DEFAULT_PRESYN_COLOR
contrast = None
timestamp = None
view_kws = None
point_column = "ctr_pt_position"
pre_pt_root_id_col = "pre_pt_root_id"
post_pt_root_id_col = "post_pt_root_id"
input_layer_name = "syns_in"
output_layer_name = "syns_out"
ngl_url = None
link_text = "Neuroglancer Link"


from nglui.statebuilder import (
    AnnotationLayerConfig,
    ChainedStateBuilder,
    PointMapper,
    StateBuilder,
)
from nglui.statebuilder.helpers import from_client

img_layer, seg_layer = from_client(client, contrast=contrast)
seg_layer.add_selection_map(selected_ids_column="root_id")


view_kws = {"position": np.array(nuc_pt_nm) / np.array([4, 4, 40])}
sb1 = StateBuilder(layers=[img_layer, seg_layer], client=client, view_kws=view_kws)

state_builders = [sb1]

level2_graph_mapper = PointMapper(
    point_column="rep_coord_nm",
    description_column="l2_id",
    split_positions=False,
    gather_linked_segmentations=False,
    set_position=False,
    collapse_groups=True,
)
level2_graph_layer = AnnotationLayerConfig(
    "level2-graph",
    data_resolution=[1, 1, 1],
    color=(0.2, 0.2, 0.2),
    mapping_rules=level2_graph_mapper,
)
level2_graph_statebuilder = StateBuilder(
    layers=[level2_graph_layer], client=client, view_kws=view_kws
)
state_builders.append(level2_graph_statebuilder)
dataframes.append(final_nf.nodes.reset_index())

key = "metaoperation_id"

colors = sns.color_palette("husl", len(edit_df[key].unique()))

for i, (operation_id, operation_data) in enumerate(edit_df.groupby(key)):
    edit_point_mapper = PointMapper(
        point_column="rep_coord_nm",
        description_column="level2_node_id",
        split_positions=False,
        gather_linked_segmentations=False,
        set_position=False,
    )
    edit_layer = AnnotationLayerConfig(
        f"level2-operation-{operation_id}",
        data_resolution=[1, 1, 1],
        color=colors[i],
        mapping_rules=edit_point_mapper,
    )
    sb_edits = StateBuilder([edit_layer], client=client, view_kws=view_kws)
    state_builders.append(sb_edits)
    dataframes.append(operation_data)


sb = ChainedStateBuilder(state_builders)

package_state(dataframes, sb, client, shorten, return_as, ngl_url, link_text)


# %%
## TODO
#
# The Plan
# --------
# - select a neuron
# - load its edits/metaedits
# - select the metaedits which are "filtered"
# - omit the metaedits which were a soma/nucleus merge - my heuristic is >= 10 nodes
#   changed and the centroid of the change is < 10 um from the nucleus
# = generate a link showing all of the meta-edits in neuroglancer
# - query the synapse table for all synapses from that neuron at latest timepoint
# - for some fraction of the edits
#   - for some number of times
#     - randomly sample a subset of the edits
#     - undo those edits in the network to get a partially rolled back network
#     - map the synapses (whichever are relevant still) onto this rolled back network
#     - store the connectivity output vector for that (neuron, fraction, sample)
# - for each neuron
#   - aggregate downstream targets by excitatory class
#   - compute the fraction of outputs onto each class for each (fraction, sample)
#   - plot these connectivities as a function of fraction of edits rolled back
#
# Questions
# ---------
# - smarter way to do this would be to map the synapses onto "segments" between merges
#   since each merge is going to turn off/on a batch of these synapses
#   - this would also make more sense semantically since merges that are "downstream" of
#     a given merge (more distal from the soma) would therefore have no effect, if that
#     upstream merge is rolled back

# root_id = query_neurons["pt_root_id"].values[1]
nuc_supervoxel = nuc.loc[root_id, "pt_supervoxel_id"]

(
    networkdeltas_by_operation,
    networkdeltas_by_metaoperation,
) = lazy_load_network_edits(root_id, client=client)

nuc_dist_threshold = 10
n_modified_nodes_threshold = 10
soma_nuc_merge_metaoperation = None

candidate_metaedits = []
for metaoperation_id, networkdelta in networkdeltas_by_metaoperation.items():
    metadata = networkdelta.metadata
    operation_ids = metadata["operation_ids"]

    has_filtered = False
    has_soma_nuc = False
    for operation_id in operation_ids:
        # check if any of the operations in this metaoperation are filtered
        is_filtered = edit_stats.loc[(root_id, operation_id), "is_filtered"]
        if is_filtered:
            has_filtered = True

        # check if any of the operations in this metaoperation are a soma/nucleus merge
        is_merge = edit_stats.loc[(root_id, operation_id), "is_merge"]
        if is_merge:
            dist_um = edit_stats.loc[
                (root_id, operation_id), "centroid_distance_to_nuc_um"
            ]
            n_modified_nodes = edit_stats.loc[
                (root_id, operation_id), "n_modified_nodes"
            ]
            if (
                dist_um < nuc_dist_threshold
                and n_modified_nodes >= n_modified_nodes_threshold
            ):
                has_soma_nuc = True
                soma_nuc_merge_metaoperation = metaoperation_id
                print("Found soma/nucleus merge metaoperation: ", metaoperation_id)
                break

    if has_filtered and metadata["any_merges"] and (not has_soma_nuc):
        candidate_metaedits.append(metaoperation_id)

# %%
candidate_metaedits = pd.Series(candidate_metaedits)

# %%

# because for this analysis we are just interested in rolling back merges from the
# most proofread version of this neuron, we can safely just use the synapses from the
# latest version
pre_synapses = client.materialize.query_table(
    "synapses_pni_2", filter_equal_dict={"pre_pt_root_id": root_id}
)
pre_synapses.set_index("id", inplace=True)
# post_synapses = client.materialize.query_table(
#     "synapses_pni_2", filter_equal_dict={"post_pt_root_id": root_id}
# )


# remove autapses
pre_synapses.query("pre_pt_root_id != post_pt_root_id", inplace=True)

pre_synapses["pre_pt_current_level2_id"] = client.chunkedgraph.get_roots(
    pre_synapses["pre_pt_supervoxel_id"], stop_layer=2
)

# %%
from pkg.utils import get_level2_nodes_edges

nodes, edges = get_level2_nodes_edges(root_id, client, positions=True)

# %%
from neuropull.graph import NetworkFrame

final_nf = NetworkFrame(nodes, edges)

final_nf.nodes["synapses"] = [[] for _ in range(len(nodes))]

for idx, synapse in pre_synapses.iterrows():
    final_nf.nodes.loc[synapse["pre_pt_current_level2_id"], "synapses"].append(idx)

# %%
n_synapses = 0
for idx, row in final_nf.nodes.iterrows():
    n_synapses += len(row["synapses"])

final_nf.nodes

print(n_synapses)
# %%

from tqdm.autonotebook import tqdm

from pkg.edits import find_supervoxel_component, reverse_edit

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

        nf = final_nf.copy()
        for metaedit in sampled_metaedits:
            networkdelta = networkdeltas_by_metaoperation[metaedit]
            reverse_edit(nf, networkdelta)

        nuc_nf = find_supervoxel_component(nuc_supervoxel, nf, client)
        if p_merge_rollback == 0.0:
            assert nuc_nf == final_nf
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

import matplotlib.pyplot as plt

# %%
p_found = connectivity_df.sum(axis=0)
p_found = p_found.sort_values(ascending=False)

connectivity_df = connectivity_df[p_found.index]
# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 9))
sns.heatmap(connectivity_df, ax=ax, cmap="Blues", xticklabels=False)


# %%
from requests import HTTPError

try:
    mtypes = client.materialize.query_view("allen_column_mtypes_v2")
    mtypes["target_id"].isin(nuc["id"]).mean()
    new_root_ids = mtypes["target_id"].map(
        nuc.reset_index().set_index("id")["pt_root_id"]
    )
    mtypes["root_id"] = new_root_ids
    mtypes.set_index("root_id", inplace=True)
    mtypes.to_csv("mtypes.csv")
except HTTPError:
    mtypes = pd.read_csv("mtypes.csv", index_col=0)
    mtypes.index = mtypes.index.astype(int)


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
import pickle

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

import seaborn.objects as so

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
sns.palplot(ctype_hues.values())

# %%
final_nf

# %%
final_nf.nodes["metaoperation_id"] = np.nan
final_nf.nodes["metaoperation_id"] = final_nf.nodes["metaoperation_id"].astype("Int64")


for metaoperation_id, networkdelta in networkdeltas_by_metaoperation.items():
    added_nodes = networkdelta.added_nodes
    # TODO more robustly deal with add->removes in the metaedits
    # TODO maybe just take the intersection with the final network as a shortcut
    #      Since some of the metaedits might not have merges which are relevant so added
    #      nodes get removed later or something...
    net_added_node_ids = networkdelta.added_nodes.index.difference(
        networkdelta.removed_nodes.index
    )

    net_added_node_ids = networkdelta.added_nodes.index.intersection(
        final_nf.nodes.index
    )

    if not final_nf.nodes.loc[net_added_node_ids, "metaoperation_id"].isna().all():
        raise AssertionError("Some nodes already exist")
    else:
        final_nf.nodes.loc[net_added_node_ids, "metaoperation_id"] = metaoperation_id

# %%
import numpy as np

np.array(nuc_pt_nm)


# %%%
from sklearn.metrics import pairwise_distances_argmin

ind = pairwise_distances_argmin(
    np.array(nuc_pt_nm).reshape(1, -1), final_nf.nodes[["x", "y", "z"]]
)[0]
nuc_level2_id = final_nf.nodes.index[ind]
final_nf.nodes["nucleus"] = False
final_nf.nodes.loc[nuc_level2_id, "nucleus"] = True

# %%
adjacency = final_nf.to_sparse_adjacency()


final_nf.nodes["iloc"] = np.arange(len(final_nf.nodes))

metaoperations_to_query = list(edit_df["metaoperation_id"].unique())
final_nf_keypoints = final_nf.nodes.query(
    "metaoperation_id.isin(@metaoperations_to_query) | nucleus", engine="python"
)
# %%


# currtime = time.time()

# dists, predecessors = shortest_path(
#     adjacency,
#     directed=False,
#     indices=final_nf_keypoints["iloc"],
#     return_predecessors=True,
# )
# print(f"{time.time() - currtime:.3f} seconds elapsed.")

# # %%
# dists = pd.DataFrame(
#     dists, index=final_nf_keypoints.index, columns=final_nf.nodes.index
# )
# predecessors = pd.DataFrame(
#     predecessors, index=final_nf_keypoints.index, columns=final_nf.nodes.index
# )

# # %%
# predecessors.loc[ind][final_nf_keypoints.index]

# %%
# TODO compute bidirectional shortest path between nucleus point and each metaedit node
# then loop through each of these paths.
# If a path from nucleus to nodes in a metaedit A contains metaedit B nodes along the
# way, then that means that metaedit A is downstream of metaedit B
# Build up a dependency tree of such things, stored as an anytree object or networkx graph

# %%

# %%

# %%

final_nf = final_nf.apply_node_features(["x", "y", "z"], axis="both")

# %%
final_nf.edges["distance_nm"] = np.sqrt(
    (final_nf.edges["source_x"] - final_nf.edges["target_x"]) ** 2
    + (final_nf.edges["source_y"] - final_nf.edges["target_z"]) ** 2
    + (final_nf.edges["source_z"] - final_nf.edges["target_z"]) ** 2
)
final_nf.edges["distance_um"] = final_nf.edges["distance_nm"] / 1000
# %%


g = final_nf.to_networkx()
g = g.to_undirected()

import networkx as nx

predecessors_by_metaoperation = {}

# final_nf_keypoints = final_nf_keypoints.groupby("metaoperation_id").first()

# for target_ind in final_nf_keypoints.index:
#     metaoperation = final_nf.nodes.loc[target_ind, "metaoperation_id"]
#     if not pd.isna(metaoperation):
#         path = nx.shortest_path(g, source=ind, target=target_ind, weight="distance_um")

#         keypoint_predecessors = final_nf_keypoints.index.intersection(path)
#         keypoint_metaoperations = final_nf_keypoints.loc[
#             keypoint_predecessors, "metaoperation_id"
#         ]
#         keypoint_metaoperations = keypoint_metaoperations[
#             ~keypoint_metaoperations.isna()
#         ]
#         keypoint_metaoperations = keypoint_metaoperations[
#             keypoint_metaoperations != metaoperation
#         ]
#         predecessors_by_metaoperation[metaoperation] = list(
#             np.unique(keypoint_metaoperations)
#         )

for metaoperation_id, metaoperation_points in final_nf_keypoints.groupby(
    "metaoperation_id", dropna=True
):
    if metaoperation_id == soma_nuc_merge_metaoperation:
        continue
    min_path_length = np.inf
    min_path = None
    for level2_id in metaoperation_points.index:
        path = nx.shortest_path(
            g, source=nuc_level2_id, target=level2_id, weight="distance_um"
        )
        path_length = nx.shortest_path_length(
            g, source=nuc_level2_id, target=level2_id, weight="distance_um"
        )
        if path_length < min_path_length:
            min_path_length = path_length
            min_path = path

    min_path = pd.Index(min_path)
    keypoint_predecessors = min_path.intersection(final_nf_keypoints.index)
    metaoperation_predecessors = final_nf_keypoints.loc[
        keypoint_predecessors, "metaoperation_id"
    ]
    print(metaoperation_predecessors)
    metaoperation_predecessors = metaoperation_predecessors[
        ~metaoperation_predecessors.isna()
    ]
    print(metaoperation_predecessors)
    metaoperation_predecessors = metaoperation_predecessors[
        metaoperation_predecessors != metaoperation_id
    ]
    print(metaoperation_predecessors)
    predecessors_by_metaoperation[metaoperation_id] = list(
        np.unique(metaoperation_predecessors)
    )

# %%
metaoperation_dependencies = nx.DiGraph()

for metaoperation, predecessors in predecessors_by_metaoperation.items():
    print(predecessors)
    path = predecessors + [metaoperation]
    print(path)
    nx.add_path(metaoperation_dependencies, path)

nx.is_tree(metaoperation_dependencies)

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
nx.draw(metaoperation_dependencies, with_labels=True, ax=ax)

# %%

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
nx.draw(nontrivial_metaoperation_dependencies, with_labels=True, ax=ax)

# %%
nx.minimum_cycle_basis(nx.to_undirected(metaoperation_dependencies))

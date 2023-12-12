# %%
import os

import caveclient as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn.objects as so
from neuropull.graph import NetworkFrame
from tqdm.autonotebook import tqdm

from pkg.edits import (
    collate_edit_info,
    get_initial_network,
    get_initial_node_ids,
    get_operation_metaoperation_map,
    lazy_load_network_edits,
    pseudo_apply_edit,
)
from pkg.morphology import (
    apply_nucleus,
    apply_synapses,
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

root_id = query_neurons["pt_root_id"].values[10]

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

for col in [
    "operation_added",
    "operation_removed",
    "metaoperation_added",
    "metaoperation_removed",
]:
    nf.nodes[col] = -1
    nf.nodes[col] = nf.nodes[col].astype(int)
    nf.edges[col] = -1
    nf.edges[col] = nf.edges[col].astype(int)

nf.edges.set_index(["source", "target"], inplace=True, drop=False)

# %%


for networkdelta in tqdm(networkdeltas_by_operation.values()):
    networkdelta.added_edges = networkdelta.added_edges.set_index(
        ["source", "target"], drop=False
    )
    networkdelta.removed_edges = networkdelta.removed_edges.set_index(
        ["source", "target"], drop=False
    )
    # TODO what happens here for multiple operations in the same metaoperation?
    # since they by definition are touching some of the same nodes?
    operation_id = networkdelta.metadata["operation_id"]
    pseudo_apply_edit(
        nf,
        networkdelta,
        operation_label=operation_id,
        metaoperation_label=operation_to_metaoperation[operation_id],
    )

# %%

node_positions = get_positions(list(nf.nodes.index), client)

nf.nodes[["rep_coord_nm", "x", "y", "z"]] = node_positions[
    ["rep_coord_nm", "x", "y", "z"]
]

# %%

nf.apply_node_features("operation_added", inplace=True)
nf.apply_node_features("metaoperation_added", inplace=True)

nf.edges["cross_operation"] = (
    nf.edges["source_operation_added"] != nf.edges["target_operation_added"]
)
nf.edges["cross_metaoperation"] = (
    nf.edges["source_metaoperation_added"] != nf.edges["target_metaoperation_added"]
)
nf.edges["was_removed"] = nf.edges["operation_removed"] != -1
nf.nodes["was_removed"] = nf.nodes["operation_removed"] != -1

# %%

meta = True
if meta:
    prefix = "meta"
else:
    prefix = ""

no_cross_nf = nf.query_edges(
    f"(~cross_{prefix}operation) & (~was_removed)"
).query_nodes("~was_removed")

# %%

n_connected_components = no_cross_nf.n_connected_components()

# %%

# create labels for these different pieces

nf.nodes["component_label"] = -1

i = 2
for component in tqdm(no_cross_nf.connected_components(), total=n_connected_components):
    if (component.nodes[f"{prefix}operation_added"] == -1).all():
        label = -1 * i
        i += 1
    else:
        label = component.nodes[f"{prefix}operation_added"].iloc[0]
    nf.nodes.loc[component.nodes.index, "component_label"] = label

# %%
nf.apply_node_features("component_label", inplace=True)

# %%
cross_nf = nf.query_edges(
    f"cross_{prefix}operation & (~was_removed)"
).remove_unused_nodes()
# %%


component_edges = cross_nf.edges[
    ["source_component_label", "target_component_label"]
].copy()
component_edges.reset_index(drop=True, inplace=True)
component_edges.rename(
    columns={"source_component_label": "source", "target_component_label": "target"},
    inplace=True,
)
component_nodelist = nf.nodes["component_label"].unique()
component_nodes = pd.DataFrame(index=component_nodelist)

component_nf = NetworkFrame(component_nodes, component_edges)


# %%
apply_nucleus(nf, root_id, client)

# %%
nuc_component = nf.nodes.query("nucleus").iloc[0]["component_label"]

# %%
component_nf.nodes["nucleus"] = False
component_nf.nodes.loc[nuc_component, "nucleus"] = True

# %%

g = component_nf.to_networkx(create_using=nx.DiGraph)
g = g.to_undirected()

# %%
all_paths = []
dependency_graph = nx.DiGraph()
for path in nx.all_simple_paths(
    g, source=nuc_component, target=nx.descendants(g, nuc_component)
):
    all_paths.append(path)
    nx.add_path(dependency_graph, path)

print(len(all_paths))

print("Is a tree?", nx.is_tree(dependency_graph))

# %%
path_df = pd.Series(all_paths, name="path").to_frame()
path_df["target"] = path_df["path"].apply(lambda x: x[-1])
path_df

# %%

operation_path_df = path_df.copy()


def remove_non_operations(path):
    return [node for node in path if node >= 0]


operation_path_df["path"] = operation_path_df["path"].apply(remove_non_operations)
operation_path_df

# %%
component_dependencies = operation_path_df.groupby("target")["path"].apply(list)
component_dependencies.name = f"{prefix}operation_dependencies"
component_dependencies.index.name = "segment"
component_dependencies = component_dependencies.to_frame()
component_dependencies["n_dependencies"] = component_dependencies[
    f"{prefix}operation_dependencies"
].apply(len)
component_dependencies

# %%
import time

from pkg.edits import find_supervoxel_component

all_metaoperations = list(networkdeltas_by_metaoperation.keys())

resolved_synapses_by_sample = {}

# still only takes ~ 3 mins for a neuron, to do 100 samples
# so 1000 samples would take 30 mins... same order as extracting the edits
choice_time = 0
query_time = 0
find_component_time = 0
record_time = 0
for i in tqdm(range(100)):
    t = time.time()
    metaoperaion_set = np.random.choice(all_metaoperations, size=10, replace=False)
    metaoperaion_set = list(metaoperaion_set)
    metaoperaion_set.append(-1)
    choice_time += time.time() - t

    t = time.time()
    sub_nf = (
        nf.query_nodes(
            f"{prefix}operation_added.isin(@metaoperaion_set)", local_dict=locals()
        )
        .query_edges(
            f"{prefix}operation_added.isin(@metaoperaion_set)", local_dict=locals()
        )
        .remove_unused_nodes()
    )
    query_time += time.time() - t

    t = time.time()
    # this takes up 90% of the time
    # i think it's from the operation of cycling through connected components
    nuc_supervoxel = nuc.loc[root_id, "pt_supervoxel_id"]
    instance_neuron_nf = find_supervoxel_component(nuc_supervoxel, sub_nf, client)
    find_component_time += time.time() - t

    t = time.time()
    found_synapses = []
    for synapses in instance_neuron_nf.nodes["synapses"]:
        found_synapses.extend(synapses)
    resolved_synapses_by_sample[i] = found_synapses
    record_time += time.time() - t

total_time = choice_time + query_time + find_component_time + record_time
print(f"Total time: {total_time}")
print(f"Choice time: {choice_time / total_time}")
print(f"Query time: {query_time / total_time}")
print(f"Find component time: {find_component_time / total_time}")
print(f"Record time: {record_time / total_time}")

# %%
for target, path_group in path_df.groupby("target"):
    if len(path_group) > 3:
        # print(len(path_group))
        print(path_group["path"].values)

# %%

# just in case dep-graph is not a tree
tree = nx.bfs_tree(dependency_graph, nuc_component)

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
pos = radial_hierarchy_pos(tree)

nx.draw_networkx(
    dependency_graph.subgraph(tree.nodes),
    nodelist=tree.nodes,
    pos=pos,
    with_labels=True,
    ax=ax,
    node_size=10,
    font_size=15,
    width=0.2,
)
ax.axis("equal")
ax.axis("off")
plt.savefig(f"merge_dependency_tree_root={root_id}.png", bbox_inches="tight", dpi=300)


# %%

pre_synapses, post_synapses = apply_synapses(
    nf,
    networkdeltas_by_operation,
    root_id,
    client,
)

apply_nucleus(nf, root_id, client)

# %%

pre_synapses["component_label"] = np.nan
pre_synapses["component_label"] = pre_synapses["component_label"].astype("Int64")
post_synapses["component_label"] = np.nan
post_synapses["component_label"] = post_synapses["component_label"].astype("Int64")

for component, component_nodes in nf.nodes.groupby("component_label"):
    all_pre_synapses = np.unique(np.concatenate(component_nodes["pre_synapses"].values))
    if len(all_pre_synapses) > 0:
        all_pre_synapses = all_pre_synapses.astype(int)
        pre_synapses.loc[all_pre_synapses, "component_label"] = component

    all_post_synapses = np.unique(
        np.concatenate(component_nodes["post_synapses"].values)
    )
    if len(all_post_synapses) > 0:
        all_post_synapses = all_post_synapses.astype(int)
        post_synapses.loc[all_post_synapses, "component_label"] = component

# %%
# need to visualize all of this to make sure it is working!

original_node_ids = list(get_initial_node_ids(root_id, client))

state_builders, dataframes = generate_neurons_base_builders(original_node_ids, client)

sbs, dfs = generate_neurons_base_builders(root_id, client, name="final")
state_builders.append(sbs[0])
dataframes.append(dfs[0])

state_builders, dataframes = add_level2_edits(
    state_builders, dataframes, modified_level2_nodes.reset_index(), client, by=None
)

from nglui.statebuilder import (
    AnnotationLayerConfig,
    LineMapper,
    PointMapper,
    StateBuilder,
)

# nf.apply_node_features('x', inplace=True)
# nf.apply_node_features('y', inplace=True)
# nf.apply_node_features('z', inplace=True)

nf.apply_node_features("rep_coord_nm", inplace=True)


# path_df = nf.nodes.loc[list(path)].copy()
# path_df.index.name = "level2_node_id"
# path_df.reset_index(inplace=True)

# path_line_df = path_df.iloc[:-1].join(
#     path_df.iloc[1:].reset_index(drop=True), rsuffix="_target", lsuffix="_source"
# )

# show_component_labels = [
#     873944,
#     -83,
#     -82,
#     -91,
#     -104,
#     -78,
#     -82,
#     -106,
#     -16,
#     -105,
#     873942,
#     -106,

# ]
show_component_labels = [54, -722, 104, -837, 118, -854, 64, -687]
n_colors = len(show_component_labels)
import seaborn as sns

colors = sns.color_palette("husl", n_colors=n_colors, desat=0.5)

for i, component_label in enumerate(show_component_labels):
    sub_nf = nf.query_nodes(f"component_label == {component_label}")
    print(sub_nf)
    print()
    if len(sub_nf) == 1:
        edit_mapper = PointMapper(
            point_column="rep_coord_nm",
            set_position=False,
        )
        dataframes.append(sub_nf.nodes)
    else:
        edit_mapper = LineMapper(
            point_column_a="source_rep_coord_nm",
            point_column_b="target_rep_coord_nm",
            set_position=False,
        )
        dataframes.append(sub_nf.edges)
    edit_layer = AnnotationLayerConfig(
        f"level2-path-lines-{component_label}",
        data_resolution=[1, 1, 1],
        color=colors[i],
        mapping_rules=edit_mapper,
    )
    sb_edit_lines = StateBuilder([edit_layer], client=client)
    state_builders.append(sb_edit_lines)


cross_nf = nf.query_nodes(
    "component_label.isin(@show_component_labels)", local_dict=locals()
).query_edges(f"cross_{prefix}operation & (~was_removed)")

edit_mapper = LineMapper(
    point_column_a="source_rep_coord_nm",
    point_column_b="target_rep_coord_nm",
    set_position=False,
)
dataframes.append(cross_nf.edges)
edit_layer = AnnotationLayerConfig(
    "cross-edges",
    data_resolution=[1, 1, 1],
    color=(0.9, 0, 0),
    mapping_rules=edit_mapper,
)
sb_edit_lines = StateBuilder([edit_layer], client=client)
state_builders.append(sb_edit_lines)


link = finalize_link(state_builders, dataframes, client)
link

# %%
nx.cycle_basis(g)

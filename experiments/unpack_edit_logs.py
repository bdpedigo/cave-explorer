# %%
import os

import caveclient as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn.objects as so
from neuropull.graph import NetworkFrame
from tqdm.autonotebook import tqdm

from pkg.edits import (
    apply_edit,
    find_supervoxel_component,
    get_detailed_change_log,
    lazy_load_initial_network,
    lazy_load_network_edits,
    reverse_edit,
)
from pkg.utils import get_level2_nodes_edges, pt_to_xyz

# %%


client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v507")

nuc = client.materialize.query_table(
    "nucleus_detection_v0",  # select_columns=["pt_supervoxel_id", "pt_root_id"]
).set_index("pt_root_id")

# %%

root_id = query_neurons["pt_root_id"].iloc[2]

os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"

nf = lazy_load_initial_network(root_id, client=client)

networkdeltas_by_operation, networkdeltas_by_metaoperation = lazy_load_network_edits(
    root_id, client=client
)

for edit in tqdm(networkdeltas_by_operation.values()):
    apply_edit(nf, edit)


nuc_supervoxel = nuc.loc[root_id, "pt_supervoxel_id"]
nuc_nf = find_supervoxel_component(nuc_supervoxel, nf, client)


nodes, edges = get_level2_nodes_edges(root_id, client=client)
final_nf = NetworkFrame(nodes, edges)

final_nf.nodes.drop(columns=["rep_coord_nm", "x", "y", "z"], inplace=True)
nuc_nf.nodes.drop(columns=["rep_coord_nm", "x", "y", "z"], inplace=True)

assert nuc_nf == final_nf


# %%


def find_relevant_metamerges(
    networkdeltas_by_operation, networkdeltas_by_metaoperation, final_nf
):
    merge_metaedit_pool = []
    for metaoperation_id, networkdelta in networkdeltas_by_metaoperation.items():
        operation_ids = networkdelta.metadata["operation_ids"]
        is_merges = []
        for operation_id in operation_ids:
            is_merges.append(
                networkdeltas_by_operation[operation_id].metadata["is_merge"]
            )
        any_is_merges = any(is_merges)
        is_relevant = networkdelta.added_nodes.index.isin(final_nf.nodes.index).any()
        if any_is_merges and is_relevant:
            merge_metaedit_pool.append(metaoperation_id)

    merge_metaedit_pool = pd.Series(merge_metaedit_pool)
    return merge_metaedit_pool


# %%

# getting a table of additional metadata for each operation

rows = []
all_modified_nodes = []
for root_id in tqdm(
    query_neurons["pt_root_id"].values, desc="Computing edit statistics"
):
    if root_id == 864691135279452833:
        continue
    (
        networkdeltas_by_operation,
        networkdeltas_by_metaoperation,
    ) = lazy_load_network_edits(root_id, client=client)

    nuc_supervoxel = nuc.loc[root_id, "pt_supervoxel_id"]
    current_nuc_level2 = client.chunkedgraph.get_roots([nuc_supervoxel], stop_layer=2)[
        0
    ]
    nuc_pt_nm = client.l2cache.get_l2data(
        [current_nuc_level2], attributes=["rep_coord_nm"]
    )[str(current_nuc_level2)]["rep_coord_nm"]

    change_log = get_detailed_change_log(root_id, client, filtered=False)

    for operation_id, networkdelta in networkdeltas_by_operation.items():
        modified_nodes = pd.concat(
            (networkdelta.added_nodes, networkdelta.removed_nodes)
        )
        n_modified_nodes = len(modified_nodes)
        n_added_nodes = len(networkdelta.added_nodes)
        n_removed_nodes = len(networkdelta.removed_nodes)
        n_added_edges = len(networkdelta.added_edges)
        n_removed_edges = len(networkdelta.removed_edges)
        n_modified_edges = n_added_edges + n_removed_edges
        # modified_node_positions = get_positions(
        #     modified_nodes.index.tolist(), client=client, n_retries=0
        # )

        modified_nodes["root_id"] = root_id
        modified_nodes["operation_id"] = operation_id

        all_modified_nodes.append(modified_nodes)

        row = {
            **networkdelta.metadata,
            "user_name": change_log.loc[operation_id, "user_name"],
            "user_id": change_log.loc[operation_id, "user_id"],
            "root_id": root_id,
            "n_modified_nodes": n_modified_nodes,
            "n_modified_edges": n_modified_edges,
            "n_added_nodes": n_added_nodes,
            "n_removed_nodes": n_removed_nodes,
            "n_added_edges": n_added_edges,
            "n_removed_edges": n_removed_edges,
            "modified_nodes": modified_nodes.index.tolist(),
            "nuc_supervoxel": nuc_supervoxel,
            "current_nuc_level2": current_nuc_level2,
            "nuc_pt_nm": nuc_pt_nm,
            "nuc_x": nuc_pt_nm[0],
            "nuc_y": nuc_pt_nm[1],
            "nuc_z": nuc_pt_nm[2],
        }
        rows.append(row)

edit_stats = pd.DataFrame(rows)
all_modified_nodes = pd.concat(all_modified_nodes)

raw_node_coords = client.l2cache.get_l2data(
    all_modified_nodes.index.to_list(), attributes=["rep_coord_nm"]
)

node_coords = pd.DataFrame(raw_node_coords).T
node_coords[["x", "y", "z"]] = pt_to_xyz(node_coords["rep_coord_nm"])
node_coords.index = node_coords.index.astype(int)

all_modified_nodes = all_modified_nodes.join(node_coords)

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


# %%

# TODO figure out a method for finding the operations that merge soma/nucleus


# fig, ax = plt.subplots(figsize=(10, 10))


so.Plot(
    edit_stats.query("is_merge & (centroid_distance_to_nuc < 1e6)"),
    x="n_modified_nodes",
    y="centroid_distance_to_nuc",
    color="user_id",
).add(so.Dot(pointsize=3, alpha=0.5))

# %%

# edit_stats.query(
#     "is_merge & (centroid_distance_to_nuc < 3e6) & (n_modified_nodes > 50)"
# )

edit_stats.query(
    "(n_modified_nodes > 20) & is_merge & (centroid_distance_to_nuc < 3e6) & was_forrest"
).sort_values("centroid_distance_to_nuc")


# %%
candidate_edits = edit_stats.query(
    "(n_modified_nodes > 10) & is_merge & (centroid_distance_to_nuc < 100_000)"
)
candidate_edits = candidate_edits.sort_values("centroid_distance_to_nuc")

# %%

from IPython.display import display
from nglui.statebuilder import make_neuron_neuroglancer_link

make_neuron_neuroglancer_link(client, [root_id])

i = 0
for idx, row in candidate_edits.iterrows():
    x, y, z = row[["nuc_x", "nuc_y", "nuc_z"]]
    x, y, z = row[["centroid_x", "centroid_y", "centroid_z"]]
    x = x / 4
    y = y / 4
    z = z / 40
    link = make_neuron_neuroglancer_link(
        client,
        [
            row["before_root_ids"][0],
            row["before_root_ids"][1],
            row["after_root_ids"][0],
        ],
        view_kws={"position": (x, y, z)},
    )
    display(link)
    i += 1
    if i > 10:
        break
# img_source =
# %%
nuc.loc[864691136966961614]

# %%
pos = np.array([128664, 202405, 24925])
pos *= np.array([4, 4, 40])
print(pos)

print(row[["nuc_x", "nuc_y", "nuc_z"]])
print(row[["centroid_x", "centroid_y", "centroid_z"]])

# %%


fig, ax = plt.subplots(figsize=(10, 10))
so.Plot(edit_stats, x="n_modified_nodes", y="n_modified_edges", color="is_merge").add(
    so.Dot(pointsize=3, alpha=0.5)
).on(ax)

# %%
fig, ax = plt.subplots(figsize=(10, 10))
so.Plot(
    edit_stats.query("is_merge"),
    x="n_modified_nodes",
    y="n_modified_edges",
    # color="is_merge",
).add(so.Dot(pointsize=3, alpha=0.5)).scale(x="log", y="log").on(ax)


# %%

merge_metaedit_pool = find_relevant_merges(networkdeltas_by_metaoperation, final_nf)
n_samples = 15
frac = 0.5

fig, axs = plt.subplots(3, 5, figsize=(15, 9))

interesting_pool = []
for i in tqdm(range(n_samples)):
    sampled_metaedit_pool = merge_metaedit_pool.sample(frac=frac).values

    partial_nf = final_nf.copy()

    for metaoperation_id in sampled_metaedit_pool:
        networkdelta = networkdeltas_by_metaoperation[metaoperation_id]
        reverse_edit(partial_nf, networkdelta)

    rooted_partial_nf = find_supervoxel_component(nuc_supervoxel, partial_nf, client)

    if len(rooted_partial_nf.nodes) < 1_000:
        interesting_pool += list(sampled_metaedit_pool)

    from pkg.plot import networkplot

    if i < 15:
        ax = axs.flat[i]
        networkplot(
            nodes=rooted_partial_nf.nodes,
            edges=rooted_partial_nf.edges,
            x="x",
            y="y",
            node_size=0.5,
            edge_linewidth=0.25,
            edge_alpha=0.5,
            edge_color="black",
            node_color="black",
            ax=ax,
        )

# %%
from pkg.edits import get_network_edits

get_network_edits(root_id, client=client)

# %%
change_log = get_detailed_change_log(root_id, client, filtered=False)
filtered_change_log = get_detailed_change_log(root_id, client, filtered=True)
change_log["is_filtered"] = False
change_log.loc[filtered_change_log.index, "is_filtered"] = True

# %%
os.environ["SKEDITS_RECOMPUTE"] = "True"
lazy_load_network_edits(root_id, client=client)

# %%

# from nglui.statebuilder import StateBuilder, ChainedStateBuilder, ImageLayerConfig, SegmentationLayerConfig, AnnotationLayerConfig

# sb = StateBuilder()

# %%
root_id = 864691135511632080
operation_id = 11249

edit_df = all_modified_nodes.query(
    "root_id == @root_id & operation_id == @operation_id"
)

# %%
# make_neuron_neuroglancer_link(
#     client,
# )
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

# sb = make_pre_post_statebuilder(
#     client,
#     show_inputs=show_inputs,
#     show_outputs=show_outputs,
#     contrast=contrast,
#     point_column=point_column,
#     view_kws=view_kws,
#     pre_pt_root_id_col=pre_pt_root_id_col,
#     post_pt_root_id_col=post_pt_root_id_col,
#     input_layer_name=input_layer_name,
#     output_layer_name=output_layer_name,
#     input_layer_color=input_color,
#     output_layer_color=output_color,
#     dataframe_resolution_pre=data_resolution_pre,
#     dataframe_resolution_post=data_resolution_post,
#     split_positions=True,
# )

from nglui.statebuilder import (
    AnnotationLayerConfig,
    ChainedStateBuilder,
    PointMapper,
    StateBuilder,
)
from nglui.statebuilder.helpers import from_client

img_layer, seg_layer = from_client(client, contrast=contrast)
seg_layer.add_selection_map(selected_ids_column="root_id")

if view_kws is None:
    view_kws = {}
sb1 = StateBuilder(layers=[img_layer, seg_layer], client=client, view_kws=view_kws)

state_builders = [sb1]

###
output_point_mapper = PointMapper(
    point_column=point_column,
    linked_segmentation_column=post_pt_root_id_col,
    split_positions=split_positions,
)
outputs_lay = AnnotationLayerConfig(
    output_layer_name,
    mapping_rules=[output_point_mapper],
    linked_segmentation_layer=seg_layer.name,
    data_resolution=dataframe_resolution_pre,
    color=output_layer_color,
)
sb_out = StateBuilder([outputs_lay], client=client)
state_builders.append(sb_out)
###

sb = ChainedStateBuilder(state_builders)

package_state(dataframes, sb, client, shorten, return_as, ngl_url, link_text)

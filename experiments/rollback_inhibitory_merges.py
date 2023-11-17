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

root_id = query_neurons["pt_root_id"].values[0]


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

from pkg.utils import get_level2_nodes_edges

nodes, edges = get_level2_nodes_edges(root_id, client, positions=True)

# %%
from neuropull.graph import NetworkFrame

final_nf = NetworkFrame(nodes, edges)

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

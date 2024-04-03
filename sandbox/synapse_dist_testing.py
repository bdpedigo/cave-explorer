# %%

from typing import Iterable

import caveclient as cc
import numpy as np
import pandas as pd
import pcg_skel
from networkframe import NetworkFrame
from nglui.statebuilder import (
    AnnotationLayerConfig,
    ChainedStateBuilder,
    PointMapper,
    StateBuilder,
)
from nglui.statebuilder.helpers import (
    make_neuron_neuroglancer_link,
    make_pre_post_statebuilder,
    package_state,
)
from scipy.sparse.csgraph import dijkstra
from scipy.stats.contingency import chi2_contingency
from sklearn.metrics.pairwise import paired_euclidean_distances

from pkg.constants import MTYPES_TABLE

DEFAULT_POSTSYN_COLOR = (0.25098039, 0.87843137, 0.81568627)  # CSS3 color turquise
DEFAULT_PRESYN_COLOR = (1.0, 0.38823529, 0.27843137)  # CSS3 color tomato

client = cc.CAVEclient("minnie65_phase3_v1")

# %%
proofreading_df = client.materialize.query_table("proofreading_status_public_release")

# %%

i = 6  # index of neuron to examine
root_id = int(proofreading_df.query("valid_id != pt_root_id")["valid_id"].iloc[i])
current_root_id = int(
    proofreading_df.query("valid_id != pt_root_id")["pt_root_id"].iloc[i]
)
# %%

# find the time when the root_id was created
root_lineage = client.chunkedgraph.get_lineage_graph(root_id)

for node in root_lineage["nodes"]:
    if root_id == node["id"]:
        operation_id = node["operation_id"]
        break

details = client.chunkedgraph.get_operation_details([operation_id])
timestamp = details[str(operation_id)]["timestamp"]
timestamp = pd.to_datetime(timestamp, utc=True)

# %%
current_root_lineage = client.chunkedgraph.get_lineage_graph(current_root_id)

# %%
# find the operations that have happened since the current root was created

operations_current = pd.DataFrame(current_root_lineage["nodes"]).dropna()
operations_old = pd.DataFrame(root_lineage["nodes"]).dropna()

new_operations = np.setdiff1d(
    operations_current["operation_id"], operations_old["operation_id"]
).astype(int)

# %%
operation_details = pd.DataFrame(
    client.chunkedgraph.get_operation_details(new_operations)
).T
if "added_edges" in operation_details.columns:
    operation_details["is_merge"] = operation_details["added_edges"].notna()
elif "removed_edges" in operation_details.columns:
    operation_details["is_merge"] = ~operation_details["removed_edges"].notna()


# %%


def make_neuron_neuroglancer_state(
    client,
    root_ids,
    return_as="html",
    shorten="always",
    show_inputs=False,
    show_outputs=False,
    input_color=DEFAULT_POSTSYN_COLOR,
    output_color=DEFAULT_PRESYN_COLOR,
    contrast=None,
    timestamp=None,
    view_kws=None,
    point_column="ctr_pt_position",
    pre_pt_root_id_col="pre_pt_root_id",
    post_pt_root_id_col="post_pt_root_id",
    input_layer_name="syns_in",
    output_layer_name="syns_out",
    ngl_url=None,
    link_text="Neuroglancer Link",
):
    """function to create a neuroglancer link view of a neuron, optionally including inputs and outputs

    Args:
        client (_type_): a CAVEclient configured for datastack to visualize
        root_ids (Iterable[int]): root_ids to build around
        return_as (str, optional): one of 'html', 'json', 'url'. (default 'html')
        shorten (str, optional): if 'always' make a state link always
                             'if_long' make a state link if the json is too long (default)
                             'never' don't shorten link
        show_inputs (bool, optional): whether to include input synapses. Defaults to False.
        show_outputs (bool, optional): whether to include output synapses. Defaults to False.
        sort_inputs (bool, optional): whether to sort inputs by presynaptic root id, ordered by synapse count.
            Defaults to True.
        sort_outputs (bool, optional): whether to sort inputs by presynaptic root id, ordered by postsynaptic synapse count.
            Defaults to True.
        sort_ascending (bool, optional): If sorting, whether to sort ascending (lowest synapse count to highest).
            Defaults to False.
        input_color (list(float) or str, optional): color of input points as rgb list [0,1],
            or hex string, or common name (see webcolors documentation)
        output_color (list(float) or str, optional): color of output points as rgb list [0,1],
            or hex string, or common name (see webcolors documentation)
        contrast (list, optional): Two elements specifying the black level and white level as
            floats between 0 and 1, by default None. If None, no contrast is set.
        timestamp (datetime.datetime, optional): timestamp to do query. Defaults to None, will use materialized version.
        view_kws (dict, optional): view_kws to configure statebuilder, see nglui.StateBuilder.
            Defaults to None.
            keys are:
                show_slices: Boolean
                    sets if slices are shown in the 3d view. Defaults to False.
                layout: str
                    `xy-3d`/`xz-3d`/`yz-3d` (sections plus 3d pane), `xy`/`yz`/`xz`/`3d` (only one pane), or `4panel` (all panes). Default is `xy-3d`.
                show_axis_lines: Boolean
                    determines if the axis lines are shown in the middle of each view.
                show_scale_bar: Boolean
                    toggles showing the scale bar.
                orthographic : Boolean
                    toggles orthographic view in the 3d pane.
                position* : 3-element vector
                    determines the centered location.
                zoom_image : float
                    Zoom level for the imagery in units of nm per voxel. Defaults to 8.
                zoom_3d : float
                    Zoom level for the 3d pane. Defaults to 2000. Smaller numbers are more zoomed in.
        point_column (str, optional): column to pull points for synapses from. Defaults to "ctr_pt_position".
        pre_pt_root_id_col (str, optional): column to pull pre synaptic ids for synapses from.
            Defaults to "pre_pt_root_id".
        post_pt_root_id_col (str, optional): column to pull post synaptic ids for synapses from.
            Defaults to "post_pt_root_id".
        input_layer_name (str, optional): name of layer for inputs. Defaults to "syns_in".
        output_layer_name (str, optional): name of layer for outputs. Defaults to "syns_out".
        ngl_url (str, optional): url to use for neuroglancer.
            Defaults to None (will use default viewer set in datastack)
        link_text (str, optional): text to use for html return.
            Defaults to 'Neuroglancer Link'
    Raises:
        ValueError: If the point column is not present in the synapse table

    Returns:
        Ipython.HTML, str, or json: a representation of the neuroglancer state.Type depends on return_as
    """
    if not isinstance(root_ids, Iterable):
        root_ids = [root_ids]
    df1 = pd.DataFrame({"root_id": root_ids})
    dataframes = [df1]
    data_resolution_pre = None
    data_resolution_post = None
    if show_inputs:
        syn_in_df = client.materialize.synapse_query(
            post_ids=root_ids,
            timestamp=timestamp,
            desired_resolution=client.info.viewer_resolution(),
            split_positions=True,
        )
        data_resolution_pre = syn_in_df.attrs["dataframe_resolution"]
        dataframes.append(syn_in_df)
    if show_outputs:
        syn_out_df = client.materialize.synapse_query(
            pre_ids=root_ids,
            timestamp=timestamp,
            desired_resolution=client.info.viewer_resolution(),
            split_positions=True,
        )
        data_resolution_post = syn_out_df.attrs["dataframe_resolution"]
        dataframes.append(syn_out_df)
    sb = make_pre_post_statebuilder(
        client,
        show_inputs=show_inputs,
        show_outputs=show_outputs,
        contrast=contrast,
        point_column=point_column,
        view_kws=view_kws,
        pre_pt_root_id_col=pre_pt_root_id_col,
        post_pt_root_id_col=post_pt_root_id_col,
        input_layer_name=input_layer_name,
        output_layer_name=output_layer_name,
        input_layer_color=input_color,
        output_layer_color=output_color,
        dataframe_resolution_pre=data_resolution_pre,
        dataframe_resolution_post=data_resolution_post,
        split_positions=True,
    )
    sbs = sb._statebuilders
    return sbs, dataframes


sbs, dfs = make_neuron_neuroglancer_state(
    client,
    current_root_id,
    show_outputs=True,  # timestamp=timestamp
)
# dfs[0].loc[1, "root_id"] = current_root_id
# dfs[0]["root_id"] = dfs[0]["root_id"].astype(int)

sb = ChainedStateBuilder(sbs)

package_state(dfs, sb, client, return_as="html")

# %%
# make_neuron_neuroglancer_link(client, [root_id, current_root_id])

# %%
neuron = pcg_skel.coord_space_meshwork(
    root_id,
    client=client,
    synapses="all",
    synapse_table=client.materialize.synapse_table,
    timestamp=timestamp,
)

# %%
skeleton_edges = neuron.skeleton.edges
skeleton_node_locations = neuron.skeleton.vertices

edge_df = pd.DataFrame(skeleton_edges, columns=["source", "target"])
node_df = pd.DataFrame(skeleton_node_locations, columns=["x", "y", "z"])
# %%

skeleton_nf = NetworkFrame(node_df, edge_df)
skeleton_nf.apply_node_features(["x", "y", "z"], inplace=True)

# %%
edges = skeleton_nf.edges
source_locs = edges[["source_x", "source_y", "source_z"]].values
target_locs = edges[["target_x", "target_y", "target_z"]].values

distances = paired_euclidean_distances(source_locs, target_locs)

edges["distance"] = distances

# %%
weighted_adj = skeleton_nf.to_sparse_adjacency(weight_col="distance")

# %%

# find the nodes w/in x nanometers of each node upstream or downstream
distance_max_nm = 50_000
downstream_path_dists = dijkstra(weighted_adj, directed=True, limit=distance_max_nm)
in_downstream = ~np.isinf(downstream_path_dists)
# ignore diagonal
in_downstream[
    np.arange(len(downstream_path_dists)), np.arange(len(downstream_path_dists))
] = False

upstream_path_dists = dijkstra(weighted_adj.T, directed=True, limit=distance_max_nm)
in_upstream = ~np.isinf(upstream_path_dists)
# ignore diagonal
in_upstream[
    np.arange(len(upstream_path_dists)), np.arange(len(upstream_path_dists))
] = False

nodes = skeleton_nf.nodes
index = nodes.index.values

downstream_subgraph_locs = []
for row in in_downstream:
    downstream_subgraph_locs.append(index[row].tolist())
downstream_subgraph_locs = pd.Series(
    downstream_subgraph_locs, index=skeleton_nf.nodes.index
)

print(
    "Median skeleton nodes downstream:",
    np.median([len(x) for x in downstream_subgraph_locs]),
)

upstream_subgraph_locs = []
for row in in_upstream:
    upstream_subgraph_locs.append(index[row].tolist())
upstream_subgraph_locs = pd.Series(
    upstream_subgraph_locs, index=skeleton_nf.nodes.index
)

print(
    "Median skeleton nodes upstream:",
    np.median([len(x) for x in upstream_subgraph_locs]),
)

subgraph_df = pd.DataFrame(
    {
        "downstream": downstream_subgraph_locs,
        "upstream": upstream_subgraph_locs,
    }
)

# %%
pre_synapses = neuron.anno["pre_syn"].df.copy()
pre_synapses["skel_index"] = neuron.anno["pre_syn"].mesh_index.to_skel_index_padded

# %%
mtypes_df = client.materialize.query_table(MTYPES_TABLE)
mtypes_df.drop_duplicates(["pt_root_id"], inplace=True)

current_roots = client.chunkedgraph.get_roots(pre_synapses["post_pt_supervoxel_id"])
pre_synapses["current_post_pt_root_id"] = current_roots
pre_synapses["post_mtype"] = pre_synapses["current_post_pt_root_id"].map(
    mtypes_df.set_index("pt_root_id")["cell_type"]
)

# %%

test_results = []
for skeleton_node in skeleton_nf.nodes.index:
    downstream_subgraph = subgraph_df.loc[skeleton_node, "downstream"]
    upstream_subgraph = subgraph_df.loc[skeleton_node, "upstream"]
    downstream_synapses = pre_synapses.query("skel_index in @downstream_subgraph")
    downstream_synapses = downstream_synapses.query("post_mtype.notna()")
    upstream_synapses = pre_synapses.query("skel_index in @upstream_subgraph")
    upstream_synapses = upstream_synapses.query("post_mtype.notna()")
    downstream_mtypes = (
        downstream_synapses["post_mtype"].value_counts()
        # .reindex(all_mtypes, fill_value=0)
    )
    downstream_mtypes.name = "downstream_mtype_counts"
    upstream_mtypes = (
        upstream_synapses["post_mtype"].value_counts()
        # reindex(all_mtypes, fill_value=0)
    )
    upstream_mtypes.name = "upstream_mtype_counts"

    mtype_count_table = pd.concat([downstream_mtypes, upstream_mtypes], axis=1).fillna(
        0
    )

    n_synapses_downstream = len(downstream_synapses)
    n_synapses_upstream = len(upstream_synapses)

    if (
        mtype_count_table.empty
        or mtype_count_table.sum().sum() < 5
        or n_synapses_downstream < 5
        or n_synapses_upstream < 5
    ):
        stat, pvalue = np.nan, np.nan
    else:
        stat, pvalue, _, _ = chi2_contingency(mtype_count_table)

    test_results.append(
        {
            "skeleton_node": skeleton_node,
            "n_synapses_downstream": n_synapses_downstream,
            "n_synapses_upstream": n_synapses_upstream,
            "stat": stat,
            "pvalue": pvalue,
        }
    )

test_results = pd.DataFrame(test_results).set_index("skeleton_node")
test_results = test_results.join(skeleton_nf.nodes)
test_results


# %%
sbs, dfs = make_neuron_neuroglancer_state(client, current_root_id, show_outputs=False)


# %%

plot_results = test_results.query("pvalue < 0.05").rename(
    columns={"x": "_x", "y": "_y", "z": "_z"}
)
if not plot_results.empty:
    mapper = PointMapper(point_column="", split_positions=True)
    line_layer = AnnotationLayerConfig(
        name="sig_points",
        data_resolution=[1, 1, 1],
        mapping_rules=mapper,
    )
    line_state = StateBuilder([line_layer], client=client)
    sbs.append(line_state)
    dfs.append(plot_results)

    # sb = ChainedStateBuilder(sbs)
    # out = package_state(dfs, sb, client, return_as="html")


seg_res = client.chunkedgraph.base_resolution
mapper = PointMapper(point_column="sink_coords", split_positions=False)
line_layer = AnnotationLayerConfig(
    name="operations_points",
    data_resolution=seg_res,
    mapping_rules=mapper,
    color="blue",
)
line_state = StateBuilder([line_layer], client=client)
sbs.append(line_state)
dfs.append(operation_details)

sb = ChainedStateBuilder(sbs)
out = package_state(dfs, sb, client, return_as="html")
out

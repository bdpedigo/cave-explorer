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

from meshparty import meshwork

from pcg_skel import features
from skeleton_plot.plot_tools import plot_mw_skel
from pkg.edits import get_detailed_change_log

import datetime

# %%
recompute = False
cloud = False

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

cg = client.chunkedgraph

# %%
meta = client.materialize.query_table("allen_v1_column_types_slanted_ref")
meta = meta.sort_values("target_id")
nuc = client.materialize.query_table("nucleus_detection_v0").set_index("id")

synapse_table = client.info.get_datastack_info()["synapse_table"]

# %%
i = 6
target_id = meta.iloc[i]["target_id"]
root_id = nuc.loc[target_id]["pt_root_id"]
root_id = client.chunkedgraph.get_latest_roots(root_id)[0]

if cloud:
    out_path = "allen-minnie-phase3/edit_info"
    cf = CloudFiles("gs://" + out_path)
else:
    out_path = OUT_PATH / "store_edit_info"
    cf = CloudFiles("file://" + str(out_path))

out_file = f"{root_id}_operations.json"

if not cf.exists(out_file) or recompute:
    print("Pulling network edits")
    networkdeltas_by_operation = get_network_edits(root_id, client, filtered=False)

    networkdelta_dicts = {}
    for operation_id, delta in networkdeltas_by_operation.items():
        networkdelta_dicts[operation_id] = delta.to_dict()

    _ = cf.put_json(out_file, networkdelta_dicts)
    print()

print("Reloading network edits")
networkdelta_dicts = cf.get_json(out_file)
networkdeltas_by_operation = {}
for operation_id, delta in networkdelta_dicts.items():
    networkdeltas_by_operation[int(operation_id)] = NetworkDelta.from_dict(delta)

print()

out_file = f"{root_id}_meta_operations.json"

if not cf.exists(out_file) or recompute:
    print("Compiling meta-edits")
    networkdeltas_by_meta_operation, meta_operation_map = get_network_metaedits(
        networkdeltas_by_operation
    )

    # remap all of the keys to strings
    networkdelta_dicts = {}
    for meta_operation_id, delta in networkdeltas_by_meta_operation.items():
        networkdelta_dicts[str(meta_operation_id)] = delta.to_dict()
    out_meta_operation_map = {}
    for meta_operation_id, operation_ids in meta_operation_map.items():
        out_meta_operation_map[str(meta_operation_id)] = operation_ids

    _ = cf.put_json(out_file, networkdelta_dicts)
    _ = cf.put_json(f"{root_id}_meta_operation_map.json", out_meta_operation_map)

    print()

print("Reloading meta-edits")
networkdelta_dicts = cf.get_json(out_file)
networkdeltas_by_meta_operation = {}
for meta_operation_id, delta in networkdelta_dicts.items():
    networkdeltas_by_meta_operation[int(meta_operation_id)] = NetworkDelta.from_dict(
        delta
    )
in_meta_operation_map = cf.get_json(f"{root_id}_meta_operation_map.json")
meta_operation_map = {}
for meta_operation_id, operation_ids in in_meta_operation_map.items():
    meta_operation_map[int(meta_operation_id)] = operation_ids

print()


def apply_edit(network_frame, network_delta):
    network_frame.add_nodes(network_delta.added_nodes, inplace=True)
    network_frame.add_edges(network_delta.added_edges, inplace=True)
    network_frame.remove_nodes(network_delta.removed_nodes, inplace=True)
    network_frame.remove_edges(network_delta.removed_edges, inplace=True)


def apply_additions(network_frame, network_delta):
    network_frame.add_nodes(network_delta.added_nodes, inplace=True)
    network_frame.add_edges(network_delta.added_edges, inplace=True)


# TODO could make this JSON serializable
nf = get_initial_network(root_id, client, positions=False)
print()

# %%

metaedit_ids = pd.Series(list(networkdeltas_by_meta_operation.keys()))
nuc_supervoxel = nuc.loc[target_id, "pt_supervoxel_id"]


def skeletonize_networkframe(
    networkframe, nan_rounds=10, require_complete=False, soma_pt=None
):
    cv = client.info.segmentation_cloudvolume()

    lvl2_eg = networkframe.edges.values.tolist()
    eg, l2dict_mesh, l2dict_r_mesh, x_ch = build_spatial_graph(
        lvl2_eg,
        cv,
        client=client,
        method="service",
        require_complete=require_complete,
    )
    mesh = trimesh_io.Mesh(
        vertices=x_ch,
        faces=[[0, 0, 0]],  # Some functions fail if no faces are set.
        link_edges=eg,
    )

    sk_utils.fix_nan_verts_mesh(mesh, nan_rounds)

    sk = skeletonize.skeletonize_mesh(
        mesh,
        invalidation_d=10_000,
        compute_radius=False,
        cc_vertex_thresh=0,
        remove_zero_length_edges=True,
        soma_pt=soma_pt,
    )
    return sk, mesh, l2dict_mesh, l2dict_r_mesh


def skeleton_to_treeneuron(skeleton):
    f = "temp.swc"
    skeleton.export_to_swc(f)
    swc = pd.read_csv(f, sep=" ", header=None)
    swc.columns = ["node_id", "structure", "x", "y", "z", "radius", "parent_id"]
    tn = navis.TreeNeuron(swc)
    return tn


# %%

ever_referenced_level2_ids = []

nf = get_initial_network(root_id, client, positions=True)
ever_referenced_level2_ids.extend(nf.nodes.index)

for edit_id, edit in networkdeltas_by_operation.items():
    ever_referenced_level2_ids.extend(edit.added_nodes)

# %%
# this takes about 10 minutes for a fairly small neuron
# probably only have to do once per neuron and stash, though?
ever_referenced_level2_ids
supervoxel_map = []
for l2_id in tqdm(ever_referenced_level2_ids):
    supervoxels = client.chunkedgraph.get_children(l2_id)
    this_map = pd.Series(index=supervoxels, data=l2_id)
    supervoxel_map.append(this_map)
supervoxel_map = pd.concat(supervoxel_map)

# %%
if supervoxel_map.index.duplicated().any():
    print("WARNING: supervoxel map has duplicates")
    supervoxel_map = supervoxel_map[~supervoxel_map.index.duplicated()]

# %%

nuc_id = root_id
nuc_table = "nucleus_detection_lookup_v1"
nuc_id_col = "id"
row = client.materialize.query_view(nuc_table, filter_equal_dict={nuc_id_col: nuc_id})


def get_soma_point(
    object_id,
    client,
    nuc_table="nucleus_detection_lookup_v1",
    id_col="pt_root_id",
    soma_point_resolution=[4, 4, 40],
):
    row = client.materialize.query_view(
        nuc_table, filter_equal_dict={id_col: object_id}
    )
    soma_point = row["pt_position"].values[0]
    soma_point_resolution = np.array(soma_point_resolution)
    soma_point = np.array(soma_point) * soma_point_resolution
    return soma_point


# %%
soma_point = get_soma_point(root_id, client)

skeleton, mesh, l2dict_mesh, l2dict_r_mesh = skeletonize_networkframe(
    nf, soma_pt=soma_point
)
nrn = meshwork.Meshwork(mesh, seg_id=root_id, skeleton=skeleton)

plot_mw_skel(nrn, plot_postsyn=False, plot_presyn=False, plot_soma=True)


# %%

syn_dfs = {}
for side in ["pre", "post"]:
    # find all of the original objects that at some point were part of this neuron
    original_roots = client.chunkedgraph.get_original_roots(root_id)

    # now get all of the latest versions of those objects
    # this will likely be a larger set of objects than we started with since those 
    # objects could have seen further editing, etc.
    latest_roots = client.chunkedgraph.get_latest_roots(original_roots)

    # get the pre/post-synapses that correspond to those objects
    syn_df = client.materialize.query_table(
        synapse_table,
        filter_in_dict={f"{side}_pt_root_id": latest_roots},
    )

    syn_df = syn_df.query("pre_pt_root_id != post_pt_root_id")

    # map the supervoxels attached to the synapses to the level 2 ids as defined by the
    # supervoxel map. this supervoxel map comes from looking up all of the level2 ids 
    # at any point in time
    # TODO I think that implies there *could* be collisions here but I don't think it 
    # will be a big problem in practice, still might break the code someday, though
    syn_df[f"{side}_pt_level2_id"] = syn_df[f"{side}_pt_supervoxel_id"].map(
        supervoxel_map
    )

    # now we can map each of the synapses to the mesh index, via the level 2 id
    syn_df = syn_df.query(f"{side}_pt_level2_id.isin(@l2dict_mesh.keys())")
    syn_df[f"{side}_pt_mesh_ind"] = syn_df[f"{side}_pt_level2_id"].map(l2dict_mesh)

    syn_dfs[side] = syn_df

    # apply these synapse -> mesh index mappings to the meshwork
    nrn.anno.add_annotations(
        f"{side}_syn",
        syn_df,
        index_column=f"{side}_pt_mesh_ind",
        point_column="ctr_pt_position",
        voxel_resolution=syn_df.attrs.get("dataframe_resolution"),
        overwrite=True,
    )

plot_mw_skel(nrn, plot_postsyn=True, plot_presyn=True, plot_soma=True)

# %%
# %%
# for metaedit_id, metaedit in networkdeltas_by_meta_operation.items():
#     apply_edit(nf, metaedit)
# %%
timestamp = client.materialize.get_timestamp()
synapse_table = client.info.get_datastack_info()["synapse_table"]

features.add_synapses(
    nrn,
    synapse_table,
    l2dict_mesh,
    client,
    root_id=root_id,
    pre=True,
    post=True,
    remove_self_synapse=True,
    timestamp=timestamp,
    live_query=False,
    metadata=False,
)
features.add_lvl2_ids(nrn, l2dict_mesh)
# %%

original_roots = client.chunkedgraph.get_original_roots(root_id)
birth_times = client.chunkedgraph.get_root_timestamps(original_roots)

tables = []
t = datetime.datetime.fromisoformat("2020-12-04 20:34:51.059000+00:00")
for og_root, birth_time in tqdm(zip(original_roots[:2], birth_times)):
    # TODO should this be live_query?
    pre_syn_df = client.materialize.live_live_query(
        synapse_table,
        filter_equal_dict={"pre_pt_root_id": og_root},
        timestamp=t,
    )
    post_synapses = client.materialize.live_live_query(
        synapse_table,
        filter_equal_dict={"post_pt_root_id": og_root},
        timestamp=t,
    )
# %%

root = 864691133186822146
t = client.chunkedgraph.get_root_timestamps(root)[0] + datetime.timedelta(seconds=1)
print(t)
currtime = time.time()
pre_syn_df = client.materialize.live_live_query(
    "synapses_pni_2",
    filter_equal_dict={"synapses_pni_2": {"pre_pt_root_id": root}},
    timestamp=t,
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")


# %%
cl = get_detailed_change_log(root_id, client, filtered=False)

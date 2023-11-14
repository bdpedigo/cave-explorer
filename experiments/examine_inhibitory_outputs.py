# %%
import os

import caveclient as cc
from tqdm.autonotebook import tqdm


def get_environment_variables():
    cloud = os.environ.get("SKEDITS_USE_CLOUD") == "True"
    recompute = os.environ.get("SKEDITS_RECOMPUTE") == "True"
    return cloud, recompute


get_environment_variables()
# %%

from pkg.edits.io import get_cloud_paths

os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"
# from pkg.workers import extract_edit_info
client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v507")
cloud = True

for root_id in tqdm(query_neurons["pt_root_id"]):
    out_file = f"{root_id}_operations.json"
    cf = get_cloud_paths(cloud)
    if not cf.exists(out_file):
        print(f"Missing {root_id}")

    # initial_nf = lazy_load_initial_network(root_id, client=client)
# %%
for root_id in tqdm(query_neurons["pt_root_id"]):
    out_file = f"{root_id}_initial_network.json"
    cf = get_cloud_paths(cloud)
    if not cf.exists(out_file):
        print(f"Missing {root_id}")

# %%

from pkg.edits import lazy_load_initial_network

os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"

for root_id in tqdm(query_neurons["pt_root_id"]):
    if root_id == 864691135279452833:
        continue
    nf = lazy_load_initial_network(root_id, client=client)
    has_null = nf.nodes[["x", "y", "z"]].isnull().any().any()
    if has_null:
        print(f"Missing positions for {root_id}")


# %%

root_id = query_neurons["pt_root_id"].iloc[0]

from pkg.utils import get_level2_nodes_edges

nodes, edges = get_level2_nodes_edges(root_id, client=client)

# %%

from neuropull.graph import NetworkFrame

final_nf = NetworkFrame(nodes, edges)

# %%
from pkg.edits import lazy_load_network_edits

networkdeltas_by_operation, networkdeltas_by_metaoperation = lazy_load_network_edits(
    root_id, client=client
)

# %%

merge_metaedit_pool = []
for metaoperation_id, networkdelta in networkdeltas_by_metaoperation.items():
    operation_ids = networkdelta.metadata["operation_ids"]
    is_merges = []
    for operation_id in operation_ids:
        is_merges.append(networkdeltas_by_operation[operation_id].metadata["is_merge"])
    any_is_merges = any(is_merges)
    if any_is_merges:
        merge_metaedit_pool.append(metaoperation_id)

# %%


def reverse_edit(network_frame: NetworkFrame, network_delta):
    network_frame.add_nodes(network_delta.removed_nodes, inplace=True)
    network_frame.add_edges(network_delta.removed_edges, inplace=True)
    network_frame.remove_nodes(network_delta.added_nodes, inplace=True)
    network_frame.remove_edges(network_delta.added_edges, inplace=True)


# %%

import pandas as pd

merge_metaedit_pool = pd.Series(merge_metaedit_pool)

# %%
frac = 0.5
sampled_metaedit_pool = merge_metaedit_pool.sample(frac=frac).values

nf = final_nf.copy()

for metaoperation_id in tqdm(sampled_metaedit_pool):
    networkdelta = networkdeltas_by_metaoperation[metaoperation_id]
    reverse_edit(nf, networkdelta)

# %%

# 156730229585347516


# query_id in final_nf.nodes.index
# query_id in networkdelta.removed_nodes.index
# query_id in networkdelta.added_nodes.index

query_id = 156659860841169753

for networkdelta in networkdeltas_by_operation.values():
    if query_id in networkdelta.removed_nodes.index:
        print("removed:", networkdelta.metadata)
    if query_id in networkdelta.added_nodes.index:
        print("added:", networkdelta.metadata)

# %%
nodes, edges = get_level2_nodes_edges(root_id, client=client)

# %%
query_id in nodes.index

# %%
nf = NetworkFrame(nodes, edges)

# %%
from pkg.edits import apply_edit

root_id = query_neurons["pt_root_id"].iloc[2]

os.environ["SKEDITS_USE_CLOUD"] = "False"
os.environ["SKEDITS_RECOMPUTE"] = "True"

nf = lazy_load_initial_network(root_id, client=client)

networkdeltas_by_operation, networkdeltas_by_metaoperation = lazy_load_network_edits(
    root_id, client=client
)

for edit in tqdm(networkdeltas_by_operation.values()):
    apply_edit(nf, edit)

nodes, edges = get_level2_nodes_edges(root_id, client=client)

server_nf = NetworkFrame(nodes, edges)

assert nf == server_nf

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

os.environ["SKEDITS_USE_CLOUD"] = "False"
os.environ["SKEDITS_RECOMPUTE"] = "True"

networkdeltas_by_operation, networkdeltas_by_metaoperation = lazy_load_network_edits(
    root_id, client=client
)

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

nf = lazy_load_initial_network(root_id, client=client)

for edit in tqdm(networkdeltas_by_operation.values()):
    apply_edit(nf, edit)

nodes, edges = get_level2_nodes_edges(root_id, client=client)

server_nf = NetworkFrame(nodes, edges)

assert nf == server_nf

# %%

from pkg.morphology import find_supervoxel_component

nuc = client.materialize.query_table(
    "nucleus_detection_v0", select_columns=["pt_supervoxel_id", "pt_root_id"]
).set_index("pt_root_id")
nuc_supervoxel = nuc.loc[root_id, "pt_supervoxel_id"]

nuc_nf = find_supervoxel_component(nuc_supervoxel, nf, client)

# %%

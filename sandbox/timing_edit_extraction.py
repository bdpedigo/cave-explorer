# %%
import time
from datetime import datetime, timedelta

import caveclient as cc
import numpy as np
import pandas as pd

from pkg.edits import NetworkDelta, get_changed_edges, get_detailed_change_log
from pkg.utils import get_all_nodes_edges

# %%


def get_network_delta_for_operation(operation_details):
    before_root_ids = operation_details["before_root_ids"]
    after_root_ids = operation_details["roots"]

    # grabbing the union of before/after nodes/edges
    # this works for merge or split
    # NOTE: this is where all the time comes from
    currtime = time.time()
    all_before_nodes, all_before_edges = get_all_nodes_edges(
        before_root_ids, client, positions=False
    )
    all_after_nodes, all_after_edges = get_all_nodes_edges(
        after_root_ids, client, positions=False
    )
    print(f"{time.time() - currtime:.3f} seconds elapsed to hit level2_chunk_graph().")

    # finding the nodes that were added or removed, simple set logic
    added_nodes_index = all_after_nodes.index.difference(all_before_nodes.index)
    added_nodes = all_after_nodes.loc[added_nodes_index]
    removed_nodes_index = all_before_nodes.index.difference(all_after_nodes.index)
    removed_nodes = all_before_nodes.loc[removed_nodes_index]

    # finding the edges that were added or removed, simple set logic again
    removed_edges, added_edges = get_changed_edges(all_before_edges, all_after_edges)

    # keep track of what changed
    metadata = {
        **operation_details.to_dict(),
        "operation_id": operation_id,
        "root_id": root_id,
        "n_added_nodes": len(added_nodes),
        "n_removed_nodes": len(removed_nodes),
        "n_modified_nodes": len(added_nodes) + len(removed_nodes),
        "n_added_edges": len(added_edges),
        "n_removed_edges": len(removed_edges),
        "n_modified_edges": len(added_edges) + len(removed_edges),
    }

    return NetworkDelta(
        removed_nodes, added_nodes, removed_edges, added_edges, metadata=metadata
    )


client = cc.CAVEclient("minnie65_phase3_v1")

root_id = 864691135697251738

change_log = get_detailed_change_log(root_id, client, filtered=False)

# %%
# running list of example cases:
# split operation where added edges and removed edges are both empty: 100606
# split operation where removed edges is not empty (i think this gets "overwritten" later): 157790

# %%
change_log[~change_log["is_merge"]]

# %%%
operation_id = 172185
details = change_log.loc[operation_id]

# %%
delta = get_network_delta_for_operation(details)

delta

currtime = time.time()
before_root_ids = details["before_root_ids"]
after_root_ids = details["roots"]
pre_l2_nodes = []
for root in before_root_ids:
    pre_l2_nodes += list(client.chunkedgraph.get_leaves(root, stop_layer=2))
pre_l2_nodes = pd.Index(pre_l2_nodes)

post_l2_nodes = []
for root in after_root_ids:
    post_l2_nodes += list(client.chunkedgraph.get_leaves(root, stop_layer=2))
post_l2_nodes = pd.Index(post_l2_nodes)
print(f"{time.time() - currtime:.3f} seconds elapsed to hit get_leaves().")

# %%

removed_nodes = pre_l2_nodes.difference(post_l2_nodes)
added_nodes = post_l2_nodes.difference(pre_l2_nodes)

print(removed_nodes)
print(added_nodes)

#####################
# %%

final_edges = client.chunkedgraph.level2_chunk_graph(root_id)
final_edges = pd.DataFrame(final_edges, columns=["source", "target"])
final_edges = final_edges.set_index(["source", "target"], drop=False)
final_node_index = np.unique(final_edges[["source", "target"]].values)
final_nodes = pd.DataFrame(index=final_node_index)

# %%


def map_l1_to_l2_edges(l1_edges, timestamp, client):
    uni_supervoxel_nodes = np.unique(l1_edges.values)

    l2_nodes = client.chunkedgraph.get_roots(
        uni_supervoxel_nodes, stop_layer=2, timestamp=timestamp
    )

    supervoxel_to_l2_map = dict(zip(uni_supervoxel_nodes, l2_nodes))

    l2_edges = l1_edges.replace(supervoxel_to_l2_map)

    l2_edges.drop_duplicates(keep="first", inplace=True)

    l2_edges = l2_edges[(l2_edges["source"] != l2_edges["target"])]

    return l2_edges


operation_timestamp = details["timestamp"]
post_timestamp = datetime.fromisoformat(operation_timestamp) + timedelta(microseconds=1)
pre_timestamp = datetime.fromisoformat(operation_timestamp) - timedelta(microseconds=1)
removed_edges = details["removed_edges"]
removed_edges = pd.DataFrame(removed_edges, columns=["source", "target"])
l2_edges = map_l1_to_l2_edges(removed_edges, pre_timestamp, client)


# %%
l1_edges = removed_edges
timestamp = pre_timestamp
uni_supervoxel_nodes = np.unique(l1_edges.values)

l2_nodes = client.chunkedgraph.get_roots(
    uni_supervoxel_nodes, stop_layer=2, timestamp=timestamp
)

supervoxel_to_l2_map = dict(zip(uni_supervoxel_nodes, l2_nodes))

l2_edges = l1_edges.replace(supervoxel_to_l2_map)
l2_edges
# %%
l2_edges.drop_duplicates(keep="first", inplace=True)

l2_edges
# %%

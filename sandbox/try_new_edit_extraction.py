# %%
import time

import caveclient as cc
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pkg.edits import (
    NetworkDelta,
    get_changed_edges,
    get_detailed_change_log,
)
from pkg.utils import get_all_nodes_edges

client = cc.CAVEclient("minnie65_phase3_v1")

# %%
proofreading_table = client.materialize.query_table(
    "proofreading_status_public_release"
)

# %%
proofreading_table = proofreading_table.set_index("pt_root_id").sort_index()

# %%


def make_bbox(bbox_halfwidth, point_in_nm, seg_resolution):
    x_center, y_center, z_center = point_in_nm

    x_start = x_center - bbox_halfwidth
    x_stop = x_center + bbox_halfwidth
    y_start = y_center - bbox_halfwidth
    y_stop = y_center + bbox_halfwidth
    z_start = z_center - bbox_halfwidth
    z_stop = z_center + bbox_halfwidth

    start_point_cg = np.array([x_start, y_start, z_start]) / seg_resolution
    stop_point_cg = np.array([x_stop, y_stop, z_stop]) / seg_resolution

    bbox_cg = np.array([start_point_cg, stop_point_cg], dtype=int)
    return bbox_cg


root_id = proofreading_table.index[0]
# out = get_network_edits(root_id, client)

# change_log = change_log.sample(n=len(change_log), replace=False)

recompute = False
if recompute:
    change_log = get_detailed_change_log(root_id, client, filtered=False)

    networkdeltas_by_operation = {}
    rows = []
    for operation_id in tqdm(
        change_log.index, desc="Finding network changes for each edit"
    ):
        row = change_log.loc[operation_id]

        before_root_ids = row["before_root_ids"]
        after_root_ids = row["roots"]

        point_in_cg = np.array(row["sink_coords"][0])
        seg_resolution = client.chunkedgraph.base_resolution
        point_in_nm = point_in_cg * seg_resolution
        BBOX_HALFWIDTH = 10_000
        bbox_cg = make_bbox(BBOX_HALFWIDTH, point_in_nm, seg_resolution).T

        # grabbing the union of before/after nodes/edges
        # NOTE: this is where all the compute time comes from
        deltas_to_compare = []
        for bounds, bounds_name in zip([bbox_cg, None], ["bounded", "full"]):
            currtime = time.time()

            all_before_nodes, all_before_edges = get_all_nodes_edges(
                before_root_ids, client, positions=False, bounds=bounds
            )
            all_after_nodes, all_after_edges = get_all_nodes_edges(
                after_root_ids, client, positions=False, bounds=bounds
            )

            # finding the nodes that were added or removed, simple set logic
            added_nodes_index = all_after_nodes.index.difference(all_before_nodes.index)
            added_nodes = all_after_nodes.loc[added_nodes_index]
            removed_nodes_index = all_before_nodes.index.difference(
                all_after_nodes.index
            )
            removed_nodes = all_before_nodes.loc[removed_nodes_index]

            # finding the edges that were added or removed, simple set logic again
            removed_edges, added_edges = get_changed_edges(
                all_before_edges, all_after_edges
            )
            elapsed = time.time() - currtime

            # keep track of what changed
            metadata = {
                **row.to_dict(),
                "operation_id": operation_id,
                "root_id": root_id,
                "n_added_nodes": len(added_nodes),
                "n_removed_nodes": len(removed_nodes),
                "n_modified_nodes": len(added_nodes) + len(removed_nodes),
                "n_added_edges": len(added_edges),
                "n_removed_edges": len(removed_edges),
                "n_modified_edges": len(added_edges) + len(removed_edges),
                "elapsed": elapsed,
                "bounds_name": bounds_name,
            }
            delta = NetworkDelta(
                removed_nodes,
                added_nodes,
                removed_edges,
                added_edges,
                metadata=metadata,
            )
            deltas_to_compare.append(delta)
            rows.append(metadata)

        # compare the two deltas
        if not deltas_to_compare[0] == deltas_to_compare[1]:
            print(f"Difference detected for operation {operation_id}")
            print(deltas_to_compare[0])
            print()
            print(deltas_to_compare[1])
            print()
            break

        networkdeltas_by_operation[operation_id] = delta

        summary = pd.DataFrame(rows)
        summary.to_csv("summary.csv")
else:
    summary = pd.read_csv("summary.csv", index_col=0)

# %%
summary["bounds_name"] = summary.index.map(
    lambda x: "bounded" if x % 2 == 0 else "full"
)

# %%
time_comparison = summary.pivot(
    index="operation_id", columns="bounds_name", values="elapsed"
)
time_comparison["ratio"] = time_comparison["bounded"] / time_comparison["full"]

# %%
import seaborn as sns

sns.histplot(time_comparison["ratio"], bins=40)

# %%
time_comparison["ratio"].median()

# %%
time_comparison["difference"] = time_comparison["full"] - time_comparison["bounded"]

# %%
sns.histplot(time_comparison["difference"], bins=40)

# %%
time_comparison["difference"].median()

# %%
close_ops = time_comparison.query("difference > 0 & difference < 0.2").index

# %%
time_comparison[["bounded", "full"]].sum() / 60

#%%
summary.query('operation_id in @close_ops')
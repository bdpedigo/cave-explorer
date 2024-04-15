# %%

import numpy as np
import pandas as pd
from pkg.edits import get_detailed_change_log

import caveclient as cc
from troglobyte.features import L2AggregateWrangler

client = cc.CAVEclient("minnie65_phase3_v1")

proofreading_df = client.materialize.query_table(
    "proofreading_status_public_release", materialization_version=661
)

# %%
nucs = client.materialize.query_table(
    "nucleus_detection_v0", materialization_version=661
)
unique_roots = proofreading_df["pt_root_id"].unique()
nucs = nucs.query("pt_root_id.isin(@unique_roots)")

# %%
proofreading_df["target_id"] = (
    proofreading_df["pt_root_id"]
    .map(nucs.set_index("pt_root_id")["id"])
    .astype("Int64")
)
# %%
extended_df = proofreading_df.query(
    "status_axon == 'extended' & status_dendrite == 'extended'"
)

# %%
root_id = extended_df["pt_root_id"].sample(n=1).values[0]
# %%
wrangler = L2AggregateWrangler(
    client,
    n_jobs=-1,
    verbose=5,
    neighborhood_hops=5,
    aggregations=["mean", "std"],
)
X = wrangler.get_features([root_id])


# %%

change_log = get_detailed_change_log(root_id, client)

# %%
splits = change_log.query("~is_merge")

# %%
before_root_ids = splits["before_root_ids"]
# %%

sink_points = splits["source_coords"].apply(lambda x: np.mean(x, axis=0))
source_points = splits["sink_coords"].apply(lambda x: np.mean(x, axis=0))
center_points_cg = (sink_points + source_points) / 2
center_points_cg = center_points_cg.apply(pd.Series).rename(
    columns={0: "x", 1: "y", 2: "z"}
)
seg_res = client.chunkedgraph.base_resolution
center_points_nm = center_points_cg * seg_res
center_points_nm = list(
    zip(center_points_nm["x"], center_points_nm["y"], center_points_nm["z"])
)
object_ids = splits["before_root_ids"].apply(lambda x: x[0])


# %%
wrangler = L2AggregateWrangler(
    client,
    n_jobs=-1,
    verbose=5,
    neighborhood_hops=5,
    box_width=10_000,
    aggregations=["mean", "std"],
)
X = wrangler.get_features(object_ids, center_points_nm)

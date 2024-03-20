# %%


import caveclient as cc
import numpy as np
import pandas as pd

client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v507")

root_options = query_neurons["pt_root_id"].values

# %%

# timestamp = pd.to_datetime("2021-07-01 00:00:00", utc=True)
timestamps = pd.date_range("2021-07-01", "2024-01-01", freq="M", tz="UTC")

# %%

object_table = pd.DataFrame()
object_table["working_root_id"] = root_options

# take my list of root IDs
# make sure I have the latest root ID for each, using `get_latest_roots`
is_current_mask = client.chunkedgraph.is_latest_roots(root_options)
outdated_roots = root_options[~is_current_mask]
root_map = dict(zip(root_options[is_current_mask], root_options[is_current_mask]))
for outdated_root in outdated_roots:
    latest_roots = client.chunkedgraph.get_latest_roots(outdated_root)
    sub_nucs = client.materialize.query_table(
        "nucleus_detection_v0", filter_in_dict={"pt_root_id": latest_roots}
    )
    if len(sub_nucs) == 1:
        root_map[outdated_root] = sub_nucs.iloc[0]["pt_root_id"]
    else:
        print(f"Multiple nuc roots for {outdated_root}")

updated_root_options = np.array([root_map[root] for root in root_options])
object_table["current_root_id"] = updated_root_options

# map to nucleus IDs
current_nucs = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"pt_root_id": updated_root_options},
    # select_columns=["id", "pt_root_id"],
).set_index("pt_root_id")["id"]
object_table["target_id"] = object_table["current_root_id"].map(current_nucs)

# %%
timestamp = timestamps[0]

nucs = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"id": object_table["target_id"].to_list()},
).set_index("id")
object_table["pt_supervoxel_id"] = object_table["target_id"].map(
    nucs["pt_supervoxel_id"]
)
object_table["timestamp_root_from_chunkedgraph"] = client.chunkedgraph.get_roots(
    object_table["pt_supervoxel_id"], timestamp=timestamp
)

past_nucs = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"id": object_table["target_id"].to_list()},
    # select_columns=["id", "pt_root_id"],
    timestamp=timestamp,
).set_index("id")["pt_root_id"]
object_table["timestamp_root_from_table"] = object_table["target_id"].map(past_nucs)

# %%
object_table["timestamp_root_from_chunkedgraph"].isin(
    object_table["timestamp_root_from_table"]
).all()

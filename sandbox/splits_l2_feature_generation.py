# %%

import time
from pathlib import Path

import caveclient as cc
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min
from skops.io import load
from tqdm.auto import tqdm
from troglobyte.features import CAVEWrangler

from pkg.edits import get_detailed_change_log

client = cc.CAVEclient("minnie65_phase3_v1")

model_path = Path("data/models/local_compartment_classifier_ej_skeletons.skops")

model = load(model_path)

proofreading_df = client.materialize.query_table("proofreading_status_public_release")

nucs = client.materialize.query_table("nucleus_detection_v0")
nucs = nucs.drop_duplicates(subset="pt_root_id", keep=False)

proofreading_df["target_id"] = (
    proofreading_df["pt_root_id"]
    .map(nucs.set_index("pt_root_id")["id"])
    .astype("Int64")
)

extended_df = proofreading_df.query(
    "status_axon == 'extended' & status_dendrite == 'extended'"
).sort_values("pt_root_id")

# %%


def get_change_log(root_id, client):
    change_log = get_detailed_change_log(root_id, client)
    change_log["timestamp"] = pd.to_datetime(
        change_log["timestamp"], utc=True, format="ISO8601"
    )
    change_log["sink_centroid"] = change_log["sink_coords"].apply(
        lambda x: np.array(x).mean(axis=0)
    )
    change_log["source_centroid"] = change_log["source_coords"].apply(
        lambda x: np.array(x).mean(axis=0)
    )
    change_log["centroid"] = (
        change_log["sink_centroid"] + change_log["source_centroid"]
    ) / 2
    change_log["centroid_nm"] = change_log["centroid"].apply(
        lambda x: x * np.array([8, 8, 40])
    )
    return change_log


out_path = Path("results/outs/split_features")

box_width = 40_000
neighborhood_hops = 5
verbose = False

shape_time = 0
ids_time = 0
synapse_time = 0
aggregate_time = 0

root_ids = extended_df["pt_root_id"].sample(100, random_state=888, replace=False)[2:]
for root_id in tqdm(root_ids):
    # get the change log
    change_log = get_change_log(root_id, client)
    splits = change_log.query("~is_merge")

    # going to query the object right before, at that edit
    before_roots = splits["before_root_ids"].explode()
    points_by_root = {}
    for operation_id, before_root in before_roots.items():
        point = splits.loc[operation_id, "centroid_nm"]
        points_by_root[before_root] = point
    points_by_root = pd.Series(points_by_root)

    # set up a query
    split_wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=verbose)
    split_wrangler.set_objects(before_roots.to_list())
    split_wrangler.set_query_boxes_from_points(points_by_root, box_width=box_width)
    t = time.time()
    split_wrangler.query_level2_ids()
    ids_time += time.time() - t
    t = time.time()
    split_wrangler.query_level2_shape_features()
    shape_time += time.time() - t
    t = time.time()
    split_wrangler.query_level2_synapse_features(method="update")
    synapse_time += time.time() - t
    split_wrangler.register_model(model, "l2class_ej_skeleton")
    t = time.time()
    split_wrangler.aggregate_features_by_neighborhood(
        aggregations=["mean", "std"], neighborhood_hops=neighborhood_hops
    )
    aggregate_time += time.time() - t
    total_time = shape_time + ids_time + synapse_time + aggregate_time

    # handle features
    split_features = split_wrangler.features_
    split_features = split_features.dropna()
    split_features["current_root_id"] = root_id
    split_features = split_features.reset_index().set_index(
        ["current_root_id", "object_id", "level2_id"]
    )

    # handle labels
    split_labels = pd.Series(
        "split", index=split_features.index, name="label"
    ).to_frame()
    _, min_dists_to_edit = pairwise_distances_argmin_min(
        split_features[["rep_coord_x", "rep_coord_y", "rep_coord_z"]],
        np.stack(points_by_root.values),
    )
    split_labels["min_dist_to_edit"] = min_dists_to_edit

    # now do the same stuff for the final cleaned neuron
    nonsplit_wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=verbose)
    nonsplit_wrangler.set_objects([root_id])
    nonsplit_wrangler.query_level2_ids()
    nonsplit_wrangler.query_level2_shape_features()
    nonsplit_wrangler.query_level2_synapse_features(method="update")
    nonsplit_wrangler.register_model(model, "l2class_ej_skeleton")
    nonsplit_wrangler.aggregate_features_by_neighborhood(
        aggregations=["mean", "std"], neighborhood_hops=neighborhood_hops
    )

    # handle features
    nonsplit_features = nonsplit_wrangler.features_
    nonsplit_features = nonsplit_features.dropna()
    nonsplit_features["current_root_id"] = root_id
    nonsplit_features = nonsplit_features.reset_index().set_index(
        ["current_root_id", "object_id", "level2_id"]
    )

    # handle labels
    nonsplit_labels = pd.Series(
        "nonsplit", index=nonsplit_features.index, name="label"
    ).to_frame()
    _, min_dists_to_edit = pairwise_distances_argmin_min(
        nonsplit_features[["rep_coord_x", "rep_coord_y", "rep_coord_z"]],
        np.stack(points_by_root.values),
    )
    nonsplit_labels["min_dist_to_edit"] = min_dists_to_edit

    # save
    features = pd.concat([split_features, nonsplit_features], axis=0)
    labels = pd.concat([split_labels, nonsplit_labels], axis=0)
    features.to_csv(out_path / f"local_features_root_id={root_id}.csv")
    labels.to_csv(out_path / f"local_labels_root_id={root_id}.csv")

print(
    f"Timing: ids={ids_time / total_time:.2f}, shape={shape_time / total_time:.2f}, synapse={synapse_time / total_time:.2f}, aggregate = {aggregate_time / total_time:.2f}"
)

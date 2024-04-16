# %%

import caveclient as cc
import numpy as np
import pandas as pd

from pkg.edits import get_detailed_change_log

client = cc.CAVEclient("minnie65_phase3_v1")

# proofreading_df = client.materialize.query_table(
#     "proofreading_status_public_release", materialization_version=661
# )

# # %%
# nucs = client.materialize.query_table(
#     "nucleus_detection_v0", materialization_version=661
# )
# unique_roots = proofreading_df["pt_root_id"].unique()
# nucs = nucs.query("pt_root_id.isin(@unique_roots)")

# # %%
# proofreading_df["target_id"] = (
#     proofreading_df["pt_root_id"]
#     .map(nucs.set_index("pt_root_id")["id"])
#     .astype("Int64")
# )
# # %%
# extended_df = proofreading_df.query(
#     "status_axon == 'extended' & status_dendrite == 'extended'"
# )

# # %%
# root_id = extended_df["pt_root_id"].sample(n=1, random_state=888).values[0]


# %%
from pkg.neuronframe import load_neuronframe
from pkg.utils import load_manifest

manifest = load_manifest()

root_id = manifest.index[0]

nf = load_neuronframe(root_id, client)

# %%
import pyvista as pv

from pkg.plot import set_up_camera

pv.set_jupyter_backend("client")
plotter = pv.Plotter()
set_up_camera(plotter, nf)
plotter.add_mesh(nf.to_skeleton_polydata(), color="black", line_width=0.1)
plotter.show()

# %%
poly = nf.to_skeleton_polydata()

# %%
edited_nf = nf.set_edits(nf.metaedits.query("has_merge").index, prefix="meta")

edited_nf.select_nucleus_component(inplace=True)
edited_nf.remove_unused_nodes(inplace=True)
edited_nf

# %%
# edited_nf.plot_pyvista()

# %%

edit_index = edited_nf.nodes.query("operation_removed != -1").index
edit_iloc = edited_nf.nodes.index.get_indexer_for(edit_index)

# %%
edited_nf.apply_edge_lengths(inplace=True)
sparse_adj = edited_nf.to_sparse_adjacency(weight_col="length")

from scipy.sparse.csgraph import dijkstra

path_length_to_edits = dijkstra(
    sparse_adj, indices=edit_iloc, min_only=True, directed=False
)

path_length_to_edits = pd.Series(path_length_to_edits, index=edited_nf.nodes.index)


# %%

from troglobyte.features import CAVEWrangler

wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=False)

wrangler.set_level2_ids(edited_nf.nodes.index)
wrangler.query_current_object_ids()
wrangler.query_level2_shape_features()
wrangler.query_level2_synapse_features()

# %%
features = wrangler.features_.droplevel("object_id")
features = features.drop(columns=["rep_coord_x", "rep_coord_y", "rep_coord_z"])

# %%
edited_nf.nodes = features
# %%
# nodes = edited_nf.nodes.join(features, how="left")
# drop_cols = [
#     "operation_added",
#     "operation_removed",
#     "metaoperation_added",
#     "metaoperation_removed",
#     "was_removed",
#     "rep_coord_nm",
#     "x",
#     "y",
#     "z",
#     "synapses",
#     "pre_synapses",
#     "post_synapses",
#     "has_synapses",
#     "nucleus",
#     "rep_coord_x",
#     "rep_coord_y",
#     "rep_coord_z",
# ]
# nodes = nodes.drop(columns=drop_cols)
# %%

neighborhood_features = edited_nf.k_hop_aggregation(k=10, aggregations=["mean", "std"])

# %%
joined_features = features.join(neighborhood_features, how="left")

# %%

close_nodes = path_length_to_edits[path_length_to_edits < 2_000].index
far_nodes = path_length_to_edits[path_length_to_edits >= 50_000].index

features_close = joined_features.loc[close_nodes]
features_far = joined_features.loc[far_nodes]

features_close = features_close.dropna()
features_far = features_far.dropna()

sub_features_far = features_far.sample(n=1 * len(features_close))


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import QuantileTransformer

X = pd.concat([features_close, sub_features_far], axis=0)
# X = X.drop(columns=[x for x in X.columns if "std" in x])
y = pd.Series(["split"] * len(features_close) + ["no split"] * len(sub_features_far))

X_transformed = QuantileTransformer(output_distribution="normal").fit_transform(X)
lda = LinearDiscriminantAnalysis()
lda.fit(X_transformed, y)

X_lda = lda.transform(X_transformed)

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

sns.histplot(x=X_lda[:, 0], hue=y, ax=ax)


from sklearn.metrics import classification_report

y_pred = lda.predict(X_transformed)

print(classification_report(y, y_pred))

# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=3)
rf.fit(X, y)

y_pred = rf.predict(X)

print(classification_report(y, y_pred))


# %%

fig, axs = plt.subplots(1, 2, figsize=(8, 10), sharey=True)
sns.barplot(x=lda.coef_[0], y=X.columns, ax=axs[0])
sns.barplot(x=rf.feature_importances_, y=X.columns, ax=axs[1])

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
before_roots = (
    splits["before_root_ids"]
    .apply(lambda x: x[0])
    .rename("root_id")
    .to_frame()
    .reset_index()
)
# after_roots1 = (
#     splits["roots"].apply(lambda x: x[0]).rename("root_id").to_frame().reset_index()
# )
# after_roots2 = (
#     splits["roots"].apply(lambda x: x[1]).rename("root_id").to_frame().reset_index()
# )
after_roots = (
    splits.explode("roots")["roots"].rename("root_id").to_frame().reset_index()
)

# %%
object_ids = pd.concat([before_roots, after_roots], axis=0)

operation_info = pd.DataFrame(index=splits.index)
operation_info["center_point_nm"] = center_points_nm

object_ids["center_point_nm"] = object_ids["operation_id"].map(
    operation_info["center_point_nm"]
)

object_ids = object_ids.drop_duplicates("root_id")
object_ids.set_index("root_id", inplace=True)

# %%
from troglobyte.features import CAVEWrangler

wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=False)

wrangler.set_objects(object_ids.index)
# wrangler.set_query_boxes_from_points(object_ids["center_point_nm"], box_width=20_000)
wrangler.query_level2_shape_features().query_synapse_features()


# %%

wrangler.aggregate_features_by_neighborhood()

# %%

features = wrangler.features_

# %%
split_pos = np.array(center_points_nm)
l2_node_pos = features[["rep_coord_x", "rep_coord_y", "rep_coord_z"]].values

# %%
from sklearn.metrics import pairwise_distances_argmin_min

split_iloc, closest_split_dist = pairwise_distances_argmin_min(l2_node_pos, split_pos)

# %%

sns.histplot(closest_split_dist, bins=100)


# %%


# %%
random_wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=False)

random_wrangler.set_objects(object_ids.index)
random_wrangler.query_level2_ids().sample_level2_nodes(100)


# %%
wrangler.features_

# %%
wrangler.features_.dropna()

import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(1, 3, figsize=(10, 10))

X = (
    wrangler.features_.dropna()[
        ["pca_unwrapped_0", "pca_unwrapped_1", "pca_unwrapped_2"]
    ]
    .iloc[:100]
    .values
)
sns.heatmap(
    X,
    ax=axs[0],
    vmin=-1,
    vmax=1,
    cmap="RdBu_r",
    center=0,
    yticklabels=False,
    cbar=False,
)

sns.heatmap(
    -1 * X,
    ax=axs[1],
    vmin=-1,
    vmax=1,
    cmap="RdBu_r",
    center=0,
    yticklabels=False,
    cbar=False,
)

multiplier = np.array([1 if x[0] >= 0 else -1 for x in X])

sns.heatmap(
    multiplier[:, None] * X,
    ax=axs[2],
    vmin=-1,
    vmax=1,
    cmap="RdBu_r",
    center=0,
    yticklabels=False,
    cbar=False,
)

X_trans = multiplier[:, None] * X
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(x=X[:, 0], y=X[:, 1], ax=ax)
sns.scatterplot(x=X_trans[:, 0], y=X_trans[:, 1], ax=ax, color="red", s=10, zorder=2)


# sns.heatmap(
#     wrangler.features_.dropna()[
#         [
#             "pca_unwrapped_0_neighbor_mean",
#             "pca_unwrapped_1_neighbor_mean",
#             "pca_unwrapped_2_neighbor_mean",
#         ]
#     ].iloc[:100],
#     ax=axs[2],
#     vmin=-1,
#     vmax=1,
#     cmap="RdBu_r",
#     center=0,
#     yticklabels=False,
# )

# %%
np.linalg.norm(
    wrangler.features_.dropna()[
        ["pca_unwrapped_0", "pca_unwrapped_1", "pca_unwrapped_2"]
    ].values,
    axis=1,
)

# %%
# # %%
# wrangler = L2AggregateWrangler(
#     client,
#     n_jobs=-1,
#     verbose=5,
#     neighborhood_hops=5,
#     box_width=10_000,
#     aggregations=["mean", "std"],
# )
# X = wrangler.get_features(object_ids, center_points_nm)

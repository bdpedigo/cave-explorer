# %%

from pathlib import Path

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from skops.io import load

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
)

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
    change_log["centroid_img"] = change_log["centroid_nm"].apply(
        lambda x: x / np.array([4, 4, 40])
    )
    return change_log


out_path = Path("results/outs/split_features")

neighborhood_hops = 5
verbose = False

shape_time = 0
ids_time = 0
synapse_time = 0
aggregate_time = 0

root_ids = extended_df["pt_root_id"].sample(100, random_state=88, replace=False)


# %%

# feature_files = glob.glob(str(out_path) + "/local_features_*.csv")
# label_files = glob.glob(str(out_path) + "/local_labels_*.csv")

feature_files = [
    out_path / f"local_features_root_id={root}.csv"
    for root in [864691135577956101, 864691135212863360]
]
label_files = [
    out_path / f"local_labels_root_id={root}.csv"
    for root in [864691135577956101, 864691135212863360]
]

features = []
for file in feature_files:
    cell_features = pd.read_csv(file, index_col=[0, 1, 2])
    features.append(cell_features)
features = pd.concat(features)

labels = []
for file in label_files:
    cell_labels = pd.read_csv(file, index_col=[0, 1, 2])
    labels.append(cell_labels)
labels = pd.concat(labels)
labels = labels.loc[features.index]

# assert features.index.equals(labels.index)

# %%
sub_labels = labels.query(
    "label.isin(['split', 'postsplit']) & min_dist_to_edit < 2_000"
)
sub_features = features.loc[sub_labels.index]

X = sub_features.drop(
    columns=[col for col in sub_features.columns if "rep_coord_" in col]
)
X = X.drop(columns=[col for col in X.columns if "pca_unwrapped_" in col])
y = sub_labels["label"]

from sklearn.preprocessing import QuantileTransformer

model = Pipeline(
    [
        ("transform", QuantileTransformer(output_distribution="normal")),
        ("lda", LinearDiscriminantAnalysis()),
    ]
)
model.fit(X, y)


# %%
labels = labels.query("(label=='nonsplit') | (min_dist_to_edit < 2_000)")
features = features.loc[labels.index]
features = features.drop(
    columns=[col for col in features.columns if "rep_coord_" in col]
)
features = features.drop(
    columns=[col for col in features.columns if "pca_unwrapped_" in col]
)

root_ids = features.index.get_level_values("current_root_id").unique()

train_root_ids, test_root_ids = train_test_split(root_ids, test_size=0.2)

X_train = features.loc[train_root_ids]
y_train = labels.loc[X_train.index, "label"]

X_test = features.loc[test_root_ids]
y_test = labels.loc[X_test.index, "label"]


model = Pipeline(
    [
        ("transform", QuantileTransformer(output_distribution="normal")),
        ("lda", LinearDiscriminantAnalysis()),
    ]
)

model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print(classification_report(y_train, y_pred_train))
print(classification_report(y_test, y_pred_test))


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=500, max_depth=5, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

print(classification_report(y_train, y_pred_train))
print(classification_report(y_test, y_pred_test))


# %%

lda = model.steps[1][1]
fig, axs = plt.subplots(1, 2, figsize=(8, 10), sharey=True)
sns.barplot(x=lda.coef_[0], y=X_train.columns, ax=axs[0])
sns.barplot(x=rf.feature_importances_, y=X_train.columns, ax=axs[1])

# %%
rf_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
rf_importances = rf_importances.sort_values(ascending=False).head(10)
for i, (name, importance) in enumerate(rf_importances.items()):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    sns.histplot(
        x=X_test[name],
        hue=y_test,
        ax=ax,
        alpha=0.5,
        bins=50,
        common_norm=False,
        stat="density",
        log_scale=True,
    )

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
sns.histplot(
    x=X_test["pca_dot_product_0_neighbor_mean"],
    hue=y_test,
    ax=ax,
    alpha=0.5,
    bins=50,
    common_norm=False,
    stat="density",
    log_scale=True,
)


# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

X_lda_test = model.transform(X_test)
sns.histplot(x=X_lda_test[:, 0], hue=y_test, ax=ax)


# %%
unused_root_ids = pd.Index(extended_df["pt_root_id"]).difference(root_ids)

# %%
new_root_id = unused_root_ids[11]
new_change_log = get_change_log(new_root_id, client)

new_splits = new_change_log.query("~is_merge")
new_splits = new_splits.sort_values("timestamp")
first_new_root_id = new_splits["before_root_ids"].iloc[0][0]


from troglobyte.features import CAVEWrangler

model_path = Path("data/models/local_compartment_classifier_ej_skeletons.skops")

ej_model = load(model_path)

wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=2)
wrangler.set_objects([first_new_root_id])
wrangler.query_level2_shape_features()
wrangler.query_level2_synapse_features(method="update")
wrangler.register_model(ej_model, "l2class_ej_skeleton")
wrangler.aggregate_features_by_neighborhood(
    aggregations=["mean", "std"], neighborhood_hops=neighborhood_hops
)


wrangler.query_level2_networks()
nf = wrangler.object_level2_networks_[first_new_root_id]


positions = np.stack(
    wrangler.features_[["rep_coord_x", "rep_coord_y", "rep_coord_z"]]
    .apply(lambda x: np.array(x))
    .values
)
split_positions = np.stack(new_splits["centroid_nm"].values)

from sklearn.metrics import pairwise_distances_argmin_min

closest, min_dist = pairwise_distances_argmin_min(split_positions, positions)

relevant_splits = new_splits.iloc[min_dist < 3_000]


features = wrangler.features_.dropna()

y_pred_new = model.predict(features[model.feature_names_in_])
y_pred_new = pd.Series(
    y_pred_new, index=features.index.get_level_values("level2_id"), name="split_pred"
)

nf.nodes["split_pred"] = "nonsplit"
nf.nodes.loc[y_pred_new.index, "split_pred"] = y_pred_new
nf.nodes["split_pred_float"] = nf.nodes["split_pred"].map(
    {"nonsplit": 0.0, "split": 1.0}
)


import json
from pathlib import Path
from typing import Optional, Union

import caveclient as cc
import cloudvolume
import numpy as np
import pandas as pd
import seaborn as sns
from networkframe import NetworkFrame
from nglui.statebuilder import (
    AnnotationLayerConfig,
    ChainedStateBuilder,
    ImageLayerConfig,
    PointMapper,
    SegmentationLayerConfig,
    StateBuilder,
)
from nglui.statebuilder.helpers import package_state
from skops.io import load


def write_networkframes_to_skeletons(
    networkframes: Union[NetworkFrame, dict[NetworkFrame]],
    client: cc.CAVEclient,
    attribute: Optional[str] = None,
    directory: str = "gs://allen-minnie-phase3/tempskel",
    spatial_columns=None,
):
    if spatial_columns is None:
        spatial_columns = ["x", "y", "z"]
    # register an info file and set up CloudVolume
    base_info = client.chunkedgraph.segmentation_info
    base_info["skeletons"] = "skeleton"
    info = base_info.copy()

    cv = cloudvolume.CloudVolume(
        f"precomputed://{directory}",
        mip=0,
        info=info,
        compress=False,
    )
    cv.commit_info()

    sk_info = cv.skeleton.meta.default_info()
    sk_info["vertex_attributes"] = [
        {"id": "radius", "data_type": "float32", "num_components": 1},
        {"id": "vertex_types", "data_type": "float32", "num_components": 1},
    ]
    cv.skeleton.meta.info = sk_info
    cv.skeleton.meta.commit_info()

    sks = []
    if isinstance(networkframes, NetworkFrame):
        networkframes = {0: networkframes}

    for name, networkframe in networkframes.items():
        # extract vertex information
        vertices = networkframe.nodes[spatial_columns].values
        edges_unmapped = networkframe.edges[["source", "target"]].values
        edges = networkframe.nodes.index.get_indexer_for(
            edges_unmapped.flatten()
        ).reshape(edges_unmapped.shape)

        vertex_types = networkframe.nodes[attribute].values.astype(np.float32)

        radius = np.ones(len(vertices), dtype=np.float32)

        sk_cv = cloudvolume.Skeleton(
            vertices,
            edges,
            radius,
            None,
            segid=name,
            extra_attributes=sk_info["vertex_attributes"],
            space="physical",
        )
        sk_cv.vertex_types = vertex_types

        sks.append(sk_cv)

    cv.skeleton.upload(sks)


write_networkframes_to_skeletons(
    {first_new_root_id: nf},
    client,
    attribute="split_pred_float",
    directory="gs://allen-minnie-phase3/tempskel",
    spatial_columns=["rep_coord_x", "rep_coord_y", "rep_coord_z"],
)

# base_state = json.loads(make_neuron_neuroglancer_link(client, first_new_root_id, return_as='json'))

dfs = []
sbs = []
viewer_resolution = client.info.viewer_resolution()
img_layer = ImageLayerConfig(
    client.info.image_source(),
)
seg_layer = SegmentationLayerConfig(
    client.info.segmentation_source(),
    alpha_3d=0.3,
    name="seg",
    color_column="color",
)
seg_layer.add_selection_map(selected_ids_column="object_id")

skel_layer = SegmentationLayerConfig(
    "precomputed://gs://allen-minnie-phase3/tempskel",
    name="skeleton",
)
skel_layer.add_selection_map(selected_ids_column="object_id")
base_sb = StateBuilder(
    [img_layer, seg_layer, skel_layer],
    client=client,
    resolution=viewer_resolution,
)

sbs.append(base_sb)
dfs.append(pd.DataFrame({"object_id": [first_new_root_id], "color": "#ffffff"}))


# skel_layer = SegmentationLayerConfig(
#     "precomputed://gs://allen-minnie-phase3/tempskel",
#     name="skeleton",
# )
# skel_layer.add_selection_map(selected_ids_column="object_id")
# sb = StateBuilder(layers=[skel_layer], base_state=base_state)
# dfs.append(pd.DataFrame({"object_id": [first_new_root_id]}))
# sbs.append(sb)

point_mapper = PointMapper(point_column="centroid_img")
point_layer = AnnotationLayerConfig(
    name="split_centroids",
    mapping_rules=point_mapper,
    data_resolution=[2, 2, 1],
)
sb = StateBuilder(layers=[point_layer])
dfs.append(relevant_splits)
sbs.append(sb)

sb = ChainedStateBuilder(sbs)
state = json.loads(package_state(dfs, sb, client, return_as="json"))

state["layers"][1]["segmentColors"] = {str(first_new_root_id): "#ffffff"}

shader = """
void main() {
    float compartment = vCustom2;
    vec4 uColor = segmentColor();
    emitRGB(vec3(0.1, 0.2, 0.4) + compartment * vec3(1, 0, 0));
}
"""
shader = """
void main() {
    float compartment = vCustom2;
    vec4 uColor = segmentColor();
    if (compartment > 0.1) {
      emitRGB(vec3(0.9, 0.1, 0.1));
    } 
    else {
      emitRGB(vec3(0.1, 0.1, 0.4));
   }
}
"""
skel_rendering_kws = {
    "shader": shader,
    "mode2d": "lines_and_points",
    "mode3d": "lines",
    "lineWidth3d": 3,
}

state["layers"][2]["skeletonRendering"] = skel_rendering_kws

StateBuilder(base_state=state, client=client).render_state(return_as="html")


# %%

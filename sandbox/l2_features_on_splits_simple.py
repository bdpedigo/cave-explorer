# %%

from pathlib import Path

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, pairwise_distances_argmin_min
from sklearn.preprocessing import QuantileTransformer
from skops.io import load
from tqdm.auto import tqdm
from troglobyte.features import CAVEWrangler

from pkg.edits import get_detailed_change_log
from pkg.plot import set_up_camera

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

out_path = Path("results/outs/split_features")

box_width = 5_000
neighborhood_hops = 5
verbose = False
all_features = []
all_labels = []
root_ids = extended_df["pt_root_id"].sample(100, random_state=88)

for root_id in tqdm(root_ids):
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

    splits = change_log.query("~is_merge")

    before_roots = splits["before_root_ids"].explode()
    # timestamps = splits["timestamp"]
    # timestamps = timestamps - timedelta(milliseconds=2)

    points_by_root = {}
    for operation_id, before_root in before_roots.items():
        point = splits.loc[operation_id, "centroid_nm"]
        points_by_root[before_root] = point
    points_by_root = pd.Series(points_by_root)

    wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=verbose)
    wrangler.set_objects(before_roots.to_list())
    wrangler.set_query_boxes_from_points(points_by_root, box_width=box_width)
    wrangler.query_level2_shape_features()
    wrangler.query_level2_synapse_features(method="update")
    wrangler.register_model(model, "l2class_ej_skeleton")
    wrangler.aggregate_features_by_neighborhood(
        aggregations=["mean", "std"], neighborhood_hops=neighborhood_hops
    )
    features = wrangler.features_
    features = features.dropna()
    features["current_root_id"] = root_id
    features = features.reset_index().set_index(
        ["current_root_id", "object_id", "level2_id"]
    )
    all_features.append(features)

    labels = pd.Series("split", index=features.index, name="label").to_frame()
    _, min_dists_to_edit = pairwise_distances_argmin_min(
        features[["rep_coord_x", "rep_coord_y", "rep_coord_z"]],
        np.stack(points_by_root.values),
    )
    labels["min_dist_to_edit"] = min_dists_to_edit
    labels["current_root_id"] = root_id
    labels = labels.reset_index().set_index(
        ["current_root_id", "object_id", "level2_id"]
    )
    all_labels.append(labels)

    # now do the same for the final cleaned neuron
    wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=verbose)
    wrangler.set_objects([root_id])
    wrangler.query_level2_shape_features()
    wrangler.query_level2_synapse_features(method="existing")
    wrangler.register_model(model, "l2class_ej_skeleton")
    wrangler.aggregate_features_by_neighborhood(
        aggregations=["mean", "std"], neighborhood_hops=neighborhood_hops
    )
    features = wrangler.features_
    features = features.dropna()
    all_features.append(features)
    labels = pd.Series("no split", index=features.index, name="label").to_frame()
    _, min_dists_to_edit = pairwise_distances_argmin_min(
        features[["rep_coord_x", "rep_coord_y", "rep_coord_z"]],
        np.stack(points_by_root.values),
    )
    labels["min_dist_to_edit"] = min_dists_to_edit
    labels["root_id"] = root_id
    all_labels.append(labels)

# %%

features = pd.concat(all_features, axis=0)
labels = pd.concat(all_labels, axis=0)

features.to_csv("feature_dump.csv")
labels.to_csv("label_dump.csv")

# %%
split_labels = labels.query("(label == 'split') & (min_dist_to_edit < 2000)")
non_split_labels = labels.query("(label == 'no split')")

# %%

X = pd.concat(
    [features.loc[split_labels.index], features.loc[non_split_labels.index]], axis=0
)
X = X.drop(columns=[col for col in X.columns if "rep_coord_" in col])
y = pd.Series(
    ["split"] * len(split_labels) + ["no split"] * len(non_split_labels),
    index=X.index,
)
y.index = X.index

# %%
from sklearn.model_selection import train_test_split

train_root_ids, test_root_ids = train_test_split(root_ids, test_size=0.2)

X_train = X
y_train = y

X_test = X
y_test = y

# %%
# try LDA
transformer = QuantileTransformer(output_distribution="normal")
X_transformed = transformer.fit_transform(X_train)
lda = LinearDiscriminantAnalysis(priors=[0.5, 0.5])
lda.fit(X_transformed, y_train)

X_lda = lda.transform(X_transformed)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

sns.histplot(x=X_lda[:, 0], hue=y_train, ax=ax)

y_pred_train = lda.predict(X_transformed)
y_pred_test = lda.predict(transformer.transform(X_test))

print(classification_report(y_train, y_pred_train))
print(classification_report(y_test, y_pred_test))

# %%
# try random forest
rf = RandomForestClassifier(n_estimators=500, max_depth=4)
rf.fit(X_train, y_train)

y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

print(classification_report(y_train, y_pred_train))
print(classification_report(y_test, y_pred_test))

# %%

# look at the feature importances

fig, axs = plt.subplots(1, 2, figsize=(8, 10), sharey=True)
sns.barplot(x=lda.coef_[0], y=X.columns, ax=axs[0])
sns.barplot(x=rf.feature_importances_, y=X.columns, ax=axs[1])

# %%
from pkg.neuronframe import load_neuronframe

root_id = test_root_ids[3]

nf = load_neuronframe(root_id, client)
edited_nf = nf.set_edits(nf.metaedits.query("has_merge").index, prefix="meta")
edited_nf.select_nucleus_component(inplace=True)
edited_nf.remove_unused_nodes(inplace=True)


# %%

features, level2_info, relevant_nf, edited_nf, final_nf = generate_features_for_root(
    root_id, client, return_networkframes=True
)

full_X = X_test.loc[root_id]
full_X_lda = lda.transform(transformer.transform(full_X))


# %%

show_neuron = True

if show_neuron:
    pv.set_jupyter_backend("client")
    plotter = pv.Plotter()
    set_up_camera(plotter, edited_nf)
    plotter.add_mesh(edited_nf.to_skeleton_polydata(), color="black", line_width=1)
    plotter.add_mesh(final_nf.to_skeleton_polydata(), color="blue", line_width=1)
    plotter.add_mesh(edited_nf.to_split_polydata(), color="red")
    plotter.show()


# %%
if show_neuron:
    pv.set_jupyter_backend("client")
    plotter = pv.Plotter()
    set_up_camera(plotter, relevant_nf)

    plotter.add_mesh(
        relevant_nf.to_skeleton_polydata(label="scalars"),
        scalars="scalars",
        line_width=0.1,
    )
    # plotter.add_mesh(final_nf.to_skeleton_polydata(), color="blue", line_width=0.3)
    plotter.add_mesh(relevant_nf.to_split_polydata(), color="red")
    plotter.show()


# %%
from sklearn.preprocessing import StandardScaler

edited_nf.nodes["lda_pred_val"] = 0.0
edited_nf.nodes.loc[
    edited_nf.nodes.index.intersection(full_X.index), "lda_pred_val"
] = 0.1 * StandardScaler().fit_transform(full_X_lda) + 0.5


# %%

pv.set_jupyter_backend("client")
plotter = pv.Plotter()
set_up_camera(plotter, edited_nf)
plotter.add_mesh(
    edited_nf.to_skeleton_polydata(label="lda_pred_val"),
    line_width=1,
    scalars="lda_pred_val",
    cmap="coolwarm",
)
plotter.add_mesh(
    edited_nf.to_split_polydata(),
    color="black",
)
plotter.show()

# %%

from pathlib import Path
from typing import Optional, Union

import caveclient as cc
import cloudvolume
import numpy as np
import pandas as pd
import seaborn as sns
from networkframe import NetworkFrame
from skops.io import load


def write_networkframes_to_skeletons(
    networkframes: Union[NetworkFrame, dict[NetworkFrame]],
    client: cc.CAVEclient,
    attribute: Optional[str] = None,
    directory: str = "gs://allen-minnie-phase3/tempskel",
):
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
        vertices = networkframe.nodes[["x", "y", "z"]].values
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
    {root_id: edited_nf},
    client,
    attribute="lda_pred_val",
    directory="gs://allen-minnie-phase3/tempskel",
)

# %%

import json
from pathlib import Path
from typing import Optional, Union

import caveclient as cc
import cloudvolume
import numpy as np
import pandas as pd
import seaborn as sns
from networkframe import NetworkFrame
from nglui import statebuilder
from skops.io import load

sbs = []
dfs = []
layers = []
viewer_resolution = client.info.viewer_resolution()
img_layer = statebuilder.ImageLayerConfig(
    client.info.image_source(),
)
seg_layer = statebuilder.SegmentationLayerConfig(
    client.info.segmentation_source(),
    alpha_3d=0.1,
    name="seg",
)
seg_layer.add_selection_map(selected_ids_column="object_id")

skel_layer = statebuilder.SegmentationLayerConfig(
    "precomputed://gs://allen-minnie-phase3/tempskel",
    name="skeleton",
)
skel_layer.add_selection_map(selected_ids_column="object_id")
base_sb = statebuilder.StateBuilder(
    [img_layer, seg_layer, skel_layer],
    client=client,
    resolution=viewer_resolution,
)

sbs.append(base_sb)
dfs.append(pd.DataFrame({"object_id": [root_id]}))

splits = edited_nf.edits.query("~is_merge").copy()
splits["centroid_x"] = splits["centroid_x"] / 4
splits["centroid_y"] = splits["centroid_y"] / 4
splits["centroid_z"] = splits["centroid_z"] / 40

point_mapper = statebuilder.PointMapper(
    point_column="centroid", description_column="operation_id", split_positions=True
)
point_layer = statebuilder.AnnotationLayerConfig(
    name="splits", linked_segmentation_layer="seg", mapping_rules=point_mapper
)
sb = statebuilder.StateBuilder([point_layer], client=client, resolution=[1, 1, 1])
sbs.append(sb)
dfs.append(splits.reset_index())

sb = statebuilder.ChainedStateBuilder(sbs)
json_out = statebuilder.helpers.package_state(dfs, sb, client=client, return_as="json")
state_dict = json.loads(json_out)

shader = """
void main() {
    float compartment = vCustom2;
    vec4 uColor = segmentColor();
    emitRGB(vec3(0.1, 0.1, 0.3) + compartment * vec3(1, 0, 0));
}
"""
skel_rendering_kws = {
    "shader": shader,
    "mode2d": "lines_and_points",
    "mode3d": "lines",
    "lineWidth3d": 2.5,
}

state_dict["layers"][2]["skeletonRendering"] = skel_rendering_kws


statebuilder.StateBuilder(base_state=state_dict, client=client).render_state(
    return_as="html"
)

# %%

# %%

import os
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


# %%
model_path = Path("data/models/local_compartment_classifier_bd_boxes.skops")

model = load(model_path)

# %%

root_ids = pd.read_csv("sandbox/l2_ml/clean_cells.csv", header=None).values
root_ids = np.unique(root_ids)

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

box_width = 15_000
neighborhood_hops = 5
verbose = False
all_features = []
all_labels = []
# root_ids = extended_df["pt_root_id"].sample(100, random_state=88)

recompute = False

for root_id in tqdm(root_ids[:]):
    if (
        os.path.exists(out_path / f"local_features_root_id={root_id}.csv")
        and not recompute
    ):
        pass
    else:
        try:
            # get the change log
            change_log = get_change_log(root_id, client)
            print("Number of edits: ", len(change_log))

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
            split_wrangler.set_query_boxes_from_points(
                points_by_root, box_width=box_width
            )
            split_wrangler.query_level2_shape_features()
            split_wrangler.prune_query_to_boxes()
            split_wrangler.query_level2_synapse_features(method="update")
            split_wrangler.register_model(model, "bd_boxes")
            split_wrangler.query_level2_edges(warn_on_missing=False)
            split_wrangler.query_level2_networks()
            split_wrangler.query_level2_graph_features()
            split_wrangler.aggregate_features_by_neighborhood(
                aggregations=["mean", "std"], neighborhood_hops=neighborhood_hops
            )

            # save features
            split_features = split_wrangler.features_
            split_features = split_features.dropna()
            split_features["current_root_id"] = root_id
            split_features = split_features.reset_index().set_index(
                ["current_root_id", "object_id", "level2_id"]
            )

            split_labels = pd.Series(
                "split", index=split_features.index, name="label"
            ).to_frame()
            _, min_dists_to_edit = pairwise_distances_argmin_min(
                split_features[["rep_coord_x", "rep_coord_y", "rep_coord_z"]],
                np.stack(points_by_root.values),
            )
            split_labels["min_dist_to_edit"] = min_dists_to_edit
            # split_labels["current_root_id"] = root_id
            split_labels = split_labels.reset_index().set_index(
                ["current_root_id", "object_id", "level2_id"]
            )

            # now do the same for the final cleaned neuron
            nonsplit_wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=verbose)
            nonsplit_wrangler.set_objects([root_id])
            nonsplit_wrangler.query_level2_shape_features()
            nonsplit_wrangler.query_level2_synapse_features(method="existing")
            nonsplit_wrangler.register_model(model, "bd_boxes")
            nonsplit_wrangler.query_level2_edges(warn_on_missing=False)
            nonsplit_wrangler.query_level2_networks()
            nonsplit_wrangler.query_level2_graph_features()
            nonsplit_wrangler.aggregate_features_by_neighborhood(
                aggregations=["mean", "std"], neighborhood_hops=neighborhood_hops
            )
            nonsplit_features = nonsplit_wrangler.features_
            nonsplit_features = nonsplit_features.dropna()
            nonsplit_features["current_root_id"] = root_id
            nonsplit_features = nonsplit_features.reset_index().set_index(
                ["current_root_id", "object_id", "level2_id"]
            )
            all_features.append(nonsplit_features)
            nonsplit_labels = pd.Series(
                "no split", index=nonsplit_features.index, name="label"
            ).to_frame()
            _, min_dists_to_edit = pairwise_distances_argmin_min(
                nonsplit_features[["rep_coord_x", "rep_coord_y", "rep_coord_z"]],
                np.stack(points_by_root.values),
            )
            nonsplit_labels["min_dist_to_edit"] = min_dists_to_edit
            # nonsplit_labels["root_id"] = root_id

            features = pd.concat([split_features, nonsplit_features], axis=0)
            labels = pd.concat([split_labels, nonsplit_labels], axis=0)
            features.to_csv(out_path / f"local_features_root_id={root_id}.csv")
            labels.to_csv(out_path / f"local_labels_root_id={root_id}.csv")
        except Exception as e:
            print("Error on root ID: ", root_id)
            print(e)
            print()


# %%
all_features = []
all_labels = []

new_root_ids = []
for temp_root_id in tqdm(root_ids[:]):
    if temp_root_id == root_id:
        continue
    if os.path.exists(out_path / f"local_features_root_id={temp_root_id}.csv"):
        features = pd.read_csv(
            out_path / f"local_features_root_id={temp_root_id}.csv", index_col=[0, 1, 2]
        )
        labels = pd.read_csv(
            out_path / f"local_labels_root_id={temp_root_id}.csv", index_col=[0, 1, 2]
        )
        all_features.append(features)
        all_labels.append(labels)
        new_root_ids.append(temp_root_id)


root_ids = new_root_ids
features = pd.concat(all_features, axis=0)
labels = pd.concat(all_labels, axis=0)

print("Number of cells:", len(root_ids))

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

drop_cols = []
drop_cols += [col for col in features.columns if "synapse_" in col]
drop_cols += ["rep_coord_x", "rep_coord_y", "rep_coord_z"]
drop_cols += [
    "rep_coord_x_neighbor_mean",
    "rep_coord_y_neighbor_mean",
    "rep_coord_z_neighbor_mean",
]

model_type = "rf"
n_splits = 1
for i in range(n_splits):
    train_roots, test_roots = train_test_split(root_ids, random_state=i)
    # sub_labels = labels.loc[train_roots]
    sub_labels = labels.query(
        '((label == "no split") & (min_dist_to_edit > 15_000)) | (min_dist_to_edit < 2000)'
    )
    train_labels = sub_labels.loc[train_roots, "label"]
    test_labels = sub_labels.loc[test_roots, "label"]

    train_features = features.loc[train_labels.index].drop(columns=drop_cols).dropna()
    test_features = features.loc[test_labels.index].drop(columns=drop_cols).dropna()
    train_labels = train_labels.loc[train_features.index]
    test_labels = test_labels.loc[test_features.index]

    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=500, max_depth=12, n_jobs=-1)
        model.fit(train_features, train_labels)
    elif model_type == "lda":
        model = LinearDiscriminantAnalysis()
        transform = QuantileTransformer(output_distribution="normal")
        model = Pipeline([("transform", transform), ("lda", model)])
        model.fit(train_features, train_labels)

    y_pred = model.predict(train_features)
    print("Train:")
    print(classification_report(train_labels, y_pred))
    print()

    y_pred = model.predict(test_features)
    print("Test:")
    print(classification_report(test_labels, y_pred))
    print()

# %%
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(8, 12), sharey=True)
if model_type == "rf":
    sns.barplot(x=model.feature_importances_, y=train_features.columns, ax=ax)
elif model_type == "lda":
    sns.barplot(x=model.steps[1][1].coef_[0], y=train_features.columns, ax=ax)
ax.set_ylabel("Feature")


# %%

from networkframe import NetworkFrame

pv.set_jupyter_backend("client")

split_wrangler.query_level2_networks()

# make a pyvista mesh for each bounding box
# for point in points_by_root.values:
boxes = split_wrangler.query_boxes_

plotter = pv.Plotter(shape=(1, 2))

plotter.subplot(0, 0)
for box in boxes:
    box_mesh = pv.Box(
        [box[0, 0], box[1, 0], box[0, 1], box[1, 1], box[0, 2], box[1, 2]]
    )
    plotter.add_mesh(box_mesh, color="red", opacity=0.5, style="wireframe")


def _edges_to_lines(nodes, edges):
    iloc_map = dict(zip(nodes.index.values, range(len(nodes))))
    iloc_edges = edges[["source", "target"]].applymap(lambda x: iloc_map[x])

    lines = np.empty((len(edges), 3), dtype=int)
    lines[:, 0] = 2
    lines[:, 1:3] = iloc_edges[["source", "target"]].values

    return lines


def to_skeleton_polydata(
    nf: NetworkFrame,
    spatial_columns=["rep_coord_x", "rep_coord_y", "rep_coord_z"],
    label=None,
    draw_lines=True,
) -> pv.PolyData:
    nodes = nf.nodes
    edges = nf.edges

    points = nodes[spatial_columns].values.astype(float)

    if draw_lines:
        lines = _edges_to_lines(nodes, edges)
    else:
        lines = None

    skeleton = pv.PolyData(points, lines=lines)

    if label is not None:
        skeleton[label] = nodes[label].values

    return skeleton


feature_name = "prediction"

for object_id, nf in split_wrangler.object_level2_networks_.items():
    # nf.nodes[feature_name] = (
    #     split_labels.droplevel(["current_root_id", "object_id"])["min_dist_to_edit"]
    #     .groupby("level2_id")
    #     .min()
    # )
    nf.nodes["prediction"] = 0.0

    nodes_to_predict = (
        nf.nodes.dropna().drop(columns=drop_cols).drop(columns=["prediction"])
    )
    if not nodes_to_predict.empty:
        predictions = model.predict_proba(nodes_to_predict)[:, 1]
        nf.nodes.loc[nodes_to_predict.index, "prediction"] = predictions
    polydata = to_skeleton_polydata(nf, label=feature_name)
    plotter.add_mesh(
        polydata,
        line_width=2,
        scalars=feature_name,
        # cmap="Reds",
    )
    plotter.add_mesh(
        polydata,
        point_size=7,
        style="points",
        scalars=feature_name,
        # cmap="Reds",
    )


nonsplit_wrangler.query_level2_networks()
nf = nonsplit_wrangler.object_level2_networks_[root_id]

nf.nodes["prediction"] = 0.0

nodes_to_predict = (
    nf.nodes.dropna().drop(columns=drop_cols).drop(columns=["prediction"])
)
if not nodes_to_predict.empty:
    predictions = model.predict_proba(nodes_to_predict)[:, 1]
    nf.nodes.loc[nodes_to_predict.index, "prediction"] = predictions


polydata = to_skeleton_polydata(nf, label=feature_name)
plotter.add_mesh(polydata, color="black", line_width=1)

plotter.subplot(0, 1)
for box in boxes:
    box_mesh = pv.Box(
        [box[0, 0], box[1, 0], box[0, 1], box[1, 1], box[0, 2], box[1, 2]]
    )
    plotter.add_mesh(box_mesh, color="red", opacity=0.5, style="wireframe")

plotter.add_mesh(
    polydata,
    # scalars=feature_name,
    line_width=2,
    color="black",
    # cmap="Reds",
)
plotter.add_mesh(
    polydata,
    point_size=7,
    style="points",
    scalars=feature_name,
    # cmap="Reds",
)

plotter.link_views()
plotter.show()

# 864691135163673901 is a good one to stare at
# %%
split_labels = labels.query("(label == 'split') & (min_dist_to_edit < 2000)")
non_split_labels = labels.query("(label == 'no split')")

# %%

X = pd.concat(
    [features.loc[split_labels.index], features.loc[non_split_labels.index]], axis=0
).dropna()
# X = X.drop(columns=[col for col in X.columns if "rep_coord_" in col])
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

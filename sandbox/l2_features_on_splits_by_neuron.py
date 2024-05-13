# %%

from pathlib import Path

import caveclient as cc
import matplotlib.pyplot as plt
import pandas as pd
import pyvista as pv
import seaborn as sns
from scipy.sparse.csgraph import dijkstra
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import QuantileTransformer
from skops.io import load
from troglobyte.features import CAVEWrangler

from pkg.neuronframe import load_neuronframe
from pkg.plot import set_up_camera
from pkg.utils import load_manifest

client = cc.CAVEclient("minnie65_phase3_v1")

model_path = Path("data/models/local_compartment_classifier_ej_skeletons.skops")

model = load(model_path)

# %%

manifest = load_manifest()
manifest["is_current"] = client.chunkedgraph.is_latest_roots(manifest.index.to_list())
manifest = manifest.query("is_current")

root_ids = manifest.index[:10]


def generate_features_for_root(
    root_id: int, client: cc.CAVEclient, return_networkframes=False
):
    # load and edit neuron
    nf = load_neuronframe(root_id, client)
    edited_nf = nf.set_edits(nf.metaedits.query("has_merge").index, prefix="meta")
    edited_nf.select_nucleus_component(inplace=True)
    edited_nf.remove_unused_nodes(inplace=True)

    final_nf = nf.set_edits(nf.edits.index)
    final_nf.select_nucleus_component(inplace=True)
    final_nf.remove_unused_nodes(inplace=True)

    # generate path lengths to closest edits
    edited_nf.apply_edge_lengths(inplace=True)
    edited_nf.to_sparse_adjacency(weight_col="length")

    splits = edited_nf.edits.query("~is_merge")
    edit_index = edited_nf.nodes.query(
        "operation_added.isin(@splits.index) | operation_removed.isin(@splits.index)"
    ).index
    edit_index = edit_index.intersection(edited_nf.nodes.index)
    edit_iloc = edited_nf.nodes.index.get_indexer_for(edit_index)
    sparse_adj = edited_nf.to_sparse_adjacency(weight_col="length")
    sparse_adj = (sparse_adj + sparse_adj.T) / 2

    path_lengths = dijkstra(
        sparse_adj, indices=edit_iloc, min_only=True, directed=False
    )

    path_lengths = pd.Series(
        path_lengths, index=edited_nf.nodes.index, name="path_length_to_edit"
    )
    # path_lengths = path_lengths.reset_index().set_index(["object_id", "level2_id"])['path_length_to_edit']
    # all_path_lengths.append(path_lengths)

    # generate training data
    close_nodes = path_lengths[path_lengths < 2_000].index
    somewhat_close_nodes = path_lengths[path_lengths < 10_000].index
    far_nodes = path_lengths[path_lengths >= 25_000].index
    far_final_nodes = far_nodes.intersection(final_nf.nodes.index)

    # relevant_index = somewhat_close_nodes.union(final_nf.nodes.index)
    # relevant_index = relevant_index.intersection(edited_nf.nodes.index)
    relevant_index = edited_nf.nodes.index

    relevant_nf = edited_nf.query_nodes(
        "index.isin(@relevant_index)", local_dict=locals()
    ).deepcopy()

    relevant_nf.nodes["dist_to_edit_class"] = "far"
    relevant_nf.nodes.loc[close_nodes, "dist_to_edit_class"] = "close"
    relevant_nf.nodes["scalars"] = relevant_nf.nodes["dist_to_edit_class"].map(
        {"close": 1.0, "far": 0.0}
    )

    level2_info = pd.DataFrame(index=relevant_index)
    level2_info.index.name = "level2_id"
    level2_info["object_id"] = root_id
    level2_info["dist_to_edit"] = path_lengths
    level2_info["dist_to_edit_class"] = relevant_nf.nodes["dist_to_edit_class"]

    # download features for the nodes
    wrangler = CAVEWrangler(client=client, n_jobs=None, verbose=False)
    wrangler.set_level2_ids(relevant_index)
    wrangler.query_current_object_ids()  # get current ids as a proxy for getting synapses
    wrangler.query_level2_shape_features()

    wrangler.query_level2_synapse_features(method="existing")

    wrangler.register_model(model, "l2class_ej_skeletons")

    # do an aggregation using the graph
    # doing this manually here since future versions could operate on the
    # non-cave graph
    features = wrangler.features_
    features = features.drop(columns=["rep_coord_x", "rep_coord_y", "rep_coord_z"])
    features = features.droplevel("object_id")

    # this can have Nan and that's chill
    features = features.reindex(relevant_nf.nodes.index)

    # # temporarily add in the features to nodes for the aggregation
    old_nodes = relevant_nf.nodes
    relevant_nf.nodes = features
    neighborhood_features = relevant_nf.k_hop_aggregation(
        k=5,
        aggregations=["mean", "std"],
        engine="scipy",
    )
    wrangler.neighborhood_features = neighborhood_features.loc[
        wrangler.features_.index.get_level_values("level2_id")
    ]
    neighborhood_features.index = wrangler.features_.index
    wrangler.neighborhood_features_ = neighborhood_features

    # stack the model and predict classes
    wrangler.stack_model_predict_proba("l2class_ej_skeletons")

    # do another round after stacking the model
    features = wrangler.predict_features_
    features = features.droplevel("object_id")
    features = features.reindex(relevant_nf.nodes.index)

    # # temporarily add in the features to nodes
    # old_nodes = relevant_nf.nodes
    relevant_nf.nodes = features
    pred_neighborhood_features = relevant_nf.k_hop_aggregation(
        k=5,
        aggregations=["mean", "std"],
        engine="scipy",
    )

    # replace the nodes to the original data
    relevant_nf.nodes = old_nodes

    pred_neighborhood_features = pred_neighborhood_features.loc[
        wrangler.predict_features_.index.get_level_values("level2_id")
    ]
    pred_neighborhood_features.index = wrangler.predict_features_.index
    wrangler.predict_features_ = wrangler.predict_features_.join(
        pred_neighborhood_features
    )

    features = wrangler.features_.droplevel("object_id").reset_index()
    features["object_id"] = root_id
    features = features.set_index(["object_id", "level2_id"])

    level2_info = level2_info.reset_index().set_index(["object_id", "level2_id"])
    level2_info = level2_info.loc[features.index]

    if not return_networkframes:
        return features, level2_info
    else:
        return features, level2_info, relevant_nf, edited_nf, final_nf


from joblib import Parallel, delayed

results = Parallel(n_jobs=1, verbose=10)(
    delayed(generate_features_for_root)(root_id, client) for root_id in root_ids
)

# %%

features = pd.concat([x[0] for x in results], axis=0)
level2_info = pd.concat([x[1] for x in results], axis=0)


# %%
features_close = features.loc[level2_info.query("dist_to_edit_class == 'close'").index]
features_far = features.loc[level2_info.query("dist_to_edit_class == 'far'").index]

features_close = features_close.dropna()
features_far = features_far.dropna()

# sub_features_far = features_far.sample(n=1 * len(features_close))
sub_features_far = features_far

# %%

X = pd.concat([features_close, sub_features_far], axis=0)
# X["object_id"] = X.index.map(level2_info["object_id"])
X = X.drop(columns=["rep_coord_x", "rep_coord_y", "rep_coord_z"])
# X = X.drop(columns=[x for x in X.columns if "std" in x])
y = pd.Series(
    ["split"] * len(features_close) + ["no split"] * len(sub_features_far),
    index=X.index,
)
# X = X.reset_index().set_index(["object_id", "level2_id"])
y.index = X.index

# %%
from sklearn.model_selection import train_test_split

train_root_ids, test_root_ids = train_test_split(root_ids, test_size=0.2)

X_train = X.loc[train_root_ids]
y_train = y.loc[train_root_ids]

X_test = X.loc[test_root_ids]
y_test = y.loc[test_root_ids]

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

root_id = test_root_ids[3]

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

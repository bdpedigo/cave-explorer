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
from tqdm.auto import tqdm
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

nfs_by_root_id = {}

l2_ids = []
root_ids = manifest.index[:10]
for root_id in root_ids:
    # load and edit neuron
    nf = load_neuronframe(root_id, client)
    edited_nf = nf.set_edits(nf.edits.index)
    edited_nf.select_nucleus_component(inplace=True)
    edited_nf.remove_unused_nodes(inplace=True)

    merges = edited_nf.edits.query("is_merge")
    # generate path lengths to closest edits
    edited_nf.apply_edge_lengths(inplace=True)
    nfs_by_root_id[root_id] = edited_nf
    l2_ids.extend(edited_nf.nodes.index)

# download features for the nodes
wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=True)
wrangler.set_level2_ids(l2_ids)
wrangler.query_current_object_ids()  # get current ids as a proxy for getting synapses
wrangler.query_level2_shape_features()
wrangler.query_level2_synapse_features(method="existing")


# %%

all_path_lengths = []
all_neighborhood_features = []
for root_id, edited_nf in tqdm(nfs_by_root_id.items()):
    merges = edited_nf.edits.query("is_merge")
    edit_index = edited_nf.nodes.query("operation_added.isin(@merges.index)").index
    edit_index = edit_index.intersection(edited_nf.nodes.index)
    edit_iloc = edited_nf.nodes.index.get_indexer_for(edit_index)
    sparse_adj = edited_nf.to_sparse_adjacency(weight_col="length")
    sparse_adj = (sparse_adj + sparse_adj.T) / 2

    path_lengths = dijkstra(
        sparse_adj, indices=edit_iloc, min_only=True, directed=False
    )

    path_lengths = pd.Series(
        path_lengths, index=edited_nf.nodes.index, name="path_length_to_edit"
    ).to_frame()
    path_lengths.index.name = "level2_id"
    path_lengths["object_id"] = root_id
    path_lengths = path_lengths.reset_index().set_index(["object_id", "level2_id"])
    all_path_lengths.append(path_lengths)

    # do an aggregation using the graph
    # doing this manually here since future versions could operate on the
    # non-cave graph
    features = wrangler.features_.loc[root_id]
    features = features.drop(columns=["rep_coord_x", "rep_coord_y", "rep_coord_z"])
    # temporarily add in the features to nodes
    old_nodes = edited_nf.nodes
    edited_nf.nodes = features
    neighborhood_features = edited_nf.k_hop_aggregation(
        k=5, aggregations=["mean", "std"], engine="scipy"
    )
    neighborhood_features.reset_index(inplace=True)
    neighborhood_features["object_id"] = root_id
    neighborhood_features.set_index(["object_id", "level2_id"], inplace=True)
    all_neighborhood_features.append(neighborhood_features)
    # joined_features = features.join(neighborhood_features, how="left")
    # joined_features.reset_index(inplace=True)
    # joined_features["root_id"] = root_id
    # joined_features.set_index(["root_id", "level2_id"], inplace=True)

    # replace the nodes to the original data
    edited_nf.nodes = old_nodes

# %%
all_neighborhood_features = pd.concat(all_neighborhood_features, axis=0)
wrangler.neighborhood_features_ = all_neighborhood_features

# %%
path_lengths = pd.concat(all_path_lengths, axis=0)["path_length_to_edit"]

# %%
wrangler.register_model(model, "l2class_ej_skeletons")
wrangler.stack_model_predict_proba("l2class_ej_skeletons")

# %%

features = wrangler.features_

# generate training data
close_nodes = path_lengths[path_lengths < 2_000].index
far_nodes = path_lengths[path_lengths >= 25_000].index

features_close = features.loc[close_nodes]
features_far = features.loc[far_nodes]

features_close = features_close.dropna()
features_far = features_far.dropna()

sub_features_far = features_far.sample(n=1 * len(features_close))

# %%

X = pd.concat([features_close, sub_features_far], axis=0)
X = X.drop(columns=["rep_coord_x", "rep_coord_y", "rep_coord_z"])
# X = X.drop(columns=[x for x in X.columns if "std" in x])
y = pd.Series(["split"] * len(features_close) + ["no split"] * len(sub_features_far), index=X.index)

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
lda = LinearDiscriminantAnalysis()
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
rf = RandomForestClassifier(n_estimators=600, max_depth=4)
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
full_X = features.loc[root_id, X_train.columns].dropna()

full_y_pred = lda.predict(transformer.transform(full_X))
full_X_lda = lda.transform(transformer.transform(full_X))

# %%
edited_nf.nodes["lda_pred_val"] = 0.0
edited_nf.nodes.loc[
    edited_nf.nodes.index.intersection(full_X.index), "lda_pred_val"
] = full_X_lda

# %%
pv.set_jupyter_backend("client")
plotter = pv.Plotter()
set_up_camera(plotter, edited_nf)
plotter.add_mesh(
    edited_nf.to_skeleton_polydata(label="lda_pred_val"),
    line_width=1.0,
    scalars="lda_pred_val",
    cmap="coolwarm",
)
plotter.add_mesh(
    edited_nf.to_merge_polydata(),
    color="black",
)
plotter.show()

# %%


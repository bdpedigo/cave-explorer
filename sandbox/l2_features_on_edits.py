# %%

from pathlib import Path

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from scipy.sparse.csgraph import dijkstra
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import QuantileTransformer
from troglobyte.features import CAVEWrangler

from pkg.neuronframe import load_neuronframe
from pkg.plot import set_up_camera
from pkg.utils import load_manifest

client = cc.CAVEclient("minnie65_phase3_v1")

model_path = Path("data/models/local_compartment_classifier_ej_skeletons.skops")

from skops.io import load

model = load(model_path)

# %%

manifest = load_manifest()

root_id = manifest.index[0]

nf = load_neuronframe(root_id, client)


# %%

edited_nf = nf.set_edits(nf.metaedits.query("has_merge").index, prefix="meta")
edited_nf.select_nucleus_component(inplace=True)
edited_nf.remove_unused_nodes(inplace=True)
edited_nf

# %%

# peek at the neuron

pv.set_jupyter_backend("client")
plotter = pv.Plotter()
set_up_camera(plotter, edited_nf)
plotter.add_mesh(edited_nf.to_skeleton_polydata(), color="black", line_width=0.1)
plotter.show()

# %%

# generate path lengths to closest edits

edited_nf.apply_edge_lengths(inplace=True)
edit_index = edited_nf.nodes.query("operation_removed != -1").index
edit_iloc = edited_nf.nodes.index.get_indexer_for(edit_index)

sparse_adj = edited_nf.to_sparse_adjacency(weight_col="length")
path_length_to_edits = dijkstra(
    sparse_adj, indices=edit_iloc, min_only=True, directed=False
)
path_length_to_edits = pd.Series(path_length_to_edits, index=edited_nf.nodes.index)


# %%

# download features for the nodes
wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=False)
wrangler.set_level2_ids(edited_nf.nodes.index)
wrangler.query_current_object_ids()  # this is necessary as a proxy for getting synapses

wrangler.query_level2_shape_features()


# %%
wrangler.query_level2_synapse_features(method="existing")


# %%
# wrangler.register_model(model, model_name="local_compartment_ej")
# wrangler.stack_model_predict("local_compartment_ej")
# %%

# do an aggregation using the graph

features = wrangler.features_.droplevel("object_id")
features = features.drop(columns=["rep_coord_x", "rep_coord_y", "rep_coord_z"])

old_nodes = edited_nf.nodes
edited_nf.nodes = features

neighborhood_features = edited_nf.k_hop_aggregation(
    k=5, aggregations=["mean", "std"], verbose=True
)
joined_features = features.join(neighborhood_features, how="left")

edited_nf.nodes = old_nodes



# %%

y_pred = model.predict(joined_features)
posteriors = model.predict_proba(joined_features)
posteriors = pd.DataFrame(
    posteriors, index=joined_features.index, columns=model.classes_
)

# %%
axon_post = posteriors["axon"] / (posteriors.sum(axis=1))


edited_nf.nodes["axon_posterior"] = 0.0
edited_nf.nodes.loc[joined_features.index, "axon_posterior"] = axon_post

#%%
plotter = pv.Plotter()
set_up_camera(plotter, edited_nf)
plotter.add_mesh(
    edited_nf.to_skeleton_polydata(label="axon_posterior"),
    line_width=0.1,
    scalars="axon_posterior",
    cmap="coolwarm",
)
plotter.show()

# %%

# generate training data
close_nodes = path_length_to_edits[path_length_to_edits < 2_000].index
far_nodes = path_length_to_edits[path_length_to_edits >= 50_000].index

features_close = joined_features.loc[close_nodes]
features_far = joined_features.loc[far_nodes]

features_close = features_close.dropna()
features_far = features_far.dropna()

sub_features_far = features_far.sample(n=1 * len(features_close))

# %%

X = pd.concat([features_close, sub_features_far], axis=0)
# X = X.drop(columns=[x for x in X.columns if "std" in x])
y = pd.Series(["split"] * len(features_close) + ["no split"] * len(sub_features_far))

# %%
# try LDA
transformer = QuantileTransformer(output_distribution="normal")
X_transformed = transformer.fit_transform(X)
lda = LinearDiscriminantAnalysis()
lda.fit(X_transformed, y)

X_lda = lda.transform(X_transformed)


fig, ax = plt.subplots(1, 1, figsize=(10, 10))

sns.histplot(x=X_lda[:, 0], hue=y, ax=ax)

y_pred = lda.predict(X_transformed)

print(classification_report(y, y_pred))

# %%
# try random forest
rf = RandomForestClassifier(n_estimators=100, max_depth=3)
rf.fit(X, y)

y_pred = rf.predict(X)

print(classification_report(y, y_pred))

# %%

# look at the feature importances

fig, axs = plt.subplots(1, 2, figsize=(8, 10), sharey=True)
sns.barplot(x=lda.coef_[0], y=X.columns, ax=axs[0])
sns.barplot(x=rf.feature_importances_, y=X.columns, ax=axs[1])

# %%
full_X = joined_features.dropna()

full_y_pred = lda.predict(transformer.transform(full_X))
full_X_lda = lda.transform(transformer.transform(full_X))

# %%
edited_nf.nodes["lda_pred_val"] = 0.0
edited_nf.nodes.loc[full_X.index, "lda_pred_val"] = full_X_lda

# %%
plotter = pv.Plotter()
set_up_camera(plotter, edited_nf)
plotter.add_mesh(
    edited_nf.to_skeleton_polydata(label="lda_pred_val"),
    line_width=0.1,
    scalars="lda_pred_val",
    cmap="coolwarm",
)
plotter.add_mesh(
    edited_nf.to_split_polydata(),
    color="black",
)
plotter.show()


# %%

# look at what the PCA features look like


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

# %%


import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

root_ids = pd.read_csv("sandbox/l2_ml/clean_cells.csv", header=None).values
root_ids = np.unique(root_ids)

out_path = Path("results/outs/split_features")

all_features = []
all_labels = []

new_root_ids = []
for root_id in tqdm(root_ids[:20]):
    if os.path.exists(out_path / f"local_features_root_id={root_id}.csv"):
        features = pd.read_csv(
            out_path / f"local_features_root_id={root_id}.csv", index_col=[0, 1, 2]
        )
        labels = pd.read_csv(
            out_path / f"local_labels_root_id={root_id}.csv", index_col=[0, 1, 2]
        )
        all_features.append(features)
        all_labels.append(labels)
        new_root_ids.append(root_id)


root_ids = new_root_ids
features = pd.concat(all_features, axis=0)
labels = pd.concat(all_labels, axis=0)

print("Number of cells:", len(root_ids))
# %%


# transform = QuantileTransformer(output_distribution="normal")
# lda = LinearDiscriminantAnalysis()

# X = features.dropna()
# X_transformed = transform.fit_transform(X)
# y = labels.loc[X.index, "label"].values

# lda.fit(X_transformed, y)

# # %%

# y_pred = lda.predict(X_transformed)
# print(classification_report(y, y_pred))

# %%

from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

n_splits = 1
for i in range(n_splits):
    train_roots, test_roots = train_test_split(root_ids, random_state=i)
    # sub_labels = labels.loc[train_roots]
    sub_labels = labels.query('label == "no split" | (min_dist_to_edit < 2000)')
    train_labels = sub_labels.loc[train_roots, "label"]
    test_labels = sub_labels.loc[test_roots, "label"]

    train_features = features.loc[train_labels.index]
    test_features = features.loc[test_labels.index]

    # transform = QuantileTransformer(output_distribution="normal")
    # lda = LinearDiscriminantAnalysis()
    # model = Pipeline([("transform", transform), ("lda", lda)])
    model = RandomForestClassifier(n_estimators=500, n_jobs=-1)

    model.fit(train_features, train_labels)

    y_pred = model.predict(train_features)
    print(classification_report(train_labels, y_pred))

    y_pred = model.predict(test_features)
    print(classification_report(test_labels, y_pred))

# %%
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(8, 12), sharey=True)
# sns.barplot(x=lda.coef_[0], y=X.columns, ax=axs[0])
sns.barplot(x=model.feature_importances_, y=train_features.columns, ax=ax)
ax.set_ylabel("Feature")

# %%

root_id = 864691136134777739

import time

import pyvista as pv
from caveclient import CAVEclient
from skops.io import load
from troglobyte.features import CAVEWrangler

from pkg.neuronframe import NeuronFrame
from pkg.utils import get_level2_nodes_edges

client = CAVEclient("minnie65_phase3_v1")

nodes, edges = get_level2_nodes_edges(root_id, client, positions=True)

# changelog = get_detailed_change_log(root_id, client)

# splits = changelog.query("~is_merge")

final_nf = NeuronFrame(nodes, edges, neuron_id=root_id)

model_path = Path("data/models/local_compartment_classifier_bd_boxes.skops")

currtime = time.time()

wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=2)
wrangler.set_objects([root_id])
wrangler.query_level2_shape_features()
wrangler.query_level2_synapse_features(method="existing")
wrangler.register_model(load(model_path), "bd_boxes")
wrangler.query_level2_networks()
wrangler.query_level2_graph_features()
wrangler.aggregate_features_by_neighborhood(
    aggregations=["mean", "std"], neighborhood_hops=5
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

final_nf.nodes["split_predict_proba"] = 0.0
node_features = wrangler.features_.loc[root_id].dropna()
node_features = node_features.loc[
    node_features.index.intersection(final_nf.nodes.index)
]
final_nf.nodes.loc[node_features.index, "split_predict_proba"] = model.predict_proba(
    node_features
)[:, 1]
pv.set_jupyter_backend("client")

# %%
plotter = pv.Plotter()
skel_poly = final_nf.to_skeleton_polydata(label="split_predict_proba")
plotter.add_mesh(
    skel_poly,
    scalars="split_predict_proba",
    cmap="coolwarm",
    render_lines_as_tubes=True,
    line_width=3,
)
plotter.add_mesh(
    skel_poly,
    scalars="split_predict_proba",
    cmap="coolwarm",
    point_size=7,
    style="points",
    render_points_as_spheres=True,
)

plotter.show()
# %%

final_nf.nodes["split_predict_proba"] = 0.0
node_features = test_features.loc[root_id].droplevel("object_id")
node_features = node_features.loc[
    node_features.index.intersection(final_nf.nodes.index)
]
final_nf.nodes.loc[node_features.index, "split_predict_proba"] = model.predict_proba(
    node_features
)[:, 1]


# %%


pv.set_jupyter_backend("client")

plotter = pv.Plotter()
skel_poly = final_nf.to_skeleton_polydata(label="split_predict_proba")
plotter.add_mesh(
    skel_poly,
    scalars="split_predict_proba",
    cmap="coolwarm",
)

plotter.show()
# %%

root_id = splits.sort_index()["before_root_ids"].iloc[0][0]
nodes, edges = get_level2_nodes_edges(root_id, client, positions=True)
old_nf = NeuronFrame(nodes, edges, neuron_id=root_id, edits=splits)

# %%

plotter = pv.Plotter()

final_skel_poly = final_nf.to_skeleton_polydata()
plotter.add_mesh(final_skel_poly, color="black")

old_skel_poly = old_nf.to_skeleton_polydata()
plotter.add_mesh(old_skel_poly, color="lightblue")

plotter.show()

# %%


# %%


old_nf.nodes["split_predict_proba"] = 0.0
node_features = wrangler.features_.loc[root_id].dropna()
node_features = node_features.loc[
    node_features.index.intersection(final_nf.nodes.index)
]
old_nf.nodes.loc[node_features.index, "split_predict_proba"] = model.predict_proba(
    node_features
)[:, 1]

plotter = pv.Plotter()

old_skel_poly = old_nf.to_skeleton_polydata(label="split_predict_proba")
plotter.add_mesh(
    old_skel_poly, scalars="split_predict_proba", cmap="coolwarm", point_size=10
)

plotter.show()

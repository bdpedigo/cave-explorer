# %%
import time

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cloudfiles import CloudFiles
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from pkg.plot import set_context

client = cc.CAVEclient("minnie65_public_v661")

DIRECTORY = "https://storage.googleapis.com/allen-minnie-phase3/minniephase3-emily-pcg-skeletons/minnie_all/v661/meshworks"

cf = CloudFiles(DIRECTORY)

proofreading_df = client.materialize.query_table(
    "proofreading_status_public_release", materialization_version=661
)

nucs = client.materialize.query_table(
    "nucleus_detection_v0", materialization_version=661
)
unique_roots = proofreading_df["pt_root_id"].unique()
nucs = nucs.query("pt_root_id.isin(@unique_roots)")

proofreading_df["target_id"] = (
    proofreading_df["pt_root_id"]
    .map(nucs.set_index("pt_root_id")["id"])
    .astype("Int64")
)

extended_df = proofreading_df.query(
    "status_axon == 'extended' & status_dendrite == 'extended'"
)

# %%
from pkg.features import L2FeatureExtractor

root_id = extended_df["valid_id"].iloc[0]
root_ids = extended_df["valid_id"].iloc[:8]

feature_extractor = L2FeatureExtractor(client, verbose=2, n_jobs=8)
node_features = feature_extractor.get_features(root_ids)

node_features.head()

# %%


currtime = time.time()

files = list(cf)
files = [file for file in files if file.endswith(".h5")]
file_manifest = pd.DataFrame(files, columns=["filename"])
file_manifest["root_id"] = (
    file_manifest["filename"].apply(lambda x: x.split("_")[0]).astype(int)
)
file_manifest["target_id"] = (
    file_manifest["filename"].apply(lambda x: x.split("_")[1].split(".")[0]).astype(int)
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

file_manifest

# %%
file_manifest["is_proofread"] = file_manifest["root_id"].isin(
    proofreading_df["pt_root_id"]
)

file_manifest["is_extended"] = file_manifest["root_id"].isin(extended_df["pt_root_id"])

# %%


def neighborhoods_to_features(neighborhoods):
    rows = []
    for node, neighborhood in neighborhoods.items():
        self_features = (
            neighborhood.nodes.loc[node].drop(["root_id", "compartment"]).to_frame().T
        )
        neighbor_features = neighborhood.nodes.drop(node).drop(
            ["root_id", "compartment"], axis=1
        )
        agg_neighbor_features = neighbor_features.mean(skipna=True).to_frame().T
        agg_neighbor_features.index = [node]
        features = self_features.join(agg_neighbor_features, rsuffix="_neighbor_agg")
        rows.append(features)

    return pd.concat(rows)


n_cells = 50
all_features = {}
all_labels = {}
nfs = {}
k_hop = 10
for _, row in tqdm(
    file_manifest.query("is_extended").iloc[:n_cells].iterrows(), total=n_cells
):
    filename = row["filename"]
    root_id = row["root_id"]
    neuron = load_mw(DIRECTORY, filename)
    neuron.reset_mask()

    nf = extract_featurized_networkframe(neuron, root_id)
    nfs[root_id] = nf

    axon_nf = nf.query_nodes("compartment=='axon'")
    dendrite_nf = nf.query_nodes("compartment=='dendrite'")

    axon_k_hop_neighborhoods = axon_nf.k_hop_decomposition(k=k_hop, directed=False)
    dendrite_k_hop_neighborhoods = dendrite_nf.k_hop_decomposition(
        k=k_hop, directed=False
    )
    axon_features = neighborhoods_to_features(axon_k_hop_neighborhoods)
    dendrite_features = neighborhoods_to_features(dendrite_k_hop_neighborhoods)
    index = pd.Index(
        np.concatenate(
            [axon_features.index.to_numpy(), dendrite_features.index.to_numpy()]
        )
    )
    labels = pd.Series(
        np.concatenate(
            [
                np.full(len(axon_features), "axon"),
                np.full(len(dendrite_features), "dendrite"),
            ]
        ),
        index=index,
    )
    all_features[root_id] = pd.concat([axon_features, dendrite_features])
    all_labels[root_id] = labels


# %%

neuron_index = pd.Index(list(all_features.keys()))

# %%

full_X_df = pd.concat(all_features)
full_y_df = pd.concat(all_labels)

full_X_df.to_csv("full_X_df.csv")
full_y_df.to_csv("full_y_df.csv")

# %%

set_context()


full_X_df["pca_val_ratio"] = (
    full_X_df["pca_val_unwrapped_0"] / full_X_df["pca_val_unwrapped_1"]
)

full_X_df["pca_val_ratio_neighbor_agg"] = (
    full_X_df["pca_val_unwrapped_0_neighbor_agg"]
    / full_X_df["pca_val_unwrapped_1_neighbor_agg"]
)


syn_features = ["pre_syn_count", "post_syn_count"]
pca_vec_features = [f"pca_unwrapped_{i}" for i in range(9)]
pca_val_features = [f"pca_val_unwrapped_{i}" for i in range(3)]
pca_val_features += ["pca_val_ratio"]
rep_coord_features = ["rep_coord_x", "rep_coord_y", "rep_coord_z"]
scalar_features = ["area_nm2", "max_dt_nm", "mean_dt_nm", "size_nm3"]

self_features = (
    syn_features
    + pca_vec_features
    + pca_val_features
    + rep_coord_features
    + scalar_features
).copy()

syn_features = syn_features + [f"{feature}_neighbor_agg" for feature in syn_features]
pca_vec_features = pca_vec_features + [
    f"{feature}_neighbor_agg" for feature in pca_vec_features
]
pca_val_features = pca_val_features + [
    f"{feature}_neighbor_agg" for feature in pca_val_features
]
rep_coord_features = rep_coord_features + [
    f"{feature}_neighbor_agg" for feature in rep_coord_features
]
scalar_features = scalar_features + [
    f"{feature}_neighbor_agg" for feature in scalar_features
]
neighbor_features = np.setdiff1d(full_X_df.columns, self_features)

# %%


drop_syn_features = False
drop_pca_vec_features = True
drop_pca_val_features = False
drop_rep_coord_features = True
drop_scalar_features = False

n_splits = 5
fig, axs = plt.subplots(
    n_splits, 1, figsize=(5, 12), constrained_layout=True, sharex=True
)

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
coeffs = {}
for fold, (train_neuron_ilocs, test_neuron_ilocs) in enumerate(kf.split(neuron_index)):
    train_neuron_index = neuron_index[train_neuron_ilocs]
    test_neuron_index = neuron_index[test_neuron_ilocs]

    select_X_df = full_X_df
    if drop_syn_features:
        select_X_df = select_X_df.drop(columns=syn_features)
    if drop_pca_vec_features:
        select_X_df = select_X_df.drop(columns=pca_vec_features)
    if drop_pca_val_features:
        select_X_df = select_X_df.drop(columns=pca_val_features)
    if drop_rep_coord_features:
        select_X_df = select_X_df.drop(columns=rep_coord_features)
    if drop_scalar_features:
        select_X_df = select_X_df.drop(columns=scalar_features)

    train_X_df = select_X_df.loc[train_neuron_index]
    train_X_df = train_X_df.dropna()
    train_y_df = full_y_df.loc[train_X_df.index]

    test_X_df = select_X_df.loc[test_neuron_index]
    test_X_df = test_X_df.dropna()
    test_y_df = full_y_df.loc[test_X_df.index]

    # transformer = QuantileTransformer(output_distribution="normal")
    # train_X = transformer.fit_transform(train_X_df)
    # test_X = transformer.transform(test_X_df)
    # train_X_df = pd.DataFrame(
    #     train_X, columns=train_X_df.columns, index=train_X_df.index
    # )
    # test_X_df = pd.DataFrame(test_X, columns=test_X_df.columns, index=test_X_df.index)

    lda = LinearDiscriminantAnalysis(n_components=1)
    train_X_lda = lda.fit_transform(train_X_df, train_y_df)
    test_X_lda = lda.transform(test_X_df)
    coeffs[fold] = pd.Series(lda.coef_[0], index=train_X_df.columns)
    train_y_pred = lda.predict(train_X_df)
    test_y_pred = lda.predict(test_X_df)

    print(f"Fold {fold}")
    print("Train")
    print(classification_report(train_y_df, train_y_pred))
    print("Test")
    print(classification_report(test_y_df, test_y_pred))
    print()

    plot_train_df = pd.DataFrame(train_X_lda, columns=["lda"])
    plot_train_df["compartment"] = train_y_df.values
    plot_train_df["data"] = "train"
    plot_test_df = pd.DataFrame(test_X_lda, columns=["lda"])
    plot_test_df["compartment"] = test_y_df.values
    plot_test_df["data"] = "test"

    plot_df = pd.concat([plot_train_df, plot_test_df])

    ax = axs[fold]
    sns.kdeplot(
        data=plot_df.query("data == 'train'"),
        x="lda",
        hue="compartment",
        common_norm=False,
        fill=True,
        alpha=0.1,
        linewidth=2,
        ax=ax,
        linestyle="--",
        label="train",
    )
    sns.kdeplot(
        data=plot_df.query("data == 'test'"),
        x="lda",
        hue="compartment",
        common_norm=False,
        fill=True,
        alpha=0.1,
        linewidth=2,
        ax=ax,
        linestyle="-",
        label="test",
    )

    ax.set_xlabel("LDA component")
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.spines["left"].set_visible(False)
    ax.set_title(f"Fold {fold}")
    if fold == 0:
        sns.move_legend(ax, "upper right", title="Compartment")
    else:
        ax.get_legend().remove()

# %%
coeff_df = pd.DataFrame(coeffs)
coeff_df.index.name = "feature"

coeff_df_long = coeff_df.melt(
    var_name="fold", value_name="coefficient", ignore_index=False
).reset_index(drop=False)
coeff_df_long["fold"] = coeff_df_long["fold"].astype(str)

fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharex=True)
ax = axs[0]
sns.stripplot(
    data=coeff_df_long.query("feature.isin(@self_features)"),
    x="coefficient",
    y="feature",
    color="white",
    linewidth=1.5,
    ax=ax,
)
ax.axvline(0, color="black", linestyle="-", alpha=0.2, zorder=-1)
ax.set(title="Self features")

ax = axs[1]
sns.stripplot(
    data=coeff_df_long.query("feature.isin(@neighbor_features)"),
    x="coefficient",
    y="feature",
    color="white",
    linewidth=1.5,
    ax=ax,
)
ax.axvline(0, color="black", linestyle="-", alpha=0.2, zorder=-1)
ax.set(title="Neighbor features")


# %%


drop_syn_features = True
drop_rep_coord_features = True

importances = {}
classification_results = []
for fold, (train_neuron_ilocs, test_neuron_ilocs) in enumerate(kf.split(neuron_index)):
    print(f"Fold {fold}")
    train_neuron_index = neuron_index[train_neuron_ilocs]
    test_neuron_index = neuron_index[test_neuron_ilocs]

    select_X_df = full_X_df
    if drop_syn_features:
        select_X_df = select_X_df.drop(columns=syn_features)
    if drop_pca_vec_features:
        select_X_df = select_X_df.drop(columns=pca_vec_features)
    if drop_pca_val_features:
        select_X_df = select_X_df.drop(columns=pca_val_features)
    if drop_rep_coord_features:
        select_X_df = select_X_df.drop(columns=rep_coord_features)
    if drop_scalar_features:
        select_X_df = select_X_df.drop(columns=scalar_features)

    train_X_df = select_X_df.loc[train_neuron_index]
    train_X_df = train_X_df.dropna()
    train_y_df = full_y_df.loc[train_X_df.index]

    test_X_df = select_X_df.loc[test_neuron_index]
    test_X_df = test_X_df.dropna()
    test_y_df = full_y_df.loc[test_X_df.index]

    model = RandomForestClassifier(n_estimators=100, max_depth=5)

    model.fit(train_X_df, train_y_df.values)
    train_y_pred = model.predict(train_X_df)
    test_y_pred = model.predict(test_X_df)

    importances[fold] = pd.Series(model.feature_importances_, index=train_X_df.columns)

    train_report = classification_report(train_y_df, train_y_pred, output_dict=True)
    test_report = classification_report(test_y_df, test_y_pred, output_dict=True)

    train_row = train_report["weighted avg"]
    test_row = test_report["weighted avg"]
    train_row["data"] = "train"
    test_row["data"] = "test"
    train_row["fold"] = fold
    test_row["fold"] = fold
    train_row["accuracy"] = train_report["accuracy"]
    test_row["accuracy"] = test_report["accuracy"]
    classification_results.append(train_row)
    classification_results.append(test_row)

    print("Train")
    print(train_row)
    print("Test")
    print(test_row)
    print()

# %%

classification_df = pd.DataFrame(classification_results)
classification_df = classification_df.drop(columns=["support", "fold"])
classification_df_long = classification_df.melt(
    id_vars="data", var_name="metric", value_name="score"
).reset_index(drop=True)

# %%

fig, ax = plt.subplots(1, 1, figsize=(5, 4))

sns.stripplot(data=classification_df_long, x="metric", hue="data", y="score", ax=ax)
sns.move_legend(ax, "upper right", title="Data", bbox_to_anchor=(1.3, 1))

# %%

importances_df = pd.DataFrame(importances)
importances_df_long = importances_df.melt(
    var_name="fold", value_name="importance", ignore_index=False
).reset_index(drop=False)
importances_df_long["fold"] = importances_df_long["fold"].astype(str)

fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharex=True)
ax = axs[0]
sns.stripplot(
    data=importances_df_long.query("feature.isin(@self_features)"),
    x="importance",
    y="feature",
    color="white",
    linewidth=1.5,
    ax=ax,
)
ax.axvline(0, color="black", linestyle="-", alpha=0.2, zorder=-1)
ax.set(title="Self features")

ax = axs[1]
sns.stripplot(
    data=importances_df_long.query("feature.isin(@neighbor_features)"),
    x="importance",
    y="feature",
    color="white",
    linewidth=1.5,
    ax=ax,
)
ax.axvline(0, color="black", linestyle="-", alpha=0.2, zorder=-1)
ax.set(title="Neighbor features")


# %%
ratio = train_X_df["pca_val_unwrapped_0"] / train_X_df["pca_val_unwrapped_1"]

fig, ax = plt.subplots(figsize=(6, 5))
sns.histplot(
    x=ratio, hue=train_y_df.values, stat="density", kde=True, common_norm=False
)

ratio = (
    train_X_df["pca_val_unwrapped_0_neighbor_agg"]
    / train_X_df["pca_val_unwrapped_1_neighbor_agg"]
)
fig, ax = plt.subplots(figsize=(6, 5))
sns.histplot(
    x=ratio,
    hue=train_y_df.values,
    stat="density",
    kde=True,
    common_norm=False,
    element="step",
)

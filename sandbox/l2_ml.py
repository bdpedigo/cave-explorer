# %%
import io
import time

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cloudfiles import CloudFiles
from meshparty import meshwork
from meshparty.meshwork import Meshwork
from networkframe import NetworkFrame
from requests import HTTPError
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from pkg.plot import set_context

client = cc.CAVEclient("minnie65_public_v661")

DIRECTORY = "https://storage.googleapis.com/allen-minnie-phase3/minniephase3-emily-pcg-skeletons/minnie_all/v661/meshworks"

cf = CloudFiles(DIRECTORY)

proofreading_df = client.materialize.query_table(
    "proofreading_status_public_release", materialization_version=661
)

# %%
nucs = client.materialize.query_table(
    "nucleus_detection_v0", materialization_version=661
)
unique_roots = proofreading_df["pt_root_id"].unique()
nucs = nucs.query("pt_root_id.isin(@unique_roots)")

# %%
proofreading_df["target_id"] = (
    proofreading_df["pt_root_id"]
    .map(nucs.set_index("pt_root_id")["id"])
    .astype("Int64")
)
# %%
extended_df = proofreading_df.query(
    "status_axon == 'extended' & status_dendrite == 'extended'"
)

# %%


def load_mw(directory, filename):
    # REF: stolen from https://github.com/AllenInstitute/skeleton_plot/blob/main/skeleton_plot/skel_io.py
    # filename = f"{root_id}_{nuc_id}/{root_id}_{nuc_id}.h5"
    '''
    """loads a meshwork file from .h5 into meshparty.meshwork object

    Args:
        directory (str): directory location of meshwork .h5 file. in cloudpath format as seen in https://github.com/seung-lab/cloud-files
        filename (str): full .h5 filename

    Returns:
        meshwork (meshparty.meshwork): meshwork object containing .h5 data
    """'''

    if "://" not in directory:
        directory = "file://" + directory

    cf = CloudFiles(directory)
    binary = cf.get([filename])

    with io.BytesIO(cf.get(binary[0]["path"])) as f:
        f.seek(0)
        mw = meshwork.load_meshwork(f)

    return mw


def label_compartment(row):
    if row["is_soma"]:
        return "soma"
    elif row["is_axon"]:
        return "axon"
    elif row["is_dendrite"]:
        return "dendrite"
    else:
        return "unknown"


def unwrap_pca(pca):
    if np.isnan(pca).all():
        return np.full(9, np.nan)
    return np.abs(np.array(pca).ravel())


def rewrap_pca(pca):
    # take the vector and transform back into 3x3 matrix
    if np.isnan(pca).all():
        return np.full((3, 3), np.nan)
    return np.abs(np.array(pca).reshape(3, 3))


def unwrap_pca_val(pca):
    if np.isnan(pca).all():
        return np.full(3, np.nan)

    return np.array(pca).ravel()


def process_node_data(node_data):
    scalar_features = node_data[
        ["area_nm2", "max_dt_nm", "mean_dt_nm", "size_nm3"]
    ].astype(float)

    pca_unwrapped = np.stack(node_data["pca"].apply(unwrap_pca).values)
    pca_unwrapped = pd.DataFrame(
        pca_unwrapped,
        columns=[f"pca_unwrapped_{i}" for i in range(9)],
        index=node_data.index,
    )

    pca_val_unwrapped = np.stack(node_data["pca_val"].apply(unwrap_pca_val).values)
    pca_val_unwrapped = pd.DataFrame(
        pca_val_unwrapped,
        columns=[f"pca_val_unwrapped_{i}" for i in range(3)],
        index=node_data.index,
    )

    rep_coord_unwrapped = np.stack(node_data["rep_coord_nm"].values)
    rep_coord_unwrapped = pd.DataFrame(
        rep_coord_unwrapped,
        columns=["rep_coord_x", "rep_coord_y", "rep_coord_z"],
        index=node_data.index,
    )

    clean_node_data = pd.concat(
        [scalar_features, pca_unwrapped, pca_val_unwrapped, rep_coord_unwrapped], axis=1
    )

    return clean_node_data


def extract_featurized_networkframe(neuron: Meshwork, root_id: int):
    nodes = neuron.anno.lvl2_ids.df.copy()
    nodes.set_index("mesh_ind_filt", inplace=True)

    nodes["is_basal_dendrite"] = False
    nodes.loc[
        neuron.anno.basal_mesh_labels["mesh_index_filt"], "is_basal_dendrite"
    ] = True

    nodes["is_apical_dendrite"] = False
    nodes.loc[
        neuron.anno.apical_mesh_labels["mesh_index_filt"], "is_apical_dendrite"
    ] = True

    nodes["is_axon"] = False
    nodes.loc[neuron.anno.is_axon_class["mesh_index_filt"], "is_axon"] = True

    nodes["is_soma"] = False
    nodes.loc[neuron.root_region, "is_soma"] = True

    nodes["n_labels"] = nodes[
        ["is_basal_dendrite", "is_apical_dendrite", "is_axon", "is_soma"]
    ].sum(axis=1)

    nodes["is_dendrite"] = nodes["is_basal_dendrite"] | nodes["is_apical_dendrite"]

    nodes["compartment"] = nodes.apply(label_compartment, axis=1)

    nodes.drop(
        [
            "is_basal_dendrite",
            "is_apical_dendrite",
            "is_axon",
            "is_soma",
            "is_dendrite",
            "mesh_ind",
            "n_labels",
        ],
        axis=1,
        inplace=True,
    )

    nodes = nodes.reset_index(drop=True).set_index("lvl2_id")

    features = [
        "area_nm2",
        "max_dt_nm",
        "mean_dt_nm",
        "pca",
        "pca_val",
        "rep_coord_nm",
        "size_nm3",
    ]
    try:
        node_data = pd.DataFrame(
            client.l2cache.get_l2data(nodes.index.to_list(), attributes=features)
        ).T
    except HTTPError:
        print(f"Error loading node data for {root_id}")

    node_data.index = node_data.index.astype(int)
    node_data.index.name = "node_id"

    clean_node_data = process_node_data(node_data)

    nodes = nodes.join(clean_node_data)
    nodes["root_id"] = root_id

    l2_pre_syn_counts = neuron.anno["pre_syn"]["pre_pt_level2_id"].value_counts()
    nodes["pre_syn_count"] = nodes.index.map(l2_pre_syn_counts).fillna(0)

    l2_post_syn_counts = neuron.anno["post_syn"]["post_pt_level2_id"].value_counts()
    nodes["post_syn_count"] = nodes.index.map(l2_post_syn_counts).fillna(0)

    edges = client.chunkedgraph.level2_chunk_graph(root_id)
    edges = pd.DataFrame(edges, columns=["source", "target"])
    nf = NetworkFrame(nodes, edges)
    return nf


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

from sklearn.metrics import classification_report

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

from sklearn.preprocessing import QuantileTransformer

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

    transformer = QuantileTransformer(output_distribution="normal")
    train_X = transformer.fit_transform(train_X_df)
    test_X = transformer.transform(test_X_df)
    train_X_df = pd.DataFrame(
        train_X, columns=train_X_df.columns, index=train_X_df.index
    )
    test_X_df = pd.DataFrame(test_X, columns=test_X_df.columns, index=test_X_df.index)

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

from sklearn.ensemble import RandomForestClassifier

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


# %%
### JUNK BELOW ###

# %%
# r= train_X_df["pca_val_unwrapped_0"] / train_X_df["pca_val_unwrapped_1"]


def create_pca_val_features(X: pd.DataFrame):
    r01 = X["pca_val_unwrapped_0"] / X["pca_val_unwrapped_1"]
    r02 = X["pca_val_unwrapped_0"] / X["pca_val_unwrapped_2"]
    r12 = X["pca_val_unwrapped_1"] / X["pca_val_unwrapped_2"]
    df = pd.concat([r01, r02, r12], axis=1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(1)
    df.columns = ["pca_val_ratio_01", "pca_val_ratio_02", "pca_val_ratio_12"]
    return df


from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(novelty=False)
lof_train = lof.fit_predict(train_X_df)
# lof_pred = lof.predict(train_X_df)
# %%

# %%
ratio_train_df = create_pca_val_features(train_X_df)
lof_train = ratio_train_df["pca_val_ratio_02"] < 100

ratio_test_df = create_pca_val_features(test_X_df)
lda = LinearDiscriminantAnalysis(n_components=1)
train_X_lda = lda.fit_transform(
    ratio_train_df[lof_train == 1], train_y_df[lof_train == 1]
)
test_y_pred = lda.predict(ratio_test_df)

print(classification_report(test_y_df, test_y_pred))

sns.scatterplot(
    data=ratio_train_df[lof_train == 1],
    x="pca_val_ratio_01",
    y="pca_val_ratio_02",
    hue=train_y_df.values[lof_train == 1],
    s=1,
    alpha=0.1,
)

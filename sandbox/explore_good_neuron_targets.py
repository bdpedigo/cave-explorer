# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hyppo.ksample import KSample
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import gaussian_kde
from standard_transform import minnie_transform_nm
from tqdm.auto import tqdm

from pkg.utils import load_casey_palette, start_client

transformer = minnie_transform_nm()

file_path = "/Users/ben.pedigo/code/skedits/skedits-app/skedits/results/outs/exc_projection_proportions_by_mtype_threshold=0.csv"

good_neurons = pd.read_csv(file_path, index_col=0).index

# %%
# root_id = good_neurons[0]
root_id = 864691136380137941

# %%

client = start_client()

# %%

nuc_table = client.materialize.query_table(
    "nucleus_detection_v0", split_positions=True, desired_resolution=[1, 1, 1]
)
nuc_table.drop_duplicates(subset="pt_root_id", inplace=True, keep=False)
nuc_table.set_index("pt_root_id", inplace=True)
nuc_table.rename(
    columns={
        "pt_position_x": "x",
        "pt_position_y": "y",
        "pt_position_z": "z",
    },
    inplace=True,
)
nuc_table[["x", "y", "z"]] = transformer.apply(nuc_table[["x", "y", "z"]])

nuc_table

# %%
mega_table = pd.read_csv(
    "/Users/ben.pedigo/code/skedits/skedits-app/skedits/data/joint_cell_table.csv",
    low_memory=False,
).set_index("target_id")

# %%
from pathlib import Path

import pandas as pd

soma_data_path = Path(
    "/Users/ben.pedigo/code/skedits/skedits-app/skedits/data/microns_SomaData_AllCells_v661.csv"
)

feature_df = pd.read_csv(soma_data_path, index_col=0, engine="python")
feature_df = feature_df.drop(
    columns=[
        "cortical_column_labels",
        "predicted_class",
        "predicted_subclass",
        "umap_embedding_x",
        "umap_embedding_y",
        "is_column",
    ]
).set_index(["nucleus_id"])

# %%
hand_labeled = mega_table[
    mega_table["cell_type_source"].isin(
        [
            "aibs_metamodel_celltypes_v661_corrections",
            "allen_v1_column_types_slanted_ref",
            "aibs_column_nonneuronal_ref",
        ]
    )
]
# %%
X = feature_df.copy().drop(columns=["soma_id"])
# y = X.join(mega_table["cell_type"], on="nucleus_id")["cell_type"]
y = X.join(hand_labeled["cell_type"], on="nucleus_id")["cell_type"]

from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution="normal")

X_trans = qt.fit_transform(X)

from sklearn.decomposition import PCA

n_components = 5
pca = PCA(n_components=n_components)

X_pca = pca.fit_transform(X_trans)

X_pca = pd.DataFrame(
    X_pca, index=X.index, columns=[f"PC{i}" for i in range(1, n_components + 1)]
)

import colorcet as cc
import seaborn as sns

palette = dict(zip(y.unique(), cc.glasbey_light))

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(
    data=X_pca,
    x="PC1",
    y="PC2",
    hue=y,
    s=10,
    linewidth=0,
    alpha=0.8,
    palette=palette,
    ax=ax,
)


# %%
sns.pairplot(
    X_pca.join(y),
    hue="cell_type",
    markers=".",
    plot_kws={"alpha": 0.7},
    palette=palette,
)

# %%
y.dropna().value_counts()

# %%

# X = feature_df.copy().drop(columns=["soma_id"])

# df = pd.read_pickle('../data/inhibitory_perisomatic_feats_v661.pkl')

data_path = "/Users/ben.pedigo/code/skedits/skedits-app/skedits/data/inhibitory_normalized_perisomatic_features.npy"
inhibitory_cells = np.load(data_path)

# %%
data_path = "/Users/ben.pedigo/code/skedits/skedits-app/skedits/data/inhibitory_perisomatic_feats_v661.pkl"
df = pd.read_pickle(data_path)

# %%
X_df = df.set_index("nucleus_id").drop(
    columns=[
        "soma_id",
        "soma_point",
        "cortical_column_labels",
        "inhib_umap_embedding_x",
        "inhib_umap_embedding_y",
    ]
)

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df.values)
n_query = 100
nn = NearestNeighbors(n_neighbors=n_query)
nn.fit(X_scaled)

chand_nuc_id = 485719

X_df.loc[chand_nuc_id]

query_x = X_df.loc[chand_nuc_id]
query_x = scaler.transform(query_x.values.reshape(1, -1))

dists, neighbor_indices = nn.kneighbors(query_x)

# %%
query_nuc_ids = df.iloc[np.squeeze(neighbor_indices)]["nucleus_id"].values

nuc_table = nuc_table.reset_index().set_index("id")
query_root_ids = nuc_table.loc[nuc_table.index.intersection(query_nuc_ids)][
    "pt_root_id"
]


# nuc_table = nuc_table.reset_index().set_index("id")
# query_root_ids = nuc_table.loc[nuc_table.index.intersection(query_nuc_ids)][
#     "pt_root_id"
# ]

# %%
data_path = Path(
    "/Users/ben.pedigo/code/skedits/skedits-app/skedits/data/state_prediction_info/inhibitory_features.csv"
)
inhib_features = pd.read_csv(data_path, index_col=0)
inhib_features.loc[query_root_ids]

# %%
import pickle

from pkg.constants import OUT_PATH

model_path = OUT_PATH / "train_state_prediction"

with open(model_path / "state_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

X = inhib_features.loc[query_root_ids]
X["n_pre_synapses"] = X["n_pre_synapses"].replace(0, 1)
X["n_post_synapses"] = X["n_post_synapses"].replace(0, 1)
X["n_pre_synapses"] = np.log10(X["n_pre_synapses"])
X["n_post_synapses"] = np.log10(X["n_post_synapses"])
X["n_nodes"] = np.log10(X["n_nodes"].astype(float))

posteriors = model.predict_proba(X[["n_pre_synapses", "n_post_synapses", "n_nodes"]])[
    :, 1
]

posterior_ratios = model.decision_function(
    X[["n_pre_synapses", "n_post_synapses", "n_nodes"]]
)

# %%
posterior_ratios = pd.Series(index=query_root_ids, data=posterior_ratios)
posterior_ratios[posterior_ratios > 0].index


# %%
chandelier_root = 864691136924072932

# X.loc[chandelier_root]
# feature_df.set_index('soma_id').loc[chandelier_root]
chandelier_nuc_id = nuc_table.loc[chandelier_root]["id"]

query_x = X.loc[chandelier_nuc_id]


dists, neighbor_indices = nn.kneighbors([query_x])

query_nuc_ids = X.index[neighbor_indices[0]]

# %%
good_nuc_ids = nuc_table.loc[good_neurons, "id"].values

# %%

nuc_table = nuc_table.reset_index().set_index("id")
query_root_ids = nuc_table.loc[nuc_table.index.intersection(query_nuc_ids)][
    "pt_root_id"
]

# %%
data_path = Path(
    "/Users/ben.pedigo/code/skedits/skedits-app/skedits/data/state_prediction_info/inhibitory_features.csv"
)
inhib_features = pd.read_csv(data_path, index_col=0)
inhib_features.loc[query_root_ids]

# %%


# %%
# this is a putative chandelier that is also in my table
# 864691136380137941
# %%


def get_pre_synapses(root_id):
    pre_synapses = client.materialize.synapse_query(
        pre_ids=root_id, split_positions=True, desired_resolution=[1, 1, 1]
    )
    pre_synapses.set_index("id", inplace=True)
    pre_synapses.rename(
        columns={
            "ctr_pt_position_x": "x",
            "ctr_pt_position_y": "y",
            "ctr_pt_position_z": "z",
        },
        inplace=True,
    )

    pre_synapses[["x", "y", "z"]] = transformer.apply(pre_synapses[["x", "y", "z"]])

    for col in ["x", "y", "z"]:
        pre_synapses[f"post_nuc_{col}"] = pre_synapses["post_pt_root_id"].map(
            nuc_table[col]
        )
    return pre_synapses


def spherical_coords(points):
    """Converts Cartesian coordinates to spherical coordinates."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arccos(-y / r)
    return r, phi


def compute_spherical_offsets(pre_synapses, inplace=True):
    valid_pre_synapses = pre_synapses.dropna(
        subset=["post_nuc_x", "post_nuc_y", "post_nuc_z"]
    )
    displacement = (
        valid_pre_synapses[["x", "y", "z"]].values
        - valid_pre_synapses[["post_nuc_x", "post_nuc_y", "post_nuc_z"]].values
    )
    rs, phis = spherical_coords(displacement)
    pre_synapses["nuc_r"] = np.nan
    pre_synapses["nuc_phi"] = np.nan
    if inplace:
        pre_synapses.loc[valid_pre_synapses.index, "nuc_r"] = rs
        pre_synapses.loc[valid_pre_synapses.index, "nuc_phi"] = phis
    else:
        return rs, phis


def plot_spherical_offsets_multi(rs, phis):
    fig, axs = plt.subplots(
        1, 3, subplot_kw={"projection": "polar"}, layout="constrained", figsize=(10, 4)
    )

    for ax, plot_type in zip(axs, ["scatter", "hist", "kde"]):
        # kde = True
        if plot_type == "scatter":
            ax.scatter(phis, rs, s=2, alpha=0.2)
        elif plot_type == "kde":
            N = 500
            interp = gaussian_kde(np.vstack((phis, rs)))
            angles_ = np.linspace(0, np.pi, N)
            radii_ = np.linspace(0, 120, N)
            mesh = np.stack(np.meshgrid(angles_, radii_), 0)
            colors = interp(mesh.reshape(2, -1)).reshape(N, N)
            ax.pcolormesh(angles_, radii_, colors, cmap="Blues")
        elif plot_type == "hist":
            (
                counts,
                bin_radius,
                bin_phi,
            ) = np.histogram2d(
                rs, phis, bins=[30, 20], range=[[0, 120], [0, np.pi]], density=True
            )
            ax.pcolormesh(bin_phi, bin_radius, counts, cmap="Blues")

        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.grid(False)
        ax.set_theta_zero_location("N")
        ax.scatter(0, 0, s=100, c="black", alpha=1)


def plot_spherical_offsets(rs, phis, ax=None, points=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "polar"}, figsize=(4, 4))

    N = 200
    interp = gaussian_kde(np.vstack((phis, rs)))
    angles_ = np.linspace(0, np.pi, N)
    radii_ = np.linspace(0, 120, N)
    mesh = np.stack(np.meshgrid(angles_, radii_), 0)
    colors = interp(mesh.reshape(2, -1)).reshape(N, N)
    ax.pcolormesh(angles_, radii_, colors, cmap="Blues")

    if points:
        ax.scatter(phis, rs, s=1, alpha=0.1, zorder=10, linewidth=0, color="black")

    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.grid(False)
    ax.set_theta_zero_location("N")
    ax.scatter(0, 0, s=100, c="black", alpha=1)
    return ax


# %%


fig, axs = plt.subplots(
    10,
    10,
    subplot_kw={"projection": "polar"},
    layout="constrained",
    figsize=(20, 20),
    sharex=True,
    sharey=True,
    gridspec_kw={"wspace": 0.0, "hspace": 0.0},
)
# index = good_neurons[:100]
index = np.random.choice(good_neurons, 100, replace=False)
root_spherical_offsets = {}
for i, root_id in enumerate(tqdm(index)):
    pre_synapses = get_pre_synapses(root_id)
    compute_spherical_offsets(pre_synapses)

    rs = pre_synapses["nuc_r"]
    phis = pre_synapses["nuc_phi"]

    mask = rs < 120

    rs = rs[mask]
    phis = phis[mask]

    ax = axs.flatten()[i]
    plot_spherical_offsets(rs, phis, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])

    root_spherical_offsets[root_id] = (rs, phis)

# %%

ks = KSample("Dcorr")

rows = []
for i, root_id_1 in enumerate(tqdm(index)):
    rs, phis = root_spherical_offsets[root_id_1]
    df1 = pd.DataFrame({"r": rs, "phi": phis})
    for j, root_id_2 in enumerate(index):
        if i >= j:
            continue
        other_rs, other_phis = root_spherical_offsets[root_id_2]
        df2 = pd.DataFrame({"r": other_rs, "phi": other_phis})
        stat, pvalue = ks.test(df1.values, df2.values)
        rows.append(
            {
                "root_id_1": root_id_1,
                "root_id_2": root_id_2,
                "stat": stat,
                "pvalue": pvalue,
            }
        )

# %%

results = pd.DataFrame(rows)
stats_square_df = (
    results.pivot(index="root_id_1", columns="root_id_2", values="stat")
    .reindex(index=index, columns=index)
    .fillna(0)
)
pvalues_square_df = (
    results.pivot(index="root_id_1", columns="root_id_2", values="pvalue")
    .reindex(index=index, columns=index)
    .fillna(1)
)

stats_square = stats_square_df.values + stats_square_df.values.T
pvalues_square = pvalues_square_df.values + pvalues_square_df.values.T

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

ax = axs[0]
sns.heatmap(
    stats_square,
    ax=ax,
    cmap="RdBu_r",
    center=0,
    xticklabels=False,
    yticklabels=False,
    square=True,
    cbar=False,
)

ax = axs[1]
sns.heatmap(
    np.log10(pvalues_square),
    ax=ax,
    cmap="RdBu",
    center=0,
    xticklabels=False,
    yticklabels=False,
    square=True,
    cbar=False,
)

# %%


stats_square = stats_square - np.diag(np.diag(stats_square))
stats_square[stats_square < 0] = 0
linkage_matrix = linkage(squareform(stats_square), method="complete", metric="cosine")

flat_labels = fcluster(linkage_matrix, 6, criterion="maxclust") - 1

colors = sns.color_palette("tab10", n_colors=6).as_hex()
flat_colors = [colors[l] for l in flat_labels]


cgrid = sns.clustermap(
    data=stats_square,
    row_linkage=linkage_matrix,
    col_linkage=linkage_matrix,
    row_colors=flat_colors,
    col_colors=flat_colors,
    xticklabels=False,
    yticklabels=False,
    cmap="RdBu_r",
    center=0,
)
cgrid.ax_cbar.set_title("DCorr 2-sample\nstatistic")
reordered_indices = cgrid.dendrogram_row.reordered_ind

labels = pd.Series(flat_labels, index=index)

plt.savefig(
    "100_inhib_spherical_targeting_clustered_heatmap.png", dpi=500, bbox_inches="tight"
)

# %%

cell_table = client.materialize.query_table("aibs_metamodel_mtypes_v661_v2").set_index(
    "pt_root_id"
)
pred_cell_types = cell_table.loc[index[reordered_indices]]["cell_type"]

# %%

casey_palette = load_casey_palette()

fig, axs = plt.subplots(
    10,
    10,
    subplot_kw={"projection": "polar"},
    layout="constrained",
    figsize=(10, 10),
    sharex=True,
    sharey=True,
    gridspec_kw={"wspace": 0.0, "hspace": 0.0},
)
for i, root_id in enumerate(tqdm(index[reordered_indices])):
    rs, phis = root_spherical_offsets[root_id]

    mask = rs < 120

    rs = rs[mask]
    phis = phis[mask]

    ax = axs.flatten()[i]
    plot_spherical_offsets(rs, phis, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    rect = Rectangle(
        (0.05, 0.05),
        0.2,
        0.2,
        fill=True,
        color=colors[labels[root_id]],
        alpha=0.5,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(rect)

    rect = Rectangle(
        (0.05, 0.75),
        0.2,
        0.2,
        fill=True,
        color=casey_palette[pred_cell_types[root_id]],
        alpha=0.5,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(rect)

plt.savefig(
    "100_inhib_spherical_targeting_clustered_unblinded.png",
    dpi=500,
    bbox_inches="tight",
)

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from caveclient import CAVEclient
from giskard.plot import MatrixGrid
from scipy.cluster.hierarchy import fcluster, linkage
from tqdm.auto import tqdm

from pkg.plot import set_context
from pkg.utils import load_mtypes

# %%

client = CAVEclient("minnie65_phase3_v1")
mtypes = load_mtypes(client)

# %%


out_path = Path("data/synapse_pull")


files = os.listdir(out_path)

pre_synapses = []
for file in tqdm(files):
    if "pre_synapses" in file:
        chunk_pre_synapses = pd.read_csv(out_path / file)
        pre_synapses.append(chunk_pre_synapses)
pre_synapses = pd.concat(pre_synapses)

# %%
pre_synapses["post_mtype"] = pre_synapses["post_pt_root_id"].map(mtypes["cell_type"])

# %%
counts_by_mtype = (
    pre_synapses.groupby(["pre_pt_root_id", "post_mtype"]).size().unstack().fillna(0)
)
mtype_cols = counts_by_mtype.columns
inhibitory_mtypes = mtype_cols[mtype_cols.str.contains("TC")]
excitatory_mtypes = mtype_cols[~mtype_cols.str.contains("TC")]

counts_by_mtype = counts_by_mtype[counts_by_mtype.sum(axis=1) > 500]

counts_by_mtype = counts_by_mtype[excitatory_mtypes]

props_by_mtype = counts_by_mtype.div(counts_by_mtype.sum(axis=1), axis=0)

label_order = [
    "L2a",
    "L2b",
    "L2c",
    "L3a",
    "L3b",
    "L4a",
    "L4b",
    "L4c",
    "L5a",
    "L5b",
    "L5ET",
    "L5NP",
    "L6short-a",
    "L6short-b",
    "L6tall-a",
    "L6tall-b",
    "L6tall-c",
]

props_by_mtype = props_by_mtype.loc[:, label_order]


# %%
from scipy.cluster.hierarchy import leaves_list

k = 22
linkage_matrix = linkage(props_by_mtype, method="ward")

leaves = leaves_list(linkage_matrix)

props_by_mtype = props_by_mtype.iloc[leaves]

linkage_matrix = linkage(props_by_mtype, method="ward")
labels = pd.Series(
    fcluster(linkage_matrix, k, criterion="maxclust"), index=props_by_mtype.index
)
# %%
weighting = props_by_mtype.copy()
weighting.columns = np.arange(len(weighting.columns))
weighting["label"] = labels

means = weighting.groupby("label").mean()
label_ranking = means.mul(means.columns).sum(axis=1).sort_values()

new_label_map = dict(zip(labels, label_ranking.index.get_indexer_for(labels)))

# props_by_mtype["label"] = labels.map(new_label_map)
# props_by_mtype = props_by_mtype.sort_values("label")
# %%

import matplotlib.pyplot as plt

# linkage_matrix = linkage_matrix[leaves]
set_context(font_scale=2)

colors = sns.color_palette("tab20", n_colors=k)
palette = dict(zip(np.arange(1, k + 1), colors))
color_labels = labels.map(palette)

cgrid = sns.clustermap(
    props_by_mtype.T,
    cmap="Reds",
    figsize=(20, 10),
    row_cluster=False,
    col_colors=color_labels,
    col_linkage=linkage_matrix,
    xticklabels=False,
)
ax = cgrid.ax_heatmap

# move the y-axis labels to the left
ax.yaxis.tick_left()
ax.set(ylabel="Excitatory Neuron Class", xlabel="Inhibitory Neuron")
ax.yaxis.set_label_position("left")

props_by_mtype["label"] = labels.map(new_label_map)
shifts = props_by_mtype["label"] != props_by_mtype["label"].shift()
shifts.iloc[0] = False
shifts = shifts[shifts].index
shift_ilocs = props_by_mtype.index.get_indexer_for(shifts)
for shift in shift_ilocs:
    ax.axvline(shift, color="black", lw=1)
ax.set(xlabel="Inhibitory Neuron", ylabel="Excitatory Neuron Class")

props_by_mtype.drop("label", axis=1, inplace=True)

y_borders = [3, 5, 8, 12, 14]
for y_border in y_borders:
    ax.axhline(y_border, color="black", lw=1)

plt.savefig("excitatory_inhibitory_clustermap_w_tree.png", bbox_inches="tight")

# %%


# column_ctypes_map = client.materialize.query_table(INHIBITORY_CTYPES_TABLE).set_index(
#     "pt_root_id"
# )["cell_type"]
# unique_ctypes = column_ctypes_map.unique()
# colors = sns.color_palette("tab20", n_colors=len(unique_ctypes))

# column_ctypes = props_by_mtype.index.map(column_ctypes_map)
# color_labels = column_ctypes.map(dict(zip(unique_ctypes, colors)))


props_by_mtype["label"] = labels.map(new_label_map)
props_by_mtype = props_by_mtype.sort_values("label")
props_by_mtype.drop("label", axis=1, inplace=True)
# fig, ax = plt.subplots(figsize=(25, 10))
mg = MatrixGrid(figsize=(25, 10))
ax = mg.ax
sns.heatmap(props_by_mtype.T, cmap="Reds", xticklabels=False, ax=ax)


props_by_mtype["label"] = labels.map(new_label_map)
shifts = props_by_mtype["label"] != props_by_mtype["label"].shift()
shifts.iloc[0] = False
shifts = shifts[shifts].index
shift_ilocs = props_by_mtype.index.get_indexer_for(shifts)
for shift in shift_ilocs:
    ax.axvline(shift, color="black", lw=1)
ax.set(xlabel="Inhibitory Neuron", ylabel="Excitatory Neuron Class")
props_by_mtype.drop("label", axis=1, inplace=True)

y_borders = [3, 5, 8, 12, 14]
for y_border in y_borders:
    ax.axhline(y_border, color="black", lw=1)

# clabel_ax = mg.append_axes("top", size="5%", pad=0.05)

# sns.heatmap(
#     column_ctypes.to_frame().values.reshape(1, -1),
#     xticklabels=False,
#     yticklabels=False,
#     cbar=False,
#     ax=clabel_ax,
#     cmap=colors,
# )

plt.savefig("excitatory_inhibitory_clustermap_layer_sort.png", bbox_inches="tight")

# %%
counts_by_mtype = (
    pre_synapses.groupby(["pre_pt_root_id", "post_mtype"]).size().unstack().fillna(0)
)
mtype_cols = counts_by_mtype.columns
inhibitory_mtypes = mtype_cols[mtype_cols.str.contains("TC")]
excitatory_mtypes = mtype_cols[~mtype_cols.str.contains("TC")]

counts_by_mtype = counts_by_mtype[counts_by_mtype.sum(axis=1) > 600]

props_by_mtype = counts_by_mtype.div(counts_by_mtype.sum(axis=1), axis=0)

props_by_mtype = props_by_mtype[inhibitory_mtypes]


# %%
sns.clustermap(props_by_mtype.T)

# %%
props_by_mtype["mtype"] = props_by_mtype.index.map(mtypes["cell_type"])
itc_props_by_mtype = props_by_mtype.query("mtype == 'ITC'")
props_by_mtype.drop("mtype", axis=1, inplace=True)
itc_props_by_mtype.drop("mtype", axis=1, inplace=True)
sns.clustermap(itc_props_by_mtype.T)

# %%

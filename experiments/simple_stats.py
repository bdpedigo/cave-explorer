# %%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pcg_skel
import seaborn as sns
from joblib import Parallel, delayed
from networkframe import NetworkFrame
from tqdm_joblib import tqdm_joblib

from pkg.constants import OUT_PATH
from pkg.neuronframe import load_neuronframe
from pkg.plot import savefig, set_context
from pkg.skeleton import extract_meshwork_node_mappings
from pkg.utils import load_manifest, start_client

set_context()
manifest = load_manifest()
client = start_client()

# %%
manifest.query("in_inhibitory_column & is_current & has_all_sequences", inplace=True)


def load_info(root_id):
    client = start_client()
    neuron = load_neuronframe(root_id, client)
    edited_neuron = neuron.set_edits(neuron.edits.index)
    edited_neuron.select_nucleus_component(inplace=True)
    edited_neuron.apply_edge_lengths(inplace=True)
    row = {
        "root_id": root_id,
        "n_edges_unedited": len(neuron.edges),
        "n_nodes_unedited": len(neuron.nodes),
        "n_edits": len(neuron.edits),
        "n_metaedits": len(neuron.metaedits),
        "n_merges": len(neuron.edits.query("is_merge")),
        "n_splits": len(neuron.edits.query("~is_merge")),
        "edge_length_sum": edited_neuron.edges["length"].sum(),
        "n_nodes": len(edited_neuron.nodes),
        "n_edges": len(edited_neuron.edges),
    }
    return row


with tqdm_joblib(total=len(manifest)) as progress_bar:
    rows = Parallel(n_jobs=8)(delayed(load_info)(root_id) for root_id in manifest.index)

summary_info = pd.DataFrame(rows).set_index("root_id")

summary_info.to_csv(OUT_PATH / "simple_stats" / "summary_info.csv")

# %%

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.histplot(summary_info["n_edits"], ax=ax)

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.histplot(summary_info["n_merges"], ax=ax)
ax.set_ylabel("Number of merge edits")

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.histplot(summary_info["n_splits"], ax=ax)
ax.set_xlabel("Number of split edits")

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.histplot(summary_info["n_metaedits"], ax=ax)

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.scatterplot(data=summary_info, x="n_nodes_unedited", y="n_edits", ax=ax)

# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.histplot(summary_info["n_merges"], ax=ax, label="Merges")
sns.histplot(summary_info["n_splits"], ax=ax, label="Splits")
ax.set_ylabel("Number of edits")

#%%
print
print(summary_info['n_merges'].mean())
summary_info['n_merges'].mean()
# summary_info['n_merges'].median()

#%%
# summary_info['n_splits'].median()

#%%
summary_info['n_splits'].mean()

# %%
counts_df = summary_info.melt(
    value_vars=["n_merges", "n_splits"], var_name="edit_type", value_name="count"
)
counts_df["edit_type"] = counts_df["edit_type"].str.replace("n_", "").str.capitalize()
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.histplot(data=counts_df, x="count", hue="edit_type", ax=ax, element="step", bins=20)
ax.set_xlabel("Number of edits")
sns.move_legend(ax, "upper right", title="Edit type")

savefig('edit_count_histogram', fig, folder='simple_stats', doc_save=True)

# %%


def extract_skeleton_nf_for_root(root_id, client):
    try:
        neuron = load_neuronframe(root_id, client)
        edited_neuron = neuron.set_edits(neuron.edits.index)
        edited_neuron.select_nucleus_component(inplace=True)
        edited_neuron.apply_edge_lengths(inplace=True)

        # get radius info
        meshwork = pcg_skel.coord_space_meshwork(root_id, client=client)
        pcg_skel.features.add_volumetric_properties(meshwork, client)
        pcg_skel.features.add_segment_properties(meshwork)
        radius_by_level2 = meshwork.anno.segment_properties["r_eff"].to_frame()
        mesh_to_level2_ids = meshwork.anno.lvl2_ids.df.set_index("mesh_ind_filt")[
            "lvl2_id"
        ]
        radius_by_level2["level2_id"] = radius_by_level2.index.map(mesh_to_level2_ids)
        radius_by_level2 = radius_by_level2.set_index("level2_id")["r_eff"]
        edited_neuron.nodes["radius"] = edited_neuron.nodes.index.map(radius_by_level2)
        modified_nodes = edited_neuron.nodes.query(
            "(operation_added != -1) | (operation_removed != -1)"
        )

        skeleton_to_level2, level2_to_skeleton = extract_meshwork_node_mappings(
            meshwork
        )
        edited_neuron.nodes["skeleton_index"] = edited_neuron.nodes.index.map(
            level2_to_skeleton
        )
        operations_by_skeleton_node = edited_neuron.nodes.groupby("skeleton_index")[
            "operation_added"
        ].unique()
        for idx, operations in operations_by_skeleton_node.items():
            operations = operations.tolist()
            if -1 in operations:
                operations.remove(-1)
            operations_by_skeleton_node[idx] = operations
        # all_modified_nodes.append(modified_nodes)

        skeleton_nodes = pd.DataFrame(
            meshwork.skeleton.vertices, columns=["x", "y", "z"]
        )
        skeleton_edges = pd.DataFrame(
            meshwork.skeleton.edges, columns=["source", "target"]
        )
        skeleton_nf = NetworkFrame(skeleton_nodes, skeleton_edges)
        skeleton_nf.nodes["operations"] = skeleton_nf.nodes.index.map(
            operations_by_skeleton_node
        )
        # HACK sure this info is somewhere in the meshwork
        skeleton_nf.nodes["radius"] = edited_neuron.nodes.groupby("skeleton_index")[
            "radius"
        ].mean()
        return skeleton_nf
    except:
        return None


root_ids = manifest.index
with tqdm_joblib(total=len(root_ids)) as progress_bar:
    all_skeleton_nfs = Parallel(n_jobs=8)(
        delayed(extract_skeleton_nf_for_root)(root_id, client) for root_id in root_ids
    )

# %%
skeleton_nfs = {
    root_id: nf for root_id, nf in zip(root_ids, all_skeleton_nfs) if nf is not None
}

# %%
bins = np.linspace(100, 500, 51)

for root_id, skeleton_nf in skeleton_nfs.items():
    skeleton_nf.apply_node_features(["x", "y", "z", "radius"], inplace=True)
    skeleton_nf.edges["radius"] = (
        skeleton_nf.edges["source_radius"] + skeleton_nf.edges["target_radius"]
    ) / 2
    skeleton_nf.edges["length"] = np.linalg.norm(
        skeleton_nf.edges[["source_x", "source_y", "source_z"]].values
        - skeleton_nf.edges[["target_x", "target_y", "target_z"]].values,
        axis=1,
    )
    skeleton_nf.nodes["radius_bin"] = pd.cut(skeleton_nf.nodes["radius"], bins=bins)
    skeleton_nf.edges["radius_bin"] = pd.cut(skeleton_nf.edges["radius"], bins=bins)

with open(OUT_PATH / "simple_stats" / "skeleton_nfs.pkl", "wb") as f:
    pickle.dump(skeleton_nfs, f)

# %%
skeleton_nodes = pd.concat([nf.nodes for nf in skeleton_nfs.values()])
skeleton_edges = pd.concat(
    [nf.edges for nf in skeleton_nfs.values()], ignore_index=True
)

# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.histplot(skeleton_edges["radius"])
ax.set_xlim(0, 1000)

# %%

# bins = np.histogram_bin_edges(skeleton_nf.nodes["radius"], bins=100)
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.histplot(
    skeleton_edges["radius"], bins=bins, element="step", ax=ax, stat="proportion"
)

# %%

ops_per_bin = {}
for idx, group_data in skeleton_nodes.groupby("radius_bin", dropna=True)["operations"]:
    group_data = np.unique(group_data.explode().dropna())
    ops_per_bin[idx] = len(group_data)
ops_per_bin = pd.Series(ops_per_bin)
n_in_bin = skeleton_nodes.groupby("radius_bin").size()
length_in_bin = skeleton_edges.groupby("radius_bin")["length"].sum()

# %%
rate_per_bin = (ops_per_bin / length_in_bin).fillna(0)

# %%


fig, axs = plt.subplots(
    2,
    1,
    figsize=(6, 6),
    gridspec_kw=dict(height_ratios=[2, 5]),
    constrained_layout=True,
    sharex=True,
)


sns.histplot(
    x=skeleton_edges["radius"],
    weights=skeleton_edges["length"],
    ax=axs[0],
    binwidth=10,
    stat="proportion",
)

ax = axs[1]
sns.scatterplot(x=rate_per_bin.index.mid, y=rate_per_bin.values * 1000, ax=ax)
ax.set_xlabel("Radius estimate (nm)")
ax.set_ylabel("Error rate (edits / um)")
ax.set_xlim(100, 500)


savefig("error_rate_vs_radius", fig, folder="simple_stats", doc_save=True)

# %%
fig, axs = plt.subplots(
    2,
    1,
    figsize=(6, 6),
    gridspec_kw=dict(height_ratios=[2, 5]),
    constrained_layout=True,
    sharex=True,
)

sns.histplot(
    x=skeleton_edges["radius"],
    weights=skeleton_edges["length"],
    ax=axs[0],
    binwidth=10,
    stat="proportion",
)

ax = axs[1]
distance_expected_errors = 1 / (rate_per_bin.values * 1000)
sns.scatterplot(x=rate_per_bin.index.mid, y=distance_expected_errors, ax=ax)
ax.set_xlabel("Radius estimate (nm)")
ax.set_ylabel("Inverse error rate (um/edit)")
ax.set_xlim(100, 500)

savefig("inverse_error_rate_vs_radius", fig, folder="simple_stats", doc_save=True)


# %%
first_skeleton_nf = skeleton_nfs[root_ids[0]]

first_skeleton_nf.edges["length"].sum() / 1000

rate_per_edge = (
    first_skeleton_nf.edges["radius_bin"].map(rate_per_bin).astype(float).fillna(0)
)
(rate_per_edge * first_skeleton_nf.edges["length"]).sum()

# %%

first_skeleton_nf.edges["radius_bin"] * first_skeleton_nf.edges["length"]

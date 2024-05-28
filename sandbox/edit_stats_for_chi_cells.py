# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pkg.neuronframe import load_neuronframe
from pkg.skeleton import extract_meshwork_node_mappings
from pkg.utils import load_manifest
from tqdm.auto import tqdm

import caveclient as cc
import pcg_skel
from networkframe import NetworkFrame

manifest = load_manifest()

client = cc.CAVEclient("minnie65_phase3_v1")

# %%
manifest.query("in_inhibitory_column & is_current", inplace=True)

# manifest = pd.DataFrame(
#     index=[
#         864691135163673901,
#         864691135617152361,
#         864691136090326071,
#         864691135565870679,
#         864691135510120201,
#         864691135214129208,
#         864691135759725134,
#         864691135256861871,
#         864691135759685966,
#         864691135946980644,
#         864691134941217635,
#         864691136275234061,
#         864691135741431915,
#         864691135361314119,
#         864691135777918816,
#         864691136137805181,
#         864691135737446276,
#         864691136451680255,
#         864691135468657292,
#         864691135578006277,
#         864691136452245759,
#         864691135916365670,
#     ]
# )
# all_edits = []
# all_metaedits = []
# rows = []
# for root_id in tqdm(manifest.index[:]):
#     out = load_neuronframe(root_id, client, only_load=True)
#     if out is None:
#         print("Failed to load neuronframe for", root_id)
#     edited_neuron = neuron.set_edits(neuron.edits.index)
#     edited_neuron.select_nucleus_component(inplace=True)
#     edited_neuron.apply_edge_lengths(inplace=True)
#     all_edits.append(neuron.edits)
#     all_metaedits.append(neuron.metaedits)
#     rows.append(
#         {
#             "root_id": root_id,
#             "n_edges_unedited": len(neuron.edges),
#             "n_nodes_unedited": len(neuron.nodes),
#             "n_edits": len(neuron.edits),
#             "n_metaedits": len(neuron.metaedits),
#             "n_merges": len(neuron.edits.query("is_merge")),
#             "n_splits": len(neuron.edits.query("~is_merge")),
#             "edge_length_sum": edited_neuron.edges["length"].sum(),
#             "n_nodes": len(edited_neuron.nodes),
#             "n_edges": len(edited_neuron.edges),
#         }
#     )
# summary_info = pd.DataFrame(rows).set_index("root_id")


# %%


def extract_skeleton_nf_for_root(root_id, client):
    try:
        neuron = load_neuronframe(root_id, client, only_load=True)
        if neuron is None:
            return None
        edited_neuron = neuron.set_edits(neuron.edits.index)
        edited_neuron.select_nucleus_component(inplace=True)
        edited_neuron.apply_edge_lengths(inplace=True)
        splits = edited_neuron.edits.query("~is_merge")
        merges = edited_neuron.edits.query("is_merge")

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
        skeleton_nf.nodes["splits"] = skeleton_nf.nodes["operations"].apply(
            lambda x: [op for op in x if op in splits.index]
        )
        skeleton_nf.nodes["merges"] = skeleton_nf.nodes["operations"].apply(
            lambda x: [op for op in x if op in merges.index]
        )

        # HACK sure this info is somewhere in the meshwork
        skeleton_nf.nodes["radius"] = edited_neuron.nodes.groupby("skeleton_index")[
            "radius"
        ].mean()
        return skeleton_nf
    except:
        return None


from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

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

# %%
skeleton_nodes = pd.concat([nf.nodes for nf in skeleton_nfs.values()])
skeleton_edges = pd.concat(
    [nf.edges for nf in skeleton_nfs.values()], ignore_index=True
)

from statsmodels.stats.weightstats import DescrStatsW

ds = DescrStatsW(data=skeleton_edges["radius"], weights=skeleton_edges["length"])

adaptive_bins = True
if adaptive_bins:
    bins = ds.quantile(np.linspace(0.0, 1.0, 21))
else:
    bins = np.linspace(50, 500, 31)

# %%
for root_id, skeleton_nf in skeleton_nfs.items():
    skeleton_nf.nodes["radius_bin"] = pd.cut(skeleton_nf.nodes["radius"], bins=bins)
    skeleton_nf.edges["radius_bin"] = pd.cut(skeleton_nf.edges["radius"], bins=bins)

skeleton_nodes = pd.concat([nf.nodes for nf in skeleton_nfs.values()])
skeleton_edges = pd.concat(
    [nf.edges for nf in skeleton_nfs.values()], ignore_index=True
)


import pickle

from pkg.constants import OUT_PATH

with open(OUT_PATH / "simple_stats" / "skeleton_nfs.pkl", "wb") as f:
    pickle.dump(skeleton_nfs, f)


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

rows = []
for idx, group_data in skeleton_nodes.groupby("radius_bin", dropna=True):
    operations = np.unique(group_data["operations"].explode().dropna())
    splits = np.unique(group_data["splits"].explode().dropna())
    merges = np.unique(group_data["merges"].explode().dropna())
    rows.append(
        {
            "radius_bin": idx,
            "n_nodes": len(group_data),
            "n_operations": len(operations),
            "n_splits": len(splits),
            "n_merges": len(merges),
            "radius_bin_mid": idx.mid,
        }
    )
results_df = pd.DataFrame(rows).set_index("radius_bin")

results_df["length_in_bin_nm"] = skeleton_edges.groupby("radius_bin")["length"].sum()

results_df["operations_per_nm"] = (
    results_df["n_operations"] / results_df["length_in_bin_nm"]
)
results_df["splits_per_nm"] = results_df["n_splits"] / results_df["length_in_bin_nm"]
results_df["merges_per_nm"] = results_df["n_merges"] / results_df["length_in_bin_nm"]

results_df["nm_per_operation"] = 1 / results_df["operations_per_nm"]
results_df["nm_per_split"] = 1 / results_df["splits_per_nm"]
results_df["nm_per_merge"] = 1 / results_df["merges_per_nm"]

results_df.reset_index(inplace=True)


# %%
value_vars = [
    "n_operations",
    "n_splits",
    "n_merges",
    "operations_per_nm",
    "splits_per_nm",
    "merges_per_nm",
    "nm_per_operation",
    "nm_per_split",
    "nm_per_merge",
]
id_vars = results_df.columns.difference(value_vars)
results_df_long = results_df.melt(
    id_vars=id_vars,
    value_vars=value_vars,
    var_name="metric",
    value_name="value",
)

# %%
from pkg.plot import set_context

set_context()

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
sns.scatterplot(
    data=results_df,
    x="radius_bin_mid",
    y=results_df["operations_per_nm"] * 1000,
    ax=ax,
)
ax.set_xlabel("Radius estimate (nm)")
ax.set_ylabel("Error rate (edits / um)")
ax.set_xlim(100, 500)


# savefig("chi_cells_error_rate_vs_radius", fig, folder="simple_stats", doc_save=True)
# %%

fig, axs = plt.subplots(
    2,
    1,
    figsize=(6, 6),
    gridspec_kw=dict(height_ratios=[2, 5]),
    constrained_layout=True,
    sharex=True,
)

ax = axs[0]
sns.histplot(
    x=skeleton_edges["radius"],
    weights=skeleton_edges["length"],
    ax=ax,
    binwidth=10,
    stat="proportion",
)
ax.set_ylabel("Proportion\nof arbor")

ax = axs[1]

sns.lineplot(
    data=results_df_long.query(
        "metric.isin(['operations_per_nm', 'splits_per_nm', 'merges_per_nm'])"
    ),
    x="radius_bin_mid",
    y="value",
    hue="metric",
    ax=ax,
    markers=True,
    style="metric",
)
ax.set_xlabel("Radius estimate (nm)")
ax.set_ylabel("Detected error rate\n(edits / nm)")
ax.set_xlim(100, 500)
label_texts = ax.get_legend().texts
label_texts[0].set_text("All")
label_texts[1].set_text("False merge")
label_texts[2].set_text("False split")
ax.get_legend().set_title("Error type")

fig.suptitle("Inhibitory neurons (column)")


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
sns.scatterplot(
    data=results_df,
    x="radius_bin_mid",
    y=results_df["nm_per_operation"] * 1000,
    ax=ax,
)
ax.set_xlabel("Radius estimate (nm)")
ax.set_ylabel("Inverse error rate (um/edit)")
ax.set_xlim(100, 500)

# savefig(
#     "chi_cells_inverse_error_rate_vs_radius", fig, folder="simple_stats", doc_save=True
# )


# %%
first_skeleton_nf = skeleton_nfs[root_ids[0]]

first_skeleton_nf.edges["length"].sum() / 1000

rate_per_edge = (
    first_skeleton_nf.edges["radius_bin"].map(rate_per_bin).astype(float).fillna(0)
)
(rate_per_edge * first_skeleton_nf.edges["length"]).sum()

# %%

first_skeleton_nf.edges["radius_bin"] * first_skeleton_nf.edges["length"]


# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.histplot(summary_info["n_edits"], ax=ax)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.histplot(summary_info["n_merges"], ax=ax)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.histplot(summary_info["n_splits"], ax=ax)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.histplot(summary_info["n_metaedits"], ax=ax)

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.scatterplot(data=summary_info, x="n_nodes_unedited", y="n_edits", ax=ax)

# %%

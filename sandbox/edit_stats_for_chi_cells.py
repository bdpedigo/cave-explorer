# %%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from pkg.constants import OUT_PATH
from pkg.neuronframe import load_neuronframe
from pkg.plot import set_context
from pkg.skeleton import extract_meshwork_node_mappings
from pkg.utils import load_manifest
from scipy.stats import pearsonr
from statsmodels.stats.weightstats import DescrStatsW
from tqdm_joblib import tqdm_joblib

import caveclient as cc
import pcg_skel
from networkframe import NetworkFrame

manifest = load_manifest()

client = cc.CAVEclient("minnie65_phase3_v1")

# %%
manifest.query("in_inhibitory_column & is_current", inplace=True)


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

        # get skeleton/radius info
        meshwork = pcg_skel.coord_space_meshwork(root_id, client=client)
        pcg_skel.features.add_volumetric_properties(meshwork, client)
        pcg_skel.features.add_segment_properties(meshwork)
        meshwork.anchor_annotations('segment_properties')
        radius_by_level2 = meshwork.anno.segment_properties["r_eff"].to_frame()
        mesh_to_level2_ids = meshwork.anno.lvl2_ids.df.set_index("mesh_ind_filt")[
            "lvl2_id"
        ]
        radius_by_level2["level2_id"] = radius_by_level2.index.map(mesh_to_level2_ids)
        radius_by_level2 = radius_by_level2.set_index("level2_id")["r_eff"]
        edited_neuron.nodes["radius"] = edited_neuron.nodes.index.map(radius_by_level2)

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

# %%


# %%


ds = DescrStatsW(data=skeleton_edges["radius"], weights=skeleton_edges["length"])

adaptive_bins = True
if adaptive_bins:
    bins = ds.quantile(np.linspace(0.0, 1.0, 21))
else:
    bins = np.linspace(100, 500, 31)

# %%

for root_id, skeleton_nf in skeleton_nfs.items():
    skeleton_nf.nodes["radius_bin"] = pd.cut(skeleton_nf.nodes["radius"], bins=bins)
    skeleton_nf.edges["radius_bin"] = pd.cut(skeleton_nf.edges["radius"], bins=bins)
    skeleton_nf.nodes["root_id"] = root_id
    skeleton_nf.edges["root_id"] = root_id

root_index = pd.Index(skeleton_nfs.keys())

skeleton_nodes = pd.concat([nf.nodes for nf in skeleton_nfs.values()])
skeleton_edges = pd.concat(
    [nf.edges for nf in skeleton_nfs.values()], ignore_index=True
)
skeleton_nodes["n_merges"] = skeleton_nodes["merges"].apply(len)
skeleton_nodes["n_splits"] = skeleton_nodes["splits"].apply(len)

# %%
with open(OUT_PATH / "simple_stats" / "skeleton_nfs.pkl", "wb") as f:
    pickle.dump(skeleton_nfs, f)

# %%

from sklearn.model_selection import train_test_split

train_root_ids, test_root_ids = train_test_split(
    root_index, test_size=0.2, random_state=888888
)

train_skeleton_nodes = skeleton_nodes.query("root_id in @train_root_ids")
test_skeleton_nodes = skeleton_nodes.query("root_id in @test_root_ids")

train_skeleton_edges = skeleton_edges.query("root_id in @train_root_ids")
test_skeleton_edges = skeleton_edges.query("root_id in @test_root_ids")

# %%


def compute_edit_morphology_stats(skeleton_nodes, skeleton_edges, units="um"):
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

    results_df[f"length_in_bin_{units}"] = skeleton_edges.groupby("radius_bin")[
        "length"
    ].sum()
    if units == "um":
        results_df[f"length_in_bin_{units}"] /= 1000

    results_df[f"operations_per_{units}"] = (
        results_df["n_operations"] / results_df[f"length_in_bin_{units}"]
    )
    results_df[f"splits_per_{units}"] = (
        results_df["n_splits"] / results_df[f"length_in_bin_{units}"]
    )
    results_df[f"merges_per_{units}"] = (
        results_df["n_merges"] / results_df[f"length_in_bin_{units}"]
    )

    results_df[f"{units}_per_operation"] = 1 / results_df[f"operations_per_{units}"]
    results_df[f"{units}_per_split"] = 1 / results_df[f"splits_per_{units}"]
    results_df[f"{units}_per_merge"] = 1 / results_df[f"merges_per_{units}"]

    results_df.reset_index(inplace=True)

    return results_df


# %%
skeleton_nodes.explode("operations").groupby(["operations", "radius_bin"]).size()


# %%
def compute_edit_morphology_stats2(skeleton_nodes, skeleton_edges, units="um"):
    item_counts = []
    for item in ["operations", "splits", "merges"]:
        bin_ops = skeleton_nodes.explode(item).groupby([item, "radius_bin"]).size()

        bin_ops = bin_ops[bin_ops > 0]
        # TODO first or last here?
        unique_bin_ops = bin_ops.reset_index().drop(columns=0).groupby(item).last()
        assert unique_bin_ops.index.is_unique

        counts_by_bin = unique_bin_ops.groupby("radius_bin").size()
        counts_by_bin.name = f"n_{item}"
        item_counts.append(counts_by_bin)

    results_df = pd.concat(item_counts, axis=1)
    results_df.reset_index(inplace=True)
    results_df["radius_bin_mid"] = results_df["radius_bin"].apply(lambda x: x.mid)

    results_df[f"length_in_bin_{units}"] = skeleton_edges.groupby("radius_bin")[
        "length"
    ].sum()
    if units == "um":
        results_df[f"length_in_bin_{units}"] /= 1000

    results_df[f"operations_per_{units}"] = (
        results_df["n_operations"] / results_df[f"length_in_bin_{units}"]
    )
    results_df[f"splits_per_{units}"] = (
        results_df["n_splits"] / results_df[f"length_in_bin_{units}"]
    )
    results_df[f"merges_per_{units}"] = (
        results_df["n_merges"] / results_df[f"length_in_bin_{units}"]
    )

    results_df[f"{units}_per_operation"] = 1 / results_df[f"operations_per_{units}"]
    results_df[f"{units}_per_split"] = 1 / results_df[f"splits_per_{units}"]
    results_df[f"{units}_per_merge"] = 1 / results_df[f"merges_per_{units}"]

    results_df.reset_index(inplace=True)
    return results_df


# %%
units = "um"

results_df = compute_edit_morphology_stats2(
    train_skeleton_nodes, train_skeleton_edges, units=units
)
value_vars = [
    "n_operations",
    "n_splits",
    "n_merges",
    f"operations_per_{units}",
    f"splits_per_{units}",
    f"merges_per_{units}",
    f"{units}_per_operation",
    f"{units}_per_split",
    f"{units}_per_merge",
]
id_vars = results_df.columns.difference(value_vars)
results_df_long = results_df.melt(
    id_vars=id_vars,
    value_vars=value_vars,
    var_name="metric",
    value_name="value",
)

# %%

set_context()

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

inverse = False
if inverse:
    query = f"metric.isin(['{units}_per_operation', '{units}_per_split', '{units}_per_merge'])"
else:
    query = f"metric.isin(['operations_per_{units}', 'splits_per_{units}', 'merges_per_{units}'])"

sns.lineplot(
    data=results_df_long.query(query),
    x="radius_bin_mid",
    y="value",
    hue="metric",
    ax=ax,
    markers=True,
    style="metric",
)
ax.set_xlabel("Radius estimate (nm)")
if inverse:
    ax.set_ylabel(f"Inverse error rate ({units} / edit)")
else:
    ax.set_ylabel(f"Detected error rate\n(edits / {units})")
ax.set_xlim(100, 500)
label_texts = ax.get_legend().texts
for text in label_texts:
    if "operation" in text.get_text():
        text.set_text("All")
    elif "split" in text.get_text():
        text.set_text("False merge")
    elif "merge" in text.get_text():
        text.set_text("False split")

ax.get_legend().set_title("Error type")

fig.suptitle("Inhibitory neurons (column)")

# %%

length_in_bin_by_root = test_skeleton_edges.groupby(["root_id", "radius_bin"])[
    "length"
].sum()
if units == "um":
    length_in_bin_by_root /= 1000

merges_per_length = results_df.set_index("radius_bin")["merges_per_um"]
splits_per_length = results_df.set_index("radius_bin")["splits_per_um"]

predicted_merges = (length_in_bin_by_root * merges_per_length).groupby("root_id").sum()
predicted_splits = (length_in_bin_by_root * splits_per_length).groupby("root_id").sum()

sns.histplot(predicted_merges, bins=20)

# %%


true_merges = test_skeleton_nodes.groupby("root_id")["n_merges"].sum()

stat, pval = pearsonr(true_merges, predicted_merges.loc[true_merges.index])

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(x=true_merges, y=predicted_merges, ax=ax)
ax.text(0.05, 0.95, f"Pearson R = {stat:.2f}", transform=ax.transAxes)

ax.set(xlabel="Number of merges", ylabel="Predicted number of merges")
max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.set(xlim=(0, max_val), ylim=(0, max_val))
ax.set_xticks([0, 100, 200, 300, 400])
ax.set_yticks([0, 100, 200, 300, 400])
ax.plot([0, 1000], [0, 1000], linestyle="--", color="grey")

true_splits = test_skeleton_nodes.groupby("root_id")["n_splits"].sum()

stat, pval = pearsonr(true_splits, predicted_splits.loc[true_splits.index])

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(x=true_splits, y=predicted_splits, ax=ax)
ax.text(0.05, 0.95, f"Pearson R = {stat:.2f}", transform=ax.transAxes)
ax.set(xlabel="Number of splits", ylabel="Predicted number of splits")

max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.set(xlim=(0, max_val), ylim=(0, max_val))
ax.set_xticks([0, 100, 200, 300, 400])
ax.set_yticks([0, 100, 200, 300, 400])
ax.plot([0, 1000], [0, 1000], linestyle="--", color="grey")


# %%

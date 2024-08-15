# %%

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pcg_skel
import seaborn as sns
from joblib import Parallel, delayed
from networkframe import NetworkFrame
from statsmodels.stats.weightstats import DescrStatsW
from tqdm_joblib import tqdm_joblib

from pkg.constants import OUT_PATH, TIMESTAMP
from pkg.neuronframe import load_neuronframe
from pkg.plot import savefig, set_context
from pkg.skeleton import extract_meshwork_node_mappings
from pkg.utils import load_manifest, start_client

set_context()
manifest = load_manifest()
client = start_client()

# %%
manifest.query("in_inhibitory_column", inplace=True)


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


if True:
    with tqdm_joblib(total=len(manifest)) as progress_bar:
        rows = Parallel(n_jobs=8)(
            delayed(load_info)(root_id) for root_id in manifest.index
        )

    summary_info = pd.DataFrame(rows).set_index("root_id")

    summary_info.to_csv(OUT_PATH / "simple_stats" / "summary_info.csv")
else:
    summary_info = pd.read_csv(
        OUT_PATH / "simple_stats" / "summary_info.csv", index_col=0
    )

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

# %%
from pkg.constants import MERGE_COLOR, SPLIT_COLOR

edit_palette = {"Merges": MERGE_COLOR, "Splits": SPLIT_COLOR}
counts_df = summary_info.melt(
    value_vars=["n_merges", "n_splits"], var_name="edit_type", value_name="count"
)
counts_df["edit_type"] = counts_df["edit_type"].str.replace("n_", "").str.capitalize()
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.histplot(
    data=counts_df,
    x="count",
    hue="edit_type",
    ax=ax,
    element="step",
    bins=20,
    palette=edit_palette,
)

ax.set_xlabel("Number of edits")
sns.move_legend(ax, "upper right", title="Edit type")

savefig("edit_count_histogram", fig, folder="simple_stats", doc_save=True, format="svg")


# %%

multiaxon_dict = {
    258362: 2,
    307059: 2,
}

recompute_skeletons = True

if recompute_skeletons:

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
            soma_point = manifest.loc[
                root_id, ["nuc_x", "nuc_y", "nuc_z"]
            ].values.astype(float)
            meshwork = pcg_skel.coord_space_meshwork(
                root_id,
                client=client,
                timestamp=TIMESTAMP,
                root_point=soma_point,
                root_point_resolution=[1, 1, 1],
                collapse_soma=True,
                require_complete=True,
                synapses="all",
                synapse_table=client.info.get_datastack_info().get("synapse_table"),
            )

            target_id = manifest.loc[root_id, "target_id"]
            if target_id in multiaxon_dict:
                n_times = multiaxon_dict[target_id]
            else:
                n_times = 1
            pcg_skel.features.add_is_axon_annotation(
                meshwork,
                pre_anno="pre_syn",
                post_anno="post_syn",
                threshold_quality=0.6,
                n_times=n_times,
            )

            pcg_skel.features.add_volumetric_properties(meshwork, client)
            pcg_skel.features.add_segment_properties(meshwork)
            meshwork.anchor_annotations("segment_properties")
            radius_by_level2 = meshwork.anno.segment_properties["r_eff"].to_frame()
            mesh_to_level2_ids = meshwork.anno.lvl2_ids.df.set_index("mesh_ind_filt")[
                "lvl2_id"
            ]
            radius_by_level2["level2_id"] = radius_by_level2.index.map(
                mesh_to_level2_ids
            )
            radius_by_level2 = radius_by_level2.set_index("level2_id")["r_eff"]
            edited_neuron.nodes["radius"] = edited_neuron.nodes.index.map(
                radius_by_level2
            )

            is_axon_level2_ilocs = meshwork.anno.is_axon["mesh_index_filt"]
            is_axon_level2_ids = mesh_to_level2_ids.loc[is_axon_level2_ilocs]
            edited_neuron.nodes["is_axon"] = False
            edited_neuron.nodes.loc[is_axon_level2_ids, "is_axon"] = True

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

            is_axon_by_skeleton_node = edited_neuron.nodes.groupby("skeleton_index")[
                "is_axon"
            ].all()

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

            skeleton_nf.nodes["is_axon"] = skeleton_nf.nodes.index.map(
                is_axon_by_skeleton_node
            )

            return skeleton_nf
        except:
            return None

    root_ids = manifest.index[:]
    with tqdm_joblib(total=len(root_ids)) as progress_bar:
        all_skeleton_nfs = Parallel(n_jobs=8)(
            delayed(extract_skeleton_nf_for_root)(root_id, client)
            for root_id in root_ids
        )

    skeleton_nfs = {
        root_id: nf for root_id, nf in zip(root_ids, all_skeleton_nfs) if nf is not None
    }
    with open(OUT_PATH / "simple_stats" / "skeleton_nfs.pkl", "wb") as f:
        pickle.dump(skeleton_nfs, f)
else:
    with open(OUT_PATH / "simple_stats" / "skeleton_nfs.pkl", "rb") as f:
        skeleton_nfs = pickle.load(f)

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

ds = DescrStatsW(data=skeleton_edges["radius"], weights=skeleton_edges["length"])

adaptive_bins = True
if adaptive_bins:
    bins = ds.quantile(np.linspace(0.0, 1.0, 21))
else:
    bins = np.linspace(100, 500, 31)
    bins = list(bins)
    bins = [0] + bins
    bins = bins + [np.inf]

# %%
for root_id, skeleton_nf in skeleton_nfs.items():
    skeleton_nf.nodes["radius_bin"] = pd.cut(
        skeleton_nf.nodes["radius"], bins=bins, include_lowest=True
    )
    skeleton_nf.edges["radius_bin"] = pd.cut(
        skeleton_nf.edges["radius"], bins=bins, include_lowest=True
    )
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


def compute_edit_morphology_stats(skeleton_nodes, skeleton_edges, units="um"):
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
    results_df.set_index("radius_bin", inplace=True)
    print(results_df)
    # print(skeleton_edges.groupby("radius_bin")["length"].sum())
    bin_lengths = skeleton_edges.groupby("radius_bin")["length"].sum()
    print(bin_lengths)

    results_df[f"length_in_bin_{units}"] = bin_lengths
    # print(results_df[f"length_in_bin_{units}"].hasnans)
    # print(results_df[results_df[f"length_in_bin_{units}"].isna()])
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


units = "um"

results_df = compute_edit_morphology_stats(skeleton_nodes, skeleton_edges, units=units)
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

inverse = True
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
ax.set_xlim(0, 500)
ax.set_ylim(0, ax.get_ylim()[1])
label_texts = ax.get_legend().texts
for text in label_texts:
    if "operation" in text.get_text():
        text.set_text("All")
    elif "split" in text.get_text():
        text.set_text("False merge")
    elif "merge" in text.get_text():
        text.set_text("False split")

ax.get_legend().set_title("Error type")

savefig(
    f"error_rate_vs_radius_inverse={inverse}", fig, folder="simple_stats", doc_save=True
)

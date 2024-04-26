# %%


import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist

from pkg.metrics import (
    annotate_mtypes,
    annotate_pre_synapses,
    compute_counts,
    compute_spatial_target_proportions,
    compute_target_counts,
    compute_target_proportions,
)
from pkg.neuronframe import NeuronFrameSequence, load_neuronframe
from pkg.plot import savefig, set_context
from pkg.utils import load_manifest, load_mtypes

# %%

set_context()

client = cc.CAVEclient("minnie65_phase3_v1")

manifest = load_manifest()
MTYPES = load_mtypes(client)

# %%
example_root_ids = manifest.query("is_sample").index

root_id = example_root_ids[14]


def compute_diffs_to_final(sequence_df):
    # the final rows the one with Nan index
    final_row_idx = -1
    final_row = sequence_df.loc[final_row_idx].fillna(0).values.reshape(1, -1)

    X = sequence_df.drop(index=final_row_idx).fillna(0)

    sample_wise_metrics = []
    for metric in ["euclidean", "cityblock", "jensenshannon", "cosine"]:
        distances = cdist(X.values, final_row, metric=metric)
        distances = pd.Series(
            distances.flatten(),
            name=metric,
            index=X.index,
        )
        sample_wise_metrics.append(distances)
    sample_wise_metrics = pd.concat(sample_wise_metrics, axis=1)

    return sample_wise_metrics


def compute_dropout_stats_for_neuron(neuron, prefix="meta"):
    sequence = NeuronFrameSequence(
        neuron,
        prefix=prefix,
        edit_label_name=f"{prefix}operation_id_dropped",
        include_initial_state=False,
    )

    if prefix == "meta":
        edits = neuron.metaedits
        filtered_edits = edits.query("has_filtered")
    else:
        edits = neuron.edits
        filtered_edits = edits.query("is_filtered")

    for operation_id in filtered_edits.index:
        edits_to_apply = edits.query(f"{prefix}operation_id != @operation_id").index
        sequence.apply_edits(
            edits_to_apply,
            label=operation_id,
            replace=True,
            warn_on_missing=False,
            # only_additions=True,
        )

    sequence.apply_edits(edits.index, label=-1, replace=True)

    sequence_feature_dfs = {}
    counts = sequence.apply_to_synapses_by_sample(
        compute_counts, which="pre", output="scalar", name="count"
    )
    sequence_feature_dfs["counts"] = counts

    counts_by_mtype = sequence.apply_to_synapses_by_sample(
        compute_target_counts, which="pre", by="post_mtype"
    )
    sequence_feature_dfs["counts_by_mtype"] = counts_by_mtype

    props_by_mtype = sequence.apply_to_synapses_by_sample(
        compute_target_proportions, which="pre", by="post_mtype"
    )
    sequence_feature_dfs["props_by_mtype"] = props_by_mtype

    spatial_props = sequence.apply_to_synapses_by_sample(
        compute_spatial_target_proportions, which="pre", mtypes=MTYPES
    )
    sequence_feature_dfs["spatial_props"] = spatial_props

    spatial_props_by_mtype = sequence.apply_to_synapses_by_sample(
        compute_spatial_target_proportions,
        which="pre",
        mtypes=MTYPES,
        by="post_mtype",
    )
    sequence_feature_dfs["spatial_props_by_mtype"] = spatial_props_by_mtype

    sequence_features_for_neuron = pd.Series(sequence_feature_dfs, name=root_id)

    diffs_by_feature = {}
    for feature_name, feature_df in sequence_features_for_neuron.items():
        diffs = compute_diffs_to_final(feature_df)
        diffs_by_feature[feature_name] = diffs

    feature_diffs_for_neuron = pd.Series(diffs_by_feature, name=root_id)

    return sequence_features_for_neuron, feature_diffs_for_neuron


def process_neuron(root_id):
    neuron = load_neuronframe(root_id, client)

    annotate_pre_synapses(neuron, MTYPES)
    annotate_mtypes(neuron, MTYPES)
    (
        sequence_features_for_neuron,
        feature_diffs_for_neuron,
    ) = compute_dropout_stats_for_neuron(neuron)

    metaedits = neuron.metaedits
    metaedits["root_id"] = root_id

    return sequence_features_for_neuron, feature_diffs_for_neuron, metaedits


from joblib import Parallel, delayed

outs = Parallel(n_jobs=8, verbose=10)(
    delayed(process_neuron)(root_id) for root_id in manifest.index[:1]
)

# %%

sequence_features_by_neuron = {}
feature_diffs_by_neuron = {}
metaedits_by_neuron = {}
for out, root_id in zip(outs, manifest.index):
    sequence_features_by_neuron[root_id] = out[0]
    feature_diffs_by_neuron[root_id] = out[1]
    metaedits_by_neuron[root_id] = out[2]

sequence_features_by_neuron = pd.concat(sequence_features_by_neuron, axis=1).T
sequence_features_by_neuron.index.name = "root_id"

feature_diffs_by_neuron = pd.concat(feature_diffs_by_neuron, axis=1).T
feature_diffs_by_neuron.index.name = "root_id"

metaedits_by_neuron = pd.concat(metaedits_by_neuron, axis=0)
metaedits_by_neuron.index.set_names(["root_id", "metaoperation_id"], inplace=True)


# %%


example_root_ids = manifest.query("is_sample").index

for root_id in example_root_ids:
    diffs = feature_diffs_by_neuron.loc[root_id]
    metaedits = metaedits_by_neuron.loc[root_id]
    sorted_diff_index = diffs["counts"].sort_values("euclidean", ascending=False).index

    fig, axs = plt.subplots(2, 5, figsize=(16, 10), constrained_layout=True)
    for i, (feature_name, diffs) in enumerate(diffs.items()):
        diffs = diffs.loc[sorted_diff_index]
        has_merge = diffs.index.map(metaedits["has_merge"]).rename("has_merge")
        ax = axs[0, i]
        sns.scatterplot(
            x=np.arange(len(diffs)), y=diffs["euclidean"], hue=has_merge, ax=ax
        )
        ax.set_title(feature_name)
        if not i == 0:
            ax.get_legend().remove()

        ax = axs[1, i]
        sns.scatterplot(
            x=np.arange(len(diffs)),
            y=diffs["euclidean"],
            hue=has_merge,
            ax=ax,
            legend=False,
        )
        ax.set_yscale("log")

    axs[1, 2].set_xlabel("Operation rank")

    savefig(
        f"edit_dropout_importance_root_id={root_id}",
        fig,
        folder="single_edit_dropout",
        doc_save=True,
        group="dropout_importance",
        caption=root_id,
    )

#
# diffs = feature_diffs_for_neuron["counts"]
# sorted_diff_index = diffs.sort_values("euclidean", ascending=False).index

# %%


pd.concat(feature_diffs_by_neuron["counts"].to_list())


# %%

for root_id in example_root_ids[:]:
    neuron = load_neuronframe(root_id, client)

    annotate_pre_synapses(neuron, MTYPES)
    annotate_mtypes(neuron, MTYPES)
    (
        sequence_features_for_neuron,
        feature_diffs_for_neuron,
    ) = compute_dropout_stats_for_neuron(neuron)

    def get_metaoperation_modified(row):
        if row["metaoperation_added"] != -1 and row["metaoperation_removed"] != -1:
            if row["metaoperation_added"] != row["metaoperation_removed"]:
                raise ValueError("Both added and removed")

        max_idx = row[["metaoperation_added", "metaoperation_removed"]].max()
        return max_idx

    modified_nodes = neuron.nodes.query(
        "(metaoperation_added != -1) | (metaoperation_removed != -1)"
    ).copy()
    modified_nodes["metaoperation_modified"] = modified_nodes.apply(
        get_metaoperation_modified, axis=1
    )

    extended_neuron = neuron  #
    # neuron.set_edits(
    #     neuron.metaedits.query("has_merge & has_filtered")
    # )
    nuc_id = extended_neuron.nucleus_id
    nuc_iloc = extended_neuron.nodes.index.get_indexer_for([nuc_id])[0]

    extended_neuron = extended_neuron.apply_edge_lengths()

    spadj = extended_neuron.to_sparse_adjacency(weight_col="length")

    dists = dijkstra(spadj, directed=False, indices=nuc_iloc, min_only=False)

    ilocs = extended_neuron.nodes.index.get_indexer_for(modified_nodes.index)

    modified_nodes["path_length_to_nuc"] = dists[ilocs] / 1_000

    metaoperation_min_dists = modified_nodes.groupby("metaoperation_modified")[
        "path_length_to_nuc"
    ].min()

    euc_count_dists = feature_diffs_for_neuron["counts"]["euclidean"]

    min_dist_l2_node_ids = modified_nodes.groupby("metaoperation_modified")[
        "path_length_to_nuc"
    ].idxmin()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    ax = axs[0]
    sns.scatterplot(
        y=euc_count_dists,
        x=metaoperation_min_dists.loc[euc_count_dists.index],
        hue=metaedits_by_neuron.loc[root_id].loc[euc_count_dists.index, "has_merge"],
        ax=ax,
    )

    ax = axs[1]
    sns.scatterplot(
        y=euc_count_dists,
        x=neuron.metaedits.loc[euc_count_dists.index, "centroid_distance_to_nuc_um"],
        hue=metaedits_by_neuron.loc[root_id].loc[euc_count_dists.index, "has_merge"],
        legend=False,
        ax=ax,
    )

    savefig(
        f"edit_dropout_importance_vs_distance-root_id={root_id}",
        fig,
        folder="single_edit_dropout",
        doc_save=True,
        group="dropout_importance_vs_distance",
        caption=root_id,
    )

# %%

from IPython.display import display
from nglui import statebuilder

colors = sns.color_palette("coolwarm", n_colors=11)
for root_id in example_root_ids[0:5]:
    neuron = load_neuronframe(root_id, client)
    diffs = feature_diffs_by_neuron.loc[root_id]
    importances = diffs["counts"]["euclidean"]
    metaedits = metaedits_by_neuron.loc[root_id].copy()
    metaedits = metaedits.query("has_filtered & has_merge")
    metaedits["importance"] = importances

    qs = np.linspace(0, 1, 11)
    quantile_bins = np.unique(metaedits["importance"].quantile(qs))
    metaedits["importance_bin"] = (
        pd.cut(metaedits["importance"], bins=quantile_bins, labels=False, right=False)
        .fillna(10)
        .astype(int)
    )

    # display(neuron.generate_neuroglancer_link(client, color_edits=False))
    sbs, dfs = neuron._generate_link_bases(client)

    annotation_mapper = statebuilder.PointMapper(
        point_column="centroid", split_positions=True
    )
    for bin_idx, bin_df in metaedits.groupby("importance_bin"):
        layer = statebuilder.AnnotationLayerConfig(
            int(bin_idx), mapping_rules=annotation_mapper, color=colors[bin_idx]
        )
        sb = statebuilder.StateBuilder(layers=[layer], resolution=[1, 1, 1])
        sbs.append(sb)
        dfs.append(bin_df)

    sb = statebuilder.ChainedStateBuilder(sbs)

    display(
        statebuilder.helpers.package_state(dfs, sb, client=client, return_as="html")
    )


# %%
from troglobyte.features import CAVEWrangler

min_dist_l2_nodes = neuron.nodes.loc[min_dist_l2_node_ids.values]
removed_min_dist_l2_nodes = min_dist_l2_nodes.query("was_removed")
# added_min_dist_l2_nodes = min_dist_l2_nodes.query("~was_removed")

# TODO not sure the right thing to be doing here
before_roots = neuron.edits.loc[removed_min_dist_l2_nodes["operation_removed"].values][
    "before_root_ids"
].apply(lambda x: x[0])


# %%
wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=5)
wrangler.set_manifest(before_roots.values, removed_min_dist_l2_nodes.index)
wrangler.query_level2_shape_features()
points = wrangler.level2_shape_features_[
    ["rep_coord_x", "rep_coord_y", "rep_coord_z"]
].droplevel(1)
points = pd.Series(data=list(zip(*zip(*points.values))), index=points.index)
wrangler.set_query_boxes_from_points(points, box_width=20_000)
wrangler.query_level2_edges()
wrangler.query_level2_ids_from_edges()
wrangler.query_level2_shape_features()
wrangler.query_level2_synapse_features()
wrangler.aggregate_features_by_neighborhood(
    aggregations=["mean", "std"], neighborhood_hops=10
)

# %%

relevant_features = wrangler.features_.loc[
    pd.MultiIndex.from_arrays([before_roots.values, removed_min_dist_l2_nodes.index])
]
relevant_features.index = removed_min_dist_l2_nodes["operation_removed"]

relevant_features = relevant_features.dropna()
# %%
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution="normal")
qt.fit(relevant_features)

transformed_relevant_features = qt.transform(relevant_features)

operation_features = neuron.edits.loc[relevant_features.index]

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()

pred = np.squeeze(lda.fit_transform(relevant_features, operation_features["is_merge"]))


sns.histplot(x=pred, hue=operation_features["is_merge"].values, kde=True)

# %%

coefs = pd.Series(data=np.squeeze(lda.coef_), index=relevant_features.columns)

fig, ax = plt.subplots(1, 1, figsize=(6, 10))
sns.barplot(x=coefs, y=coefs.index, ax=ax)

# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=1000, max_depth=3)

rf.fit(relevant_features, operation_features["is_merge"])

pred = rf.predict_proba(relevant_features)[:, 1]

sns.histplot(x=pred, hue=operation_features["is_merge"].values, kde=True)

# %%
feature_importances = pd.Series(
    data=rf.feature_importances_, index=relevant_features.columns
)

fig, ax = plt.subplots(1, 1, figsize=(6, 10))
sns.barplot(x=feature_importances, y=feature_importances.index, ax=ax)

# %%

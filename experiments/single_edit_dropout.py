# %%


import caveclient as cc
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm.autonotebook import tqdm

from pkg.metrics import (
    annotate_mtypes,
    annotate_pre_synapses,
    compute_counts,
    compute_spatial_target_proportions,
    compute_target_counts,
    compute_target_proportions,
)
from pkg.neuronframe import NeuronFrameSequence, load_neuronframe
from pkg.plot import set_context
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


def compute_dropout_stats_for_neuron(neuron):
    sequence = NeuronFrameSequence(
        neuron,
        prefix="meta",
        edit_label_name="metaoperation_id_dropped",
        include_initial_state=False,
    )

    metaedits = neuron.metaedits
    filtered_metaedits = metaedits.query("has_filtered")
    for metaoperation_id in tqdm(filtered_metaedits.index):
        edits_to_apply = metaedits.query("metaoperation_id != @metaoperation_id").index
        sequence.apply_edits(
            edits_to_apply, label=metaoperation_id, replace=True, warn_on_missing=False
        )

    sequence.apply_edits(metaedits.index, label=-1, replace=True)

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

    return sequence_features_for_neuron, feature_diffs_for_neuron, neuron.metaedits


from joblib import Parallel, delayed

outs = Parallel(n_jobs=-1, verbose=5)(
    delayed(process_neuron)(root_id) for root_id in manifest.index
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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pkg.plot import savefig

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

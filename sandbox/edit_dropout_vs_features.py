# %%


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
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
from pkg.utils import get_nucleus_point_nm, load_manifest, load_mtypes
from scipy.spatial.distance import cdist

import caveclient as cc
import pcg_skel

# %%

set_context()

client = cc.CAVEclient("minnie65_phase3_v1")

manifest = load_manifest()
MTYPES = load_mtypes(client)

# %%


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
    root_id = neuron.neuron_id
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
    try:
        neuron = load_neuronframe(root_id, client)

        annotate_pre_synapses(neuron, MTYPES)
        annotate_mtypes(neuron, MTYPES)
        (
            sequence_features_for_neuron,
            feature_diffs_for_neuron,
        ) = compute_dropout_stats_for_neuron(neuron)

        metaedits = neuron.metaedits
        metaedits["root_id"] = root_id

        root_point = get_nucleus_point_nm(root_id, client=client)
        # generate a skeleton and map it back to the level2 nodes
        meshwork = pcg_skel.coord_space_meshwork(
            root_id,
            client=client,
            root_point=root_point,
            root_point_resolution=[1, 1, 1],
        )
        pcg_skel.features.add_volumetric_properties(meshwork, client)
        pcg_skel.features.add_segment_properties(meshwork)

        level2_nodes = meshwork.anno.lvl2_ids.df.copy()
        level2_nodes.set_index("mesh_ind_filt", inplace=True)
        level2_nodes[
            "skeleton_index"
        ] = meshwork.anno.lvl2_ids.mesh_index.to_skel_index_padded
        level2_nodes = level2_nodes.rename(columns={"lvl2_id": "level2_id"}).drop(
            columns="mesh_ind"
        )
        # skeleton_to_level2 = level2_nodes.groupby("skeleton_index")["level2_id"].unique()

        radius_by_level2 = meshwork.anno.segment_properties["r_eff"].to_frame()
        # radius_by_skeleton = radius_by_level2.groupby(level2_nodes["skeleton_index"]).mean()

        radius_by_level2["level2_id"] = radius_by_level2.index.map(
            level2_nodes["level2_id"]
        )
        radius_by_level2 = radius_by_level2.set_index("level2_id")["r_eff"]

        importance_by_metaoperation = feature_diffs_for_neuron["counts"]["euclidean"]

        neuron = load_neuronframe(root_id, client)
        neuron.set_edits(neuron.edits.index, inplace=True)
        neuron.select_nucleus_component(inplace=True)

        neuron.nodes["radius"] = radius_by_level2

        radius_by_metaoperation = (
            neuron.nodes.groupby("metaoperation_added")["radius"].mean().drop(-1)
        )

        metaoperation_stats = pd.concat(
            [radius_by_metaoperation, importance_by_metaoperation], axis=1
        )

        metaoperation_stats["root_id"] = root_id

        return metaoperation_stats
    except:
        return None


root_ids = manifest.query("in_inhibitory_column & has_all_sequences").index

# metaoperation_stats_for_neuron = process_neuron(example_root_ids[0])


outs = Parallel(n_jobs=8, verbose=10)(
    delayed(process_neuron)(root_id) for root_id in root_ids
)

# %%

outs = [out for out in outs if out is not None]
metaoperation_stats = pd.concat(outs).reset_index().dropna()


# %%

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.scatterplot(
    data=metaoperation_stats,
    x="radius",
    y="euclidean",
    # hue="root_id",
    ax=ax,
    # palette="tab20",
    alpha=0.3,
    s=5,
    legend=False,
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set(
    ylabel="Edit importance (# synapses modified)", xlabel="Radius estimate of arbor"
)
ax.set_xlim(50, 1000)


# %%

from scipy.stats import spearmanr

spearmanr(metaoperation_stats["radius"], metaoperation_stats["euclidean"])

# %%
from scipy.stats import pearsonr

pearsonr(metaoperation_stats["radius"], metaoperation_stats["euclidean"])

# %%
metaoperation_stats_non_soma = metaoperation_stats.query("radius < 1000")
pearsonr(
    metaoperation_stats_non_soma["radius"], metaoperation_stats_non_soma["euclidean"]
)

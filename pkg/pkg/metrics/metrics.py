import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from ..neuronframe import NeuronFrame

SPATIAL_BINS = np.linspace(0, 1_000_000, 31)


def annotate_pre_synapses(neuron: NeuronFrame, mtypes: pd.DataFrame) -> None:
    # annotating with classes
    neuron.pre_synapses["post_mtype"] = neuron.pre_synapses["post_pt_root_id"].map(
        mtypes["cell_type"]
    )

    # locations of the post-synaptic soma
    post_locs = (
        neuron.pre_synapses["post_pt_root_id"]
        .map(mtypes["pt_position"])
        .dropna()
        .to_frame(name="post_nuc_loc")
    )
    post_locs["post_nuc_x"] = post_locs["post_nuc_loc"].apply(lambda x: x[0])
    post_locs["post_nuc_y"] = post_locs["post_nuc_loc"].apply(lambda x: x[1])
    post_locs["post_nuc_z"] = post_locs["post_nuc_loc"].apply(lambda x: x[2])
    neuron.pre_synapses = neuron.pre_synapses.join(post_locs)

    # euclidean distance to post-synaptic soma
    nuc_loc = neuron.nodes.loc[neuron.nucleus_id, ["x", "y", "z"]]
    X = neuron.pre_synapses[["post_nuc_x", "post_nuc_y", "post_nuc_z"]].dropna()
    euclidean_distances = pairwise_distances(
        X, nuc_loc.values.reshape(1, -1), metric="euclidean"
    )
    euclidean_distances = pd.Series(
        euclidean_distances.flatten(), index=X.index, name="euclidean"
    )

    # radial (x-z only) distance to post-synaptic soma
    X_radial = neuron.pre_synapses[["post_nuc_x", "post_nuc_z"]].dropna()
    nuc_loc_radial = nuc_loc[["x", "z"]]
    radial_distances = pairwise_distances(
        X_radial, nuc_loc_radial.values.reshape(1, -1), metric="euclidean"
    )
    radial_distances = pd.Series(
        radial_distances.flatten(), index=X_radial.index, name="radial"
    )
    distance_df = pd.concat([euclidean_distances, radial_distances], axis=1)
    neuron.pre_synapses = neuron.pre_synapses.join(distance_df)

    neuron.pre_synapses["radial_to_nuc_bin"] = pd.cut(
        neuron.pre_synapses["radial"], SPATIAL_BINS
    )

    return None


def annotate_mtypes(neuron: NeuronFrame, mtypes: pd.DataFrame):
    mtypes["post_mtype"] = mtypes["cell_type"]
    mtypes["x"] = mtypes["pt_position"].apply(lambda x: x[0])
    mtypes["y"] = mtypes["pt_position"].apply(lambda x: x[1])
    mtypes["z"] = mtypes["pt_position"].apply(lambda x: x[2])
    nuc_loc = neuron.nodes.loc[neuron.nucleus_id, ["x", "y", "z"]]
    distance_to_nuc = pairwise_distances(
        mtypes[["x", "y", "z"]], nuc_loc.values.reshape(1, -1), metric="euclidean"
    )
    mtypes["euclidean_to_nuc"] = distance_to_nuc

    nuc_loc = neuron.nodes.loc[neuron.nucleus_id, ["x", "z"]]
    distance_to_nuc = pairwise_distances(
        mtypes[["x", "z"]], nuc_loc.values.reshape(1, -1), metric="euclidean"
    )
    mtypes["radial_to_nuc"] = distance_to_nuc

    mtypes["radial_to_nuc_bin"] = pd.cut(mtypes["radial_to_nuc"], SPATIAL_BINS)

    return None


def compute_spatial_target_proportions(synapses_df, mtypes=None, by=None):
    if by is not None:
        spatial_by = ["radial_to_nuc_bin", by]
    else:
        spatial_by = ["radial_to_nuc_bin"]

    cells_hit = synapses_df.groupby(spatial_by)["post_pt_root_id"].nunique()

    cells_available = mtypes.groupby(spatial_by).size()

    p_cells_hit = cells_hit / cells_available

    return p_cells_hit


def compute_target_counts(synapses_df: pd.DataFrame, by=None):
    result = synapses_df.groupby(by).size()
    return result


def compute_target_proportions(synapses_df: pd.DataFrame, by=None):
    result = synapses_df.groupby(by).size()
    result = result / result.sum()
    return result


def compute_counts(synapses_df: pd.DataFrame):
    return len(synapses_df)


def compute_precision_recall(sequence: pd.DataFrame, which="pre"):
    synapses: pd.Series = sequence.sequence_info[f"{which}_synapses"]
    final_synapses = synapses.iloc[-1]

    results = pd.DataFrame(
        index=synapses.index,
        columns=[f"{which}_synapse_recall", f"{which}_synapse_precision"],
    )
    for idx, synapses in synapses.items():
        n_intersection = len(np.intersect1d(final_synapses, synapses))

        # recall: the proportion of synapses in the final state that show up in the current
        if len(final_synapses) == 0:
            recall = np.nan
        else:
            recall = n_intersection / len(final_synapses)
            results.loc[idx, f"{which}_synapse_recall"] = recall

        # precision: the proportion of synapses in the current state that show up in the final
        if len(synapses) == 0:
            precision = np.nan
        else:
            precision = n_intersection / len(synapses)
            results.loc[idx, f"{which}_synapse_precision"] = precision

    results[f"{which}_synapse_f1"] = (
        2
        * (results[f"{which}_synapse_recall"] * results[f"{which}_synapse_precision"])
        / (results[f"{which}_synapse_recall"] + results[f"{which}_synapse_precision"])
    )

    return results

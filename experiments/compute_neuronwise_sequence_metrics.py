# %%

import pickle

import caveclient as cc
import numpy as np
import pandas as pd
from cloudfiles import CloudFiles
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances

from pkg.constants import OUT_PATH
from pkg.neuronframe import NeuronFrame, load_neuronframe
from pkg.sequence import create_merge_and_clean_sequence, create_time_ordered_sequence
from pkg.utils import load_manifest, load_mtypes

cloud_bucket = "allen-minnie-phase3"
folder = "edit_sequences"

cf = CloudFiles(f"gs://{cloud_bucket}/{folder}")

files = list(cf.list())
files = pd.DataFrame(files, columns=["file"])

# pattern is root_id=number as the beginning of the file name
# extract the number from the file name and store it in a new column
files["root_id"] = files["file"].str.split("=").str[1].str.split("-").str[0].astype(int)
files["order_by"] = files["file"].str.split("=").str[2].str.split("-").str[0]
files["random_seed"] = files["file"].str.split("=").str[3].str.split("-").str[0]

file_counts = files.groupby("root_id").size()
has_all = file_counts[file_counts == 12].index

files["scheme"] = "historical"
files.loc[files["order_by"].notna(), "scheme"] = "clean-and-merge"

files_finished = files.query("root_id in @has_all")
manifest = load_manifest()

# %%


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


def apply_metadata(df, key):
    index_name = df.index.name
    df["root_id"] = key[0]
    df["scheme"] = key[1]
    df["order_by"] = key[2]
    df["random_seed"] = key[3]
    df["order"] = np.arange(len(df))
    df.reset_index(drop=False, inplace=True)
    df.set_index(
        ["root_id", "scheme", "order_by", "random_seed", index_name, "order"],
        inplace=True,
    )
    return df


# %%

client = cc.CAVEclient("minnie65_phase3_v1")
mtypes = load_mtypes(client)

# %%

recompute = True
save = True

root_ids = manifest.index
all_infos = []
all_sequence_features = {}


def process_for_neuron(root_id, rows):
    neuron_infos = []
    neuron_sequence_features = {}

    neuron = load_neuronframe(root_id, client)

    annotate_pre_synapses(neuron, mtypes)
    annotate_mtypes(neuron, mtypes)

    for keys, sub_rows in rows.groupby(
        ["scheme", "order_by", "random_seed"], dropna=False
    ):
        scheme, order_by, random_seed = keys

        if scheme == "clean-and-merge":
            sequence = create_merge_and_clean_sequence(
                neuron, root_id, order_by=order_by, random_seed=random_seed
            )
            sequence = sequence.select_by_bout("has_merge", keep="last")
        elif scheme == "historical":
            sequence = create_time_ordered_sequence(neuron, root_id)
        else:
            raise ValueError(f"Scheme {scheme} not recognized.")

        sequence_key = (root_id, scheme, order_by, random_seed)

        sequence_feature_dfs = {}
        counts_by_mtype = sequence.apply_to_synapses_by_sample(
            compute_target_counts, which="pre", by="post_mtype"
        )
        counts_by_mtype = apply_metadata(counts_by_mtype, sequence_key)
        sequence_feature_dfs["counts_by_mtype"] = counts_by_mtype

        props_by_mtype = sequence.apply_to_synapses_by_sample(
            compute_target_proportions, which="pre", by="post_mtype"
        )
        props_by_mtype = apply_metadata(props_by_mtype, sequence_key)
        sequence_feature_dfs["props_by_mtype"] = props_by_mtype

        spatial_props = sequence.apply_to_synapses_by_sample(
            compute_spatial_target_proportions, which="pre", mtypes=mtypes
        )
        spatial_props = apply_metadata(spatial_props, sequence_key)
        sequence_feature_dfs["spatial_props"] = spatial_props

        spatial_props_by_mtype = sequence.apply_to_synapses_by_sample(
            compute_spatial_target_proportions,
            which="pre",
            mtypes=mtypes,
            by="post_mtype",
        )
        spatial_props_by_mtype = apply_metadata(spatial_props_by_mtype, sequence_key)
        sequence_feature_dfs["spatial_props_by_mtype"] = spatial_props_by_mtype

        neuron_sequence_features[sequence_key] = sequence_feature_dfs

        info = sequence.sequence_info
        info["root_id"] = root_id
        info["scheme"] = scheme
        info["order_by"] = order_by
        info["random_seed"] = random_seed
        neuron_infos.append(
            info.drop(["pre_synapses", "post_synapses", "applied_edits"], axis=1)
        )

    return neuron_sequence_features, neuron_infos


# %%

inputs = [
    (root_id, rows)
    for root_id, rows in files_finished.query("root_id.isin(@root_ids)").groupby(
        "root_id"
    )
]

# %%

inputs = [
    (root_id, rows)
    for root_id, rows in files_finished.query("root_id.isin(@root_ids)").groupby(
        "root_id"
    )
]

results = Parallel(n_jobs=8, verbose=10)(
    delayed(process_for_neuron)(*input) for input in inputs
)

all_sequence_features = {}
all_infos = []
for result in results:
    neuron_sequence_features, neuron_infos = result
    all_sequence_features.update(neuron_sequence_features)
    all_infos.extend(neuron_infos)


# %%

all_infos_df = pd.concat(all_infos)

meta_features_df = pd.DataFrame(all_sequence_features).T
meta_features_df.index.names = ["root_id", "scheme", "order_by", "random_seed"]

if save:
    with open(OUT_PATH / "sequence_metrics" / "all_infos.pkl", "wb") as f:
        pickle.dump(all_infos_df, f)
    with open(OUT_PATH / "sequence_metrics" / "meta_features_df.pkl", "wb") as f:
        pickle.dump(meta_features_df, f)

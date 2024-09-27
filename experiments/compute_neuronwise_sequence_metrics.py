# %%

import pickle

import numpy as np
import pandas as pd
from cloudfiles import CloudFiles
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances

from pkg.constants import OUT_PATH
from pkg.neuronframe import NeuronFrame, load_neuronframe
from pkg.sequence import (
    create_lumped_time_sequence,
    create_merge_and_clean_sequence,
    create_time_ordered_sequence,
)
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
has_all = file_counts[file_counts >= 12].index

# files["scheme"] = "historical"
# files.loc[files["order_by"].notna(), "scheme"] = "clean-and-merge"

files["scheme"] = files["file"].str.split("-").str[-1].str.split(".").str[0].str[:-9]
files['scheme'] = files['scheme'].str.replace('_', '-')
files['scheme'] = files['scheme'].str.replace('merge-and-clean', 'clean-and-merge')
files['scheme'] = files['scheme'].str.replace('time-ordered', 'historical')

files_finished = files.query("root_id in @has_all")
manifest = load_manifest()
manifest = manifest.query("in_inhibitory_column")

print(len(np.intersect1d(files_finished["root_id"].unique(), manifest.index)))

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

    return results


def compute_synapses_in_final(sequence: pd.DataFrame, which="pre"):
    synapses: pd.Series = sequence.sequence_info[f"{which}_synapses"]
    final_synapses = synapses.iloc[-1]

    results = pd.Series(
        index=synapses.index,
        name=f"{which}_synapses_in_final",
    )
    for idx, synapses in synapses.items():
        n_intersection = len(np.intersect1d(final_synapses, synapses))
        results.loc[idx] = n_intersection
    return results.to_frame()


def compute_partners(synapses_df: pd.DataFrame, by=None):
    result = synapses_df["post_pt_root_id"].unique()
    result = pd.Series(index=result, data=1)
    return result


# %%

from pkg.utils import start_client

client = start_client()
mtypes = load_mtypes(client)


# %%

# root_id = 864691134886016762
# neuron = load_neuronframe(root_id, client)
# annotate_pre_synapses(neuron, mtypes)
# annotate_mtypes(neuron, mtypes)


# %%

# sequence = create_time_ordered_sequence(neuron, root_id)
# partner_sequence = sequence.apply_to_synapses_by_sample(compute_partners, which="pre")


# %%
# X = partner_sequence.fillna(0).values

# from scipy.spatial.distance import cdist

# distances = cdist(X, X[-1, :].reshape(1, -1), metric="hamming")

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
        elif scheme == "lumped-time":
            sequence = create_lumped_time_sequence(neuron, root_id)
        elif scheme == "historical":
            sequence = create_time_ordered_sequence(neuron, root_id)
        else:
            raise ValueError(f"Scheme {scheme} not recognized.")

        sequence_key = (root_id, scheme, order_by, random_seed)

        sequence_feature_dfs = {}
        # counts of outputs
        counts_by_mtype = sequence.apply_to_synapses_by_sample(
            compute_target_counts, which="pre", by="post_mtype"
        )
        counts_by_mtype = apply_metadata(counts_by_mtype, sequence_key)
        sequence_feature_dfs["counts_by_mtype"] = counts_by_mtype

        # proportion of outputs by mtype
        props_by_mtype = sequence.apply_to_synapses_by_sample(
            compute_target_proportions, which="pre", by="post_mtype"
        )
        props_by_mtype = apply_metadata(props_by_mtype, sequence_key)
        sequence_feature_dfs["props_by_mtype"] = props_by_mtype

        # spatial props
        spatial_props = sequence.apply_to_synapses_by_sample(
            compute_spatial_target_proportions, which="pre", mtypes=mtypes
        )
        spatial_props = apply_metadata(spatial_props, sequence_key)
        sequence_feature_dfs["spatial_props"] = spatial_props

        # spatial props by mtype
        spatial_props_by_mtype = sequence.apply_to_synapses_by_sample(
            compute_spatial_target_proportions,
            which="pre",
            mtypes=mtypes,
            by="post_mtype",
        )
        spatial_props_by_mtype = apply_metadata(spatial_props_by_mtype, sequence_key)
        sequence_feature_dfs["spatial_props_by_mtype"] = spatial_props_by_mtype

        # partners
        partners = sequence.apply_to_synapses_by_sample(compute_partners, which="pre")
        partners = partners.fillna(0)
        partners = apply_metadata(partners, sequence_key)
        sequence_feature_dfs["partners"] = partners

        # precision recall
        pre_precision_recall = compute_precision_recall(sequence, which="pre")
        pre_precision_recall = apply_metadata(pre_precision_recall, sequence_key)
        post_precision_recall = compute_precision_recall(sequence, which="post")
        post_precision_recall = apply_metadata(post_precision_recall, sequence_key)
        sequence_feature_dfs["pre_precision_recall"] = pre_precision_recall
        sequence_feature_dfs["post_precision_recall"] = post_precision_recall

        # synapses in final
        pre_synapses_in_final = compute_synapses_in_final(sequence, which="pre")
        pre_synapses_in_final = apply_metadata(pre_synapses_in_final, sequence_key)
        post_synapses_in_final = compute_synapses_in_final(sequence, which="post")
        post_synapses_in_final = apply_metadata(post_synapses_in_final, sequence_key)
        sequence_feature_dfs["pre_synapses_in_final"] = pre_synapses_in_final
        sequence_feature_dfs["post_synapses_in_final"] = post_synapses_in_final

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

from tqdm_joblib import tqdm_joblib

with tqdm_joblib(total=len(inputs)) as pbar:
    results = Parallel(n_jobs=8)(
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

#%%
save = True
if save:
    with open(OUT_PATH / "sequence_metrics" / "all_infos.pkl", "wb") as f:
        pickle.dump(all_infos_df, f)
    with open(OUT_PATH / "sequence_metrics" / "meta_features_df.pkl", "wb") as f:
        pickle.dump(meta_features_df, f)

# %%

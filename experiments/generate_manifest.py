# %%

import numpy as np
import pandas as pd
from cloudfiles import CloudFiles

from pkg.constants import (
    COLUMN_MTYPES_TABLE,
    INHIBITORY_CTYPES_TABLE,
    MTYPES_TABLE,
    NUCLEUS_TABLE,
    OUT_PATH,
    PROOFREADING_TABLE,
    TIMESTAMP,
)
from pkg.io import write_variable
from pkg.utils import load_manifest, start_client

client = start_client()

# %%

proofreading_df = client.materialize.query_table(PROOFREADING_TABLE)
inhibitory_column_df = client.materialize.query_table(INHIBITORY_CTYPES_TABLE)
mtypes_df = client.materialize.query_table(MTYPES_TABLE)
mtypes_lookup = mtypes_df.drop_duplicates("pt_root_id").set_index("pt_root_id")[
    "cell_type"
]
column_mtypes_df = client.materialize.query_table(COLUMN_MTYPES_TABLE)
column_mtypes_lookup = column_mtypes_df.drop_duplicates("pt_root_id").set_index(
    "pt_root_id"
)["cell_type"]
nuc_df = client.materialize.query_table(NUCLEUS_TABLE)
nuc_df.drop_duplicates("pt_root_id", inplace=True)

# %%
nuc_lookup = nuc_df.set_index("pt_root_id")["id"]

# # %%

# inhibitory_column_df[
#     ~inhibitory_column_df["target_id"].isin(
#         proofreading_df["pt_root_id"].map(nuc_lookup)
#     )
# ]

# %%
roots = (
    pd.Index(proofreading_df["pt_root_id"].unique())
    .union(pd.Index(inhibitory_column_df["pt_root_id"]))
    .rename("root_id")
)
neuron_manifest = pd.DataFrame(index=roots)

neuron_manifest["in_inhibitory_column"] = False
neuron_manifest.loc[inhibitory_column_df["pt_root_id"], "in_inhibitory_column"] = True

neuron_manifest["in_proofreading_table"] = False
neuron_manifest.loc[proofreading_df["pt_root_id"], "in_proofreading_table"] = True
# %%

# join the two mtype lookup series such that the column mtypes take precedence
# over the mtypes

joined_mtypes_lookup = column_mtypes_lookup.reindex(
    mtypes_lookup.index.union(column_mtypes_lookup.index)
).combine_first(mtypes_lookup)

assert (
    joined_mtypes_lookup.loc[column_mtypes_lookup.index] == column_mtypes_lookup
).all()

# %%
neuron_manifest["mtype"] = joined_mtypes_lookup


# %%
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

# TODO add something about being current

files_finished.to_csv(
    OUT_PATH / "manifest" / "edit_sequences_manifest.csv", index=False
)

# %%
finished_roots = files_finished["root_id"].unique()[:]
n_finished = len(finished_roots)
write_variable(n_finished, "manifest-n_finished")

neuron_manifest["has_all_sequences"] = False
neuron_manifest.loc[
    neuron_manifest.index.intersection(finished_roots), "has_all_sequences"
] = True

# %%

is_current_mask = client.chunkedgraph.is_latest_roots(
    roots.tolist(), timestamp=TIMESTAMP
)
neuron_manifest["is_current"] = is_current_mask

outdated_roots = roots[~is_current_mask]
root_map = dict(zip(roots[is_current_mask], roots[is_current_mask]))
for outdated_root in outdated_roots:
    latest_roots = client.chunkedgraph.get_latest_roots(
        outdated_root, timestamp=TIMESTAMP
    )
    possible_nucs = client.materialize.query_table(
        NUCLEUS_TABLE, filter_in_dict={"pt_root_id": latest_roots}
    )
    if len(possible_nucs) == 1:
        root_map[outdated_root] = possible_nucs.iloc[0]["pt_root_id"]
    elif len(possible_nucs) == 0:
        print(f"No nuc roots for {outdated_root}")
        root_map[outdated_root] = None
    else:
        print(f"Multiple nuc roots for {outdated_root}")

updated_root_options = np.array([root_map[root] for root in roots])
neuron_manifest["current_root_id"] = updated_root_options
neuron_manifest.dropna(subset=["current_root_id"], inplace=True)

# map to nucleus IDs
current_nucs = client.materialize.query_table(
    NUCLEUS_TABLE,
    filter_in_dict={"pt_root_id": updated_root_options},
    # select_columns=["id", "pt_root_id"],
).set_index("pt_root_id")["id"]
neuron_manifest["target_id"] = (
    neuron_manifest["current_root_id"].map(current_nucs).astype("Int64")
)
neuron_manifest.dropna(subset=["target_id"], inplace=True)

neuron_manifest["current_root_id"] = neuron_manifest["current_root_id"].astype(int)
neuron_manifest["target_id"] = neuron_manifest["target_id"].astype(int)

# %%


n_samples = 20

write_variable(n_samples, "manifest-n_samples")

# sample_neurons = (
#     neuron_manifest.query("is_current")
#     .sort_index()
#     .sample(n=n_samples, random_state=8888)
# )
# examples = [
#     864691135082840567,
#     864691135132887456,
#     864691134886016762,
#     864691135213953920,
#     864691135292201142,
#     864691135359413848,
#     864691135502190941,
#     864691135503003997,
#     864691135518510218,
#     864691135561619681,
#     864691135865244030,
#     864691135386650965,
#     864691135660772080,
#     864691135697284250,
#     864691135808473885,
#     864691135919630768,
#     864691135995786154,
#     864691136066728600,
#     864691136618908301,
#     864691136903387826,
# ]
example_targets = [
    264920,
    298930,
    291122,
    271886,
    307066,
    262692,
    262555,
    267293,
    298796,
    255137,
    265035,
    292670,
    260519,
    301085,
    267068,
    260505,
    258281,
    301218,
    258293,
    307264,
]

neuron_manifest = neuron_manifest.reset_index().set_index("target_id")
neuron_manifest["is_sample"] = False
neuron_manifest.loc[example_targets, "is_sample"] = True
neuron_manifest["is_example"] = False
neuron_manifest.loc[example_targets, "is_example"] = True

neuron_manifest = neuron_manifest.reset_index().set_index("root_id")
# TODO add in all of the neuron target id logic here

neuron_manifest.to_csv(OUT_PATH / "manifest" / "neuron_manifest.csv")

# %%
loaded_manifest = load_manifest()

# %%
assert loaded_manifest.equals(neuron_manifest)

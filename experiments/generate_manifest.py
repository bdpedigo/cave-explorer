# %%

import caveclient as cc
import numpy as np
import pandas as pd
from cloudfiles import CloudFiles

from pkg.constants import NUCLEUS_TABLE, OUT_PATH
from pkg.io import write_variable
from pkg.utils import load_manifest

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
roots = files_finished["root_id"].unique()[:]
n_finished = len(roots)
write_variable(n_finished, "manifest-n_finished")
neuron_manifest = pd.DataFrame(index=roots)
neuron_manifest.index.name = "working_root_id"

# %%
client = cc.CAVEclient("minnie65_phase3_v1")

# latest_roots = client.chunkedgraph.get_latest_roots(roots)
# neuron_manifest["latest_root_id"] = latest_roots

# # %%
# current_nucs = client.materialize.query_table(
#     "nucleus_detection_v0",
#     filter_in_dict={"pt_root_id": neuron_manifest.index.tolist()},
# )


# neuron_manifest["target_id"] = neuron_manifest.index.map(
#     current_nucs.set_index("pt_root_id")["id"]
# ).astype("Int64")

is_current_mask = client.chunkedgraph.is_latest_roots(roots)
neuron_manifest["is_current"] = is_current_mask

outdated_roots = roots[~is_current_mask]
root_map = dict(zip(roots[is_current_mask], roots[is_current_mask]))
for outdated_root in outdated_roots:
    latest_roots = client.chunkedgraph.get_latest_roots(outdated_root)
    possible_nucs = client.materialize.query_table(
        NUCLEUS_TABLE, filter_in_dict={"pt_root_id": latest_roots}
    )
    if len(possible_nucs) == 1:
        root_map[outdated_root] = possible_nucs.iloc[0]["pt_root_id"]
    else:
        print(f"Multiple nuc roots for {outdated_root}")

updated_root_options = np.array([root_map[root] for root in roots])
neuron_manifest["current_root_id"] = updated_root_options

# map to nucleus IDs
current_nucs = client.materialize.query_table(
    NUCLEUS_TABLE,
    filter_in_dict={"pt_root_id": updated_root_options},
    # select_columns=["id", "pt_root_id"],
).set_index("pt_root_id")["id"]
neuron_manifest["target_id"] = neuron_manifest["current_root_id"].map(current_nucs)

# %%


n_samples = 20

write_variable(n_samples, "manifest-n_samples")

sample_neurons = (
    neuron_manifest.query("is_current")
    .sort_index()
    .sample(n=n_samples, random_state=8888)
)

neuron_manifest["is_sample"] = False
neuron_manifest.loc[sample_neurons.index, "is_sample"] = True

# TODO add in all of the neuron target id logic here

neuron_manifest.to_csv(OUT_PATH / "manifest" / "neuron_manifest.csv")

# %%
loaded_manifest = load_manifest()

# %%
assert loaded_manifest.equals(neuron_manifest)

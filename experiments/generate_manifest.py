# %%

import pandas as pd
from cloudfiles import CloudFiles

from pkg.io import write_variable
from pkg.paths import OUT_PATH

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
root_ids = files_finished["root_id"].unique()[:]
neuron_manifest = pd.DataFrame(index=root_ids)
neuron_manifest.index.name = "root_id"

# %%
n_samples = 20

write_variable(n_samples, "manifest-n_samples")

sample_neurons = neuron_manifest.sort_index().sample(n=n_samples, random_state=8888)

neuron_manifest["is_sample"] = False
neuron_manifest.loc[sample_neurons.index, "is_sample"] = True

# TODO add in all of the neuron target id logic here

neuron_manifest.to_csv(OUT_PATH / "manifest" / "neuron_manifest.csv")

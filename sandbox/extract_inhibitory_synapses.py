# %%
import os
import time
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

from pkg.utils import load_joint_table, load_mtypes, start_client

client = start_client()

mtypes = load_mtypes(client)

joint_table = load_joint_table()

# %%
coarse_type_cols = joint_table.columns[joint_table.columns.str.contains("coarse_type")]

any_inhib_mask = (joint_table[coarse_type_cols] == "inhibitory").any(axis=1)

inhibitory_root_ids = joint_table[any_inhib_mask].index.sort_values()

# %%

out_path = Path("data/synapse_pull")

files = os.listdir(out_path)

max_chunk_label = -1
pre_synapses = []
for file in tqdm(files):
    if "pre_synapses" in file:
        chunk_pre_synapses = pd.read_csv(out_path / file)
        pre_synapses.append(chunk_pre_synapses)
    chunk_label = int(file.split("_")[0])
    if chunk_label > max_chunk_label:
        max_chunk_label = chunk_label

pre_synapses = pd.concat(pre_synapses)

post_synapses = []
for file in tqdm(files):
    if "post_synapses" in file:
        chunk_post_synapses = pd.read_csv(out_path / file)
        post_synapses.append(chunk_post_synapses)
    chunk_label = int(file.split("_")[0])
    if chunk_label > max_chunk_label:
        max_chunk_label = chunk_label

post_synapses = pd.concat(post_synapses)

pre_synapses["post_mtype"] = pre_synapses["post_pt_root_id"].map(mtypes["cell_type"])

pre_synapses = pre_synapses.query("pre_pt_root_id != post_pt_root_id")
post_synapses = post_synapses.query("pre_pt_root_id != post_pt_root_id")

# %%

represented_ids = pre_synapses["pre_pt_root_id"].unique()
represented_ids = np.union1d(represented_ids, post_synapses["post_pt_root_id"].unique())

missing_root_ids = np.setdiff1d(inhibitory_root_ids, represented_ids)


# %%

chunk_size = 50

inhibitory_root_chunks = np.array_split(
    missing_root_ids, np.ceil(len(missing_root_ids) / chunk_size)
)

# %%

out_path = Path("data/synapse_pull")


def query_synapses_for_chunk(chunk_ids, chunk_label, n_attempts=4):
    client = start_client()

    while n_attempts > 0:
        try:
            pre_synapses = client.materialize.query_table(
                "synapses_pni_2",
                filter_in_dict={
                    "pre_pt_root_id": chunk_ids,
                },
                select_columns=[
                    "id",
                    "size",
                    "pre_pt_supervoxel_id",
                    "post_pt_supervoxel_id",
                    "pre_pt_root_id",
                    "post_pt_root_id",
                    "pre_pt_position",
                    "post_pt_position",
                    "ctr_pt_position",
                ],
            )
            pre_synapses.to_csv(
                out_path / f"{chunk_label}_pre_synapses.csv", index=False
            )
            break
        except Exception:
            sleep(10)
            n_attempts -= 1
            continue

    while n_attempts > 0:
        try:
            post_synapses = client.materialize.query_table(
                "synapses_pni_2",
                filter_in_dict={
                    "post_pt_root_id": chunk_ids,
                },
                select_columns=[
                    "id",
                    "size",
                    "pre_pt_supervoxel_id",
                    "post_pt_supervoxel_id",
                    "pre_pt_root_id",
                    "post_pt_root_id",
                    "pre_pt_position",
                    "post_pt_position",
                    "ctr_pt_position",
                ],
            )
            post_synapses.to_csv(
                out_path / f"{chunk_label}_post_synapses.csv", index=False
            )
            break
        except Exception:
            sleep(10)
            n_attempts -= 1
            continue

    return pre_synapses, post_synapses


# %%

currtime = time.time()

with tqdm_joblib(total=len(inhibitory_root_chunks)) as pbar:
    synapses_by_chunk = Parallel(n_jobs=-1, verbose=True)(
        delayed(query_synapses_for_chunk)(chunk_ids, i + max_chunk_label + 1)
        for i, chunk_ids in enumerate(inhibitory_root_chunks[:])
    )
print(f"{time.time() - currtime:.3f} seconds elapsed.")

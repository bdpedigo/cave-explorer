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

inhibitory_root_ids = joint_table[any_inhib_mask]["pt_root_id"].sort_values()

inhibitory_root_ids.is_unique

# %%

out_path = Path("data/state_prediction_info")


def load_pre_post_synapses():
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

    pre_synapses["post_mtype"] = pre_synapses["post_pt_root_id"].map(
        mtypes["cell_type"]
    )

    pre_synapses = pre_synapses.query("pre_pt_root_id != post_pt_root_id")
    post_synapses = post_synapses.query("pre_pt_root_id != post_pt_root_id")

    return pre_synapses, post_synapses, max_chunk_label


pre_synapses, post_synapses, max_chunk_label = load_pre_post_synapses()

# %%

represented_ids = pre_synapses["pre_pt_root_id"].unique()
represented_ids = np.union1d(represented_ids, post_synapses["post_pt_root_id"].unique())

missing_root_ids = np.setdiff1d(inhibitory_root_ids, represented_ids)


# %%

chunk_size = 50

if len(missing_root_ids) > 0:
    inhibitory_root_chunks = np.array_split(
        missing_root_ids, np.ceil(len(missing_root_ids) / chunk_size)
    )
else:
    inhibitory_root_chunks = []

# %%


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

# %%

pre_synapses, post_synapses, _ = load_pre_post_synapses()

n_pre_synapses = pre_synapses.groupby("pre_pt_root_id").size().fillna(0)
n_post_synapses = post_synapses.groupby("post_pt_root_id").size().fillna(0)

inhib_features = pd.DataFrame(
    {
        "n_pre_synapses": n_pre_synapses,
        "n_post_synapses": n_post_synapses,
    }
)
inhib_features = inhib_features.reindex(inhibitory_root_ids).fillna(0).astype(int)

inhib_features.index.is_unique


# %%


def get_n_leaves_for_root(root_id):
    return len(client.chunkedgraph.get_leaves(root_id, stop_layer=2))


if os.path.exists(out_path / "n_leaves.csv"):
    n_leaves = pd.read_csv(out_path / "n_leaves.csv", index_col=0)["n_nodes"]
    n_leaves.name = "n_nodes"
else:
    n_leaves = pd.Series(name="n_nodes")
    n_leaves.index.name = "root_id"

missing_root_ids = np.setdiff1d(inhibitory_root_ids, n_leaves.index)

with tqdm_joblib(desc="Getting n_leaves", total=len(missing_root_ids)) as progress:
    new_n_leaves = Parallel(n_jobs=-1)(
        delayed(get_n_leaves_for_root)(root_id) for root_id in missing_root_ids
    )
new_n_leaves = pd.Series(new_n_leaves, index=missing_root_ids, name="n_nodes")
new_n_leaves.index.name = "root_id"

n_leaves = pd.concat([n_leaves, new_n_leaves], axis=0)

n_leaves = n_leaves.loc[inhib_features.index]
n_leaves.to_csv(out_path / "n_leaves.csv")

inhib_features = inhib_features.join(n_leaves)

inhib_features.index.is_unique


# %%


def get_change_info(root_id: int, n_tries: int = 3) -> dict:
    try:
        out = client.chunkedgraph.get_change_log(root_id, filtered=True)
        out.pop("operations_ids")
        out.pop("past_ids")
        out.pop("user_info")
        return out
    except Exception:
        if n_tries > 0:
            return get_change_info(root_id, n_tries - 1)
        else:
            return {
                "n_mergers": 0,
                "n_splits": 0,
            }


if os.path.exists(out_path / "change_info.csv"):
    changes_df = pd.read_csv(out_path / "change_info.csv", index_col=0).dropna()
    changes_df["n_operations"] = changes_df["n_mergers"] + changes_df["n_splits"]
else:
    changes_df = pd.DataFrame()

missing_root_ids = np.setdiff1d(inhibitory_root_ids, changes_df.index)

# missing_root_ids = np.setdiff1d(
#     missing_root_ids,
#     [864691135463752254],
# )

if len(missing_root_ids) > 0:
    with tqdm_joblib(
        desc="Getting change info", total=len(missing_root_ids)
    ) as progress:
        changelog_infos = Parallel(n_jobs=-1)(
            delayed(get_change_info)(root_id) for root_id in missing_root_ids
        )
    new_changes_df = pd.DataFrame(changelog_infos, index=missing_root_ids)
    new_changes_df["n_operations"] = (
        new_changes_df["n_mergers"] + new_changes_df["n_splits"]
    )

    changes_df = pd.concat([changes_df, new_changes_df])
    changes_df.to_csv(out_path / "change_info.csv")

inhib_features = inhib_features.join(changes_df)

# %%
inhib_features = inhib_features.dropna().sort_index()
inhib_features = inhib_features.astype(int)

# %%
inhib_features.to_csv(out_path / "inhibitory_features.csv")

#%%
inhib_features
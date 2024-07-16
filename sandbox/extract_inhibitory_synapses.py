# %%
import time
from pathlib import Path

import numpy as np
from caveclient import CAVEclient
from joblib import Parallel, delayed

from pkg.utils import load_mtypes, send_message, start_client

client = start_client()

mtypes = load_mtypes(client)

# %%
inhibitory_root_ids = mtypes.query("classification_system == 'inhibitory_neuron'").index

# %%

chunk_size = 50

inhibitory_root_chunks = np.array_split(
    inhibitory_root_ids, np.ceil(len(inhibitory_root_ids) / chunk_size)
)

# %%

out_path = Path("data/synapse_pull")


def query_synapses_for_chunk(chunk_ids, chunk_label, n_attempts=3):
    client = CAVEclient("minnie65_phase3_v1")

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
            n_attempts -= 1
            continue

    if chunk_label % 10 == 0:
        send_message(f"Finished chunk {chunk_label}.")

    return pre_synapses, post_synapses


currtime = time.time()

synapses_by_chunk = Parallel(n_jobs=8, verbose=True)(
    delayed(query_synapses_for_chunk)(chunk_ids, i)
    for i, chunk_ids in enumerate(inhibitory_root_chunks[:])
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%

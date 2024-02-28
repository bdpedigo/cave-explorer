# %%
import time
from datetime import datetime

import caveclient as cc
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

client = cc.CAVEclient("minnie65_phase3_v1")


def collect_all_operations(op_ids, cg, use_steps=True):
    # op_ids: list or arraylike of operation ids to iterate through
    # cg: client.chunkedgraph for the given cave client
    # use_steps: break operation list into smaller steps to iterate through

    if use_steps:
        # Create slice list to iterate through operations log
        all_op_ids = op_ids.copy()
        step_size = 500
        len_data = len(all_op_ids)

        # iterate through slize list
        slice_list = []
        n_steps = np.floor(len_data / step_size).astype(int)
        for rr in range(n_steps):
            slice_list.append(slice(rr * step_size, (rr + 1) * step_size, 1))

        slice_list.append(slice((rr + 1) * step_size, len_data, 1))

        # For each iteration
        df_list = []
        for ss in range(0, len(slice_list)):
            op_ids = all_op_ids[slice_list[ss]].astype(int)

            op_df, _ = accumulate_operations(op_ids, cg)

            df_list.append(op_df)

            # Convert to dataframe
            operation_log_all = pd.concat(df_list)

    else:
        op_df, _ = accumulate_operations(op_ids, cg)

        # Convert to dataframe
        operation_log_all = op_df.copy()

    return operation_log_all


cg = client.chunkedgraph

op_ids = np.arange(0, 500)




# op_ids: list or arraylike of operation ids to iterate through
# cg: client.chunkedgraph for the given cave client


currtime = time.time()
# Get operations from chunked graph
op_dict = cg.get_operation_details(op_ids)
operations = pd.DataFrame(op_dict).T

print(f"{time.time() - currtime:.3f} seconds elapsed.")

operations

#%%
# Turn dict of dict into dataframe
dict_list = []
troublesome_ids = []
for op in tqdm(op_ids):
    if str(op) not in op_dict:
        troublesome_ids.append(op)
    else:
        inner_dict = op_dict[str(op)]

        iso_timestamp = inner_dict["timestamp"]
        timestamp = int(datetime.fromisoformat(iso_timestamp).timestamp() * 1000)
        if "added_edges" in inner_dict:
            is_merge = True
        else:
            is_merge = False

        user_id = inner_dict["user"]
        after_root_ids = inner_dict["roots"]

        loop_dict = {
            "operation_id": op,
            "timestamp": timestamp,
            "user_id": user_id,
            "after_root_ids": after_root_ids,
            "is_merge": is_merge,
            "date": iso_timestamp[:16],
        }
        dict_list.append(loop_dict)


print(f"{time.time() - currtime:.3f} seconds elapsed.")

#%%
pd.DataFrame(dict_list)

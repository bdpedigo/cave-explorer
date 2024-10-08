# %%
import pandas as pd
from caveclient import CAVEclient

client = CAVEclient("minnie65_phase3_v1")

root_id = 864691135639556411

change_log = client.chunkedgraph.get_tabular_change_log(root_id)[root_id]

change_log = pd.DataFrame(change_log).set_index("operation_id")

# %%
merge_id = change_log.index[0]
split_id = change_log.index[1]

# %%
import datetime

import numpy as np
import pytz

details = client.chunkedgraph.get_operation_details([merge_id, split_id])

merge_details = details[str(merge_id)]
added_edges = details[str(merge_id)]["added_edges"]
nodes_added = np.unique(np.concatenate([list(edge) for edge in added_edges]))
merge_row = change_log.loc[merge_id]
x = merge_row["timestamp"]
timestamp = datetime.datetime.fromtimestamp(x / 1000, pytz.UTC)
delta = datetime.timedelta(microseconds=1)

pre_l2_ids = client.chunkedgraph.get_roots(
    nodes_added, stop_layer=2, timestamp=timestamp - delta
)
post_l2_ids = client.chunkedgraph.get_roots(
    nodes_added, stop_layer=2, timestamp=timestamp + delta
)


# %%

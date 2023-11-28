# %%
import caveclient as cc

client = cc.CAVEclient("minnie65_phase3_v1")

root_id = 864691135915450982
out = client.chunkedgraph.get_tabular_change_log(root_id)[root_id]

# %%
out["timestamp"]

import datetime
import pytz

# datetime.datetime.fromisoformat(out["timestamp"][0])

ts = out["timestamp"][0]

datetime.datetime.utcfromtimestamp(ts / 1000)
# %%
operation_id = 22342
client.chunkedgraph.get_operation_details([operation_id])

# %%
client.chunkedgraph.get_root_timestamps([root_id])

# %%
leaves = client.chunkedgraph.get_leaves(root_id)
# %%
client.chunkedgraph.get_root_id(int(leaves[0]))

# %%
client.chunkedgraph.get_roots(leaves)
# %%
import datetime

client.chunkedgraph.get_user_operations("161", datetime.datetime(2021, 1, 1))

# %%
client.chunkedgraph.get_children(root_id)

# %%
client.chunkedgraph.get_lineage_graph(root_id, as_nx_graph=True)

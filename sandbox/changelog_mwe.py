# %%
import caveclient as cc
import pandas as pd

client = cc.CAVEclient("minnie65_phase3_v1")

cg = client.chunkedgraph

root_id = 864691136143786292

change_log = cg.get_tabular_change_log(root_id)[root_id]
change_log.set_index("operation_id", inplace=True)

splits = change_log.query("~is_merge")

splits["after_root_ids"].head()

# %%
details = cg.get_operation_details(splits.index.to_list())
details = pd.DataFrame(details).T
details["roots"].head()

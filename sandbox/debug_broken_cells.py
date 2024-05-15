# %%
import caveclient as cc
from tqdm.auto import tqdm

from pkg.neuronframe import load_neuronframe

client = cc.CAVEclient("minnie65_phase3_v1")
# %%
client.materialize.get_tables()

# %%
ctype_table = client.materialize.query_table("connectivity_groups_v795")
mtype_table = client.materialize.query_table("allen_column_mtypes_v2")

itc_table = mtype_table.query('cell_type == "ITC"')

# %%

from pkg.sequence import create_time_ordered_sequence

missing = 0
for root_id in tqdm(itc_table["pt_root_id"]):
    neuron = load_neuronframe(root_id, client=client, only_load=True)
    sequence = create_time_ordered_sequence(neuron, root_id, only_load=True)
    if sequence is None or isinstance(neuron, (str)):
        missing += 1


# %%
import caveclient as cc
from tqdm.auto import tqdm

from pkg.neuronframe import load_neuronframe
from pkg.sequence import create_time_ordered_sequence

client = cc.CAVEclient("minnie65_phase3_v1")
# %%
client.materialize.get_tables()

# %%
ctype_table = client.materialize.query_table("connectivity_groups_v795")
mtype_table = client.materialize.query_table("allen_column_mtypes_v2")

itc_table = mtype_table.query('cell_type == "ITC"')

# %%

for root_id in tqdm(itc_table["pt_root_id"]):
    neuron = load_neuronframe(root_id, client=client)
    if isinstance(neuron, str): 
        neuron = load_neuronframe(root_id, client=client, use_cache=False)
    sequence = create_time_ordered_sequence(neuron, root_id)

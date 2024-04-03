# %%

import caveclient as cc

from pkg.neuronframe import load_neuronframe
from pkg.sequence import create_merge_and_clean_sequence
from pkg.utils import load_manifest

manifest = load_manifest()

client = cc.CAVEclient("minnie65_phase3_v1")

# %%
for root_id in manifest.query("is_sample").index:
    neuron = load_neuronframe(root_id, client)
    sequence = create_merge_and_clean_sequence(neuron, root_id)
    break

#%% 


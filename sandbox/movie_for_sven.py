# %%
from caveclient import CAVEclient

from pkg.neuronframe import load_neuronframe
from pkg.sequence import create_time_ordered_sequence

client = CAVEclient("minnie65_phase3_v1")

use_cache = False
root_id = 864691136137805181
nf = load_neuronframe(root_id, client, use_cache=use_cache)
sequence = create_time_ordered_sequence(nf, root_id, use_cache=use_cache)

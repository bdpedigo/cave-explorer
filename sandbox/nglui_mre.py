#%%
import caveclient as cc
from nglui.statebuilder import make_neuron_neuroglancer_link
import pandas as pd 

client = cc.CAVEclient("minnie65_phase3_v1")

# works
make_neuron_neuroglancer_link(client, 864691135941359220)

timestamp = pd.to_datetime("2021-07-01 00:00:00", utc=True)

# breaks
make_neuron_neuroglancer_link(client, 864691135941359220, show_inputs=True, timestamp=timestamp)

# also breaks
make_neuron_neuroglancer_link(client, 864691135941359220, show_outputs=True, timestamp=timestamp)

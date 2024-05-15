# %%
import caveclient as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from pkg.neuronframe import load_neuronframe
from pkg.utils import load_manifest

manifest = load_manifest()

client = cc.CAVEclient("minnie65_phase3_v1")

# %%
manifest.query("in_inhibitory_column & is_current", inplace=True)

# %%


all_edits = []
all_metaedits = []
rows = []
for root_id in tqdm(manifest.index[:]):
    neuron = load_neuronframe(root_id, client)
    edited_neuron = neuron.set_edits(neuron.edits.index)
    edited_neuron.select_nucleus_component(inplace=True)
    edited_neuron.apply_edge_lengths(inplace=True)
    all_edits.append(neuron.edits)
    all_metaedits.append(neuron.metaedits)
    rows.append(
        {
            "root_id": root_id,
            "n_edges_unedited": len(neuron.edges),
            "n_nodes_unedited": len(neuron.nodes),
            "n_edits": len(neuron.edits),
            "n_metaedits": len(neuron.metaedits),
            "n_merges": len(neuron.edits.query("is_merge")),
            "n_splits": len(neuron.edits.query("~is_merge")),
            "edge_length_sum": edited_neuron.edges["length"].sum(),
            "n_nodes": len(edited_neuron.nodes),
            "n_edges": len(edited_neuron.edges),
        }
    )
summary_info = pd.DataFrame(rows).set_index("root_id")

# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.histplot(summary_info["n_edits"], ax=ax)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.histplot(summary_info["n_merges"], ax=ax)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.histplot(summary_info["n_splits"], ax=ax)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.histplot(summary_info["n_metaedits"], ax=ax)

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.scatterplot(data=summary_info, x="n_nodes_unedited", y="n_edits", ax=ax)

#%%

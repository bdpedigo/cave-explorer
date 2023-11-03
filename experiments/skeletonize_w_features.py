# %%

import time

t0 = time.time()

import os

import matplotlib.pyplot as plt
from pkg.edits import (
    apply_edit,
    get_initial_network,
    lazy_load_network_edits,
)
from pkg.morphology import (
    apply_compartments,
    apply_synapses,
    get_pre_post_synapses,
    get_soma_point,
    map_synapses,
    skeletonize_networkframe,
)
from pkg.plot import clean_axis
from tqdm.autonotebook import tqdm

import caveclient as cc
from meshparty import meshwork
from pcg_skel import features
from skeleton_plot.plot_tools import plot_mw_skel

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

# %%
meta = client.materialize.query_table("allen_v1_column_types_slanted_ref")
meta = meta.sort_values("target_id")
nuc = client.materialize.query_table("nucleus_detection_v0").set_index("id")

synapse_table = client.info.get_datastack_info()["synapse_table"]

# %%

os.environ["SKEDITS_USE_CLOUD"] = "False"
os.environ["SKEDITS_RECOMPUTE"] = "False"

# %%
i = 7
target_id = meta.iloc[i]["target_id"]
root_id = nuc.loc[target_id]["pt_root_id"]
root_id = client.chunkedgraph.get_latest_roots(root_id)[0]

# %%
networkdeltas_by_operation, networkdeltas_by_meta_operation = lazy_load_network_edits(
    root_id, client
)

# %%


nf = get_initial_network(root_id, client, positions=True)

# %%
for metaedit_id, metaedit in tqdm(
    networkdeltas_by_meta_operation.items(), desc="Playing meta-edits"
):
    apply_edit(nf, metaedit)


soma_point = get_soma_point(root_id, client)
skeleton, mesh, l2dict_mesh, l2dict_r_mesh = skeletonize_networkframe(
    nf, client, soma_pt=soma_point
)

nrn = meshwork.Meshwork(mesh, seg_id=root_id, skeleton=skeleton)
features.add_lvl2_ids(nrn, l2dict_mesh)

plot_mw_skel(nrn, plot_postsyn=False, plot_presyn=False, plot_soma=True)

# %%

from pkg.edits.changes import get_level2_lineage_components
from pkg.morphology.synapses import map_synapse_level2_ids

pre_synapses, post_synapses = get_pre_post_synapses(root_id, client)

level2_lineage_component_map = get_level2_lineage_components(networkdeltas_by_operation)

# %%
import pandas as pd

synapses = post_synapses
side = "post"
verbose = True



# %%
for idx, row in pre_synapses.iterrows():
    current_l2 = row["pre_pt_current_level2_id"]

    # this means that that level2 node was never part of a merge or split
    # therefore we can safely keep the current mapping from supervoxel to level2 ID
    if current_l2 in level2_lineage_component_map:
        print(idx)
        break

# %%

side = "pre"
idx = 36
supervoxel = pre_synapses.loc[idx][f"{side}_pt_supervoxel_id"]
current_l2 = client.chunkedgraph.get_roots([supervoxel], stop_layer=2)[0]
current_l2 in level2_lineage_component_map.index
component = level2_lineage_component_map[current_l2]

candidate_level2_ids = level2_lineage_component_map[
    level2_lineage_component_map == component
].index

candidate_level2_ids = candidate_level2_ids[candidate_level2_ids.isin(nf.nodes.index)]

winning_level2_id = None
for candidate_level2_id in candidate_level2_ids:
    candidate_supervoxels = client.chunkedgraph.get_children(candidate_level2_id)
    if supervoxel in candidate_supervoxels:
        winning_level2_id = candidate_level2_id
        break

# %%
map_synapse_level2_ids(pre_synapses, level2_lineage_component_map, "pre", client)

# %%
pre_synapses, post_synapses = map_synapses(
    pre_synapses, post_synapses, networkdeltas_by_operation, l2dict_mesh, client
)
apply_synapses(nrn, pre_synapses, post_synapses)


fig, axs = plt.subplots(
    1,
    2,
    figsize=(8, 5),
    sharex=True,
    sharey=True,
    gridspec_kw=dict(hspace=0, wspace=0),
)

plot_mw_skel(nrn, plot_postsyn=True, plot_presyn=True, plot_soma=True, ax=axs[0])

apply_compartments(nrn, root_id, client)

plot_mw_skel(
    nrn,
    plot_postsyn=True,
    plot_presyn=True,
    plot_soma=True,
    pull_compartment_colors=True,
    ax=axs[1],
)


clean_axis(axs[0])
clean_axis(axs[1])

axs[0].autoscale()

axs[0].plot(
    [1.1, 1.1],
    [0.1, 0.9],
    color="darkgrey",
    linestyle="-",
    linewidth=2,
    clip_on=False,
    transform=axs[0].transAxes,
)

axs[0].set_title("Before compartment\nlabeling/masking")
axs[1].set_title("After compartment\nlabeling/masking")

# %%

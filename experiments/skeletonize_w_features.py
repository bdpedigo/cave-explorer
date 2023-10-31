# %%

import time

t0 = time.time()

import os

import caveclient as cc
from meshparty import meshwork
from pkg.edits import (
    apply_edit,
    get_initial_network,
    lazy_load_network_edits,
    lazy_load_supervoxel_level2_map,
)
from pkg.morphology import (
    apply_synapses_to_meshwork,
    get_pre_post_synapses,
    get_soma_point,
    map_synapses,
    skeletonize_networkframe,
)
from tqdm.autonotebook import tqdm

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
i = 6
target_id = meta.iloc[i]["target_id"]
root_id = nuc.loc[target_id]["pt_root_id"]
root_id = client.chunkedgraph.get_latest_roots(root_id)[0]

# %%
networkdeltas_by_operation, networkdeltas_by_meta_operation = lazy_load_network_edits(
    root_id, client
)

# %%

nf = get_initial_network(root_id, client, positions=True)

for metaedit_id, metaedit in tqdm(
    networkdeltas_by_meta_operation.items(), desc="Playing meta-edits"
):
    apply_edit(nf, metaedit)


soma_point = get_soma_point(root_id, client)
skeleton, mesh, l2dict_mesh, l2dict_r_mesh = skeletonize_networkframe(
    nf, soma_pt=soma_point
)
nrn = meshwork.Meshwork(mesh, seg_id=root_id, skeleton=skeleton)

plot_mw_skel(nrn, plot_postsyn=False, plot_presyn=False, plot_soma=True)

# %%

supervoxel_level2_map = lazy_load_supervoxel_level2_map(root_id, client)

pre_synapses, post_synapses = get_pre_post_synapses(root_id, client)
pre_synapses, post_synapses = map_synapses(
    pre_synapses, post_synapses, supervoxel_level2_map, l2dict_mesh
)
apply_synapses_to_meshwork(nrn, pre_synapses, post_synapses)

plot_mw_skel(nrn, plot_postsyn=True, plot_presyn=True, plot_soma=True)

# %%
is_axon, split_quality = meshwork.algorithms.split_axon_by_annotation(
    nrn, "pre_syn", "post_syn"
)
print(is_axon)
print(split_quality)
nrn.anno.add_annotations("is_axon", is_axon, mask=True, overwrite=True)
print(is_axon)


plot_mw_skel(nrn, plot_postsyn=True, plot_presyn=True, plot_soma=True)

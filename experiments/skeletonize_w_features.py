# %%

import time

t0 = time.time()

import os

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

import caveclient as cc
from meshparty import meshwork
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

from pcg_skel import features

nf = get_initial_network(root_id, client, positions=True)

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


# %%

plot_mw_skel(nrn, plot_postsyn=False, plot_presyn=False, plot_soma=True)

# %%

supervoxel_level2_map = lazy_load_supervoxel_level2_map(
    root_id, networkdeltas_by_operation, client
)

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

nrn.anno.add_annotations("is_axon", is_axon, mask=True, overwrite=True)

plot_mw_skel(nrn, plot_postsyn=True, plot_presyn=True, plot_soma=True)

# %%

# compartment_labels = np.array([2]*len(nrn.skeleton.node_mask))
# compartment_labels[nrn.skeleton.node_mask] = 3
# compartment_labels[og_nrn.skeleton.root] = 1


# create compartment labels
# compartment_labels = np.zeros(len(nrn.skeleton.vertices))
# compartment_labels[nrn.anno.apical_mesh_labels.skel_index] = 4
# compartment_labels[nrn.anno.basal_mesh_labels.skel_index] = 3
# compartment_labels[nrn.skeleton.root] = 1
# compartment_labels[nrn.anno.is_axon.skel_index] = 2

# %%
nrn.skeleton.root

# %%
axon_indices = nrn.anno["is_axon"]["mesh_index_filt"]

# %%
import joblib

import apical_classifier.apical_model_utils as amu

apical_model_dir = "apical_classifier/models"
# add apical labels and remove axons with peel back
# load up the models for apical annotation
rfc = joblib.load(f"{apical_model_dir}/point_model_current.pkl.gz")
feature_cols = joblib.load(f"{apical_model_dir}/feature_cols_current.pkl")
branch_params = joblib.load(f"{apical_model_dir}/branch_params_current.pkl")
# create branch classifier
BranchClassifier = amu.BranchClassifierFactory(rfc, feature_cols)
branch_classifier = BranchClassifier(**branch_params)

# %%
import pcg_skel

pcg_skel.features.add_volumetric_properties(nrn, client, overwrite=True)
pcg_skel.features.add_segment_properties(nrn, overwrite=True)

# %%

from pkg.morphology import get_soma_row

row = get_soma_row(root_id, client)
nrn.anno.add_annotations(name="soma_row", data=row, overwrite=True, anchored=False)

# %%

import numpy as np

axon_m1 = "axon_id/axon_id/ml_models/new_models/rf1.joblib"
axon_m2 = "axon_id/axon_id/ml_models/new_models/rf2.joblib"


def unmasked_to_masked(nrn, inds):
    """
    converts nodes from masked skeleton indices to unmasked skeleton indices

    """
    unmasked_inds = []
    for ind in inds:
        unmasked_inds.append(np.where(nrn.skeleton.indices_unmasked == ind)[0][0])

    return unmasked_inds


mask_out_ax = False

# generate apical features (also removes axons)
point_features_df, axons_remaining_unmasked = amu.peel_axon_id_apical(
    nrn, axon_m1, axon_m2, mask_out_ax=mask_out_ax
)


# %%
branch_df = branch_classifier.fit_predict_data(point_features_df, "base_skind")
branch_df["masked_base_skind"] = unmasked_to_masked(nrn, branch_df["base_skind"])
# add apical features to annotations
nrn.anno.add_annotations(
    name="apical_segments", data=branch_df, overwrite=True, anchored=False
)

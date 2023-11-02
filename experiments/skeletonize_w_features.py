# %%

import time

t0 = time.time()

import os

import joblib
import numpy as np
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
    get_soma_row,
    map_synapses,
    skeletonize_networkframe,
)
from tqdm.autonotebook import tqdm

import apical_classifier.apical_model_utils as amu
import caveclient as cc
import pcg_skel
from apical_classifier.apical_features import generate_apical_features
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
    nf, client, soma_pt=soma_point
)

# %%
nrn = meshwork.Meshwork(mesh, seg_id=root_id, skeleton=skeleton)
features.add_lvl2_ids(nrn, l2dict_mesh)


plot_mw_skel(nrn, plot_postsyn=False, plot_presyn=False, plot_soma=True)


supervoxel_level2_map = lazy_load_supervoxel_level2_map(
    root_id, networkdeltas_by_operation, client
)

pre_synapses, post_synapses = get_pre_post_synapses(root_id, client)
pre_synapses, post_synapses = map_synapses(
    pre_synapses, post_synapses, supervoxel_level2_map, l2dict_mesh
)
apply_synapses_to_meshwork(nrn, pre_synapses, post_synapses)

plot_mw_skel(nrn, plot_postsyn=True, plot_presyn=True, plot_soma=True)


is_axon, split_quality = meshwork.algorithms.split_axon_by_annotation(
    nrn, "pre_syn", "post_syn"
)

nrn.anno.add_annotations("is_axon", is_axon, mask=True, overwrite=True)

plot_mw_skel(nrn, plot_postsyn=True, plot_presyn=True, plot_soma=True)


apical_model_dir = "apical_classifier/models"
# add apical labels and remove axons with peel back
# load up the models for apical annotation
rfc = joblib.load(f"{apical_model_dir}/point_model_current.pkl.gz")
feature_cols = joblib.load(f"{apical_model_dir}/feature_cols_current.pkl")
branch_params = joblib.load(f"{apical_model_dir}/branch_params_current.pkl")
# create branch classifier
BranchClassifier = amu.BranchClassifierFactory(rfc, feature_cols)
branch_classifier = BranchClassifier(**branch_params)


pcg_skel.features.add_volumetric_properties(nrn, client, overwrite=True)
pcg_skel.features.add_segment_properties(nrn, overwrite=True)
row = get_soma_row(root_id, client)
nrn.anno.add_annotations(name="soma_row", data=row, overwrite=True, anchored=False)


def unmasked_to_masked(nrn, inds):
    """
    converts nodes from masked skeleton indices to unmasked skeleton indices

    """
    unmasked_inds = []
    for ind in inds:
        unmasked_inds.append(np.where(nrn.skeleton.indices_unmasked == ind)[0][0])

    return unmasked_inds


axon_m1 = "axon_id/axon_id/ml_models/new_models/rf1.joblib"
axon_m2 = "axon_id/axon_id/ml_models/new_models/rf2.joblib"

# TODO this was not working, some error in numpy
# generate apical features (also removes axons)
mask_out_ax = True
point_features_df, axons_remaining_unmasked = amu.peel_axon_id_apical(
    nrn, axon_m1, axon_m2, mask_out_ax=mask_out_ax
)

# generate apical features
point_features_df = generate_apical_features(nrn)

branch_df = branch_classifier.fit_predict_data(point_features_df, "base_skind")
branch_df["masked_base_skind"] = unmasked_to_masked(nrn, branch_df["base_skind"])
# add apical features to annotations
nrn.anno.add_annotations(
    name="apical_segments", data=branch_df, overwrite=True, anchored=False
)


import pandas as pd


def flatten_list(nested_lst):
    return [item for sublist in nested_lst for item in sublist]


def axon_segment_df(nrn, segments):
    """
    returns a dataframe of remaining axon segments to be added to a neuron as an annotation

    """
    # rewrite to use mesh from beginning
    mesh_idx = np.arange(nrn.mesh.n_vertices)
    skel_segment_nodes_masked = flatten_list(
        [unmasked_to_masked(nrn, seg) for seg in segments]
    )
    classified_axon = np.array([False] * len(mesh_idx))
    if len(skel_segment_nodes_masked) > 0:
        remaining_axon_mesh_mask = nrn.SkeletonIndex(
            skel_segment_nodes_masked
        ).to_mesh_index
        classified_axon[remaining_axon_mesh_mask] = True

    # put in df
    remaining_axon = pd.DataFrame()
    remaining_axon["mesh_index"] = mesh_idx
    remaining_axon["classified_axon"] = classified_axon

    """masked_skel_vertices = list(range(len(nrn.skeleton.vertices)))
    classified_axon = np.array([False]*len(masked_skel_vertices))
    # get masked nodes that are classified axon
    segment_nodes_masked = flatten_list([unmasked_to_masked(nrn, seg) for seg in segments])
    classified_axon[segment_nodes_masked] = True

    # get mesh indices 
    skel_nodes_to_mesh_index = nrn.SkeletonIndex(segment_nodes_masked).to_mesh_region
    # put in df
    remaining_axon = pd.DataFrame()
    remaining_axon['masked_skel_vertices'] = masked_skel_vertices
    remaining_axon['mesh_index'] = skel_nodes_to_mesh_index
    remaining_axon['classified_axon'] = classified_axon"""

    return remaining_axon


def make_node_labels(nrn, apical_labels=True):
    """
    takes a neuron (with annotations indicating radius and apical)
    and returns node labels (radius and compartment type) for swc
    """
    branch_df = nrn.anno.apical_segments.df
    # now get apical nodes - to get correct downstream, we need to convert to unmasked
    apical_initial_nodes = unmasked_to_masked(
        nrn, branch_df[branch_df["is_apical"] == True]["base_skind"].tolist()
    )

    # now get the downstream nodes for this
    apical_nodes = []
    if apical_labels:
        for node in apical_initial_nodes:
            apical_nodes.append(nrn.skeleton.downstream_nodes(node).tolist())
        apical_nodes = np.array(flatten_list(apical_nodes))

    # ok now create compartment list and fill
    compartment_labels = np.array([0] * len(nrn.skeleton.vertices))
    # fill apical labels

    if len(apical_nodes) > 0 and apical_labels:
        compartment_labels[apical_nodes] = 4
    # add soma label
    compartment_labels[int(nrn.skeleton.root)] = 1
    # the rest are basal dendrite
    compartment_labels[compartment_labels == 0] = 3

    # add compartment labels annotation, separately for each comp
    compartment_3 = nrn.SkeletonIndex(
        list(np.where(compartment_labels == 3)[0])
    ).to_mesh_index
    compartment_4 = nrn.SkeletonIndex(apical_nodes).to_mesh_index
    nrn.anno.add_annotations(
        name="apical_mesh_labels", data=compartment_4, mask=True, overwrite=True
    )
    nrn.anno.add_annotations(
        name="basal_mesh_labels", data=compartment_3, mask=True, overwrite=True
    )

    # now get volume labels
    # convert the mesh indices to skel indices
    volume_df = nrn.anno.segment_properties.df
    # add column indicating skel index
    volume_df[
        "skel_index"
    ] = nrn.anno.segment_properties.mesh_index.to_skel_index_padded
    sk_volume_df = (
        volume_df.drop_duplicates("skel_index").sort_values("skel_index").reset_index()
    )
    # pull out map for skel index -> radius
    radius_labels = np.array(sk_volume_df["r_eff"])
    # volume info already in anno

    return nrn


add_apical_labels = True

remaining_axon_df = axon_segment_df(nrn, axons_remaining_unmasked)

nrn.anno.add_annotations(
    name="remaining_axon",
    data=remaining_axon_df,
    overwrite=True,
    index_column="mesh_index",
)
# return remaining_axon_df

make_node_labels(nrn, apical_labels=add_apical_labels)
print("compartment labels added")

# %%
plot_mw_skel(
    nrn,
    plot_postsyn=True,
    plot_presyn=True,
    plot_soma=True,
    pull_compartment_colors=True,
)

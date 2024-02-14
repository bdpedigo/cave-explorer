# everything in this file stolen from Emily's skeleton_tools code

import joblib
import numpy as np
import pandas as pd

# import apical_classifier.apical_model_utils as amu
# import pcg_skel
from meshparty import meshwork

from .morphology import get_soma_row


def unmasked_to_masked(nrn, inds):
    """
    converts nodes from masked skeleton indices to unmasked skeleton indices

    """
    unmasked_inds = []
    for ind in inds:
        unmasked_inds.append(np.where(nrn.skeleton.indices_unmasked == ind)[0][0])

    return unmasked_inds


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


def apply_axon_label(nrn):
    # add simple axon/dendrite split based on synapse locations/synapse flow centrality
    is_axon, split_quality = meshwork.algorithms.split_axon_by_annotation(
        nrn, "pre_syn", "post_syn"
    )
    nrn.anno.add_annotations("is_axon", is_axon, mask=True, overwrite=True)
    return split_quality


def generate_apical_features_mask_axon(nrn, mask_out_ax=True):
    # TODO un-hardcode these paths
    axon_m1 = "axon_id/axon_id/ml_models/new_models/rf1.joblib"
    axon_m2 = "axon_id/axon_id/ml_models/new_models/rf2.joblib"
    point_features_df, axons_remaining_unmasked = amu.peel_axon_id_apical(
        nrn, axon_m1, axon_m2, mask_out_ax=mask_out_ax
    )
    return point_features_df, axons_remaining_unmasked


def apply_apical_classifier(nrn, point_features_df):
    apical_model_dir = "apical_classifier/models"
    # add apical labels and remove axons with peel back
    # load up the models for apical annotation
    rfc = joblib.load(f"{apical_model_dir}/point_model_current.pkl.gz")
    feature_cols = joblib.load(f"{apical_model_dir}/feature_cols_current.pkl")
    branch_params = joblib.load(f"{apical_model_dir}/branch_params_current.pkl")
    # create branch classifier
    BranchClassifier = amu.BranchClassifierFactory(rfc, feature_cols)
    branch_classifier = BranchClassifier(**branch_params)

    branch_df = branch_classifier.fit_predict_data(point_features_df, "base_skind")
    branch_df["masked_base_skind"] = unmasked_to_masked(nrn, branch_df["base_skind"])
    # add apical features to annotations
    nrn.anno.add_annotations(
        name="apical_segments", data=branch_df, overwrite=True, anchored=False
    )


def label_unmasked_axon(nrn, axons_remaining_unmasked):
    remaining_axon_df = axon_segment_df(nrn, axons_remaining_unmasked)
    nrn.anno.add_annotations(
        name="remaining_axon",
        data=remaining_axon_df,
        overwrite=True,
        index_column="mesh_index",
    )


def apply_compartments(
    nrn: meshwork.Meshwork, root_id: int, client, mask_axon=True, apical_labels=True
):
    """
    Takes a neuron and adds annotations for what compartment type (e.g. axon/dendrite,
    basal/apical) each part of the neuron is.

    This assumes that synapses have already been added to the neuron as annotations.
    """

    split_quality = apply_axon_label(nrn)

    pcg_skel.features.add_volumetric_properties(nrn, client, overwrite=True)

    pcg_skel.features.add_segment_properties(nrn, overwrite=True)

    row = get_soma_row(root_id, client)
    nrn.anno.add_annotations(name="soma_row", data=row, overwrite=True, anchored=False)

    point_features_df, axons_remaining_unmasked = generate_apical_features_mask_axon(
        nrn, mask_out_ax=mask_axon
    )

    apply_apical_classifier(nrn, point_features_df)

    label_unmasked_axon(nrn, axons_remaining_unmasked)

    make_node_labels(nrn, apical_labels=apical_labels)

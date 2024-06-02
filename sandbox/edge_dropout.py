# %%

import pickle
from itertools import chain
from typing import Callable, Optional, Union

import cloudvolume
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from pkg.constants import OUT_PATH
from pkg.metrics import (
    annotate_mtypes,
    annotate_pre_synapses,
    compute_counts,
    compute_spatial_target_proportions,
    compute_target_counts,
    compute_target_proportions,
)
from pkg.neuronframe import NeuronFrame, load_neuronframe
from pkg.plot import savefig, set_context
from pkg.utils import get_nucleus_point_nm, load_manifest, load_mtypes
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances_argmin
from tqdm.auto import tqdm

import caveclient as cc
import pcg_skel
from networkframe import NetworkFrame


def apply_to_synapses_by_sample(
    func: Callable,
    synapses_df: pd.DataFrame,
    resolved_synapses: Union[dict, pd.Series],
    output="series",
    name: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Apply a function which takes in a DataFrame of synapses and returns a DataFrame
    of results.

    Parameters
    ----------
    func
        A function which takes in a DataFrame of synapses and returns a DataFrame
        of results.
    which
        Whether to apply the function to the pre- or post-synaptic synapses.
    kwargs
        Additional keyword arguments to pass to `func`.
    """

    results_by_sample = []
    for i, key in enumerate(resolved_synapses.keys()):
        sample_resolved_synapses = resolved_synapses[key]

        input = synapses_df.loc[sample_resolved_synapses]
        result = func(input, **kwargs)
        if output == "dataframe":
            result["edit"] = key
        elif output == "series":
            result.name = key
        elif output == "scalar":
            pass

        results_by_sample.append(result)

    if output == "dataframe":
        results_df = pd.concat(results_by_sample, axis=0)
        return results_df
    elif output == "series":
        results_df = pd.concat(results_by_sample, axis=1).T
        results_df.index.name = "edit"
        return results_df
    else:
        results_df = pd.Series(
            results_by_sample, index=resolved_synapses.keys()
        ).to_frame()
        results_df.index.name = "edit"
        if name is not None:
            results_df.columns = [name]
        return results_df


def compute_diffs_to_final(sequence_df):
    # the final rows the one with Nan index
    final_row_idx = -1
    final_row = sequence_df.loc[final_row_idx].fillna(0).values.reshape(1, -1)

    X = sequence_df.drop(index=final_row_idx).fillna(0)

    sample_wise_metrics = []
    for metric in ["euclidean", "cityblock", "jensenshannon", "cosine"]:
        distances = cdist(X.values, final_row, metric=metric)
        distances = pd.Series(
            distances.flatten(),
            name=metric,
            index=X.index,
        )
        sample_wise_metrics.append(distances)
    sample_wise_metrics = pd.concat(sample_wise_metrics, axis=1)

    return sample_wise_metrics


# %%

set_context()

client = cc.CAVEclient("minnie65_phase3_v1")

MTYPES = load_mtypes(client)

manifest = load_manifest()
manifest = manifest.query("is_example")

# %%
root_id = manifest.index[0]
root_point = get_nucleus_point_nm(root_id, client=client)

# %%

root_ids = manifest.index[:1]
for root_id in tqdm(root_ids):
    root_point = get_nucleus_point_nm(root_id, client=client)

    # set the neuronframe to the final state
    level2_nf = load_neuronframe(root_id, client=client)
    final_nf = level2_nf.set_edits(level2_nf.edits.index)
    final_nf.select_nucleus_component(inplace=True)
    final_nf.remove_unused_nodes(inplace=True)
    final_nf.remove_unused_synapses(inplace=True)

    # generate a skeleton and map it back to the level2 nodes
    meshwork = pcg_skel.coord_space_meshwork(
        root_id, client=client, root_point=root_point, root_point_resolution=[1, 1, 1]
    )
    pcg_skel.features.add_volumetric_properties(meshwork, client)
    pcg_skel.features.add_segment_properties(meshwork)

    level2_nodes = meshwork.anno.lvl2_ids.df.copy()
    level2_nodes.set_index("mesh_ind_filt", inplace=True)
    level2_nodes[
        "skeleton_index"
    ] = meshwork.anno.lvl2_ids.mesh_index.to_skel_index_padded
    level2_nodes = level2_nodes.rename(columns={"lvl2_id": "level2_id"}).drop(
        columns="mesh_ind"
    )
    skeleton_to_level2 = level2_nodes.groupby("skeleton_index")["level2_id"].unique()

    radius_by_level2 = meshwork.anno.segment_properties["r_eff"].to_frame()
    radius_by_skeleton = radius_by_level2.groupby(level2_nodes["skeleton_index"]).mean()

    # skeleton networkframe
    skeleton_nodes = pd.DataFrame(meshwork.skeleton.vertices, columns=["x", "y", "z"])
    skeleton_nodes["radius"] = radius_by_skeleton
    # note the convention reversal here
    skeleton_edges = pd.DataFrame(meshwork.skeleton.edges, columns=["source", "target"])
    skel_nuc_id = pairwise_distances_argmin(
        skeleton_nodes[["x", "y", "z"]], [root_point], axis=0
    )[0]
    skeleton_nf = NeuronFrame(skeleton_nodes, skeleton_edges, nucleus_id=skel_nuc_id)
    g = skeleton_nf.to_networkx(create_using=nx.DiGraph)
    sorted_nodes = list(nx.topological_sort(g))
    sorted_nodes_map = {node: i for i, node in enumerate(sorted_nodes)}
    skeleton_nf.nodes["topological_order"] = skeleton_nf.nodes.index.map(
        sorted_nodes_map
    )
    skeleton_nf.apply_node_features("topological_order", inplace=True)
    skeleton_nf.edges.sort_values("source_topological_order", inplace=True)
    skeleton_nf.edges["source_indexer"] = skeleton_nf.nodes.index.get_indexer_for(
        skeleton_nf.edges["source"]
    )
    skeleton_nf.edges["target_indexer"] = skeleton_nf.nodes.index.get_indexer_for(
        skeleton_nf.edges["target"]
    )
    skeleton_nf.apply_node_features("radius", inplace=True)
    skeleton_nf.edges["radius"] = (
        skeleton_nf.edges["source"] + skeleton_nf.edges["target"]
    ) / 2

    with open(OUT_PATH / "edge_dropout" / f"skeleton_root_id={root_id}.pkl", "wb") as f:
        pickle.dump(skeleton_nf, f)

    # remove each edge one by one
    # HACK:there is a much much smarter way to do this starting from tips and removing edges
    # one by one. could do this in a way that avoids recomputing the adjacency matrix
    # every time, also.

    # def drop_edge(edge_id, skeleton_nf):

    # return dropped_level2_nf.pre_synapses.index

    n_edges = len(skeleton_nf.edges)
    pre_synapse_ids_by_edge = {}

    soma_index = skeleton_nf.nodes["topological_order"].idxmax()

    lil_adjacency = skeleton_nf.to_sparse_adjacency(format="lil")
    pre_syns = final_nf.pre_synapses
    for edge_id, row in tqdm(skeleton_nf.edges.iterrows(), total=n_edges):
        # drop the edge
        i = int(row["source_indexer"])
        j = int(row["target_indexer"])

        lil_adjacency[i, j] = 0

        mask = dijkstra(lil_adjacency, directed=False, indices=soma_index)

        reachable_nodes_at_iter = skeleton_nf.nodes.index[np.isfinite(mask)]
        # reachable_nodes[edge_id] = reachable_nodes_at_iter

        # used_level2_nodes = skeleton_to_level2_long[reachable_nodes_at_iter].values
        used_level2_nodes = skeleton_to_level2[reachable_nodes_at_iter]
        used_level2_nodes = list(chain.from_iterable(used_level2_nodes))
        # reachable_level2_nodes_by_drop[edge_id] = used_level2_nodes

        # replace
        lil_adjacency[i, j] = 1

        # dropped_skeleton_nf.edges.drop(index=edge_id, inplace=True)
        # dropped_skeleton_nf.select_nucleus_component(inplace=True)
        # dropped_skeleton_nf.remove_unused_nodes(inplace=True)
        # used_level2_nodes = (
        #     skeleton_to_level2[dropped_skeleton_nf.nodes.index]
        #     .explode()
        #     .values.astype(int)
        # )
        # # map the new skeleton to the level2 nodes
        # dropped_level2_nf = final_nf.reindex_nodes(used_level2_nodes)
        # dropped_level2_nf.select_nucleus_component(inplace=True)
        # dropped_level2_nf.remove_unused_nodes(inplace=True)
        # dropped_level2_nf.remove_unused_synapses(inplace=True)

        pre_synapse_ids_by_edge[edge_id] = pre_syns.query(
            "pre_pt_level2_id.isin(@used_level2_nodes)"
        ).index

    # from joblib import Parallel, delayed

    # outs = Parallel(n_jobs=-1, verbose=10)(
    #     delayed(drop_edge)(edge_id, skeleton_nf) for edge_id in skeleton_nf.edges.index
    # )
    # pre_synapse_ids_by_edge = dict(zip(skeleton_nf.edges.index, outs))

    pre_synapse_ids_by_edge[-1] = final_nf.pre_synapses.index

    # features to compute
    annotate_mtypes(level2_nf, MTYPES)
    annotate_pre_synapses(level2_nf, MTYPES)

    sequence_feature_dfs = {}

    counts = apply_to_synapses_by_sample(
        compute_counts,
        level2_nf.pre_synapses,
        pre_synapse_ids_by_edge,
        output="scalar",
        name="counts",
    )
    sequence_feature_dfs["counts"] = counts

    counts_by_mtype = apply_to_synapses_by_sample(
        compute_target_counts,
        level2_nf.pre_synapses,
        pre_synapse_ids_by_edge,
        by="post_mtype",
    )
    sequence_feature_dfs["counts_by_mtype"] = counts_by_mtype

    props_by_mtype = apply_to_synapses_by_sample(
        compute_target_proportions,
        level2_nf.pre_synapses,
        pre_synapse_ids_by_edge,
        by="post_mtype",
    )
    sequence_feature_dfs["props_by_mtype"] = props_by_mtype

    spatial_props = apply_to_synapses_by_sample(
        compute_spatial_target_proportions,
        level2_nf.pre_synapses,
        pre_synapse_ids_by_edge,
        mtypes=MTYPES,
    )
    sequence_feature_dfs["spatial_props"] = spatial_props

    spatial_props_by_mtype = apply_to_synapses_by_sample(
        compute_spatial_target_proportions,
        level2_nf.pre_synapses,
        pre_synapse_ids_by_edge,
        mtypes=MTYPES,
        by="post_mtype",
    )
    sequence_feature_dfs["spatial_props_by_mtype"] = spatial_props_by_mtype

    sequence_features_for_neuron = pd.Series(sequence_feature_dfs, name=root_id)
    with open(
        OUT_PATH / "edge_dropout" / f"sequence_features_root_id={root_id}.pkl", "wb"
    ) as f:
        pickle.dump(sequence_features_for_neuron, f)

    # sequence_features_by_neuron.append(sequence_features_for_neuron)

    diffs_by_feature = {}
    for feature_name, feature_df in sequence_features_for_neuron.items():
        diffs = compute_diffs_to_final(feature_df)
        diffs_by_feature[feature_name] = diffs

    feature_diffs_for_neuron = pd.Series(diffs_by_feature, name=root_id)

    with open(
        OUT_PATH / "edge_dropout" / f"feature_diffs_root_id={root_id}.pkl", "wb"
    ) as f:
        pickle.dump(feature_diffs_for_neuron, f)

    # feature_diffs_by_neuron.append(feature_diffs_for_neuron)
# %%

import seaborn as sns 

select_diffs = diffs_by_feature["counts"]["euclidean"]
radius = skeleton_edges['radius']

sns.scatterplot(x=radius, y=select_diffs, alpha=0.8, s=3)

# %%


sequence_features_by_neuron = []
feature_diffs_by_neuron = []
skeleton_nfs = []
for root_id in root_ids:
    with open(
        OUT_PATH / "edge_dropout" / f"sequence_features_root_id={root_id}.pkl", "rb"
    ) as f:
        sequence_features_for_neuron = pickle.load(f)
        sequence_features_by_neuron.append(sequence_features_for_neuron)

    with open(
        OUT_PATH / "edge_dropout" / f"feature_diffs_root_id={root_id}.pkl", "rb"
    ) as f:
        feature_diffs_for_neuron = pickle.load(f)
        feature_diffs_by_neuron.append(feature_diffs_for_neuron)

    with open(OUT_PATH / "edge_dropout" / f"skeleton_root_id={root_id}.pkl", "rb") as f:
        skeleton_nf = pickle.load(f)
        skeleton_nfs.append(skeleton_nf)

# %%
from pkg.utils import load_casey_palette

fig, axs = plt.subplots(1, 5, figsize=(20, 5), constrained_layout=True, sharex=True)

mtype_palette = load_casey_palette()
for root_id, diffs in zip(root_ids, feature_diffs_by_neuron):
    mtype = manifest.loc[root_id, "mtype"]
    color = mtype_palette[mtype]
    sorted_diff_index = diffs["counts"].sort_values("euclidean", ascending=False).index
    for i, (feature_name, diffs) in enumerate(diffs.items()):
        feat_diffs = diffs.loc[sorted_diff_index]
        ax = axs[i]
        sns.scatterplot(
            x=np.arange(len(feat_diffs)),
            y=feat_diffs["euclidean"],
            ax=ax,
            s=1,
            linewidth=0,
            alpha=0.2,
            color=color,
        )
        ax.set_title(feature_name)
        ax.set_yscale("log")
# %%
fig, axs = plt.subplots(len(root_ids), 1, figsize=(5, 0.5 * len(root_ids)), sharex=True)
for i, (root_id, diffs) in enumerate(zip(root_ids, feature_diffs_by_neuron)):
    color = mtype_palette[manifest.loc[root_id, "mtype"]]
    sns.histplot(diffs["counts"]["euclidean"], log_scale=True, ax=axs[i], color=color)
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

# %%
manifest = load_manifest()

# %%

diffs = feature_diffs_for_neuron
sorted_diff_index = diffs["counts"].sort_values("euclidean", ascending=False).index

fig, axs = plt.subplots(2, 5, figsize=(16, 10), constrained_layout=True, sharex=True)
for i, (feature_name, diffs) in enumerate(diffs.items()):
    feat_diffs = diffs.loc[sorted_diff_index]
    ax = axs[0, i]
    sns.scatterplot(
        x=np.arange(len(feat_diffs)),
        y=feat_diffs["euclidean"],
        ax=ax,
        s=10,
        linewidth=0,
        alpha=0.3,
    )
    ax.set_title(feature_name)

    ax = axs[1, i]
    sns.scatterplot(
        x=np.arange(len(feat_diffs)),
        y=feat_diffs["euclidean"],
        ax=ax,
        legend=False,
        s=10,
        linewidth=0,
        alpha=0.2,
    )
    ax.set_yscale("log")

axs[1, 2].set_xlabel("Skeleton edge rank")


target_id = manifest.loc[root_id, "target_id"]
savefig(f"edge_dropout_importances_target_id={target_id}", fig, folder="edge_dropout")

# %%

pv.set_jupyter_backend("client")

plotter = pv.Plotter(shape=(4, 4))

for i, (root_id, skeleton_nf, feature_diffs_for_neuron) in enumerate(
    zip(root_ids[:16], skeleton_nfs, feature_diffs_by_neuron)
):
    skeleton_nf.edges["importance"] = feature_diffs_for_neuron["counts"]["euclidean"]

    spadj = skeleton_nf.to_sparse_adjacency(weight_col="importance")
    spadj = spadj + spadj.T

    node_importances = np.sum(spadj, axis=0)

    skeleton_nf.nodes["importance"] = node_importances

    skeleton_polydata = skeleton_nf.to_skeleton_polydata(label="importance")
    plotter.subplot(*np.unravel_index(i, (4, 4)))
    plotter.add_mesh(
        skeleton_polydata, scalars="importance", cmap="Reds", line_width=5, opacity=0.5
    )
    plotter.add_mesh(skeleton_polydata, color="grey", line_width=0.5)

    # skeleton_nf.plot_pyvista(scalar="importance", cmap="Reds", line_width=3)
plotter.link_views()
plotter.show_axes()
plotter.show()
# %%
sns.histplot(node_importances, log_scale=True)


# %%


def write_networkframes_to_skeletons(
    networkframes: Union[NetworkFrame, dict[NetworkFrame]],
    client: cc.CAVEclient,
    attribute: Optional[str] = None,
    directory: str = "gs://allen-minnie-phase3/tempskel",
):
    # register an info file and set up CloudVolume
    base_info = client.chunkedgraph.segmentation_info
    base_info["skeletons"] = "skeleton"
    info = base_info.copy()

    cv = cloudvolume.CloudVolume(
        f"precomputed://{directory}",
        mip=0,
        info=info,
        compress=False,
    )
    cv.commit_info()

    sk_info = cv.skeleton.meta.default_info()
    sk_info["vertex_attributes"] = [
        {"id": "radius", "data_type": "float32", "num_components": 1},
        {"id": "vertex_types", "data_type": "float32", "num_components": 1},
    ]
    cv.skeleton.meta.info = sk_info
    cv.skeleton.meta.commit_info()

    sks = []
    if isinstance(networkframes, NetworkFrame):
        networkframes = {0: networkframes}

    for name, networkframe in networkframes.items():
        # extract vertex information
        vertices = networkframe.nodes[["x", "y", "z"]].values
        edges_unmapped = networkframe.edges[["source", "target"]].values
        edges = networkframe.nodes.index.get_indexer_for(
            edges_unmapped.flatten()
        ).reshape(edges_unmapped.shape)

        vertex_types = networkframe.nodes[attribute].values.astype(np.float32)

        radius = np.ones(len(vertices), dtype=np.float32)

        sk_cv = cloudvolume.Skeleton(
            vertices,
            edges,
            radius,
            None,
            segid=name,
            extra_attributes=sk_info["vertex_attributes"],
            space="physical",
        )
        sk_cv.vertex_types = vertex_types

        sks.append(sk_cv)

    cv.skeleton.upload(sks)


write_networkframes_to_skeletons(
    {root_id: skeleton_nf}, client=client, attribute="importance"
)

# %%

import json
from typing import Optional, Union

import cloudvolume
import numpy as np
import pandas as pd
import seaborn as sns

import caveclient as cc
from networkframe import NetworkFrame
from nglui import statebuilder

sbs = []
dfs = []
layers = []
viewer_resolution = client.info.viewer_resolution()
img_layer = statebuilder.ImageLayerConfig(
    client.info.image_source(),
)
seg_layer = statebuilder.SegmentationLayerConfig(
    client.info.segmentation_source(),
    alpha_3d=0.1,
    name="seg",
)
seg_layer.add_selection_map(selected_ids_column="object_id")

skel_layer = statebuilder.SegmentationLayerConfig(
    "precomputed://gs://allen-minnie-phase3/tempskel",
    name="skeleton",
)
skel_layer.add_selection_map(selected_ids_column="object_id")
base_sb = statebuilder.StateBuilder(
    [img_layer, seg_layer, skel_layer],
    client=client,
    resolution=viewer_resolution,
)

sbs.append(base_sb)
dfs.append(pd.DataFrame({"object_id": [root_id]}))

sb = statebuilder.ChainedStateBuilder(sbs)
json_out = statebuilder.helpers.package_state(dfs, sb, client=client, return_as="json")
state_dict = json.loads(json_out)

shader = """
void main() {
    float compartment = vCustom2;
    vec4 uColor = segmentColor();
    emitRGB(vec3(0.1, 0.1, 0.3) + compartment * vec3(1, 0, 0));
}
"""
skel_rendering_kws = {
    "shader": shader,
    "mode2d": "lines_and_points",
    "mode3d": "lines",
    "lineWidth3d": 2.5,
}

state_dict["layers"][2]["skeletonRendering"] = skel_rendering_kws


statebuilder.StateBuilder(base_state=state_dict, client=client).render_state(
    return_as="html"
)

# %%

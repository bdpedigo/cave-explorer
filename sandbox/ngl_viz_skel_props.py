# %%
import json
from pathlib import Path
from typing import Optional

import caveclient as cc
import cloudvolume
import numpy as np
import pandas as pd
from nglui import statebuilder
from nglui import statebuilder as sb
from skops.io import load
from troglobyte.features import CAVEWrangler

from pkg.neuronframe import load_neuronframe
from pkg.utils import load_manifest

client = cc.CAVEclient("minnie65_phase3_v1")

manifest = load_manifest()

root_id = manifest.index[1]

nf = load_neuronframe(root_id, client)

edited_nf = nf.set_edits(nf.metaedits.query("has_merge").index, prefix="meta")
edited_nf.select_nucleus_component(inplace=True)
edited_nf.remove_unused_nodes(inplace=True)
edited_nf

model_path = Path("data/models/local_compartment_classifier_ej_skeletons.skops")

model = load(model_path)


# %%

# download features for the nodes
wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=False)
wrangler.set_level2_ids(edited_nf.nodes.index)
wrangler.query_current_object_ids()  # get current ids as a proxy for getting synapses
wrangler.query_level2_shape_features()
wrangler.query_level2_synapse_features(method="existing")

# do an aggregation using the graph

features = wrangler.features_.droplevel("object_id")
features = features.drop(columns=["rep_coord_x", "rep_coord_y", "rep_coord_z"])

old_nodes = edited_nf.nodes
edited_nf.nodes = features

# %%
edited_nf

nodes = edited_nf.nodes.copy()
directed: bool = False
k = 5
drop_non_numeric = True
if drop_non_numeric:
    nodes = nodes.select_dtypes(include=[np.number])

sparse_adjacency = edited_nf.to_sparse_adjacency()

from scipy.sparse.csgraph import dijkstra

# TODO add a check for interaction of directed and whether the graph has any
# bi-directional edges
dists = dijkstra(sparse_adjacency, directed=directed, limit=k, unweighted=True)
mask = ~np.isinf(dists)

drop_self_in_neighborhood = True
if drop_self_in_neighborhood:
    mask[np.diag_indices_from(mask)] = False

print("sparsity of k-hop neighborhood graph:")
print(mask.sum() / mask.size)

# %%
from scipy.sparse import csr_array

mask = csr_array(mask)

# %%

feature_matrix = features.fillna(0).values

# %%
neighborhood_sum_matrix = mask @ feature_matrix

# %%

agg = "mean"

if agg == "mean":
    divisor_matrix = mask @ features.notna().astype(int)
    divisor_matrix[divisor_matrix == 0] = 1
    neighborhood_mean_matrix = neighborhood_sum_matrix / divisor_matrix
    neighborhood_mean_matrix = pd.DataFrame(
        neighborhood_mean_matrix, index=features.index, columns=features.columns
    )

# %%
old_nodes = edited_nf.nodes
edited_nf.nodes = features

# %%

import time

currtime = time.time()
test_agg = edited_nf.k_hop_aggregation(
    k=5, aggregations=["mean", "std"], verbose=True, engine="scipy"
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")


# %%

neighborhood_features = edited_nf.k_hop_aggregation(
    k=5, aggregations=["mean", "std"], verbose=True
)

#%%
np.abs(neighborhood_features - test_agg).max()

# %%
# feature = "pca_ratio_01_neighbor_std"
feature = "pre_synapse_count_neighbor_std"
diff = neighborhood_features[feature] - test_agg[feature].fillna(0)

diff_from_mean = neighborhood_features[feature] - neighborhood_features[feature].mean()

# sns.scatterplot(x=np.arange(len(diff)), y=diff)
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(x=diff_from_mean, y=diff)
ax.set(xlabel="diff from mean of feature", ylabel="diff from scipy answer")

# %%[
diff.sort_values().index[:100]

#%%
neighborhood_features.loc[diff.sort_values().index[:1000], feature]


# %%
joined_features = features.join(neighborhood_features, how="left")

edited_nf.nodes = old_nodes


# %%

y_pred = model.predict(joined_features)
class_to_int_map = {c: i for i, c in enumerate(model.classes_)}
y_pred_int = np.array([class_to_int_map[c] for c in y_pred])

edited_nf.nodes["predicted_compartment"] = y_pred_int


# %%

from typing import Union

from networkframe import NetworkFrame


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
    edited_nf,
    client,
    attribute="predicted_compartment",
    directory="gs://allen-minnie-phase3/tempskel",
)


# %%
# can just add level2 IDs in a new segmentation layer
# both in same source layer would mean they get loaded up together
# could just color level2 IDs in the mesh on my own?
# speulunker might be undocumented for doing this
# SegmentColor dict seg ID to hex
# this is in the state, not in the layer

# %%


sbs = []
dfs = []
viewer_resolution = client.info.viewer_resolution()
img_layer = statebuilder.ImageLayerConfig(
    client.info.image_source(),
)
seg_layer = statebuilder.SegmentationLayerConfig(
    client.info.segmentation_source(), alpha_3d=0.3
)
skel_layer = statebuilder.SegmentationLayerConfig(
    "precomputed://gs://allen-minnie-phase3/tempskel"
)
skel_layer.add_selection_map(selected_ids_column="skel_id")

base_sb = statebuilder.StateBuilder(
    [img_layer, seg_layer, skel_layer],
    client=client,
    resolution=viewer_resolution,
)
base_df = pd.DataFrame({"skel_id": [0]})

sbs.append(base_sb)
dfs.append(base_df)

sb = statebuilder.ChainedStateBuilder(sbs)
json_out = statebuilder.helpers.package_state(dfs, sb, client=client, return_as="json")
state_dict = json.loads(json_out)


shader = """
void main() {
    float compartment = vCustom2;
    vec4 uColor = segmentColor();
    emitRGB(uColor.rgb*0.5 + vec3(0.5, 0.5, 0.5) * compartment);
}
"""
skel_rendering_kws = {
    "shader": shader,
    "mode2d": "lines_and_points",
    "mode3d": "lines",
    "lineWidth3d": 1,
}

state_dict["layers"][1]["skeletonRendering"] = skel_rendering_kws

statebuilder.StateBuilder(base_state=state_dict, client=client).render_state(
    return_as="html"
)

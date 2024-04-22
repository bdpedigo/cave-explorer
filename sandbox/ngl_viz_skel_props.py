# %%
from pathlib import Path

import caveclient as cc
import cloudvolume
import neuroglancer
import neuroglancer.static_file_server
import numpy as np
import pandas as pd
from skops.io import load
from troglobyte.features import CAVEWrangler

from pkg.neuronframe import load_neuronframe
from pkg.utils import load_manifest

client = cc.CAVEclient("minnie65_phase3_v1")

manifest = load_manifest()

root_id = manifest.index[2]

nf = load_neuronframe(root_id, client)

edited_nf = nf.set_edits(nf.metaedits.query("has_merge").index, prefix="meta")
# edited_nf = nf.set_edits([], prefix="meta")
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

neighborhood_features = edited_nf.k_hop_aggregation(
    k=5, aggregations=["mean", "std"], verbose=True
)
joined_features = features.join(neighborhood_features, how="left")

edited_nf.nodes = old_nodes


# %%

y_pred = model.predict(joined_features)
class_to_int_map = {c: i for i, c in enumerate(model.classes_)}
y_pred_int = np.array([class_to_int_map[c] for c in y_pred])

edited_nf.nodes["predicted_compartment"] = y_pred_int

# posteriors = model.predict_proba(joined_features)
# posteriors = pd.DataFrame(
#     posteriors, index=joined_features.index, columns=model.classes_
# )
# axon_post = posteriors["axon"] / (posteriors.sum(axis=1))

# edited_nf.nodes["axon_posterior"] = 0.0
# edited_nf.nodes.loc[joined_features.index, "axon_posterior"] = axon_post


# %%

from typing import Optional


def write_skeleton(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    client: cc.CAVEclient,
    attribute: Optional[str] = None,
    directory: str = "./tempskel",
):
    # register an info file and set up CloudVolume
    base_info = client.chunkedgraph.segmentation_info
    base_info["skeletons"] = "skeleton"
    info = base_info.copy()

    cv = cloudvolume.CloudVolume(
        f"precomputed://file://{directory}",
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
    for i in range(1):
        # extract vertex information
        vertices = edited_nf.nodes[["x", "y", "z"]].values
        edges_unmapped = edited_nf.edges[["source", "target"]].values
        edges = edited_nf.nodes.index.get_indexer_for(edges_unmapped.flatten()).reshape(
            edges_unmapped.shape
        )

        vertex_types = nodes[attribute].values.astype(np.float32)

        radius = np.ones(len(vertices), dtype=np.float32)

        sk_cv = cloudvolume.Skeleton(
            vertices,
            edges,
            radius,
            None,
            segid=i,
            extra_attributes=sk_info["vertex_attributes"],
            space="physical",
        )
        sk_cv.vertex_types = vertex_types

        sks.append(sk_cv)

        cv.skeleton.upload(sks)

        # cf = CloudFiles(cv.cloudpath, progress=True)


write_skeleton(
    edited_nf.nodes, edited_nf.edges, client, attribute="predicted_compartment"
)

# %%

server = neuroglancer.static_file_server.StaticFileServer(
    static_dir=".", bind_address="127.0.0.1", daemon=True
)
viewer = neuroglancer.Viewer()
coordinate_space = neuroglancer.CoordinateSpace(
    names=["x", "y", "z"],
    units=["nm", "nm", "nm"],
    scales=[1, 1, 1],  # was 25 25 25
)

# """
shader = """
void main() {
  float compartment = vCustom2;
  vec4 uColor = segmentColor();
#   emitRGB(uColor.rgb*0.5 + vec4(0.5, 0.5, 0.5, 1) * compartment);
}
"""
shader = """
void main() {

  float compartment = vCustom2;
  vec4 uColor = segmentColor();

  if (compartment < 0.5){
  	emitRGB(uColor.rgb);
  } else {
    emitRGB(uColor.rgb*.5);
  }
}
"""
shader = """
void main() {

  float compartment = vCustom2;
  vec4 uColor = segmentColor();

  if (compartment <= 0.5){
  	emitRGB(uColor.rgb);
  }

  else{
    emitRGB(vec3(1.0, 1.0, 1.0));
  }
}
"""
shader = """
void main() {
  float compartment = vCustom2;
  vec4 uColor = segmentColor();
  emitRGB(colormapJet(compartment));
}
"""

shader = """
void main() {
    float compartment = vCustom2;
    vec4 uColor = segmentColor();
    emitRGB(uColor.rgb*0.5 + vec3(0.5, 0.5, 0.5) * compartment);
}
"""


with viewer.txn() as s:
    s.dimensions = coordinate_space
    seg_layer = neuroglancer.SegmentationLayer(
        source=f"precomputed://{server.url}/tempskel"
    )
    skeleton_render = neuroglancer.viewer_state.SkeletonRenderingOptions(shader=shader)

    seg_layer.skeleton_rendering = skeleton_render
    # seg_layer.segment_colors = {root_id: "red"}

    s.layers["seg"] = seg_layer

viewer


# %%

import pyvista as pv

pv.set_jupyter_backend("client")

lines = np.empty((len(edges), 3), dtype=int)
lines[:, 0] = 2
lines[:, 1:3] = edges

poly = pv.PolyData(vertices.astype(float), lines=lines)

plotter = pv.Plotter()
plotter.add_mesh(poly, scalars=vertex_types, cmap="coolwarm", line_width=10)
plotter.show()
# %%

# info = cloudvolume.CloudVolume.create_new_info(
#     num_channels=1,
#     layer_type="segmentation",
#     data_type="uint64",  # Channel images might be 'uint8'
#     # raw, png, jpeg, compressed_segmentation, fpzip, kempressed, zfpc, compresso, crackle
#     encoding="raw",
#     resolution=[1, 1, 1],  # Voxel scaling, units are in nanometers
#     voxel_offset=[0, 0, 0],  # x,y,z offset in voxels from the origin
#     mesh="mesh",
#     skeletons="skeleton",
#     # Pick a convenient size for your underlying chunk representation
#     # Powers of two are recommended, doesn't need to cover image exactly
#     # chunk_size=[512, 512, 512],  # units are voxels
#     volume_size=[500_000, 500_000, 500_000],  # e.g. a cubic millimeter dataset
# )

# %%

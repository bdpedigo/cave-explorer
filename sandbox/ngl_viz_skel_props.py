# %%
import caveclient as cc
import cloudvolume
import neuroglancer
import neuroglancer.static_file_server
import numpy as np
from cloudfiles import CloudFiles

from pkg.neuronframe import load_neuronframe
from pkg.utils import load_manifest

client = cc.CAVEclient("minnie65_phase3_v1")

manifest = load_manifest()

root_id = manifest.index[0]

nf = load_neuronframe(root_id, client)


# %%

base_info = client.chunkedgraph.segmentation_info
base_info["skeletons"] = "skeleton"
info = base_info.copy()

cv = cloudvolume.CloudVolume(
    "precomputed://file://./tempskel",
    mip=0,
    info=info,
    compress=False,
)
cv.commit_info()

# %%
sk_info = cv.skeleton.meta.default_info()

# sk_info["transform"] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
sk_info["vertex_attributes"] = [
    {"id": "radius", "data_type": "float32", "num_components": 1},
    {"id": "vertex_types", "data_type": "float32", "num_components": 1},
]
cv.skeleton.meta.info = sk_info
cv.skeleton.meta.commit_info()

# %%
vertices = nf.nodes[["x", "y", "z"]].values
edges_unmapped = nf.edges[["source", "target"]].values
edges = nf.nodes.index.get_indexer_for(edges_unmapped.flatten()).reshape(
    edges_unmapped.shape
)


# %%
vertex_types = (nf.nodes["x"] > nf.nodes["x"].mean()).values.astype(np.float32)

radius = 1000 * np.ones(len(vertices), dtype=np.float32)

sks = []
for _ in range(1):
    sk_cv = cloudvolume.Skeleton(
        vertices,
        edges,
        radius,
        vertex_types,
        segid=root_id,
        extra_attributes=sk_info["vertex_attributes"],
        space="physical",
    )
    # sk_cv.vertex_typ .es =

    sks.append(sk_cv)

# %%

cv.skeleton.upload(sks)
# %%

cf = CloudFiles(cv.cloudpath, progress=True)

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
shader = """
void main() {

  float compartment = vCustom2;
  vec4 uColor = segmentColor();
  if (compartment == 1.0){    
  	emitRGB(uColor.rgb);
  }

  else{
    emitRGB(vec3(1.0, 1.0, 1.0));
  }
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

# lines = np.empty((len(edges), 3), dtype=int)
# lines[:, 0] = 2
# lines[:, 1:3] = edges

# poly = pv.PolyData(vertices.astype(float), lines=lines)

# plotter = pv.Plotter()
# plotter.add_mesh(poly)
# plotter.show()
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

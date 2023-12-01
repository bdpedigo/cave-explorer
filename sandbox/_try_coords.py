# %%
import numpy as np

import caveclient as cc

client = cc.CAVEclient("minnie65_phase3_v1")

operation_id = 9408

out = client.chunkedgraph.get_operation_details([operation_id])[str(operation_id)]

sink_coords_point = out["sink_coords"][0]

print("Point (seg-voxel space) for sink_coords:", sink_coords_point)

modified_supervoxel = out["added_edges"][0][0]

l2_id = client.chunkedgraph.get_roots([modified_supervoxel], stop_layer=2)[0]

l2_point = client.l2cache.get_l2data(l2_id, attributes=["rep_coord_nm"])[str(l2_id)][
    "rep_coord_nm"
]
print("Point (nm) from level 2 cache:", l2_point)

print(
    "Point for sink_coords scaled by (8x8x40):",
    np.array(sink_coords_point) * np.array([8, 8, 40]),
)

# %%

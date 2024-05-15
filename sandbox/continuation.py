# %%
import caveclient as cc
import pandas as pd
import pcg_skel
from networkframe import NetworkFrame

from pkg.utils import load_manifest

client = cc.CAVEclient("minnie65_phase3_v1")

manifest = load_manifest()
manifest.query("in_inhibitory_column", inplace=True)

# %%
root_id = manifest.index[0]
meshwork = pcg_skel.coord_space_meshwork(root_id, client=client)


level2_nodes = meshwork.anno.lvl2_ids.df.copy()
level2_nodes.set_index("mesh_ind_filt", inplace=True)
level2_nodes["skeleton_index"] = meshwork.anno.lvl2_ids.mesh_index.to_skel_index_padded
level2_nodes = level2_nodes.rename(columns={"lvl2_id": "level2_id"}).drop(
    columns="mesh_ind"
)
# %%
skeleton_to_level2 = level2_nodes.groupby("skeleton_index")["level2_id"].unique()

# %%

skeleton_nodes = pd.DataFrame(meshwork.skeleton.vertices, columns=["x", "y", "z"])
skeleton_edges = pd.DataFrame(meshwork.skeleton.edges, columns=["source", "target"])

# %%
nf = NetworkFrame(skeleton_nodes, skeleton_edges)

# %%

end_points = []
for end_point_index in meshwork.skeleton.end_points:
    level2s_at_end = skeleton_to_level2[end_point_index]
    end_point = meshwork.skeleton.vertices[end_point_index]
    end_points.append(end_point)

end_points = pd.Series(data=end_points, index=len(end_points) * [root_id])

# %%
from tqdm.auto import tqdm
from troglobyte.features import CAVEWrangler

for point in tqdm(end_points[5:6]):
    wrangler = CAVEWrangler(client, n_jobs=-1)
    wrangler.set_query_box_from_point(point, box_width=5_000)
    wrangler.query_objects_from_box(mip=5, size_threshold=10)
    wrangler.query_level2_ids()
    wrangler.query_level2_shape_features()
    wrangler.aggregate_features_by_neighborhood(aggregations=["mean", "std"])


# %%
wrangler.object_ids_ = wrangler.object_ids_.reindex(
    [root_id] + wrangler.object_ids_.difference([root_id]).tolist()
)[0]

# %%
wrangler.visualize_query()


# %%

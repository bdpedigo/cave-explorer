# %%
import numpy as np
import pyvista as pv
from pkg.utils import get_nucleus_point_nm, load_manifest
from skops.io import load

import pcg_skel
from caveclient import CAVEclient
from troglobyte.features import CAVEWrangler

client = CAVEclient("minnie65_phase3_v1")

cv = client.info.segmentation_cloudvolume()

manifest = load_manifest()

root_id = manifest.index[0]

# %%
mesh = cv.mesh.get(root_id)[root_id]

# %%
vertices = mesh.vertices
faces = mesh.faces


# add a column of all 3s to the faces

padded_faces = np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1)

mesh_poly = pv.PolyData(vertices, faces=padded_faces)


# %%


root_point = get_nucleus_point_nm(root_id, client=client)

meshwork = pcg_skel.pcg_meshwork(
    root_id,
    client=client,
    collapse_soma=True,
    root_point=root_point,
    root_point_resolution=[1, 1, 1],
)
pcg_skel.features.add_volumetric_properties(meshwork, client)
pcg_skel.features.add_segment_properties(meshwork)

# %%
level2_nodes = meshwork.anno.lvl2_ids.df.copy()
level2_nodes.set_index("mesh_ind_filt", inplace=True)
level2_nodes["skeleton_index"] = meshwork.anno.lvl2_ids.mesh_index.to_skel_index_padded
level2_nodes = level2_nodes.rename(columns={"lvl2_id": "level2_id"}).drop(
    columns="mesh_ind"
)
features_by_level2 = meshwork.anno.segment_properties["r_eff"].to_frame()
features_by_level2["level2_id"] = features_by_level2.index.map(
    level2_nodes["level2_id"]
)
features_by_level2 = features_by_level2.set_index("level2_id")["r_eff"].to_frame()
features_by_level2["skeleton_index"] = level2_nodes.set_index("level2_id")[
    "skeleton_index"
]
radius_by_skeleton = features_by_level2.groupby("skeleton_index")["r_eff"].mean()

# %%

model = load("skedits/data/models/local_compartment_classifier_ej_skeletons.skops")
wrangler = CAVEWrangler(client)
wrangler.set_objects([root_id])
wrangler.query_level2_shape_features()
wrangler.query_level2_synapse_features()
wrangler.query_level2_edges()
wrangler.register_model(model, "bd_boxes")
wrangler.aggregate_features_by_neighborhood(
    aggregations=["mean", "std"], neighborhood_hops=5, drop_self_in_neighborhood=True
)

# %%

axon_posteriors = wrangler.features_["bd_boxes_axon_neighbor_mean"].droplevel(
    "object_id"
)

features_by_level2 = features_by_level2.join(axon_posteriors)

axon_posterior_by_skeleton = (
    features_by_level2.groupby("skeleton_index")["bd_boxes_axon_neighbor_mean"]
    .mean()
    .sort_index()
)

# %%

vertices = meshwork.skeleton.vertices
padded_lines = np.concatenate(
    [np.full((meshwork.skeleton.edges.shape[0], 1), 2), meshwork.skeleton.edges], axis=1
)
skel_poly = pv.PolyData(vertices, lines=padded_lines)
skel_poly["radius"] = radius_by_skeleton.values
skel_poly["axon_posterior"] = axon_posterior_by_skeleton
# %%

pv.set_jupyter_backend("client")

plotter = pv.Plotter()
plotter.add_mesh(mesh_poly, opacity=0.3)
# plotter.add_mesh(skel_poly, line_width=5, scalars="radius")
min_radius = radius_by_skeleton.min()
max_radius = radius_by_skeleton.max()
tube = skel_poly.strip().tube(
    scalars="radius",
    radius=min_radius,
    absolute=True,  # radius_factor=max_radius / min_radius
)
# tube["axon_posterior"] = axon_posterior_by_skeleton.values
plotter.add_mesh(tube, scalars="axon_posterior", opacity=0.3)
# plotter.fly_to_mouse_position()
plotter.enable_fly_to_right_click()
plotter.show()
# plotter.export_html('pv.html')

# %%
level2_ids = level2_nodes["level2_id"].values


# %%
plotter = pv.Plotter()
plotter.add_mesh(mesh_poly, opacity=0.3)
# plotter.add_mesh(skel_poly, line_width=5, scalars="radius")
min_radius = radius_by_skeleton.min()
max_radius = radius_by_skeleton.max()
tube = skel_poly.strip().tube(
    scalars="radius",
    radius=min_radius,
    absolute=True,  # radius_factor=max_radius / min_radius
)
plotter.add_mesh(tube, opacity=0.3)
# plotter.fly_to_mouse_position()
plotter.enable_fly_to_right_click()
plotter.show()

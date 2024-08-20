# %%
from pathlib import Path

import numpy as np
import pyvista as pv
from neurovista import center_camera, to_line_polydata, to_mesh_polydata
from pcg_skel import pcg_skeleton
from tqdm.auto import tqdm

from pkg.constants import MERGE_COLOR, SPLIT_COLOR
from pkg.neuronframe import load_neuronframe
from pkg.utils import get_nucleus_point_nm, load_manifest, start_client

# %%
client = start_client()

target_id = 305182

manifest = load_manifest()
manifest = manifest.query("in_inhibitory_column")

root_id = manifest.query(f"target_id == {target_id}").index[0]

nf = load_neuronframe(root_id, client)

# %%
cv = client.info.segmentation_cloudvolume()
cv.cache.enabled = True

# mesh = cv.mesh.get(root_id)[root_id]

# %%

split_iloc = 10
split_row = nf.edits.iloc[split_iloc]
split_loc = split_row[["centroid_x", "centroid_y", "centroid_z"]]

this_root_after = 864691135925825294
other_root_after = 864691135725760575

# meshes = cv.mesh.get([this_root_after, other_root_after])

# mesh_polys = {}
# for this_root_id, mesh in meshes.items():
#     vertices = mesh.vertices
#     faces = mesh.faces
#     mesh_poly = to_mesh_polydata(vertices, faces)
#     mesh_polys[this_root_id] = mesh_poly

merge_iloc = 15
merge_row = nf.edits.iloc[merge_iloc]
merge_loc = merge_row[["centroid_x", "centroid_y", "centroid_z"]]

root_before_1 = merge_row["before_root_ids"][0]
root_before_2 = merge_row["before_root_ids"][1]

meshes = cv.mesh.get([root_before_1, root_before_2, other_root_after])

mesh_polys = {}
for this_root_id, mesh in meshes.items():
    vertices = mesh.vertices
    faces = mesh.faces
    mesh_poly = to_mesh_polydata(vertices, faces)
    mesh_polys[this_root_id] = mesh_poly

# %%

pv.set_jupyter_backend("client")

font_size = 300
camera_pos = [
    (75046.54147921813, 892734.4784498038, 1975869.7306010728),
    (584010.0, 767487.0, 961926.0),
    (-0.19587889797889652, -0.980363577146987, 0.02277529209773961),
]
plotter = pv.Plotter()
plotter.add_mesh(mesh_polys[root_before_1], color="grey", opacity=0.5)
plotter.add_mesh(mesh_polys[root_before_2], color=MERGE_COLOR)
plotter.add_mesh(mesh_polys[other_root_after], color=SPLIT_COLOR)


direction = np.array(camera_pos[1]) - np.array(camera_pos[0])

split_cylinder = pv.Cylinder(
    center=split_loc.values.astype(float),
    direction=direction,
    height=1000,
    radius=10_000,
)
plotter.add_mesh(
    split_cylinder, color=SPLIT_COLOR, opacity=1, style="wireframe", line_width=10
)

plotter.add_point_labels(
    split_cylinder.points[110, :],
    ["B"],
    point_size=0,
    font_size=font_size,
    shape=None,
    text_color=SPLIT_COLOR,
)

merge_cylinder = pv.Cylinder(
    center=merge_loc.values.astype(float),
    direction=direction,
    height=1000,
    radius=10_000,
)
plotter.add_mesh(
    merge_cylinder, color=MERGE_COLOR, opacity=1, style="wireframe", line_width=10
)

plotter.add_point_labels(
    merge_cylinder.points[110, :],
    ["C"],
    point_size=0,
    font_size=font_size,
    shape=None,
    text_color=MERGE_COLOR,
)

plotter.camera.zoom(1.5)

scale = 1500
window_size = (scale * np.array([2.58, 3.08])).astype(int)
plotter.window_size = window_size
plotter.camera_position = camera_pos

out_path = Path("docs/result_images/show_neuron_edits")
plotter.save_graphic(out_path / "whole_neuron.svg")

# %%

plotter = pv.Plotter()
plotter.add_mesh(mesh_polys[root_before_1].smooth(50), color="grey", opacity=0.5)
plotter.add_mesh(mesh_polys[other_root_after].smooth(50), color=SPLIT_COLOR)

center_camera(plotter, split_loc, distance=100_000)

plotter.enable_depth_of_field()
plotter.camera.zoom(5)

plotter.window_size = (scale * np.array([1.29, 1.54])).astype(int)
plotter.save_graphic(out_path / "split_example.svg")

# %%

plotter = pv.Plotter()
plotter.add_mesh(mesh_polys[root_before_1].smooth(50), color="grey", opacity=0.5)
plotter.add_mesh(mesh_polys[root_before_2].smooth(50), color=MERGE_COLOR)

center_camera(plotter, merge_loc, distance=100_000)

plotter.enable_depth_of_field()
plotter.camera.zoom(5)

plotter.window_size = (scale * np.array([1.29, 1.54])).astype(int)
plotter.save_graphic(out_path / "merge_example.svg")

# %%

manifest = load_manifest()
manifest = manifest.query("is_sample").sort_values("nuc_y")

root_ids = manifest.index

skel_polys = {}
point_polys = {}
for root_id in tqdm(root_ids):
    nf = load_neuronframe(root_id, client)
    edits = nf.edits
    edits = edits.query("is_filtered")
    root_point = get_nucleus_point_nm(root_id, client)
    skel = pcg_skeleton(root_id, client=client, root_point=root_point)

    skel_poly = to_line_polydata(skel.vertices, skel.edges)

    point_poly = pv.PolyData(edits[["centroid_x", "centroid_y", "centroid_z"]].values)
    point_poly["is_merge"] = edits["is_merge"]

    skel_polys[root_id] = skel_poly
    point_polys[root_id] = point_poly

# %%

pv.set_jupyter_backend("client")

# shape = (2, 10)
# window_size = [8000, 2000]
# shape = (4, 5)
# window_size = [2000, 4000]

shape = (4, 5)
# window_size = [3000, 2500]
window_size = (scale * np.array([3.87, 3.08])).astype(int)

plotter = pv.Plotter(shape=shape, window_size=window_size, border_width=0)

nuc_loc_centroid = manifest.loc[root_ids, ["nuc_x", "nuc_y", "nuc_z"]].values.mean(
    axis=0
)
for i, root_id in enumerate(root_ids[: shape[0] * shape[1]]):
    skel_poly = skel_polys[root_id]
    point_poly = point_polys[root_id]

    row, col = np.unravel_index(i, shape, order="C")
    plotter.subplot(row, col)
    plotter.add_mesh(
        skel_poly,
        color="black",
        line_width=1.5,
        opacity=0.6,
        show_scalar_bar=False,
        style="wireframe",
    )
    plotter.add_mesh(
        point_poly,
        point_size=20,
        render_points_as_spheres=True,
        scalars="is_merge",
        cmap=[SPLIT_COLOR, MERGE_COLOR],
        show_scalar_bar=False,
    )
    nuc_loc_centroid = manifest.loc[root_id, ["nuc_x", "nuc_y", "nuc_z"]].values
    center_camera(plotter, nuc_loc_centroid, distance=1_250_000)

plotter.remove_bounding_box()

plotter.save_graphic(out_path / "neuron_gallery.svg")
plotter.save_graphic(out_path / "neuron_gallery.pdf")

# %%

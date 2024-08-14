# %%
from pathlib import Path

import numpy as np
import pyvista as pv
import seaborn as sns
from neurovista import center_camera, to_line_polydata, to_mesh_polydata
from pcg_skel import pcg_skeleton
from tqdm.auto import tqdm

from pkg.neuronframe import load_neuronframe
from pkg.utils import get_nucleus_point_nm, load_manifest, start_client

colors = sns.color_palette("Dark2").as_hex()
MERGE_COLOR = colors[0]
SPLIT_COLOR = colors[1]

# %%
client = start_client()

target_id = 305182

manifest = load_manifest()
manifest = manifest.query("in_inhibitory_column")

root_id = manifest.query(f"target_id == {target_id}").index[0]

nf = load_neuronframe(root_id, client)

# nfs = {}
# rows = []
# for i, edit_id in tqdm(enumerate(nf.edits.index), total=len(nf.edits)):
#     edited_nf = nf.set_edits(nf.edits.index[: i + 1])
#     edited_nf = resolve_neuron(edited_nf, nf, warn_on_missing=False)

#     info = {
#         "order": i,
#         "edit_id": edit_id,
#         "n_nodes": len(edited_nf.nodes),
#         "n_pre_synapses": len(edited_nf.pre_synapses),
#         "n_post_synapses": len(edited_nf.post_synapses),
#     }

#     nfs[edit_id] = edited_nf
#     rows.append(info)

# final_nf = edited_nf

# # %%


# pv.set_jupyter_backend("trame")

# # 9
# plotter = pv.Plotter()
# i = 8
# skel1 = nfs[nf.edits.index[i]].to_skeleton_polydata()
# skel2 = nfs[nf.edits.index[i + 1]].to_skeleton_polydata()
# # plotter.add_mesh(skel1, color="grey", line_width=2)
# # plotter.add_mesh(skel2, color="red", line_width=0.5)
# merges = final_nf.to_merge_polydata()
# plotter.add_mesh(merges, color="blue", line_width=2)

# splits = nf.to_split_polydata(filter="is_filtered")
# plotter.add_mesh(splits, color="red", line_width=2)

# final_skel = final_nf.to_skeleton_polydata()
# plotter.add_mesh(final_skel, color="black", line_width=1)
# plotter.show()
# %%
cv = client.info.segmentation_cloudvolume()
cv.cache.enabled = True

mesh = cv.mesh.get(root_id)[root_id]

# %%

split_iloc = 10
split_row = nf.edits.iloc[split_iloc]
split_loc = split_row[["centroid_x", "centroid_y", "centroid_z"]]

this_root_after = 864691135925825294
other_root_after = 864691135725760575

meshes = cv.mesh.get([this_root_after, other_root_after])

mesh_polys = {}
for this_root_id, mesh in meshes.items():
    vertices = mesh.vertices
    faces = mesh.faces
    mesh_poly = to_mesh_polydata(vertices, faces)
    mesh_polys[this_root_id] = mesh_poly

plotter = pv.Plotter()
plotter.add_mesh(mesh_polys[this_root_after], color="grey")
plotter.add_mesh(mesh_polys[other_root_after], color="red")
center_camera(plotter, split_loc, distance=100_000)
plotter.show()

# %%

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

camera_pos = [
    (75046.54147921813, 892734.4784498038, 1975869.7306010728),
    (584010.0, 767487.0, 961926.0),
    (-0.19587889797889652, -0.980363577146987, 0.02277529209773961),
]
plotter = pv.Plotter()
plotter.add_mesh(mesh_polys[root_before_1], color="grey", opacity=0.5)
plotter.add_mesh(mesh_polys[root_before_2], color="blue")
plotter.add_mesh(mesh_polys[other_root_after], color="red")


direction = np.array(camera_pos[1]) - np.array(camera_pos[0])

split_cylinder = pv.Cylinder(
    center=split_loc.values.astype(float),
    direction=direction,
    height=1000,
    radius=10_000,
)
plotter.add_mesh(
    split_cylinder, color="red", opacity=1, style="wireframe", line_width=10
)

plotter.add_point_labels(
    split_cylinder.points[110, :],
    ["B"],
    point_size=0,
    font_size=150,
    shape=None,
    text_color="red",
)

merge_cylinder = pv.Cylinder(
    center=merge_loc.values.astype(float),
    direction=direction,
    height=1000,
    radius=10_000,
)
plotter.add_mesh(
    merge_cylinder, color="blue", opacity=1, style="wireframe", line_width=10
)

plotter.add_point_labels(
    merge_cylinder.points[110, :],
    ["C"],
    point_size=0,
    font_size=150,
    shape=None,
    text_color="blue",
)

plotter.camera.zoom(1.5)

plotter.window_size = [3840, 3840]
plotter.camera_position = camera_pos

out_path = Path("docs/result_images/show_neuron_edits")
plotter.save_graphic(out_path / "whole_neuron.svg")

# %%

plotter = pv.Plotter()
plotter.add_mesh(mesh_polys[root_before_1].smooth(50), color="grey", opacity=0.5)
plotter.add_mesh(mesh_polys[other_root_after].smooth(50), color="red")

center_camera(plotter, split_loc, distance=100_000)

plotter.enable_depth_of_field()
plotter.camera.zoom(5)

plotter.window_size = [3840, 3840]
plotter.save_graphic(out_path / "split_example.svg")

# %%

plotter = pv.Plotter()
plotter.add_mesh(mesh_polys[root_before_1].smooth(50), color="grey", opacity=0.5)
plotter.add_mesh(mesh_polys[root_before_2].smooth(50), color="blue")

center_camera(plotter, merge_loc, distance=100_000)

plotter.enable_depth_of_field()
plotter.camera.zoom(5)

plotter.window_size = [3840, 3840]
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
window_size = [8000, 2000]
# shape = (4, 5)
# window_size = [2000, 4000]

shape = (3, 5)
window_size = [3000, 2500]

plotter = pv.Plotter(shape=shape, window_size=window_size, border_width=0)

nuc_loc_centroid = manifest.loc[root_ids, ["nuc_x", "nuc_y", "nuc_z"]].values.mean(
    axis=0
)
for i, root_id in enumerate(root_ids[:15]):
    skel_poly = skel_polys[root_id]
    point_poly = point_polys[root_id]

    row, col = np.unravel_index(i, shape, order="C")
    plotter.subplot(row, col)
    plotter.add_mesh(
        skel_poly,
        color="black",
        line_width=1,
        opacity=0.6,
        show_scalar_bar=False,
        style="wireframe",
    )
    plotter.add_mesh(
        point_poly,
        point_size=10,
        render_points_as_spheres=True,
        scalars="is_merge",
        cmap=[SPLIT_COLOR, MERGE_COLOR],
        show_scalar_bar=False,
    )
    nuc_loc_centroid = manifest.loc[root_id, ["nuc_x", "nuc_y", "nuc_z"]].values
    center_camera(plotter, nuc_loc_centroid, distance=1_250_000)

plotter.remove_bounding_box()
# plotter.link_views()

# plotter.view_isometric()

# plotter.show()

plotter.save_graphic(out_path / "neuron_gallery.svg")
plotter.save_graphic(out_path / "neuron_gallery.pdf")


# %%

skel.vertices

# %%
skel.edges

# %%
from neurovista import to_line_polydata

skel_poly = to_line_polydata(skel.vertices, skel.edges)

plotter = pv.Plotter()

plotter.add_mesh(skel_poly, color="black", line_width=2)

center_camera(plotter, root_point, distance=1_000_000)

plotter.show()

# %%
mesh_w_normals = skel_poly.compute_normals(inplace=False)
mesh_w_normals.active_vectors_name = "Normals"
arrows = mesh_w_normals.arrows
arrows.plot(show_scalar_bar=False)

# %%
import matplotlib.pyplot as plt
import skeleton_plot as skelplot
from skeleton_plot.skel_io import load_mw

rid = 864691134884752122
sid = 612904

filename = f"{rid}_{sid}.h5"
skel_path = "https://storage.googleapis.com/allen-minnie-phase3/minniephase3-emily-pcg-skeletons/minnie_all/v661/meshworks"

mw = load_mw(skel_path, filename)
f, ax = plt.subplots(figsize=(7, 10))

skelplot.plot_tools.plot_mw_skel(
    mw,
    pull_radius=True,
    invert_y=True,
    line_width=5,
    plot_soma=True,
    pull_compartment_colors=True,
    plot_presyn=True,
    plot_postsyn=True,
)

# %%
print("mw.skeleton.mesh_to_skel_map")
print(mw.skeleton.mesh_to_skel_map)
print()
# %%
print("mw.mesh.vertices.shape")
print(mw.mesh.vertices.shape)
print()

# %%
print("len(mw.skeleton.mesh_to_skel_map)")
print(len(mw.skeleton.mesh_to_skel_map))
print()
# %%
print("len(mw.skeleton.vertices)")
print(len(mw.skeleton.vertices))
print()

# %%
print("max(mw.skeleton.mesh_to_skel_map)")
print(max(mw.skeleton.mesh_to_skel_map))

# %%
mw.vertex_properties
# %%
mw.reset_mask()
mw.mesh.vertices.shape
# %%
mw.mesh.vertices

# %%
mw.mesh.index_map
# %%
mw.mesh.metadata

# %%
mw.anno.lvl2_ids.df
# %%
mw.skeleton
# %%
mw.anno.segment_properties
# %%
mw.skeleton_indices.to_mesh_region
# %%
dir(mw.skeleton)

# %%
mw.anno.apical_mesh_labels

# %%
mw.anno.basal_mesh_labels

# %%
mw.anno.soma_row

# %%
mw.anno.remaining_axon

# %%
mw.root_region


# %%
nodes = mw.anno.lvl2_ids.df.copy()
nodes.set_index("mesh_ind_filt", inplace=True)

nodes["is_basal_dendrite"] = False
nodes.loc[mw.anno.basal_mesh_labels["mesh_index_filt"], "is_basal_dendrite"] = True

nodes["is_apical_dendrite"] = False
nodes.loc[mw.anno.apical_mesh_labels["mesh_index_filt"], "is_apical_dendrite"] = True

nodes["is_axon"] = False
nodes.loc[mw.anno.is_all_axon["mesh_index_filt"], "is_axon"] = True

nodes["is_soma"] = False
nodes.loc[mw.root_region, "is_soma"] = True

nodes["n_labels"] = nodes[
    ["is_basal_dendrite", "is_apical_dendrite", "is_axon", "is_soma"]
].sum(axis=1)

print(len(nodes.query("n_labels == 0")))
print(len(nodes.query("n_labels > 1")))

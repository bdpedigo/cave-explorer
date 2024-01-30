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

#%%
print("max(mw.skeleton.mesh_to_skel_map)")
print(max(mw.skeleton.mesh_to_skel_map))
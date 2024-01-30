# %%
import caveclient as cc
from cloudfiles import CloudFiles

bucket_path = "gs://allen-minnie-phase3/minniephase3-emily-pcg-skeletons/minnie_all/v661/skeletons"

cf = CloudFiles(bucket_path)

client = cc.CAVEclient("minnie65_phase3_v1")

file_name = "864691134884807418_518848.swc"
cf.exists(file_name)

# %%
read_bytes = cf.get(file_name)

swc_str = read_bytes.decode("utf-8")

import navis

neuron = navis.read_swc(swc_str)
neuron


# %%
from skeleton_plot.skel_io import read_skeleton, read_swc

full_path = "gs://allen-minnie-phase3/minniephase3-emily-pcg-skeletons/minnie_all/v661/skeletons/864691134884807418_518848.swc"
read_swc(full_path)


# %%
read_skeleton(bucket_path, file_name)


# %%
from meshparty.skeleton_io import read_skeleton_h5_by_part

files_generator = cf.list()
i = 0
while i < 2:
    file_name = next(files_generator)
    root_id = int(file_name.split("_")[0])
    binary = cf.get(file_name)
    with open("temp.binary", "wb") as f:
        f.write(binary)
    sk = read_skeleton_h5_by_part("temp.binary")
    i += 1

# %%
import matplotlib.pyplot as plt
import skeleton_plot as skelplot
from skeleton_plot.skel_io import load_mw

rid = 864691134884807418
sid = 518848
filename = f"{rid}_{sid}/{rid}_{sid}.h5"
skel_path = "https://storage.googleapis.com/allen-minnie-phase3/minniephase3-emily-pcg-skeletons/minnie_all/BIL_neurons/file_groups/"


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

#%%
dir(mw)

#%%
mw.skeleton.mesh_to_skel_map

#%%
mw.mesh.vertices.shape
#%%
len(mw.skeleton.mesh_to_skel_map)

#%%
len(mw.skeleton.vertices)
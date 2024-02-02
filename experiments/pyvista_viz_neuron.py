# %%
import os

os.environ["LAZYCLOUD_USE_CLOUD"] = "True"
os.environ["LAZYCLOUD_RECOMPUTE"] = "False"
os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"

import pickle

import caveclient as cc

from pkg.neuronframe import load_neuronframe

# %%
palette_file = "/Users/ben.pedigo/code/skedits/skedits-app/skedits/data/ctype_hues.pkl"

with open(palette_file, "rb") as f:
    ctype_hues = pickle.load(f)

ctype_hues = {ctype: tuple(ctype_hues[ctype]) for ctype in ctype_hues.keys()}

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v507")
query_neurons.sort_values("id", inplace=True)


root_id = 864691135697251738

prefix = "meta"


full_neuron = load_neuronframe(root_id, client)


# %%
final_neuron = full_neuron.set_edits(full_neuron.edits.index, inplace=False)
final_neuron.select_nucleus_component(inplace=True)
final_neuron.remove_unused_synapses(inplace=True)

# %%

# %%
import numpy as np
import pyvista as pv

pv.set_jupyter_backend("client")

points = final_neuron.nodes[["x", "y", "z"]].values
index = final_neuron.nodes.index
iloc_map = dict(zip(index.values, range(len(index))))

line_locs = final_neuron.edges[["source", "target"]]
source_ilocs = line_locs["source"].map(iloc_map).values
target_ilocs = line_locs["target"].map(iloc_map).values

lines = np.empty((len(line_locs), 3), dtype=int)

lines[:, 0] = 2
lines[:, 1] = source_ilocs
lines[:, 2] = target_ilocs

mesh = pv.PolyData(points, lines=lines)
mesh["scalars"] = final_neuron.nodes["has_synapses"]

mesh.plot(cmap="viridis", line_width=5)

# %%
p = pv.Plotter()
p.add_mesh(mesh, lighting=False)
# p.camera.zoom(1.5)
path = p.generate_orbital_path(n_points=60, shift=mesh.length, viewup=[0, -1, 0])
p.open_gif("orbit.gif")
p.orbit_on_path(path, write_frames=True, viewup=[0, -1, 0])
p.close()

# pl = pv.Plotter()

# pl.add_mesh(mesh, show_edges=True, color="black", line_width=2)
# pl.add_points(mesh.points, color="red", point_size=2)
# pl.show()

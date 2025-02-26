# %%

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm

from pkg.metrics import compute_target_proportions
from pkg.neuronframe import (
    NeuronFrameSequence,
    load_neuronframe,
)
from pkg.plot import animate_neuron_edit_sequence
from pkg.utils import load_casey_palette, load_manifest, load_mtypes

# pv.set_jupyter_backend("client")

# %%
sns.set_context("notebook", font_scale=1.25)

client = cc.CAVEclient("minnie65_phase3_v1")

ctype_hues = load_casey_palette()
mtypes = load_mtypes(client)
manifest = load_manifest()

folder = "animations"
verbose = False

# for root_id in manifest.query("is_sample").index[3:4]:
root_id = 864691135213953920
full_neuron = load_neuronframe(root_id, client)

#%%
prefix = "meta"
# simple time-ordered case
neuron_sequence = NeuronFrameSequence(
    full_neuron,
    prefix=prefix,
    edit_label_name=f"{prefix}operation_id",
    warn_on_missing=verbose,
)

if prefix == "meta":
    key = 'has_merge'
else: 
    key = 'is_merge'

order_by = "time"
random_seed = 0
if order_by == "time":
    neuron_sequence.edits.sort_values([key, "time"], inplace=True)
elif order_by == "random":
    rng = np.random.default_rng(random_seed)
    neuron_sequence.edits["random"] = rng.random(len(neuron_sequence.edits))
    neuron_sequence.edits.sort_values([key, "random"], inplace=True)


# splits = neuron_sequence.edits.query("~is_merge").index
# neuron_sequence.apply_edits(splits)


i = 0
next_operation = True
pbar = tqdm(total=len(neuron_sequence.edits), desc="Applying edits...")
while next_operation is not None:
    possible_edit_ids = neuron_sequence.find_incident_edits()
    if len(possible_edit_ids) == 0:
        next_operation = None
    else:
        next_operation = possible_edit_ids[0]
        neuron_sequence.apply_edits(next_operation, only_additions=False)
    i += 1
    pbar.update(1)
pbar.close()

if not neuron_sequence.is_completed:
    raise UserWarning("Neuron is not completed.")
#%%
neuron_sequence.final_neuron.nodes.index.difference(neuron_sequence.current_resolved_neuron.nodes.index)

#%%

for neuron in neuron_sequence.resolved_sequence.values():
    print(len(neuron.pre_synapses))
#%%

neuron_sequence.current_resolved_neuron.to_skeleton_polydata().plot()

# %%
# neuron_sequence.edits.sort_values("time", inplace=True)

# for i in tqdm(
#     range(len(neuron_sequence.edits)),
#     leave=False,
#     desc="Applying edits...",
#     disable=not verbose,
# ):
#     operation_id = neuron_sequence.edits.index[i]
#     neuron_sequence.apply_edits(operation_id, warn_on_missing=verbose)

# %%
max_dist = 0
for _, neuron in neuron_sequence.resolved_sequence.items():
    positions = neuron.nodes[["x", "y", "z"]]
    soma_pos = full_neuron.nodes[["x", "y", "z"]].loc[full_neuron.nucleus_id]
    positions_values = positions.values
    soma_positions_values = soma_pos.values.reshape(1, -1)

    distances = np.squeeze(pairwise_distances(positions_values, soma_positions_values))
    max_dist = max(max_dist, distances.max())

target_id = manifest.loc[root_id, "target_id"]
name = f"all_edits_by_time-target_id={target_id}"

pre_synapses = neuron_sequence.base_neuron.pre_synapses
pre_synapses["post_mtype"] = pre_synapses["post_pt_root_id"].map(mtypes["cell_type"])

neuron_sequence.sequence_info["pre_synapses"]

output_proportions = neuron_sequence.apply_to_synapses_by_sample(
    compute_target_proportions, which="pre", by="post_mtype"
)
# neuron_sequence.select_by_bout(
by = "has_merge"
keep = "last"
bouts = neuron_sequence.sequence_info[by].fillna(False).cumsum()
bouts.name = "bout"
if keep == "first":
    keep_ind = 0
else:
    keep_ind = -1
bout_exemplars = (
    neuron_sequence.sequence_info.index.to_series()
    .groupby(bouts, sort=False)
    .apply(lambda x: x.iloc[keep_ind])
).values
# bout_exemplars = pd.Index(bout_exemplars, name='metaoperation_id')
bout_exemplars = neuron_sequence.sequence_info.index

output_proportions = output_proportions.loc[bout_exemplars]

cumulative_n_operations_map = neuron_sequence.sequence_info.loc[bout_exemplars]['cumulative_n_operations']

output_proportions_long = (
    output_proportions
    .fillna(0)
    .reset_index()
    .melt(value_name="proportion", id_vars=f"{prefix}operation_id")
)

output_proportions_long["cumulative_n_operations"] = output_proportions_long[f"{prefix}operation_id"].map(cumulative_n_operations_map)

sns.lineplot(
    data=output_proportions_long,
    x="cumulative_n_operations",
    y="proportion",
    hue="post_mtype",
    palette=ctype_hues,
)

#%%

print(output_proportions['L6short-a'].tolist())

#%%
diffs = output_proportions['L6short-a'].diff().abs()
changes = diffs[diffs > 0.1].index
pre_changes = diffs.index[diffs.index.get_indexer(changes) - 1]


#%%
import pyvista as pv

from neurovista import center_camera
pv.set_jupyter_backend("client")

#%%
plotter = pv.Plotter(shape=(1, 1))

pre = pre_changes[0]
post = changes[0]

neuron_sequence.edits.loc[post]

neuron_before = neuron_sequence.resolved_sequence[pre]
neuron_after = neuron_sequence.resolved_sequence[post]

skeleton_before = neuron_before.to_skeleton_polydata()
skeleton_after = neuron_after.to_skeleton_polydata()

plotter.add_mesh(skeleton_before, color="blue", line_width=1.5, opacity=0.2)
plotter.add_mesh(skeleton_after, color="red", line_width=1.5, opacity=0.2)

edit_loc = neuron_sequence.base_neuron.nodes.query(f"{prefix}operation_added == {post}")[['x', 'y', 'z']].mean(axis=0)

center_camera(plotter, edit_loc.values, max_dist)

plotter.show()

#%%
plotter = pv.Plotter(shape=(1, 1))

pre = pre_changes[1]
post = changes[1]

neuron_sequence.edits.loc[post]

neuron_before = neuron_sequence.resolved_sequence[pre]
neuron_after = neuron_sequence.resolved_sequence[post]

skeleton_before = neuron_before.to_skeleton_polydata()
skeleton_after = neuron_after.to_skeleton_polydata()

plotter.add_mesh(skeleton_before, color="blue", line_width=1.5, opacity=0.2)
plotter.add_mesh(skeleton_after, color="red", line_width=1.5, opacity=0.2)

edit_loc = neuron_sequence.base_neuron.nodes.query(f"{prefix}operation_added == {post}")[['x', 'y', 'z']].mean(axis=0)

center_camera(plotter, edit_loc.values, max_dist)

plotter.show()

#%%

plotter = pv.Plotter(shape=(1, 2))

plotter.subplot(0, 0)   
pre = pre_changes[0]
post = changes[0]

neuron_sequence.edits.loc[post]

neuron_before = neuron_sequence.resolved_sequence[pre]
neuron_after = neuron_sequence.resolved_sequence[post]

skeleton_before = neuron_before.to_skeleton_polydata()
skeleton_after = neuron_after.to_skeleton_polydata()

plotter.add_mesh(skeleton_before, color="blue", line_width=1.5, opacity=0.2)
plotter.add_mesh(skeleton_after, color="red", line_width=1.5, opacity=0.2)

edit_loc = neuron_sequence.base_neuron.nodes.query(f"{prefix}operation_added == {post}")[['x', 'y', 'z']].mean(axis=0)

focus_sphere = pv.Sphere(radius=5_000, center=edit_loc.values)
plotter.add_mesh(focus_sphere, color="black", opacity=0.1)

plotter.subplot(0, 1)

pre = pre_changes[1]
post = changes[1]

neuron_sequence.edits.loc[post]

neuron_before = neuron_sequence.resolved_sequence[pre]
neuron_after = neuron_sequence.resolved_sequence[post]

skeleton_before = neuron_before.to_skeleton_polydata()
skeleton_after = neuron_after.to_skeleton_polydata()

plotter.add_mesh(skeleton_before, color="blue", line_width=1.5, opacity=0.2)
plotter.add_mesh(skeleton_after, color="red", line_width=1.5, opacity=0.2)

edit_loc = neuron_sequence.base_neuron.nodes.query(f"{prefix}operation_added == {post}")[['x', 'y', 'z']].mean(axis=0)
focus_sphere = pv.Sphere(radius=5_000, center=edit_loc.values)
plotter.add_mesh(focus_sphere, color="black", opacity=0.1)

plotter.link_views()

center_camera(plotter, edit_loc.values, max_dist)

plotter.show()


#%%
neuron_sequence._sequence_info[3]

#%%
neuron_sequence._sequence_info[5]

#%%
datas = neuron_sequence.sequence_info.iloc[[21, 22]]['pre_synapses']
pre_before = datas.iloc[0]
pre_after = datas.iloc[1]

#%%


#%%
# neurons = neuron_sequence.resolved_sequence
# last_neuron = next(iter(neurons.values()))
# relevant_edits = []
# for i, (sample_id, neuron) in enumerate(neurons.items()):
#     if neuron.nodes.index.equals(last_neuron.nodes.index) and i != 0:
#         continue
#     else:
#         relevant_edits.append(sample_id)
#     last_neuron = neuron
# relevant_edits = pd.Index(pd.Series(relevant_edits, dtype="Int64"), name="operation_id")


output_proportions_long = (
    output_proportions.loc[]
    .fillna(0)
    .reset_index()
    .melt(value_name="proportion", id_vars="operation_id")
)
output_proportions_long["cumulative_n_operations"] = output_proportions_long[
    "operation_id"
].map(neuron_sequence.sequence_info["cumulative_n_operations"])
output_proportions_long["order"] = relevant_edits.get_indexer_for(
    output_proportions_long["operation_id"]
)


from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(tight_layout=True)

sns.lineplot(
    data=output_proportions_long,
    x="cumulative_n_operations",
    y="proportion",
    hue="post_mtype",
    ax=ax,
    legend=False,
    palette=ctype_hues,
)
ax.set_ylabel("Proportion of \noutput synapses")
ax.set_xlabel("Number of operations")
ax.spines[["top", "right"]].set_visible(False)

children = ax.get_children()
xdata_by_line = {}
ydata_by_line = {}
for i, child in enumerate(children):
    if isinstance(child, plt.Line2D):
        xdata_by_line[i] = child.get_xdata()
        ydata_by_line[i] = child.get_ydata()


#%%
def update(sample_id):
    children = ax.get_children()
    for i, child in enumerate(children):
        if isinstance(child, plt.Line2D):
            if sample_id is None:
                child.set_xdata(xdata_by_line[i][:1])
                child.set_ydata(ydata_by_line[i][:1])
            else:
                child.set_xdata(
                    xdata_by_line[i][: relevant_edits.get_loc(sample_id) + 1]
                )
                child.set_ydata(
                    ydata_by_line[i][: relevant_edits.get_loc(sample_id) + 1]
                )


# %%
from pkg.sequence import create_merge_and_clean_sequence

order_by = "time"
sequence = create_merge_and_clean_sequence(neuron, root_id, order_by=order_by)

# %%

sequence


#%%


# %%
# print(sequence.sequence_info.shape)
# sequence = sequence.select_by_bout("has_merge", keep="last")
# print(sequence.sequence_info.shape)


# def compute_target_counts(synapses_df: pd.DataFrame, by=None):
#     result = synapses_df.size()
#     return result


# counts_by_mtype = sequence.apply_to_synapses_by_sample(
#     compute_target_counts, which="pre"
# )

# %%

from matplotlib import animation

new_relevant_edits = relevant_edits.values.copy()
new_relevant_edits = np.repeat(relevant_edits, 8)
new_relevant_edits = pd.Index(new_relevant_edits)
ani = FuncAnimation(fig, update, frames=new_relevant_edits, repeat=True, interval=1)
writer = animation.PillowWriter(fps=20)
ani.save("scatter.gif", writer=writer)
plt.show()

# %%
# target_id
name = f"all_edits_merge_clean_by_time-root_id={root_id}"


# plotter = pv.Plotter()
# pv.global_theme.transparent_background = False

# plotter.background_color = pv.Color()

animate_neuron_edit_sequence(
    neuron_sequence,
    folder=folder,
    name=name,
    window_size=(1152, 1152),
    n_rotation_steps=8,
    setback=-3 * max_dist,
    azimuth_step_size=0.5,
    line_width=1.5,
    fps=20,
    highlight_last=3,
    highlight_decay=0.95,
    highlight_point_size=4,
    doc_save=True,
    caption=target_id,
    group="all_edits_by_time_animation",
    verbose=verbose,
    edit_point_size=1,
    merge_color="black",
    split_color="black",
    fig=fig,
    update=update,
)


# %%
name = f"all_edits_by_time_with_plots-target_id={target_id}"
animate_neuron_edit_sequence(
    neuron_sequence,
    folder=folder,
    name=name,
    window_size=(1600, 900),
    n_rotation_steps=5,
    setback=-4 * max_dist,
    azimuth_step_size=0.5,
    line_width=1.5,
    fps=20,
    highlight_last=3,
    highlight_decay=0.95,
    fig=fig,
    update=update,
    doc_save=True,
    caption=target_id,
    group="all_edits_by_time_animation_with_plots",
    verbose=verbose,
)

# %%

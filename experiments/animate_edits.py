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

for root_id in manifest.query("is_sample").index[3:4]:
    full_neuron = load_neuronframe(root_id, client)

    prefix = ""
    # simple time-ordered case
    neuron_sequence = NeuronFrameSequence(
        full_neuron,
        prefix=prefix,
        edit_label_name="operation_id",
        warn_on_missing=verbose,
    )
    neuron_sequence.edits.sort_values("time", inplace=True)

    for i in tqdm(
        range(len(neuron_sequence.edits)),
        leave=False,
        desc="Applying edits...",
        disable=not verbose,
    ):
        operation_id = neuron_sequence.edits.index[i]
        neuron_sequence.apply_edits(operation_id, warn_on_missing=verbose)

    max_dist = 0
    for _, neuron in neuron_sequence.resolved_sequence.items():
        positions = neuron.nodes[["x", "y", "z"]]
        soma_pos = full_neuron.nodes[["x", "y", "z"]].loc[full_neuron.nucleus_id]
        positions_values = positions.values
        soma_positions_values = soma_pos.values.reshape(1, -1)

        distances = np.squeeze(
            pairwise_distances(positions_values, soma_positions_values)
        )
        max_dist = max(max_dist, distances.max())

    target_id = manifest.loc[root_id, "target_id"]
    name = f"all_edits_by_time-target_id={target_id}"

    pre_synapses = neuron_sequence.base_neuron.pre_synapses
    pre_synapses["post_mtype"] = pre_synapses["post_pt_root_id"].map(
        mtypes["cell_type"]
    )

    neuron_sequence.sequence_info["pre_synapses"]

    output_proportions = neuron_sequence.apply_to_synapses_by_sample(
        compute_target_proportions, which="pre", by="post_mtype"
    )

    neurons = neuron_sequence.resolved_sequence
    last_neuron = next(iter(neurons.values()))
    relevant_edits = []
    for i, (sample_id, neuron) in enumerate(neurons.items()):
        if neuron.nodes.index.equals(last_neuron.nodes.index) and i != 0:
            continue
        else:
            relevant_edits.append(sample_id)
        last_neuron = neuron
    relevant_edits = pd.Index(
        pd.Series(relevant_edits, dtype="Int64"), name="operation_id"
    )

    output_proportions_long = (
        output_proportions.loc[relevant_edits]
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

# %%
X = output_proportions.loc[relevant_edits].values
from sklearn.metrics import pairwise_distances

dists = pairwise_distances(X, metric="cityblock")
dists = dists[-1]
dists = pd.Series(
    dists, index=output_proportions.loc[relevant_edits].index, name="distance_to_final"
).to_frame()
dists["order"] = relevant_edits.get_indexer_for(dists.index)
dists["cumulative_n_operations"] = dists.index.map(
    neuron_sequence.sequence_info["cumulative_n_operations"]
)

# %%

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


from matplotlib import animation

new_relevant_edits = relevant_edits.values.copy()
new_relevant_edits = np.repeat(relevant_edits, 9)
new_relevant_edits = pd.Index(new_relevant_edits)
ani = FuncAnimation(fig, update, frames=new_relevant_edits, repeat=True, interval=1)
writer = animation.PillowWriter(fps=20)
# ani.save("../../../../talks/docs/slides/berlin-2024/images", writer=writer)
ani.save("../../../talks/docs/slides/berlin-2024/images/scatter.gif", writer=writer)
plt.show()

# %%
name = f"all_edits_by_time-target_id={target_id}"

import pyvista as pv

# plotter = pv.Plotter()
pv.global_theme.transparent_background = False

# plotter.background_color = pv.Color()

animate_neuron_edit_sequence(
    neuron_sequence,
    # folder=folder,
    # folder = "../../../talks/docs/slides/berlin-2024/images/",
    # name=name,
    # name =
    window_size=(1152, 1152),
    n_rotation_steps=4,
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
    loop=1,
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


from matplotlib.animation import FuncAnimation

from pkg.plot import set_context

fig, axs = plt.subplots(2, 1, tight_layout=True, figsize=(5, 8))

set_context(font_scale=2.5)
ax = axs[0]
sns.lineplot(
    data=output_proportions_long,
    x="cumulative_n_operations",
    y="proportion",
    hue="post_mtype",
    ax=ax,
    legend=False,
    palette=ctype_hues,
    linewidth=3,
)
ax.set_ylabel("Proportion of outputs")
ax.set_xlabel("Number of operations")
ax.spines[["top", "right"]].set_visible(False)

children = ax.get_children()
xdata_by_line = {}
ydata_by_line = {}
for i, child in enumerate(children):
    if isinstance(child, plt.Line2D):
        xdata_by_line[i] = child.get_xdata()
        ydata_by_line[i] = child.get_ydata()

ax = axs[1]
sns.lineplot(
    dists,
    ax=ax,
    x="cumulative_n_operations",
    y="distance_to_final",
    linewidth=3,
    color="black",
)
ax.set_ylabel("Distance to final state")
ax.set_xlabel("Number of operations")
ax.spines[["top", "right"]].set_visible(False)

children = ax.get_children()
xdata2_by_line = {}
ydata2_by_line = {}
for i, child in enumerate(children):
    if isinstance(child, plt.Line2D):
        xdata2_by_line[i] = child.get_xdata()
        ydata2_by_line[i] = child.get_ydata()


def update(sample_id):
    ax = axs[0]
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

    ax = axs[1]
    children = ax.get_children()
    for i, child in enumerate(children):
        if isinstance(child, plt.Line2D):
            if sample_id is None:
                child.set_xdata(xdata2_by_line[i][:1])
                child.set_ydata(ydata2_by_line[i][:1])
            else:
                child.set_xdata(
                    xdata2_by_line[i][: relevant_edits.get_loc(sample_id) + 1]
                )
                child.set_ydata(
                    ydata2_by_line[i][: relevant_edits.get_loc(sample_id) + 1]
                )


theme = pv.global_theme
theme.font.family = "arial"
name = f"all_edits_by_time_with_plots-target_id={target_id}"
animate_neuron_edit_sequence(
    neuron_sequence,
    folder=folder,
    # folder="../../../talks/docs/slides/berlin-2024/images/",
    name=name,
    window_size=(2400, 1350),
    # window_size=(1600, 900),
    # window_size=(800, 450),
    # n_rotation_steps=2,
    n_rotation_steps=2,
    setback=-3 * max_dist,
    azimuth_step_size=0.5,
    line_width=1.5,
    fps=20,
    highlight_last=3,
    highlight_decay=0.95,
    fig=fig,
    update=update,
    doc_save=True,
    caption=target_id,
    # group="all_edits_by_time_animation_with_plots",
    verbose=verbose,
    chart_size=(0.9, 0.8),
    chart_loc=(0.0, 0.05),
    loop=0,
)
# %%

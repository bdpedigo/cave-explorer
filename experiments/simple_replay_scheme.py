# %%

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import pairwise_distances

from pkg.metrics import compute_precision_recall, compute_target_proportions
from pkg.neuronframe import NeuronFrameSequence, load_neuronframe
from pkg.utils import load_casey_palette, load_manifest, load_mtypes

# %%
sns.set_context("notebook", font_scale=1.25)

client = cc.CAVEclient("minnie65_phase3_v1")

ctype_hues = load_casey_palette()
mtypes = load_mtypes(client)
manifest = load_manifest()

folder = "animations"
verbose = False


prefix = "meta"
order_by = "time"
# key = "not_has_split"
hide = False

cv = client.info.segmentation_cloudvolume()
cv.cache.enabled = True

for root_id in manifest.query("is_sample").index[:]:
    full_neuron = load_neuronframe(root_id, client)

    neuron_sequence = NeuronFrameSequence(
        full_neuron,
        prefix=prefix,
        edit_label_name=f"{prefix}operation_id",
        warn_on_missing=verbose,
    )

    if prefix == "meta":
        key = "has_merge"
    else:
        key = "is_merge"

    random_seed = 0
    neuron_sequence.edits["not_has_split"] = ~neuron_sequence.edits["has_split"]
    neuron_sequence.edits["is_all_merge"] = (
        neuron_sequence.edits["has_merge"] & ~neuron_sequence.edits["has_split"]
    )

    if order_by == "time":
        neuron_sequence.edits.sort_values(["time"], inplace=True)
    # elif order_by == "random":
    #     rng = np.random.default_rng(random_seed)
    #     neuron_sequence.edits["random"] = rng.random(len(neuron_sequence.edits))
    #     neuron_sequence.edits.sort_values([key, "random"], inplace=True)

    for i, edit in neuron_sequence.edits.iterrows():
        neuron_sequence.apply_edits(i)

    if not neuron_sequence.is_completed:
        raise UserWarning("Neuron is not completed.")

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

    precision_recall = compute_precision_recall(neuron_sequence, which="pre")

    # neuron_sequence.select_by_bout(
    if hide:
        by = "is_all_merge"
        keep = "last"
        bouts = neuron_sequence.sequence_info[by].fillna(False).cumsum() + 1
        bouts.iloc[0] = 0
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
    else:
        bout_exemplars = neuron_sequence.sequence_info.index

    precision_recall = precision_recall.join(neuron_sequence.sequence_info)
    precision_recall = precision_recall.loc[bout_exemplars]
    precision_recall["cumulative_n_operations"] = precision_recall[
        "n_operations"
    ].cumsum()

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(
        data=precision_recall.reset_index(),
        x="cumulative_n_operations",
        y="pre_synapse_recall",
        ax=ax,
        color="blue",
        linestyle="--",
    )
    sns.lineplot(
        data=precision_recall.reset_index(),
        x="cumulative_n_operations",
        y="pre_synapse_precision",
        ax=ax,
        color="red",
        linestyle=":",
    )
    sns.lineplot(
        data=precision_recall.reset_index(),
        x="cumulative_n_operations",
        y="pre_synapse_f1",
        ax=ax,
        color="purple",
    )
    ax.set_title(manifest.loc[root_id, "target_id"])
    ax.set_ylim((0, 1))
    plt.show()

# %%

candidates = neuron_sequence.edits.query("n_operations == 2")
# candidates = candidates.query("is_merges == [False, True]")
candidates = candidates[candidates["is_merges"].apply(tuple) == (False, True)]

# %%
neuron_sequence.base_neuron.edges.query("metaoperation_added == @candidates.index[]")

#%%
neuron_sequence.base_neuron.nodes.query("metaoperation_added == @candidates.index[0]")

#%%



# %%

import pyvista as pv

skels = {
    i: skel.to_skeleton_polydata()
    for i, skel in enumerate(neuron_sequence.resolved_sequence.values())
}

state_range = np.arange(len(skels))
state_index = pd.Series(skels.keys(), index=state_range)

plotter = pv.Plotter()

actors = []


def plot_skeleton_at_index(index):
    index = int(index)

    for actor in actors:
        plotter.remove_actor(actor)

    actor = plotter.add_mesh(skels[index], color="black", line_width=1.5)
    actors.append(actor)


plotter.add_slider_widget(
    plot_skeleton_at_index, [0, len(state_index) - 1], value=0, fmt="%.0f"
)

plotter.show()

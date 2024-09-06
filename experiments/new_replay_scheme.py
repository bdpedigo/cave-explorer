# %%

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm

from pkg.metrics import compute_target_proportions
from pkg.neuronframe import (
    NeuronFrameSequence,
    load_neuronframe,
)
from pkg.utils import load_casey_palette, load_manifest, load_mtypes

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

prefix = "meta"
order_by = "time"
key = "not_has_split"


for root_id in manifest.query("is_sample").index:
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
    # key = "has_merge"

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
            row = neuron_sequence.edits.loc[next_operation]
            # if row["has_merge"]:
            #     print(row)
            neuron_sequence.apply_edits(next_operation, only_additions=False)
        i += 1
        pbar.update(1)
        # break
    pbar.close()

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
    # neuron_sequence.select_by_bout(
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
    # bout_exemplars = pd.Index(bout_exemplars, name='metaoperation_id')
    # bout_exemplars = neuron_sequence.sequence_info.index

    output_proportions = output_proportions.loc[bout_exemplars]

    cumulative_n_operations_map = neuron_sequence.sequence_info.loc[bout_exemplars][
        "cumulative_n_operations"
    ]

    output_proportions_long = (
        output_proportions.fillna(0)
        .reset_index()
        .melt(value_name="proportion", id_vars=f"{prefix}operation_id")
    )

    output_proportions_long["cumulative_n_operations"] = output_proportions_long[
        f"{prefix}operation_id"
    ].map(cumulative_n_operations_map)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(
        data=output_proportions_long,
        x="cumulative_n_operations",
        y="proportion",
        hue="post_mtype",
        palette=ctype_hues,
        legend=False,
        ax=ax,
    )
    plt.show()

# %%
bout_exemplars = (
    neuron_sequence.sequence_info.index.to_series()
    .groupby(bouts, sort=False)
    .apply(lambda x: x.iloc[keep_ind])
).values

skeleton_poly_by_index = {}
for i, neuron in enumerate(neuron_sequence.resolved_sequence.values()):
    skeleton_poly = neuron.to_skeleton_polydata()
    skeleton_poly_by_index[i] = skeleton_poly

import pandas as pd
import pyvista as pv

pv.set_jupyter_backend("trame")
plotter = pv.Plotter()


actors = []


def plot_skeleton_at_index(index):
    index = int(index)

    operation = neuron_sequence.sequence_info.index[index]
    if pd.isna(operation):
        color = "black"
    else:
        if np.isin(operation, bout_exemplars[1:]):
            color = "green"
        else:
            color = "black"

    for actor in actors:
        plotter.remove_actor(actor)

    actor = plotter.add_mesh(skeleton_poly_by_index[index], color=color, line_width=1.5)
    actors.append(actor)


plotter.add_slider_widget(
    plot_skeleton_at_index, [0, len(skeleton_poly_by_index) - 1], value=0, fmt="%.0f"
)

plotter.show()


# %%

print(output_proportions["L6short-a"].tolist())

# %%
diffs = output_proportions["L6short-a"].diff().abs()
changes = diffs[diffs > 0.1].index
pre_changes = diffs.index[diffs.index.get_indexer(changes) - 1]


# %%
import pyvista as pv
from neurovista import center_camera

pv.set_jupyter_backend("client")

# %%
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

edit_loc = neuron_sequence.base_neuron.nodes.query(
    f"{prefix}operation_added == {post}"
)[["x", "y", "z"]].mean(axis=0)

center_camera(plotter, edit_loc.values, max_dist)

plotter.show()

# %%
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

edit_loc = neuron_sequence.base_neuron.nodes.query(
    f"{prefix}operation_added == {post}"
)[["x", "y", "z"]].mean(axis=0)

center_camera(plotter, edit_loc.values, max_dist)

plotter.show()

# %%

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

edit_loc = neuron_sequence.base_neuron.nodes.query(
    f"{prefix}operation_added == {post}"
)[["x", "y", "z"]].mean(axis=0)

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

edit_loc = neuron_sequence.base_neuron.nodes.query(
    f"{prefix}operation_added == {post}"
)[["x", "y", "z"]].mean(axis=0)
focus_sphere = pv.Sphere(radius=5_000, center=edit_loc.values)
plotter.add_mesh(focus_sphere, color="black", opacity=0.1)

plotter.link_views()

center_camera(plotter, edit_loc.values, max_dist)

plotter.show()


# %%

prefix = "meta"
# simple time-ordered case
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

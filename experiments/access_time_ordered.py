# %%
import os

os.environ["LAZYCLOUD_USE_CLOUD"] = "True"
os.environ["LAZYCLOUD_RECOMPUTE"] = "False"
os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"

import pickle

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import pairwise_distances_argmin
from tqdm.auto import tqdm

from pkg.edits import count_synapses_by_sample
from pkg.neuronframe import load_neuronframe
from pkg.paths import OUT_PATH
from pkg.plot import savefig

# %%
palette_file = "/Users/ben.pedigo/code/skedits/skedits-app/skedits/data/ctype_hues.pkl"

with open(palette_file, "rb") as f:
    ctype_hues = pickle.load(f)

ctype_hues = {ctype: tuple(ctype_hues[ctype]) for ctype in ctype_hues.keys()}

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v507")
query_neurons.sort_values("id", inplace=True)

prefix = "meta"
path = OUT_PATH / "access_time_ordered"

# %%
mtypes = client.materialize.query_table("aibs_metamodel_mtypes_v661_v2")
root_id_counts = mtypes["pt_root_id"].value_counts()
root_id_singles = root_id_counts[root_id_counts == 1].index
mtypes = mtypes.query("pt_root_id in @root_id_singles")
mtypes.set_index("pt_root_id", inplace=True)

# %%


def find_closest_point(df, point):
    if not isinstance(point, np.ndarray):
        point = np.array(point)
    X = df.loc[:, ["x", "y", "z"]].values
    min_iloc = pairwise_distances_argmin(point.reshape(1, -1), X)[0]
    return df.index[min_iloc]


# TODO
# something weird going on w/ 0 -
# it seems like the nucleus node is later removed by a subsequent edit
# maybe need to look up in the table again to make sure everything is still there
# and the nucleus hasn't changed?

for i, root_id in enumerate(query_neurons["pt_root_id"].values[:20]):
    if i == 8:
        continue
    print("---")
    print("ROOT ID", root_id)
    print("i", i)
    print("---")
    print()
    full_neuron = load_neuronframe(root_id, client)
    if full_neuron == "Not for now!":
        continue

    metaedits = full_neuron.metaedits.sort_values("time")

    # split_metaedits = metaedits.query("~has_merge")
    split_metaedits = metaedits.query("has_split")

    merge_metaedits = metaedits.query("has_merge")

    merge_op_ids = merge_metaedits.index
    split_op_ids = split_metaedits.index
    applied_op_ids = list(split_op_ids)

    # edge case where the neuron's ultimate soma location is itself a merge node
    if full_neuron.nodes.loc[full_neuron.nucleus_id, f"{prefix}operation_added"] != -1:
        applied_op_ids.append(
            full_neuron.nodes.loc[full_neuron.nucleus_id, f"{prefix}operation_added"]
        )

    neuron_list = []
    applied_merges = []
    resolved_synapses = {}

    for i in tqdm(
        range(len(merge_op_ids) + 1), desc="Applying edits and resolving synapses..."
    ):
        # apply the next operation
        current_neuron = full_neuron.set_edits(
            applied_op_ids, inplace=False, prefix=prefix
        )

        if full_neuron.nucleus_id in current_neuron.nodes.index:
            current_neuron.select_nucleus_component(inplace=True)
        else:
            point_id = find_closest_point(
                current_neuron.nodes,
                full_neuron.nodes.loc[full_neuron.nucleus_id, ["x", "y", "z"]],
            )
            current_neuron.select_component_from_node(point_id, inplace=True)

        current_neuron.remove_unused_synapses(inplace=True)
        neuron_list.append(current_neuron)
        resolved_synapses[i] = {
            "resolved_pre_synapses": current_neuron.pre_synapses.index.to_list(),
            "resolved_post_synapses": current_neuron.post_synapses.index.to_list(),
            f"{prefix}operation_added": applied_op_ids[-1] if i > 0 else None,
        }

        # select the next operation to apply
        out_edges = full_neuron.edges.query(
            "source.isin(@current_neuron.nodes.index) | target.isin(@current_neuron.nodes.index)"
        )

        out_edges = out_edges.drop(current_neuron.edges.index)

        possible_operations = out_edges[f"{prefix}operation_added"].unique()

        ordered_ops = merge_op_ids[merge_op_ids.isin(possible_operations)]

        # HACK?
        ordered_ops = ordered_ops[~ordered_ops.isin(applied_merges)]

        if len(ordered_ops) == 0:
            break

        applied_op_ids.append(ordered_ops[0])
        applied_merges.append(ordered_ops[0])

    print(f"No remaining merges, stopping ({(i+1) / len(merge_op_ids):.2f})")

    final_neuron = full_neuron.set_edits(full_neuron.edits.index, inplace=False)
    final_neuron.select_nucleus_component(inplace=True)
    final_neuron.remove_unused_synapses(inplace=True)

    assert final_neuron.nodes.index.sort_values().equals(
        current_neuron.nodes.index.sort_values()
    )

    assert final_neuron.edges.index.sort_values().equals(
        current_neuron.edges.index.sort_values()
    )

    pre_synapses = full_neuron.pre_synapses
    post_synapses = full_neuron.post_synapses

    pre_synapses["post_mtype"] = pre_synapses["post_pt_root_id"].map(
        mtypes["cell_type"]
    )
    post_synapses["pre_mtype"] = post_synapses["pre_pt_root_id"].map(
        mtypes["cell_type"]
    )

    resolved_synapses = pd.DataFrame(resolved_synapses).T

    resolved_pre_synapses = resolved_synapses["resolved_pre_synapses"]
    post_mtype_counts = count_synapses_by_sample(
        pre_synapses, resolved_pre_synapses, "post_mtype"
    )

    counts_table = post_mtype_counts
    post_mtype_stats_tidy = counts_table.reset_index().melt(
        var_name="post_mtype", value_name="count", id_vars="sample"
    )

    post_mtype_probs = counts_table / counts_table.sum(axis=1).values[:, None]
    post_mtype_probs.fillna(0, inplace=True)
    post_mtype_probs_tidy = post_mtype_probs.reset_index().melt(
        var_name="post_mtype", value_name="prob", id_vars="sample"
    )

    post_mtype_stats_tidy["prob"] = post_mtype_probs_tidy["prob"]

    post_mtype_stats_tidy[f"{prefix}operation_added"] = post_mtype_stats_tidy[
        "sample"
    ].map(resolved_synapses[f"{prefix}operation_added"])

    post_mtype_stats_tidy = post_mtype_stats_tidy.join(
        metaedits, on=f"{prefix}operation_added"
    )

    final_probs = post_mtype_probs.iloc[-1]

    diffs = (
        ((((post_mtype_probs - final_probs) ** 2).sum(axis=1)) ** 0.5)
        .to_frame("diff")
        .reset_index()
    )

    sns.set_context("talk")

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    sns.lineplot(
        data=post_mtype_stats_tidy,
        x="sample",
        y="count",
        hue="post_mtype",
        legend=False,
        palette=ctype_hues,
        ax=ax,
    )
    ax.set_xlabel("Metaoperation added")
    ax.set_ylabel("# output synapses")
    ax.spines[["top", "right"]].set_visible(False)
    savefig(
        f"output_synapses_access_time_ordered-root_id={root_id}",
        fig,
        folder="access_time_ordered",
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    sns.lineplot(
        data=post_mtype_stats_tidy,
        x="sample",
        y="prob",
        hue="post_mtype",
        legend=False,
        palette=ctype_hues,
        ax=ax,
    )
    ax.set_xlabel("Metaoperation added")
    ax.set_ylabel("Proportion of output synapses")
    ax.spines[["top", "right"]].set_visible(False)
    savefig(
        f"output_proportion_access_time_ordered-root_id={root_id}",
        fig,
        folder="access_time_ordered",
    )
    print()

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    sns.lineplot(
        data=post_mtype_stats_tidy,
        x="sample",
        y="centroid_distance_to_nuc_um",
        hue="post_mtype",
        legend=False,
        palette=ctype_hues,
        ax=ax,
    )
    ax.set_xlabel("Metaoperation added")
    ax.set_ylabel("Distance to nucleus (nm)")
    ax.spines[["top", "right"]].set_visible(False)
    savefig(
        f"distance_access_time_ordered-root_id={root_id}",
        fig,
        folder="access_time_ordered",
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    sns.lineplot(
        data=diffs,
        x="sample",
        y="diff",
    )
    ax.set_xlabel("Metaoperation added")
    ax.set_ylabel("Distance from final")
    ax.spines[["top", "right"]].set_visible(False)

    savefig(
        f"distance_from_final_access_time_ordered-root_id={root_id}",
        fig,
        folder="access_time_ordered",
    )

    resolved_synapses.to_csv(path / f"resolved_synapses-root_id={root_id}.csv")

    post_mtype_stats_tidy.to_csv(path / f"post_mtype_stats_tidy-root_id={root_id}.csv")

    diffs.to_csv(path / f"diffs-root_id={root_id}.csv")


# %%
import os
import pickle

import caveclient as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from pkg.edits import count_synapses_by_sample
from pkg.neuronframe import load_neuronframe
from pkg.plot import savefig

# %%

os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"

# %%
palette_file = "/Users/ben.pedigo/code/skedits/skedits-app/skedits/data/ctype_hues.pkl"

with open(palette_file, "rb") as f:
    ctype_hues = pickle.load(f)

ctype_hues = {ctype: tuple(ctype_hues[ctype]) for ctype in ctype_hues.keys()}

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v507")
query_neurons.sort_values("id", inplace=True)

# 10 looked "unstable"
# 11 looked "stable"
root_id = query_neurons["pt_root_id"].values[11]
# root_id = 864691135865971164
# root_id = 864691135992790209
prefix = "meta"


# %%
import numpy as np
from sklearn.metrics import pairwise_distances_argmin


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

for root_id in query_neurons["pt_root_id"].values[0:12]:
    full_neuron = load_neuronframe(root_id, client)

    metaedits = full_neuron.metaedits.sort_values("time")

    pure_split_metaedits = metaedits.query("~has_merge")

    merge_metaedits = metaedits.query("has_merge")

    merge_op_ids = merge_metaedits.index
    split_op_ids = pure_split_metaedits.index
    applied_op_ids = list(split_op_ids)

    # edge case where the neuron's ultimate soma location is itself a merge node
    if full_neuron.nodes.loc[full_neuron.nucleus_id, f"{prefix}operation_added"] != -1:
        applied_op_ids.append(
            full_neuron.nodes.loc[full_neuron.nucleus_id, f"{prefix}operation_added"]
        )

    neuron_list = []
    applied_merges = []
    resolved_synapses = {}

    for i in tqdm(range(len(merge_op_ids) + 1)):
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
            "metaoperation_added": applied_op_ids[-1] if i > 0 else None,
        }

        # select the next operation to apply
        out_edges = full_neuron.edges.query(
            "source.isin(@current_neuron.nodes.index) | target.isin(@current_neuron.nodes.index)"
        )
        # print(len(out_edges), "out edges")

        out_edges = out_edges.drop(current_neuron.edges.index)

        # print(len(out_edges), "out edges after removing current edges")

        possible_operations = out_edges[f"{prefix}operation_added"].unique()
        # print(len(possible_operations), "possible operations")

        ordered_ops = merge_op_ids[merge_op_ids.isin(possible_operations)]

        # HACK
        ordered_ops = ordered_ops[~ordered_ops.isin(applied_merges)]

        if len(ordered_ops) == 0:
            print(f"no remaining merges, stopping ({i / len(merge_op_ids):.2f})")
            break

        applied_op_ids.append(ordered_ops[0])
        applied_merges.append(ordered_ops[0])

    # current_neuron.generate_neuroglancer_link(client)

    final_neuron = full_neuron.set_edits(full_neuron.edits.index, inplace=False)
    final_neuron.select_nucleus_component(inplace=True)
    final_neuron.remove_unused_synapses(inplace=True)

    final_neuron.nodes.index.sort_values().equals(
        current_neuron.nodes.index.sort_values()
    )

    final_neuron.edges.index.sort_values().equals(
        current_neuron.edges.index.sort_values()
    )

    mtypes = client.materialize.query_table("aibs_metamodel_mtypes_v661_v2")
    root_id_counts = mtypes["pt_root_id"].value_counts()
    root_id_singles = root_id_counts[root_id_counts == 1].index
    mtypes = mtypes.query("pt_root_id in @root_id_singles")
    mtypes.set_index("pt_root_id", inplace=True)

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
    savefig(f"output_synapses_access_time_ordered-root_id={root_id}", fig)

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
    savefig(f"output_proportion_access_time_ordered-root_id={root_id}", fig)


# %%

# TODO i bet these are all things such that filtered = False
# metaedits.loc[merge_op_ids[~np.isin(merge_op_ids, applied_merges)]]

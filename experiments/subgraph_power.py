# %%

import time

import caveclient as cc
import numpy as np
import pandas as pd
from cloudfiles import CloudFiles
from joblib import Parallel, delayed
from networkframe import NetworkFrame

from pkg.neuronframe import load_neuronframe
from pkg.plot import set_context
from pkg.sequence import create_time_ordered_sequence

# %%

set_context("paper", font_scale=1.5)

client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v795")

root_options = query_neurons["pt_root_id"].values


# %%

nodes = pd.DataFrame()
nodes["working_root_id"] = root_options

# take my list of root IDs
# make sure I have the latest root ID for each, using `get_latest_roots`
is_current_mask = client.chunkedgraph.is_latest_roots(root_options)
outdated_roots = root_options[~is_current_mask]
root_map = dict(zip(root_options[is_current_mask], root_options[is_current_mask]))
for outdated_root in outdated_roots:
    latest_roots = client.chunkedgraph.get_latest_roots(outdated_root)
    sub_nucs = client.materialize.query_table(
        "nucleus_detection_v0", filter_in_dict={"pt_root_id": latest_roots}
    )
    if len(sub_nucs) == 1:
        root_map[outdated_root] = sub_nucs.iloc[0]["pt_root_id"]
    else:
        print(f"Multiple nuc roots for {outdated_root}")

updated_root_options = np.array([root_map[root] for root in root_options])
nodes["current_root_id"] = updated_root_options

# map to nucleus IDs
current_nucs = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"pt_root_id": updated_root_options},
    # select_columns=["id", "pt_root_id"],
).set_index("pt_root_id")["id"]
nodes["target_id"] = nodes["current_root_id"].map(current_nucs)


# %%
timestamp = pd.to_datetime("2021-07-01 00:00:00", utc=True)

nucs = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"id": nodes["target_id"].to_list()},
).set_index("id")
nodes["pt_supervoxel_id"] = nodes["target_id"].map(nucs["pt_supervoxel_id"])
nodes["timestamp_root_from_chunkedgraph"] = client.chunkedgraph.get_roots(
    nodes["pt_supervoxel_id"], timestamp=timestamp
)
nodes["nuc_depth"] = nodes["target_id"].map(nucs["pt_position"].apply(lambda x: x[1]))

past_nucs = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"id": nodes["target_id"].to_list()},
    # select_columns=["id", "pt_root_id"],
    timestamp=timestamp,
).set_index("id")["pt_root_id"]
nodes["timestamp_root_from_table"] = nodes["target_id"].map(past_nucs)

mtypes = client.materialize.query_table(
    "allen_column_mtypes_v2", filter_in_dict={"target_id": nodes["target_id"].to_list()}
)
nodes["mtype"] = nodes["target_id"].map(mtypes.set_index("target_id")["cell_type"])

# %%
cloud_bucket = "allen-minnie-phase3"
folder = "edit_sequences"

cf = CloudFiles(f"gs://{cloud_bucket}/{folder}")

files = list(cf.list())
files = pd.DataFrame(files, columns=["file"])

# pattern is root_id=number as the beginning of the file name
# extract the number from the file name and store it in a new column
files["root_id"] = files["file"].str.split("=").str[1].str.split("-").str[0].astype(int)
files["order_by"] = files["file"].str.split("=").str[2].str.split("-").str[0]
files["random_seed"] = files["file"].str.split("=").str[3].str.split("-").str[0]


file_counts = files.groupby("root_id").size()
has_all = file_counts[file_counts == 12].index

files_finished = files.query("root_id in @has_all")

root_options = files_finished["root_id"].unique()

# %%
nodes["has_sequence"] = nodes["current_root_id"].isin(root_options)
nodes["ctype"] = nodes["target_id"].map(
    query_neurons.set_index("target_id")["cell_type"]
)
# %%
root_options = nodes.query("has_sequence")["working_root_id"]

# %%
# timestamps = pd.date_range("2022-07-01", "2024-01-01", freq="M", tz="UTC")


# loop over the entire set of nodes to consider and get the collection of pre
# and post synapses for each, across time


def get_pre_post_synapse_ids(root_id):
    neuron = load_neuronframe(root_id, client, cache_verbose=False)
    return neuron.pre_synapses.index, neuron.post_synapses.index


currtime = time.time()
pre_posts = Parallel(n_jobs=8, verbose=10)(
    delayed(get_pre_post_synapse_ids)(root_id) for root_id in root_options
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

all_pre_synapses = []
all_post_synapses = []
for pre, post in pre_posts:
    all_pre_synapses.extend(pre)
    all_post_synapses.extend(post)

# %%

# the induced synapses is the set of all synapses which show up at least once pre, and
# at least once post across all time

induced_synapses = np.intersect1d(all_pre_synapses, all_post_synapses)

print("Number of induced synapses:", len(induced_synapses))

# %%

# now, extract the synapses that are in this induced set for each neuron.


def process_resolved_synapses(sequence, root_id, which="pre"):
    synapse_sets = sequence.sequence_info[f"{which}_synapses"]
    synapse_sets = synapse_sets.apply(
        lambda x: np.intersect1d(x, induced_synapses)
    ).to_frame()
    synapse_sets["time"] = sequence.edits["time"]
    synapse_sets["time"] = synapse_sets["time"].fillna("2019-07-01 00:00:00")
    synapse_sets["datetime"] = pd.to_datetime(synapse_sets["time"], utc=True)
    synapse_sets[f"{which}_root_id"] = root_id

    breaks = list(synapse_sets["datetime"])
    breaks.append(
        pd.to_datetime("2070-01-01 00:00:00", utc=True)
    )  # TODO could make this explicit about now
    intervals = pd.IntervalIndex.from_breaks(breaks, closed="left")
    synapse_sets["interval"] = intervals

    synapse_sets = synapse_sets.reset_index(drop=False)
    synapses = synapse_sets.explode(f"{which}_synapses").rename(
        columns={f"{which}_synapses": "synapse_id"}
    )
    return synapses


def get_info_by_time(root_id):
    neuron = load_neuronframe(root_id, client, cache_verbose=False)
    sequence = create_time_ordered_sequence(neuron, root_id)
    pre_synapse_sets = process_resolved_synapses(sequence, root_id, which="pre")
    post_synapse_sets = process_resolved_synapses(sequence, root_id, which="post")
    edits = neuron.edits.copy()
    edits["root_id"] = root_id
    return pre_synapse_sets, post_synapse_sets, edits


outs = Parallel(n_jobs=8, verbose=10)(
    delayed(get_info_by_time)(root_id) for root_id in root_options
)


pre_synapselist = []
post_synapselist = []
all_edit_tables = []

for pre, post, edit_table in outs:
    pre_synapselist.append(pre)
    post_synapselist.append(post)
    all_edit_tables.append(edit_table)

pre_synapses = pd.concat(pre_synapselist)
post_synapses = pd.concat(post_synapselist)
all_edits = pd.concat(all_edit_tables)

# %%


# %%
def synapselist_at_time(timestamp, remove_loops=True):
    # pre_synapses_at_time = pre_synapses.query("@timestamp in interval")
    # post_synapses_at_time = post_synapses.query("@timestamp in interval")
    pre_synapses_at_time = pre_synapses[
        pd.IntervalIndex(pre_synapses.interval).contains(timestamp)
    ]
    post_synapses_at_time = post_synapses[
        pd.IntervalIndex(post_synapses.interval).contains(timestamp)
    ]

    pre_synapses_at_time = pre_synapses_at_time.set_index("synapse_id")
    post_synapses_at_time = post_synapses_at_time.set_index("synapse_id")
    synapselist = pre_synapses_at_time.join(
        post_synapses_at_time, how="inner", lsuffix="_pre", rsuffix="_post"
    )
    synapselist["source"] = synapselist["pre_root_id"]
    synapselist["target"] = synapselist["post_root_id"]
    if remove_loops:
        synapselist = synapselist.query("source != target")
    return synapselist


synapselist = synapselist_at_time(pd.to_datetime("2021-07-01 00:00:00", utc=True))

# %%
from sklearn.model_selection import StratifiedShuffleSplit

p_effort = 0.5

sss = StratifiedShuffleSplit(n_splits=10, train_size=0.5, random_state=88)

for sub_ilocs, _ in sss.split(nodes.index, nodes["mtype"]):
    sub_nodes = nodes.iloc[sub_ilocs]
    sub_nodes.groupby("mtype").size()

# %%
root_id = sub_nodes.iloc[0]["working_root_id"]
neuron = load_neuronframe(root_id, client, cache_verbose=False)
sequence = create_time_ordered_sequence(neuron, root_id)
pre_synapse_sets = process_resolved_synapses(sequence, root_id, which="pre")
post_synapse_sets = process_resolved_synapses(sequence, root_id, which="post")
edits = neuron.edits.copy()
edits["root_id"] = root_id

# %%


from pkg.sequence import create_merge_and_clean_sequence

working_nodes = nodes.query("has_sequence")


def load_for_neuron(root_id):
    client = cc.CAVEclient("minnie65_phase3_v1")
    neuron = load_neuronframe(root_id, client, cache_verbose=False)
    rows = []
    # sequence = create_time_ordered_sequence(neuron, root_id)
    rng = np.random.default_rng(8888)
    for i in range(10):
        seed = rng.integers(0, np.iinfo(np.int32).max, dtype=np.int32)
        sequence = create_merge_and_clean_sequence(
            neuron, root_id, order_by="random", random_seed=seed
        )
        # sequence = create_merge_and_clean_sequence(neuron, root_id, order_by='random', random_seed=)
        for p_neuron_effort in [0, 0.25, 0.5, 0.75, 1.0]:
            # n_total_edits = len(sequence) - 1
            # n_select_edits = np.floor(n_total_edits * p_neuron_effort).astype(int)
            # selected_state = sequence.sequence_info.iloc[n_select_edits]
            n_total_edits = sequence.sequence_info["cumulative_n_operations"].max()
            n_select_edits = np.floor(n_total_edits * p_neuron_effort).astype(int)
            selected_state_idx = np.abs(
                sequence.sequence_info["cumulative_n_operations"] - n_select_edits
            ).idxmin()
            selected_state = sequence.sequence_info.loc[selected_state_idx]

            row = {
                "root_id": root_id,
                "p_neuron_effort": p_neuron_effort,
                "n_total_edits": n_total_edits,
                "n_select_edits": n_select_edits,
                "order": selected_state["order"],
                "n_outputs": len(neuron.pre_synapses),
                "n_inputs": len(neuron.post_synapses),
                "seed": seed,
                "i": i,
            }

            for which in ["pre", "post"]:
                selected_synapses = selected_state[f"{which}_synapses"]
                selected_synapses = np.intersect1d(selected_synapses, induced_synapses)
                row[f"{which}_synapses"] = selected_synapses.tolist()
            rows.append(row)
    return rows


rows_by_neuron = Parallel(n_jobs=8, verbose=10)(
    delayed(load_for_neuron)(root_id)
    for root_id in working_nodes["working_root_id"].unique()
)
rows = []
for rbn in rows_by_neuron:
    rows.extend(rbn)

# %%
synapse_selection_df = pd.DataFrame(rows)

# %%
synapse_selection_df.query("p_neuron_effort == 0.5")["n_select_edits"].sum()

# %%
neuron_outputs = synapse_selection_df.groupby("root_id")["n_outputs"].first()


# %%

nfs_by_strategy = {}
stats = {}
for p_neuron_effort in [0, 0.25, 0.5, 0.75, 1.0]:
    synapse_selections_at_effort = synapse_selection_df.query(
        "p_neuron_effort == @p_neuron_effort"
    )
    for p_neurons in [0.25, 0.5, 0.75, 1.0]:
        if p_neurons == 1.0:
            index_list = [(np.arange(len(working_nodes)), None)]
        else:
            sss = StratifiedShuffleSplit(
                n_splits=25, train_size=p_neurons, random_state=88
            )
            index_list = sss.split(working_nodes.index, working_nodes["mtype"])

        for i, (sub_ilocs, _) in enumerate(index_list):
            sub_nodes = working_nodes.iloc[sub_ilocs]
            sub_roots = sub_nodes["working_root_id"]
            sub_synapse_selections = synapse_selections_at_effort.query(
                "root_id in @sub_roots"
            )
            for sequence in range(10):
                seq_synapse_selections = sub_synapse_selections.query("i == @sequence")

                pre_synapses_long = (
                    seq_synapse_selections.explode("pre_synapses")[
                        ["root_id", "pre_synapses"]
                    ]
                    .rename({"root_id": "source"}, axis=1)
                    .dropna()
                    .set_index("pre_synapses")
                )
                post_synapses_long = (
                    seq_synapse_selections.explode("post_synapses")[
                        ["root_id", "post_synapses"]
                    ]
                    .rename({"root_id": "target"}, axis=1)
                    .dropna()
                    .set_index("post_synapses")
                )

                sub_edges = pre_synapses_long.join(
                    post_synapses_long, how="inner", lsuffix="_pre", rsuffix="_post"
                )
                sub_edges = sub_edges.query("source != target")

                nfs_by_strategy[
                    (p_neurons, p_neuron_effort, i, sequence)
                ] = NetworkFrame(
                    nodes=sub_nodes.set_index("working_root_id"), edges=sub_edges
                )
                row = {
                    "n_edges": len(sub_edges),
                    "n_nodes": len(sub_nodes),
                    "n_select_edits": seq_synapse_selections["n_select_edits"].sum(),
                    "sequence": sequence,
                }
                stats[(p_neurons, p_neuron_effort, i, sequence)] = row

# %%
stats_by_strategy = pd.DataFrame(stats).T
stats_by_strategy.index.set_names(
    ["p_neurons", "p_neuron_effort", "split", "sequence"], inplace=True
)

# %%
synapse_group_counts_by_strategy = []
for key, nf in nfs_by_strategy.items():
    groupby = nf.groupby_nodes("mtype")
    node_counts = nf.nodes.groupby("mtype").size()
    synapse_counts = groupby.apply_edges("size")
    avg_synapses_per_type = synapse_counts.unstack() / node_counts
    avg_synapses_per_type = avg_synapses_per_type.stack().rename("avg_synapses")

    # synapse_counts.name = "n_synapses"
    synapse_counts = avg_synapses_per_type.to_frame()
    synapse_counts["p_neurons"] = key[0]
    synapse_counts["p_neuron_effort"] = key[1]
    synapse_counts["split"] = key[2]
    synapse_counts["sequence"] = key[3]
    # synapse_counts.name = key
    synapse_counts.index.set_names(["source_mtype", "target_mtype"], inplace=True)
    synapse_group_counts_by_strategy.append(synapse_counts)
    # synapse_group_counts_by_strategy.append(synapse_counts)

synapse_group_counts_by_strategy = pd.concat(synapse_group_counts_by_strategy)
synapse_group_counts_by_strategy

# %%
synapse_group_counts_by_strategy.reset_index(inplace=True)

# %%
synapse_group_counts_by_strategy["connection"] = list(
    zip(
        synapse_group_counts_by_strategy["source_mtype"],
        synapse_group_counts_by_strategy["target_mtype"],
    )
)

# %%
synapse_group_counts_by_strategy[
    "n_select_edits"
] = synapse_group_counts_by_strategy.set_index(
    ["p_neurons", "p_neuron_effort", "split", "sequence"]
).index.map(stats_by_strategy["n_select_edits"])

# %%
synapse_group_counts_by_strategy["strategy"] = list(
    zip(
        synapse_group_counts_by_strategy["p_neurons"],
        synapse_group_counts_by_strategy["p_neuron_effort"],
    )
)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

set_context()
fig, axs = plt.subplots(
    4, 4, figsize=(12, 10), sharex=True, sharey=False, constrained_layout=False
)
row_index = pd.Index(synapse_group_counts_by_strategy["source_mtype"].unique())
col_index = pd.Index(synapse_group_counts_by_strategy["target_mtype"].unique())

for source_mtype, sub_df in synapse_group_counts_by_strategy.groupby("source_mtype"):
    for target_mtype, sub_sub_df in sub_df.groupby("target_mtype"):
        i = row_index.get_loc(source_mtype)
        j = col_index.get_loc(target_mtype)
        ax = axs[i, j]
        if i == 0 and j == 3:
            show_legend = True
        else:
            show_legend = False
        sns.scatterplot(
            data=sub_sub_df,
            x="n_select_edits",
            y="avg_synapses",
            style="p_neurons",
            hue="p_neuron_effort",
            palette="tab10",
            ax=ax,
            legend=show_legend,
            s=20,
        )
        if show_legend:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

        ax.set(title=f"{source_mtype}" + r"$\rightarrow$" + f"{target_mtype}")

# for i in range(4):
#     axs[i, 4].axis("off")
#     # i += 1
# plt.tight_layout()
# Need to generalize this better for arbitrary metrics on the subgraph, or something like that
# redo this but using the average number of synapses from a group k neuron to a groul l neuron
# as the metric of interest

# %%

from scipy.stats import pearsonr

reciprocal_ratios_by_strategy = []
for key, nf in nfs_by_strategy.items():
    edges = nf.edges.copy()
    edges = edges.groupby(["source", "target"]).size().reset_index(name="weight")
    edge_index = edges.set_index(["source", "target"]).index.unique()
    sources = edges["source"]
    targets = edges["target"]
    reverse_edge_index = pd.MultiIndex.from_tuples(
        zip(targets, sources), names=["source", "target"]
    ).unique()

    reciprocal_edges = edge_index.intersection(reverse_edge_index)
    if len(reciprocal_edges) == 0:
        reciprocal_ratio = 0
        stat = np.nan
        pvalue = np.nan
    else:
        edge_weights_forward = edges.set_index(["source", "target"]).loc[
            reciprocal_edges
        ]["weight"]
        edge_weights_reverse = edges.set_index(["source", "target"]).loc[
            reciprocal_edges.reorder_levels(["target", "source"])
        ]["weight"]
        stat, pvalue = pearsonr(
            edge_weights_forward.values, edge_weights_reverse.values
        )

        reciprocal_ratio = len(reciprocal_edges) / len(edge_index)

    reciprocal_ratios_by_strategy.append(
        {
            "p_neurons": key[0],
            "p_neuron_effort": key[1],
            "split": key[2],
            "sequence": key[3],
            "reciprocal_ratio": reciprocal_ratio,
            "reciprocal_weight_corr": stat,
            "reciprocal_weight_pvalue": pvalue,
        }
    )

reciprocal_ratios_by_strategy = pd.DataFrame(reciprocal_ratios_by_strategy)
reciprocal_ratios_by_strategy[
    "n_select_edits"
] = reciprocal_ratios_by_strategy.set_index(
    ["p_neurons", "p_neuron_effort", "split", "sequence"]
).index.map(stats_by_strategy["n_select_edits"])

# %%
sns.scatterplot(x=edge_weights_forward.values, y=edge_weights_reverse.values, alpha=0.2)

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

sns.scatterplot(
    data=reciprocal_ratios_by_strategy,
    x="p_neuron_effort",
    y="reciprocal_ratio",
    style="p_neurons",
    hue="p_neuron_effort",
    palette="tab10",
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

sns.scatterplot(
    data=reciprocal_ratios_by_strategy,
    x="n_select_edits",
    y="reciprocal_ratio",
    style="p_neurons",
    hue="p_neuron_effort",
    palette="tab10",
)
ax.set(xlabel="Number of edits used", ylabel="Reciprocal ratio")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.scatterplot(
    data=reciprocal_ratios_by_strategy,
    x="n_select_edits",
    y="reciprocal_weight_corr",
    style="p_neurons",
    hue="p_neuron_effort",
    palette="tab10",
)
ax.set(xlabel="Number of edits used", ylabel="Reciprocal weight correlation")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

sns.scatterplot(
    data=reciprocal_ratios_by_strategy,
    x="n_select_edits",
    y="reciprocal_weight_pvalue",
    style="p_neurons",
    hue="p_neuron_effort",
    palette="tab10",
)

sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
# %%

for key, nf in nfs_by_strategy.items():
    nf = nf.apply_node_features("mtype")
    nf = nf.query_edges("source_mtype == 'ITC'")
    edges = nf.edges
    outputs_by_type = edges.groupby(["source", "target_mtype"]).size()
    outputs_by_type = outputs_by_type.reset_index(name="n_synapses")
    # outputs_by_type["prop_synapses"] = outputs_by_type["n_synapses"] / outputs_by_type[
    #     "source"
    # ].map(neuron_outputs)
    outputs_by_type["prop_synapses"] = outputs_by_type["n_synapses"] / outputs_by_type[
        "source"
    ].map(edges.groupby("source").size())
    outputs_by_type.set_index(["source", "target_mtype"], inplace=True)

    # if (key[0] == 1 and key[1] == 0.5) or (key[0] == 0.5 and key[1] == 1):
    #     props_by_type_wide = outputs_by_type["prop_synapses"].unstack().fillna(0)
    #     grid = sns.clustermap(props_by_type_wide.T, cmap="viridis")
    #     grid.figure.suptitle(f"p_neurons = {key[0]}, p_neuron_effort = {key[1]}")

# %%

from scipy.cluster.hierarchy import dendrogram, linkage

from pkg.plot import set_context

set_context(font_scale=2)

keys = []
# keys += [(0.5, 1.0, 0, 0), (1.0, 0.5, 0, 0)]
# keys += [(0.5, 1.0, 1, 1), (1.0, 0.5, 0, 1)]
# keys += [(0.5, 1.0, 2, 2), (1.0, 0.5, 0, 2)]
# keys += [(0.5, 1.0, 3, 3), (1.0, 0.5, 0, 3)]
keys += [(0.5, 1.0, 4, 4), (1.0, 0.5, 0, 4)]
keys += [(0.5, 1.0, 5, 5), (1.0, 0.5, 0, 5)]
keys += [(0.5, 1.0, 6, 6), (1.0, 0.5, 0, 6)]
keys += [(0.5, 1.0, 7, 7), (1.0, 0.5, 0, 7)]

fig, axs = plt.subplots(2, 4, figsize=(25, 10), sharey=True, constrained_layout=False)
for i, key in enumerate(keys):
    nf = nfs_by_strategy[key]
    nf = nf.apply_node_features("mtype")
    nf = nf.query_edges("source_mtype == 'ITC'")
    edges = nf.edges
    outputs_by_type = edges.groupby(["source", "target_mtype"]).size()
    outputs_by_type = outputs_by_type.reset_index(name="n_synapses")
    outputs_by_type["prop_synapses"] = outputs_by_type["n_synapses"] / outputs_by_type[
        "source"
    ].map(edges.groupby("source").size())
    outputs_by_type.set_index(["source", "target_mtype"], inplace=True)

    props_by_type_wide = outputs_by_type["prop_synapses"].unstack().fillna(0)
    # grid = sns.clustermap(
    #     props_by_type_wide.T, cmap="viridis", xticklabels=False, figsize=(5, 3)
    # )
    # grid.figure.suptitle(f"p_neurons = {key[0]}, p_neuron_effort = {key[1]}")
    mat = props_by_type_wide.T.loc[["DTC", "PTC", "ITC", "STC"]]

    # do a seaborn style sorting of the columns using scipy dendrogram, but don't
    # plot the dendrogram

    Z = linkage(mat.T, method="ward")
    indices = dendrogram(Z, no_plot=True)["leaves"]
    mat = mat.iloc[:, indices]
    if mat.iloc[0, 0] < 0.4:
        mat_values = mat.values
        mat_values = mat_values[:, ::-1]
        mat = pd.DataFrame(mat_values, index=mat.index, columns=mat.columns)

    ax = axs.T.flat[i]
    sns.heatmap(
        mat,
        xticklabels=False,
        cbar=False,
        ax=ax,
        cmap="RdBu_r",
        center=0,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

ax = axs[0, 0]
ax.text(
    -0.2,
    0.5,
    "Half neurons,\nall edits",
    ha="right",
    va="center",
    transform=ax.transAxes,
    fontsize="xx-large",
)
plt.setp(ax.get_yticklabels(), rotation=0)
ax = axs[1, 0]
ax.text(
    -0.2,
    0.5,
    "All neurons,\nhalf edits",
    ha="right",
    va="center",
    transform=ax.transAxes,
    fontsize="xx-large",
)
plt.setp(ax.get_yticklabels(), rotation=0)

for i in range(4):
    ax = axs[0, i]
    ax.set_title("Sample " + str(i + 1))
    ax = axs[1, i]
    ax.set_title("Sample " + str(i + 1))


# %%

key = (1.0, 1.0, 0, 0)
nf = nfs_by_strategy[key]
nf = nf.apply_node_features("mtype")
nf = nf.query_edges("source_mtype == 'ITC'")
edges = nf.edges
outputs_by_type = edges.groupby(["source", "target_mtype"]).size()
outputs_by_type = outputs_by_type.reset_index(name="n_synapses")
outputs_by_type["prop_synapses"] = outputs_by_type["n_synapses"] / outputs_by_type[
    "source"
].map(edges.groupby("source").size())
outputs_by_type.set_index(["source", "target_mtype"], inplace=True)

props_by_type_wide = outputs_by_type["prop_synapses"].unstack().fillna(0)
# grid = sns.clustermap(
#     props_by_type_wide.T, cmap="viridis", xticklabels=False, figsize=(5, 3)
# )
# grid.figure.suptitle(f"p_neurons = {key[0]}, p_neuron_effort = {key[1]}")
mat = props_by_type_wide.T.loc[["DTC", "PTC", "ITC", "STC"]]

# do a seaborn style sorting of the columns using scipy dendrogram, but don't
# plot the dendrogram

Z = linkage(mat.T, method="ward")
indices = dendrogram(Z, no_plot=True)["leaves"]
mat = mat.iloc[:, indices]
if mat.iloc[0, 0] < 0.4:
    mat_values = mat.values
    mat_values = mat_values[:, ::-1]
    mat = pd.DataFrame(mat_values, index=mat.index, columns=mat.columns)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.heatmap(
    mat,
    xticklabels=False,
    cbar=False,
    ax=ax,
    cmap="RdBu_r",
    center=0,
)
ax.set_xlabel("")
ax.set_ylabel("")
plt.setp(ax.get_yticklabels(), rotation=0)

# %%

sns.clustermap(props_by_type_wide.T)
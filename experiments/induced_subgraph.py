# %%

import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import MatrixGrid
from graspologic.embed import ClassicalMDS
from graspologic.utils import pass_to_ranks
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from networkframe import NetworkFrame
from tqdm.auto import tqdm

from pkg.io import write_variable
from pkg.metrics import MotifFinder
from pkg.neuronframe import load_neuronframe
from pkg.plot import savefig, set_context
from pkg.sequence import create_time_ordered_sequence
from pkg.utils import load_manifest, start_client

# %%

set_context("paper", font_scale=1.5)

client = start_client()

manifest = load_manifest()

nodes = manifest.query("in_inhibitory_column & has_all_sequences").copy()

root_options = nodes.index

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

# nodes = pd.DataFrame()
# nodes.index = root_options
timestamps = pd.date_range("2020-04-01", "2024-07-01", freq="M", tz="UTC")

used_nodes = nodes.query("index in @root_options")
nfs_by_time = {}
for timestamp in timestamps:
    synapselist = synapselist_at_time(timestamp)
    nf = NetworkFrame(used_nodes.copy(), synapselist.copy())
    nfs_by_time[timestamp] = nf

# %%
nf.nodes

# %%

rows = []
all_edits["timestamp"] = pd.to_datetime(all_edits["time"], utc=True)
for timestamp in timestamps:
    applied_edits = all_edits.query("timestamp <= @timestamp")
    n_appled_edits = applied_edits.shape[0]
    rows.append({"timestamp": timestamp, "n_edits": n_appled_edits})
    n_edits_per_neuron = applied_edits.groupby("root_id").size()
    nf = nfs_by_time[timestamp]
    nf.nodes["n_edits"] = nf.nodes.index.map(n_edits_per_neuron)
    # nf.nodes['ctype'] = nf.nodes.index.map(nodes['ctype'÷])

time_df = pd.DataFrame(rows)

# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.lineplot(data=time_df, x="timestamp", y="n_edits", ax=ax)

# %%
timebins = pd.cut(all_edits["timestamp"], bins=timestamps, right=True)
n_edits_by_time = timebins.value_counts().sort_index().cumsum()

# %%
final_nf = nfs_by_time[timestamps[-1]]

ordering = {"PTC": 0, "DTC": 1, "STC": 2, "ITC": 3}
final_nf.nodes["mtype_order"] = final_nf.nodes["mtype"].map(ordering)
final_nf.nodes = final_nf.nodes.sort_values(["mtype_order", "ctype", "nuc_y"])

n_nodes = final_nf.nodes.shape[0]
write_variable(n_nodes, "induced_subgraph/n_nodes")

# %%

# nf = nfs_by_time[timestamps[4]]

MAX_EDITS = final_nf.nodes["n_edits"].max()


def plot_adjacency(nf, ax):
    nf = nf.deepcopy()
    mg = MatrixGrid(row_ticks=False, col_ticks=False, ax=ax)

    nf.nodes = nf.nodes.loc[final_nf.nodes.index]
    nf.nodes["position"] = np.arange(nf.nodes.shape[0]) + 0.5
    adj = nf.to_sparse_adjacency(weight_col=None).todense()
    adj = pass_to_ranks(adj)
    sns.heatmap(
        adj,
        ax=mg.ax,
        square=True,
        cbar=False,
        cmap="RdBu_r",
        center=0,
        xticklabels=False,
        yticklabels=False,
    )
    ax_left = mg.append_axes("left", size="10%", pad=0.05)
    ax_top = mg.append_axes("top", size="10%", pad=0.05)
    ax_left.barh(nf.nodes["position"], nf.nodes["n_edits"], color="black")
    ax_left.invert_xaxis()
    ax_left.spines[["left", "top", "bottom"]].set_visible(False)
    ax_left.set_xticks([])
    ax_left.set_xlim(MAX_EDITS, 0)
    # ax_left.set_xlabel("# edits")
    # ax_left.set_xticks([0, 100, 200, 300])
    # ax_left.set_xticklabels([0, 100, 200, 300], rotation=45)

    ax_top.bar(nf.nodes["position"], nf.nodes["n_edits"], color="black")
    ax_top.spines[["top", "right", "left"]].set_visible(False)
    # ax_top.set_ylabel("# edits")
    # ax_top.set_yticks([0, 100, 200, 300])
    ax_top.set_yticks([])
    ax_top.set_ylim(0, MAX_EDITS)

    return mg
    # mg.ax.set_xlabel("Neuron")


# %%

fig, axs = plt.subplots(3, 5, figsize=(15, 9))
for i, timestamp in enumerate(timestamps[:15]):
    nf = nfs_by_time[timestamp]
    ax = axs.flatten()[i]
    mg = plot_adjacency(nf, ax)
    mg.set_title(timestamp.strftime("%Y-%m"))
    mg.left_axs[0].spines["right"].set_visible(True)

plt.tight_layout()

savefig("adjacency_by_time", fig, folder="induced_subgraph", doc_save=True)


# %%
diffs_by_time = pd.DataFrame(index=timestamps, columns=timestamps, dtype=float)
for i, timestamp1 in enumerate(timestamps):
    nf1 = nfs_by_time[timestamp1]
    nf1.nodes = nf1.nodes.loc[final_nf.nodes.index]
    adj1 = nf1.to_sparse_adjacency(weight_col=None)
    adj1 = adj1.todense()
    for j, timestamp2 in enumerate(timestamps):
        nf2 = nfs_by_time[timestamp2]
        nf2.nodes = nf2.nodes.loc[final_nf.nodes.index]
        assert (nf1.nodes.index == nf2.nodes.index).all()
        adj2 = nf2.to_sparse_adjacency(weight_col=None)
        adj2 = adj2.todense()
        diff = np.linalg.norm(adj1 - adj2, ord="fro")
        diffs_by_time.loc[timestamps[i], timestamps[j]] = diff
        diffs_by_time.loc[timestamps[j], timestamps[i]] = diff

diffs_by_time.fillna(0.0, inplace=True)


# %%

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(
    diffs_by_time, cmap="RdBu_r", center=0, square=True, ax=ax, cbar_kws={"shrink": 0.5}
)
labels = diffs_by_time.columns.strftime("%Y-%m")
tick_locs = np.arange(0, len(labels)) + 0.5
ax.set_yticks(tick_locs)
ax.set_yticklabels(labels, rotation=0, fontsize=10)
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_title("Network dissimilarity (Frobenius norm)")

savefig("network_dissimilarity", fig, folder="induced_subgraph", doc_save=True)

# %%

cmds = ClassicalMDS(n_components=2, dissimilarity="precomputed")
embedding = cmds.fit_transform(diffs_by_time.values)

embedding_df = pd.DataFrame(
    embedding, index=diffs_by_time.index, columns=["MDS1", "MDS2"]
)
embedding_df["timestamp"] = timestamps

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(
    data=embedding_df, x="MDS1", y="MDS2", hue="timestamp", ax=ax, legend=False
)

# connect with lines in time
for i in range(embedding_df.shape[0] - 1):
    x1, y1 = embedding_df.iloc[i, :2]
    x2, y2 = embedding_df.iloc[i + 1, :2]
    ax.plot([x1, x2], [y1, y2], color="black", alpha=0.5, zorder=-1, linewidth=1)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# timestamps on every 5th point
for i, timestamp in enumerate(embedding_df["timestamp"]):
    if i % 5 == 0:
        ax.text(
            embedding_df.iloc[i, 0],
            embedding_df.iloc[i, 1],
            timestamp.strftime("%Y-%m"),
            fontsize=8,
        )


# %%
iloc_index = np.arange(0, len(timestamps))

month_diffs = diffs_by_time.values[iloc_index[:-1], iloc_index[1:]]

fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, constrained_layout=True)

n_edits = pd.cut(all_edits["timestamp"], bins=timestamps).value_counts().sort_index()

ax = axs[0]
sns.lineplot(x=n_edits.index.categories.right, y=n_edits, ax=ax)
ax.get_xaxis().set_tick_params(rotation=45)
ax.set_ylabel("# edits")
ax.set_xlabel("Time")

ax = axs[1]
sns.scatterplot(y=month_diffs, x=diffs_by_time.index[1:], ax=ax)
ax.get_xaxis().set_tick_params(rotation=45)
ax.set_ylabel("Network dissimilarity (F-norm)")
ax.set_xlabel("Time")

savefig("n-edits-net-dissimilarity", fig, folder="induced_subgraph", doc_save=True)

# %%

synapse_counts_by_time = {}
for timestamp in timestamps:
    nf: NetworkFrame = nfs_by_time[timestamp]
    groupby = nf.groupby_nodes("mtype")
    synapse_counts = groupby.apply_edges("size")
    synapse_counts_by_time[timestamp] = synapse_counts
    synapse_counts.name = timestamp

    node_counts = nf.nodes.groupby("mtype").size()
    possible_edges = node_counts.values[:, None] * node_counts.values[None, :]
    possible_edges = pd.DataFrame(
        possible_edges, index=node_counts.index, columns=node_counts.index
    )

# %%
synapse_counts_by_time = pd.concat(synapse_counts_by_time, axis=1)

# %%
synapse_counts_by_time_long = synapse_counts_by_time.melt(
    ignore_index=False, var_name="timestamp", value_name="n_synapses"
).reset_index()
synapse_counts_by_time_long["connection"] = list(
    zip(
        synapse_counts_by_time_long["source_mtype"],
        synapse_counts_by_time_long["target_mtype"],
    )
)
synapse_counts_by_time_long["cumulative_n_operations"] = (
    synapse_counts_by_time_long["timestamp"].map(n_edits_by_time).fillna(0).astype(int)
)
# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.lineplot(
    data=synapse_counts_by_time_long,
    x="timestamp",
    y="n_synapses",
    hue="connection",
    legend=False,
    ax=ax,
)
# rotate x labels
ax.get_xaxis().set_tick_params(rotation=45)
ax.set(ylabel="# synapses (groupwise)", xlabel="Time")

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.lineplot(
    data=synapse_counts_by_time_long,
    x="cumulative_n_operations",
    y="n_synapses",
    hue="connection",
    legend=False,
    ax=ax,
)
# rotate x labels
ax.get_xaxis().set_tick_params(rotation=45)
ax.set(ylabel="# synapses (groupwise)", xlabel="# operations")


# %%
# %%

mc = MotifFinder(orders=[2, 3], backend="grandiso", ignore_isolates=True)
# %%

motif_matches_by_time = {}
motif_counts_by_time = {}
for i, (time_label, nf) in tqdm(enumerate(nfs_by_time.items()), total=len(timestamps)):
    # if i < 15:
    #     continue
    adj = nf.to_sparse_adjacency(weight_col=None)
    g = nx.from_scipy_sparse_array(adj, create_using=nx.DiGraph)
    matches = mc.find_motif_monomorphisms(g)
    motif_matches_by_time[time_label] = matches
    motif_counts_by_time[time_label] = [
        len(matches_by_motif) for matches_by_motif in matches
    ]


# %%
last_motif_matches = motif_matches_by_time[timestamps[-1]]


def matches_to_index(matches):
    matches = pd.DataFrame(matches)
    matches = matches.set_index(matches.columns.to_list()).index
    return matches


rows = []
pbar = tqdm(total=len(timestamps))
for time_label, matches_by_motif in motif_matches_by_time.items():
    for motif_id, matches in enumerate(matches_by_motif):
        matches = matches_to_index(matches)

        last_matches = last_motif_matches[motif_id]
        last_matches = matches_to_index(last_matches)

        recall = len(last_matches.intersection(matches)) / len(last_matches)

        precision = len(matches.intersection(last_matches)) / len(matches)

        rows.append(
            {
                "timestamp": time_label,
                "motif_id": motif_id,
                "recall": recall,
                "precision": precision,
            }
        )
    pbar.update(1)

# %%
motif_recall_precision = pd.DataFrame(rows)
time_map = dict(zip(timestamps, np.arange(len(timestamps))))
motif_recall_precision["time_order"] = motif_recall_precision["timestamp"].map(time_map)
motif_recall_precision["n_edits"] = (
    motif_recall_precision["timestamp"].map(n_edits_by_time).fillna(0)
)
# %%
motif_counts_by_time = pd.DataFrame(motif_counts_by_time).T
# motif_counts_by_time /= motif_counts_by_time.min()
norm_motif_counts_by_time = motif_counts_by_time.copy()
norm_motif_counts_by_time = (
    norm_motif_counts_by_time / norm_motif_counts_by_time.iloc[-1]
)


unique_subgraphs = mc.motifs
scale = 0.75
fig, axs = plt.subplots(
    len(unique_subgraphs),
    2,
    figsize=(
        10,
        len(unique_subgraphs) * scale,
    ),
    constrained_layout=True,
    gridspec_kw={"width_ratios": [scale, 10]},
    sharex="col",
)

pos = {0: (-1, 0), 1: (1, 0), 2: (0, np.sqrt(2))}
pad = 0.3
for i, subgraph in enumerate(unique_subgraphs):
    nx.draw(subgraph, ax=axs[i, 0], node_size=50, pos=pos, width=3)
    axs[i, 0].set_axis_off()
    axs[i, 0].set(xlim=(-1 - pad, 1 + pad), ylim=(-pad, np.sqrt(2) + pad))

    ax = axs[i, 1]
    data = motif_counts_by_time[i]
    sns.lineplot(x=np.arange(len(data.values))[5:], y=data.values[5:], ax=ax)
ax.set(xlabel="Month")

# %%


unique_subgraphs = mc.motifs
scale = 0.75
fig, axs = plt.subplots(
    len(unique_subgraphs),
    2,
    figsize=(
        10,
        len(unique_subgraphs) * scale,
    ),
    constrained_layout=True,
    gridspec_kw={"width_ratios": [scale, 10]},
    sharex="col",
)

pos = {0: (-1, 0), 1: (1, 0), 2: (0, np.sqrt(2))}
pad = 0.3
for i, subgraph in enumerate(unique_subgraphs):
    nx.draw(subgraph, ax=axs[i, 0], node_size=50, pos=pos, width=3)
    axs[i, 0].set_axis_off()
    axs[i, 0].set(xlim=(-1 - pad, 1 + pad), ylim=(-pad, np.sqrt(2) + pad))

    ax = axs[i, 1]

    # data = motif_counts_by_time[i]
    # sns.lineplot(x=np.arange(len(data.values))[5:], y=data.values[5:], ax=ax)
ax.set(xlabel="Month")

# %%

set_context(font_scale=2)
n_motifs = len(mc.motifs)
fig, axs = plt.subplots(
    3, 5, figsize=(25, 15), sharex=True, sharey=True, constrained_layout=True
)

pos = {0: (-1, 0), 1: (1, 0), 2: (0, np.sqrt(2))}
pad = 0.3
hue = "n_edits"
for i, subgraph in enumerate(unique_subgraphs):
    ax = axs.flatten()[i]
    data = motif_recall_precision.query("motif_id == @i")

    sns.scatterplot(
        data=data,
        x="recall",
        y="precision",
        hue=hue,
        legend=False if i != 0 else "brief",
        ax=ax,
        palette="coolwarm",
        s=50,
    )
    sns.lineplot(
        data=data,
        x="recall",
        y="precision",
        ax=ax,
    )
    ax.set_title(f"Motif {i}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    if i == 0:
        sns.move_legend(ax, "upper left", title="# edits")

    sub_ax = ax.inset_axes([0.6, 0, 0.4, 0.4])
    nx.draw(
        subgraph,
        ax=sub_ax,
        node_size=300,
        pos=pos,
        width=5,
        arrowsize=20,
        node_color="black",
        edge_color="grey",
    )
    # sub_ax.set_axis_off()
    sub_ax.set(xlim=(-1 - pad, 1 + pad), ylim=(-pad, np.sqrt(2) + pad))

    ax.set_xticks(np.linspace(0, 1, 5))
    # ax.axis("on")
    # ax.axison = True
    # ax.set_xticklabels(xticklabels)
    # ax.set_yticklabels(yticklabels)

savefig("motif_recall_precision", fig, folder="induced_subgraph", doc_save=True)

# %%

mc = MotifFinder(orders=[2, 3, 4], backend="grandiso", ignore_isolates=True)
# %%

motif_matches_by_time = {}
motif_counts_by_time = {}
for i, (time_label, nf) in tqdm(enumerate(nfs_by_time.items()), total=len(timestamps)):
    # if i < 15:
    #     continue
    adj = nf.to_sparse_adjacency(weight_col=None)
    g = nx.from_scipy_sparse_array(adj, create_using=nx.DiGraph)
    matches = mc.find_motif_monomorphisms(g)
    motif_matches_by_time[time_label] = matches
    motif_counts_by_time[time_label] = [
        len(matches_by_motif) for matches_by_motif in matches
    ]


# %%
last_motif_matches = motif_matches_by_time[timestamps[-1]]


def matches_to_index(matches):
    matches = pd.DataFrame(matches)
    matches = matches.set_index(matches.columns.to_list()).index
    return matches


rows = []
pbar = tqdm(total=len(timestamps))
for time_label, matches_by_motif in motif_matches_by_time.items():
    for motif_id, matches in enumerate(matches_by_motif):
        matches = matches_to_index(matches)

        last_matches = last_motif_matches[motif_id]
        last_matches = matches_to_index(last_matches)

        recall = len(last_matches.intersection(matches)) / len(last_matches)

        precision = len(matches.intersection(last_matches)) / len(matches)

        rows.append(
            {
                "timestamp": time_label,
                "motif_id": motif_id,
                "recall": recall,
                "precision": precision,
            }
        )
    pbar.update(1)

# %%
motif_recall_precision = pd.DataFrame(rows)
time_map = dict(zip(timestamps, np.arange(len(timestamps))))
motif_recall_precision["time_order"] = motif_recall_precision["timestamp"].map(time_map)
motif_recall_precision["n_edits"] = (
    motif_recall_precision["timestamp"].map(n_edits_by_time).fillna(0)
)
# %%
motif_counts_by_time = pd.DataFrame(motif_counts_by_time).T
# motif_counts_by_time /= motif_counts_by_time.min()
norm_motif_counts_by_time = motif_counts_by_time.copy()
norm_motif_counts_by_time = (
    norm_motif_counts_by_time / norm_motif_counts_by_time.iloc[-1]
)
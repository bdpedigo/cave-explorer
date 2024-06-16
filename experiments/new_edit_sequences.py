# %%
import numpy as np
from caveclient import CAVEclient
from tqdm.auto import tqdm

from pkg.metrics import compute_target_proportions
from pkg.neuronframe import NeuronFrameSequence, load_neuronframe
from pkg.utils import load_manifest, load_mtypes

client = CAVEclient("minnie65_phase3_v1")
manifest = load_manifest()
manifest = manifest.query("in_inhibitory_column")

# %%

verbose = False
prefix = ""

mtypes = load_mtypes(client)

from pathlib import Path

from pkg.utils import load_casey_palette

palette = load_casey_palette()

out_path = Path("results/outs/new_edit_sequences")


# for root_id in manifest.query("is_sample").index[:5]:
def run_for_neuron(root_id):
    full_neuron = load_neuronframe(root_id, client, only_load=True)
    if full_neuron is None:
        return None

    full_neuron.pre_synapses["post_mtype"] = full_neuron.pre_synapses[
        "post_pt_root_id"
    ].map(mtypes["cell_type"])

    full_neuron.pre_synapses.to_csv(out_path / f"pre_synapses-root_id={root_id}.csv")
    full_neuron.post_synapses.to_csv(out_path / f"post_synapses-root_id={root_id}.csv")
    return
    # simple time-ordered case
    neuron_sequence = NeuronFrameSequence(
        full_neuron,
        prefix=prefix,
        edit_label_name=f"{prefix}operation_id",
        warn_on_missing=verbose,
    )

    order_by = "time"
    if order_by == "time":
        neuron_sequence.edits.sort_values(["is_merge", "time"], inplace=True)
    elif order_by == "random":
        rng = np.random.default_rng()
        neuron_sequence.edits["random"] = rng.random(len(neuron_sequence.edits))
        neuron_sequence.edits.sort_values(["is_merge", "random"], inplace=True)

    i = 0
    next_operation = True
    pbar = tqdm(
        total=len(neuron_sequence.edits), desc="Applying edits...", disable=True
    )
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

    neuron_sequence.sequence_info.to_csv(
        out_path / f"merge-clean-by-time-sequence_info-root_id={root_id}.csv"
    )

    if not neuron_sequence.is_completed:
        print("Neuron is not completed.")

    output_proportions = neuron_sequence.apply_to_synapses_by_sample(
        compute_target_proportions, which="pre", by="post_mtype"
    )

    output_proportions_long = (
        output_proportions.fillna(0)
        .reset_index()
        .melt(value_name="proportion", id_vars="operation_id")
    )
    output_proportions_long["cumulative_n_operations"] = output_proportions_long[
        "operation_id"
    ].map(neuron_sequence.sequence_info["cumulative_n_operations"])

    by = "is_merge"
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
    # bout_exemplars = neuron_sequence.sequence_info.index

    # output_proportions_long = output_proportions_long.query(
    #     "operation_id in @bout_exemplars"
    # )

    output_proportions_long["is_exemplar"] = False
    output_proportions_long.loc[
        output_proportions_long["operation_id"].isin(bout_exemplars), "is_exemplar"
    ] = True

    output_proportions_long.to_csv(
        out_path / f"merge-clean-by-time-output_proportions-root_id={root_id}.csv"
    )

    # fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    # sns.lineplot(
    #     data=output_proportions_long,
    #     x="cumulative_n_operations",
    #     y="proportion",
    #     hue="post_mtype",
    #     palette=palette,
    #     ax=ax,
    # )


from joblib import Parallel, delayed

Parallel(n_jobs=6, verbose=10)(
    delayed(run_for_neuron)(root_id) for root_id in manifest.index
)

# %%

import pandas as pd
from sklearn.metrics import pairwise_distances

output_proportions_by_root = []
sequence_info_by_root = []
distances_by_root = []
distances_to_next_by_root = []

for file in out_path.glob("*.csv"):
    # print(.head())
    root_id = int(file.name.split("root_id=")[1].split(".")[0])
    df = pd.read_csv(file, index_col=0)
    df["root_id"] = root_id
    if "output_proportions" in file.name:
        output_proportions_by_root.append(df)
        df = df.pivot(
            values="proportion",
            index=["root_id", "operation_id", "cumulative_n_operations", "is_exemplar"],
            columns="post_mtype",
        )
        df = df.sort_index(level="cumulative_n_operations")
        X = df.values
        dists = pairwise_distances(X, metric="manhattan")
        dists_df = pd.Series(
            data=dists[-1], index=df.index, name="cityblock"
        ).to_frame()
        distances_by_root.append(dists_df)

        square_dists_df = pd.DataFrame(data=dists, index=df.index, columns=df.index)
        # square_dists_df.values
        offdiag_dists = dists[np.arange(1, len(dists)), np.arange(len(dists) - 1)]

    else:
        sequence_info_by_root.append(df)

# %%
output_proportions = pd.concat(output_proportions_by_root)
sequence_infos = pd.concat(sequence_info_by_root)
distances = pd.concat(distances_by_root).reset_index()
distances = distances.sort_values(["root_id", "cumulative_n_operations"])
orders = [np.arange(n) for n in distances.groupby("root_id").size()]
orders = np.concatenate(orders)
distances["order"] = orders
# %%
out_clean = output_proportions.query("is_exemplar")
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

sns.lineplot(
    out_clean, x="cumulative_n_operations", y="proportion", hue="post_mtype", ax=ax
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.lineplot(
    distances.query("is_exemplar"),
    # x="cumulative_n_operations",
    x="order",
    y="cityblock",
    units="root_id",
    color="black",
    alpha=0.2,
    ax=ax,
    estimator=None,
)
ax.set(xlim=(0, 400))


# %%
clean_distances = distances.query("is_exemplar").copy()
clean_distances["order"] = clean_distances.groupby("root_id").cumcount()
max_order = clean_distances["order"].max()
clean_distances["padded_order"] = clean_distances["order"].copy()
clean_distances.set_index(["root_id", "padded_order"], inplace=True)
clean_distances = (
    clean_distances.reindex(
        pd.MultiIndex.from_product(
            [clean_distances.index.levels[0], np.arange(max_order + 1)],
            names=clean_distances.index.names,
        )
    )
    .reset_index()
    .ffill()
)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.lineplot(
    clean_distances,
    # x="cumulative_n_operations",
    x="padded_order",
    y="cityblock",
    units="root_id",
    color="black",
    alpha=0.1,
    ax=ax,
    estimator=None,
)
sns.lineplot(
    clean_distances,
    # x="cumulative_n_operations",
    x="padded_order",
    y="cityblock",
    ax=ax,
    color="red",
)
ax.set(xlim=(0, 200), xlabel="Number of merge edits")

# %%
clean_distances["p_merges"] = clean_distances["order"] / clean_distances.groupby(
    "root_id"
)["order"].transform("max")

# %%

clean_distances["p_merges_bin"] = pd.cut(clean_distances["p_merges"], bins=20)
clean_distances["p_merges_bin_mid"] = clean_distances["p_merges_bin"].apply(
    lambda x: x.mid
)
# clean_distances['p_merges_bin'] = clean_distances['p_merges_bin'].cat.codes

# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5), tight_layout=True)


sns.lineplot(
    clean_distances,
    # x="cumulative_n_operations",
    x="p_merges_bin_mid",
    y="cityblock",
    color="red",
    linewidth=3,
    ax=ax,
)
sns.lineplot(
    clean_distances,
    # x="cumulative_n_operations",
    x="p_merges",
    y="cityblock",
    units="root_id",
    color="black",
    alpha=0.1,
    ax=ax,
    estimator=None,
    zorder=-10,
)

ax.set(xlabel="Proportion of merge edits", ylabel="Cityblock distance to final")


lines = ax.get_lines()[1:]
for line in lines:
    line.set_visible(False)

red_lines = ax.get_children()[:2]
for line in red_lines:
    line.set_visible(False)


def update(i):
    if i <= 40:
        for line in lines[:i]:
            line.set_visible(True)
            line.set_alpha(0.2)
        for line in red_lines:
            line.set_visible(False)
        lines[i].set_visible(True)
        lines[i].set_alpha(1)
    if i > 40:
        for line in lines:
            line.set_visible(True)
            line.set_alpha(0.2)
        for line in red_lines:
            line.set_visible(True)


from matplotlib import animation

ani = animation.FuncAnimation(
    fig, update, frames=range(0, 80), interval=100, repeat=False
)

writer = animation.PillowWriter(fps=5)

ani.save("distance-lines-animation.gif", writer=writer)


# %%
offdiag_dists_by_root = []
for file in out_path.glob("*output_proportions*.csv"):
    root_id = int(file.name.split("root_id=")[1].split(".")[0])
    df = pd.read_csv(file, index_col=0)
    df["root_id"] = root_id
    df = df.query("is_exemplar")
    df = df.pivot(
        values="proportion",
        index=["root_id", "operation_id", "cumulative_n_operations", "is_exemplar"],
        columns="post_mtype",
    )
    df = df.sort_index(level="cumulative_n_operations")
    X = df.values
    dists = pairwise_distances(X, metric="manhattan")
    dists_df = pd.Series(data=dists[-1], index=df.index, name="cityblock").to_frame()
    distances_by_root.append(dists_df)

    square_dists_df = pd.DataFrame(data=dists, index=df.index, columns=df.index)
    # square_dists_df.values
    offdiag_dists = dists[np.arange(1, len(dists)), np.arange(len(dists) - 1)]
    offdiag_dists = pd.Series(
        offdiag_dists, index=df.index[1:], name="cityblock_to_next"
    ).to_frame()
    offdiag_dists_by_root.append(offdiag_dists)

# %%
offdiag_dists = pd.concat(offdiag_dists_by_root)
offdiag_dists["order"] = offdiag_dists.groupby("root_id").cumcount()
offdiag_dists["p_merges"] = offdiag_dists["order"] / offdiag_dists.groupby("root_id")[
    "order"
].transform("max")
# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.lineplot(
    offdiag_dists,
    x="order",
    y="cityblock_to_next",
    units="root_id",
    color="black",
    alpha=0.1,
    ax=ax,
    estimator=None,
)
ax.set(xlim=(0, 200))

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.lineplot(
    offdiag_dists,
    x="p_merges",
    y="cityblock_to_next",
    units="root_id",
    color="black",
    alpha=0.1,
    ax=ax,
    estimator=None,
)
# ax.set(xlim=(0, 200))

# %%
sequence_infos

# %%
pre_synapses_by_root = []
for file in out_path.glob("pre_synapses*.csv"):
    root_id = int(file.name.split("root_id=")[1].split(".")[0])
    syn_df = pd.read_csv(file, index_col=0)
    syn_df["pre_root"] = root_id
    pre_synapses_by_root.append(syn_df)

post_synapses_by_root = []
for file in out_path.glob("post_synapses*.csv"):
    root_id = int(file.name.split("root_id=")[1].split(".")[0])
    syn_df = pd.read_csv(file, index_col=0)
    syn_df["post_root"] = root_id
    post_synapses_by_root.append(syn_df)

pre_synapses = pd.concat(pre_synapses_by_root)
post_synapses = pd.concat(post_synapses_by_root)
synapses = pre_synapses.merge(post_synapses, on="id", how="inner")

# %%
root_ids = manifest.index
synapses = synapses.query("pre_root.isin(@root_ids) & post_root.isin(@root_ids)")

# %%
sequence_infos["is_merge"].fillna(False, inplace=True)
sequence_infos["cumulative_merges"] = sequence_infos.groupby("root_id")[
    "is_merge"
].cumsum()
sequence_infos["total_merges"] = sequence_infos.groupby("root_id")[
    "is_merge"
].transform("sum")
sequence_infos["p_merges"] = (
    sequence_infos["cumulative_merges"] / sequence_infos["total_merges"]
)

# %%
sequence_infos.set_index(["root_id", "order"], inplace=True)

# %%
target_p = 0.75

sequence_infos["dist_from_target"] = (sequence_infos["p_merges"] - target_p).abs()

query_index = sequence_infos.groupby("root_id")["dist_from_target"].idxmin()

# %%
# query_index = (
#     sequence_infos.reset_index()
#     .set_index(["root_id", "cumulative_n_operations"])
#     .query("is_exemplar")
#     .groupby("root_id")["dist_from_target"]
#     .idxmin()
# )
target_p = 1
labels_by_p = {}
n_ops_by_p = []
for target_p in np.linspace(0, 1, 21):
    output_proportions = output_proportions.reset_index().set_index(
        ["root_id", "cumulative_n_operations"]
    )
    sequence_infos = sequence_infos.reset_index().set_index(
        ["root_id", "cumulative_n_operations"]
    )
    output_proportions["p_merges"] = sequence_infos["p_merges"]
    # output_proportions['dist_from_target'] = sequence_infos['dist_from_target']
    output_proportions["dist_from_target"] = (
        output_proportions["p_merges"] - target_p
    ).abs()
    query_output_proportions = output_proportions.query("is_exemplar")

    query_output_proportions_index = query_output_proportions.groupby("root_id")[
        "dist_from_target"
    ].idxmin()

    query_output_props_long = output_proportions.loc[query_output_proportions_index]
    n_ops = (
        query_output_props_long.reset_index()
        .groupby("root_id")["cumulative_n_operations"]
        .first()
        .sum()
    )
    query_output_props_wide = (
        query_output_props_long.reset_index()
        .pivot(
            values="proportion",
            index=["root_id", "operation_id", "cumulative_n_operations", "is_exemplar"],
            columns="post_mtype",
        )
        .fillna(0)
    )

    X = query_output_props_wide.values

    from scipy.cluster.hierarchy import fcluster, linkage

    Z = linkage(X, method="average", metric="cityblock")

    labels = fcluster(Z, 18, criterion="maxclust")
    labels = pd.Series(labels, index=query_output_props_wide.index, name="cluster")

    labels_by_p[target_p] = labels
    n_ops_by_p.append(n_ops)


from sklearn.metrics import adjusted_rand_score

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
aris_by_p = []
for p, labels in labels_by_p.items():
    ari = adjusted_rand_score(labels, labels_by_p[1])
    ax.plot(p, ari, "o", color="black")
    aris_by_p.append(ari)

# %%
from pkg.plot import set_context

set_context()
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.scatterplot(x=n_ops_by_p, y=aris_by_p, ax=ax)
ax.set(xlabel="Number of merge operations", ylabel="ARI to final clustering")

# %%
rows_by_trial = {}
labels_by_trial = {}
n_samples = 5

mats_by_target_p = {}
for target_p in np.linspace(0, 1, 11):
    output_proportions = output_proportions.reset_index().set_index(
        ["root_id", "cumulative_n_operations"]
    )
    sequence_infos = sequence_infos.reset_index().set_index(
        ["root_id", "cumulative_n_operations"]
    )
    output_proportions["p_merges"] = sequence_infos["p_merges"]
    # output_proportions['dist_from_target'] = sequence_infos['dist_from_target']
    output_proportions["dist_from_target"] = (
        output_proportions["p_merges"] - target_p
    ).abs()
    query_output_proportions = output_proportions.query("is_exemplar")

    query_output_proportions_index = query_output_proportions.groupby("root_id")[
        "dist_from_target"
    ].idxmin()

    query_output_props_long = output_proportions.loc[query_output_proportions_index]
    n_ops = (
        query_output_props_long.reset_index()
        .groupby("root_id")["cumulative_n_operations"]
        .first()
        .sum()
    )
    query_output_props_wide = (
        query_output_props_long.reset_index()
        .pivot(
            values="proportion",
            index=["root_id", "operation_id", "cumulative_n_operations", "is_exemplar"],
            columns="post_mtype",
        )
        .fillna(0)
    )
    mats_by_target_p[target_p] = query_output_props_wide
    for p_neurons in np.linspace(0.1, 1, 10):
        for i in range(n_samples):
            choice_roots = np.random.choice(
                query_output_props_wide.index.get_level_values("root_id").unique(),
                replace=False,
                size=int(np.floor(query_output_props_wide.shape[0] * p_neurons)),
            )

            X = query_output_props_wide.loc[choice_roots]
            n_ops = (
                X.reset_index()
                .groupby("root_id")["cumulative_n_operations"]
                .first()
                .sum()
            )

            from scipy.cluster.hierarchy import fcluster, linkage

            Z = linkage(X.values, method="average", metric="cityblock")

            labels = fcluster(Z, 18, criterion="maxclust")
            labels = pd.Series(
                labels, index=X.index.get_level_values("root_id"), name="cluster"
            )
            labels_by_trial[(target_p, p_neurons, i)] = labels
            rows_by_trial[(target_p, p_neurons, i)] = {
                "target_p": target_p,
                "p_neurons": p_neurons,
                "sample": i,
                "n_ops": n_ops,
                "labels": labels,
            }


# %%
aris_by_trial = []

for (target_p, p_neurons, i), labels in labels_by_trial.items():
    info = rows_by_trial[(target_p, p_neurons, i)]
    ari = adjusted_rand_score(labels, labels_by_trial[(1, 1, 0)].loc[labels.index])
    aris_by_trial.append(
        {
            "target_p": target_p,
            "p_neurons": p_neurons,
            "ari": ari,
            "n_ops": info["n_ops"],
        }
    )


# %%
aris_df = pd.DataFrame(aris_by_trial)


# %%
aris_square = aris_df.pivot_table(index="target_p", columns="p_neurons", values="ari")

sns.heatmap(aris_square, cmap="coolwarm", center=0, annot=True)


# %%
from sklearn.metrics import adjusted_rand_score

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
aris_by_p = []
for p, labels in labels_by_p.items():
    ari = adjusted_rand_score(labels, labels_by_p[1])
    ax.plot(p, ari, "o", color="black")
    aris_by_p.append(ari)

# %%

sns.clustermap(
    mats_by_target_p[0.5].T,
    xticklabels=False,
    yticklabels=False,
    row_cluster=False,
)
sns.clustermap(
    mats_by_target_p[1].T,
    xticklabels=False,
    yticklabels=False,
    row_cluster=False,
)

# %%
full_neurons_aris = aris_df.query("p_neurons == 1.0")
some_neurons_aris = aris_df.query("target_p == 1.0")
sns.scatterplot(data=full_neurons_aris, x="n_ops", y="ari")
sns.scatterplot(data=some_neurons_aris, x="n_ops", y="ari")

# %%
full_neurons_aris = aris_df.query("p_neurons == 1.0")

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

set_context()

sns.scatterplot(data=full_neurons_aris, x="target_p", y="ari", ax=ax, s=40)

ax.set(xlabel="Proportion of merge edits", ylabel="ARI to final clustering")

#%%





#%%

# %%

sns.clustermap(query_output_props_wide.T)

# query_output_props_long = output_proportions.set_index(
#     ["root_id", "cumulative_n_operations"]
# ).loc[query_index]
# %%
query_output_props = query_output_props_long.reset_index().pivot(
    values="proportion",
    index=["root_id", "operation_id", "cumulative_n_operations", "is_exemplar"],
    columns="post_mtype",
)

# %%

# sequence_infos['pre_synapses'] = sequence_infos['pre_synapses'].apply(eval)
# sequence_infos['post_synapses'] = sequence_infos['post_synapses'].apply(eval)

# %%
query_sequences = sequence_infos.loc[query_index].copy()
query_sequences["pre_synapses"] = query_sequences["pre_synapses"].apply(eval)
# %%
query_sequences["post_synapses"] = query_sequences["post_synapses"].apply(eval)

# %%
uni_pres = query_sequences["pre_synapses"].explode().unique().astype(int)

# %%
uni_posts = query_sequences["post_synapses"].explode().unique().astype(int)

# %%
subgraph_synapses = np.unique(np.intersect1d(uni_pres, uni_posts).astype(int))
# %%
subgraph_synapse_table = (
    synapses.loc[subgraph_synapses][["pre_root", "post_root"]].groupby("id").first()
)

# %%

# %%
from pkg.metrics import MotifFinder

mf = MotifFinder()
mf.find_motif_monomorphisms()

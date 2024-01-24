# %%
import os
import pickle
from typing import Union

import caveclient as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pkg.edits import (
    apply_edit_history,
    apply_synapses,
    collate_edit_info,
    count_synapses_by_sample,
    get_initial_network,
    get_operation_metaoperation_map,
    lazy_load_network_edits,
    resolve_synapses_from_edit_selections,
)
from pkg.morphology import (
    apply_nucleus,
    apply_positions,
)
from pkg.plot import rotate_set_labels, savefig

# %%

os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"


# %%

client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v507")
query_neurons.sort_values("id", inplace=True)

# 10 looked "unstable"
# 11 looked "stable"
root_id = query_neurons["pt_root_id"].values[10]
# root_id = 864691135865971164
# root_id = 864691135992790209

# %%
palette_file = "/Users/ben.pedigo/code/skedits/skedits-app/skedits/data/ctype_hues.pkl"

with open(palette_file, "rb") as f:
    ctype_hues = pickle.load(f)

ctype_hues = {ctype: tuple(ctype_hues[ctype]) for ctype in ctype_hues.keys()}


# %%
(
    networkdeltas_by_operation,
    networkdeltas_by_metaoperation,
) = lazy_load_network_edits(root_id, client=client)

# %%
operation_to_metaoperation = get_operation_metaoperation_map(
    networkdeltas_by_metaoperation
)

# %%
edit_stats, metaoperation_stats, modified_level2_nodes = collate_edit_info(
    networkdeltas_by_operation, operation_to_metaoperation, root_id, client
)


# %%
initial_nf = get_initial_network(root_id, client, positions=False)

# %%


# go through all of the edits/metaedits
# add nodes that were added, but don't remove any nodes
# mark nodes/edges with when they were added/removed
# things that were never removed/added get -1


nf = initial_nf.copy()

apply_edit_history(nf, networkdeltas_by_operation, operation_to_metaoperation)


apply_positions(nf, client)


# %%

pre_synapses, post_synapses = apply_synapses(
    nf,
    networkdeltas_by_operation,
    root_id,
    client,
)

# %%

nuc_level2_id = apply_nucleus(nf, root_id, client)

# TODO is it worth just caching the whole networkframe at this stage?

# %%

from typing import Optional

from networkframe import NetworkFrame


class NeuronFrame(NetworkFrame):
    def __init__(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        pre_synapses: Optional[pd.DataFrame] = None,
        post_synapses: Optional[pd.DataFrame] = None,
        edits: Optional[pd.DataFrame] = None,
        nucleus_id: Optional[int] = None,
        neuron_id: Optional[int] = None,
        pre_synapse_mapping_col: str = "pre_pt_level2_id",
        post_synapse_mapping_col: str = "post_pt_level2_id",
        **kwargs,
    ):
        super().__init__(nodes, edges, **kwargs)

        if pre_synapses is None:
            pre_synapses = pd.DataFrame()
        if post_synapses is None:
            post_synapses = pd.DataFrame()
        if edits is None:
            edits = pd.DataFrame()

        self.pre_synapses = pre_synapses
        self.post_synapses = post_synapses
        self.edits = edits
        self.nucleus_id = nucleus_id
        self.neuron_id = neuron_id

        # TODO if the pre/post synapses are not empty, then we should check that the
        # column exists
        self.pre_synapse_mapping_col = pre_synapse_mapping_col
        self.post_synapse_mapping_col = post_synapse_mapping_col

    def __repr__(self) -> str:
        out = (
            "NeuronFrame(\n"
            + f"    neuron_id={self.neuron_id},\n"
            + f"    nodes={self.nodes.shape},\n"
            + f"    edges={self.edges.shape},\n"
            + f"    pre_synapses={self.pre_synapses.shape},\n"
            + f"    post_synapses={self.post_synapses.shape},\n"
            + f"    edits={self.edits.shape},\n"
            + f"    nucleus_id={self.nucleus_id}\n"
            + ")"
        )
        return out

    @property
    def nucleus_id(self) -> int:
        return self._nucleus_id

    @nucleus_id.setter
    def nucleus_id(self, nucleus_id):
        if nucleus_id not in self.nodes.index:
            raise ValueError(f"nucleus_id {nucleus_id} not in nodes table index")
        self._nucleus_id = nucleus_id

    @property
    def has_pre_synapses(self) -> bool:
        return not self.pre_synapses.empty

    @property
    def has_post_synapses(self) -> bool:
        return not self.post_synapses.empty

    @property
    def has_edits(self) -> bool:
        return not self.edits.empty

    @property
    def pre_synapse_mapping_col(self) -> str:
        return self._pre_synapse_mapping_col

    @pre_synapse_mapping_col.setter
    def pre_synapse_mapping_col(self, pre_synapse_mapping_col: str):
        # check if the column exists
        if self.has_pre_synapses and (
            pre_synapse_mapping_col not in self.pre_synapses.columns
        ):
            raise ValueError(
                f"pre_synapse_mapping_col '{pre_synapse_mapping_col}' not in pre_synapses table columns"
            )

        # check if all elements in the column are in the nodes index
        if not self.pre_synapses[pre_synapse_mapping_col].isin(self.nodes.index).all():
            raise ValueError(
                f"pre_synapse_mapping_col '{pre_synapse_mapping_col}' contains values not in nodes index"
            )

        self._pre_synapse_mapping_col = pre_synapse_mapping_col

    @property
    def post_synapse_mapping_col(self) -> str:
        return self._post_synapse_mapping_col

    @post_synapse_mapping_col.setter
    def post_synapse_mapping_col(self, post_synapse_mapping_col: str):
        # check if the column exists
        if self.has_post_synapses and (
            post_synapse_mapping_col not in self.post_synapses.columns
        ):
            raise ValueError(
                f"post_synapse_mapping_col '{post_synapse_mapping_col}' not in post_synapses table columns"
            )

        # check if all elements in the column are in the nodes index
        if (
            not self.post_synapses[post_synapse_mapping_col]
            .isin(self.nodes.index)
            .all()
        ):
            raise ValueError(
                f"post_synapse_mapping_col '{post_synapse_mapping_col}' contains values not in nodes index"
            )

        self._post_synapse_mapping_col = post_synapse_mapping_col

    def activate_edits(self, edit_ids: Union[list[int], int]):
        """activate edits by id"""
        pass
        # self.edits.loc[edit_ids, "active"] = True
    
    def deactivate_edits(self, edit_ids: Union[list[int], int]):
        """deactivate edits by id"""
        pass
        # self.edits.loc[edit_ids, "active"] = False
    
    


NeuronFrame(nodes=nf.nodes, edges=nf.edges, nucleus_id=nuc_level2_id, neuron_id=root_id)

# %%
mtypes = client.materialize.query_table("aibs_metamodel_mtypes_v661_v2")
root_id_counts = mtypes["pt_root_id"].value_counts()
root_id_singles = root_id_counts[root_id_counts == 1].index
mtypes = mtypes.query("pt_root_id in @root_id_singles")
mtypes.set_index("pt_root_id", inplace=True)

# %%
pre_synapses["post_mtype"] = pre_synapses["post_pt_root_id"].map(mtypes["cell_type"])
post_synapses["pre_mtype"] = post_synapses["pre_pt_root_id"].map(mtypes["cell_type"])

# %%
metaoperation_stats = metaoperation_stats.sort_values("time")

# %%
edit_selections = {}
for i in range(len(metaoperation_stats) + 1):
    edit_selections[i] = metaoperation_stats.index[:i].tolist()

# %%
resolved_pre_synapses, resolved_post_synapses = resolve_synapses_from_edit_selections(
    nf, edit_selections, root_id, client
)

# %%
post_mtype_counts = count_synapses_by_sample(
    pre_synapses, resolved_pre_synapses, "post_mtype"
)

operation_feature_key = "time"
operation_key = "metaoperation_added"
fillna = metaoperation_stats[operation_feature_key].min()


def wrangle_counts_by_edit_sample(
    metaoperation_stats, counts_table, operation_feature_key, operation_key, fillna
):
    post_mtype_stats_tidy = counts_table.reset_index().melt(
        var_name="post_mtype", value_name="count", id_vars="sample"
    )

    post_mtype_probs = counts_table / counts_table.sum(axis=1).values[:, None]
    post_mtype_probs.fillna(0, inplace=True)
    post_mtype_probs_tidy = post_mtype_probs.reset_index().melt(
        var_name="post_mtype", value_name="prob", id_vars="sample"
    )

    post_mtype_stats_tidy["metaoperation_added"] = (
        (post_mtype_stats_tidy["sample"] - 1)
        .map(metaoperation_stats.index.to_series().reset_index(drop=True))
        .fillna(-1)
        .astype(int)
    )

    post_mtype_stats_tidy["prob"] = post_mtype_probs_tidy["prob"]

    post_mtype_stats_tidy["time"] = (
        post_mtype_stats_tidy[operation_key]
        .map(metaoperation_stats[operation_feature_key])
        .fillna(fillna)
    )

    if "time" in operation_feature_key:
        post_mtype_stats_tidy[operation_feature_key] = pd.to_datetime(
            post_mtype_stats_tidy[operation_feature_key]
        )

    return post_mtype_stats_tidy


post_mtype_stats_tidy = wrangle_counts_by_edit_sample(
    metaoperation_stats, post_mtype_counts, operation_feature_key, operation_key, fillna
)


# %%
sns.set_context("talk")

name_map = {
    "count": "Synapse count",
    "prob": "Output proportion",
    "time": "Time",
    "centroid_distance_to_nuc_um": "Distance to nucleus (um)",
}


def apply_name(name):
    if name in name_map:
        return name_map[name]
    else:
        return name


def editplot(stats, x, y, hue="post_mtype", figsize=(6, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=stats,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
        legend=False,
        linewidth=1,
        palette=ctype_hues,
    )
    sns.scatterplot(
        data=stats,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
        legend=False,
        s=10,
        palette=ctype_hues,
    )

    ax.set_xlabel(apply_name(ax.get_xlabel()))
    ax.set_ylabel(apply_name(ax.get_ylabel()))

    ax.spines[["right", "top"]].set_visible(False)

    return fig, ax


# %%
root_id_time_map = {
    864691135995711402: (pd.to_datetime("2023-04-17"), pd.to_datetime("2023-04-27"))
}

if root_id in root_id_time_map:
    spans = [None, root_id_time_map[root_id]]
else:
    spans = [None]

x = "time"
hue = "post_mtype"

for y in ["count", "prob"]:
    for span in spans:
        name = f"{y}_vs_{x}_by_{hue}-root_id={root_id}"
        fig, ax = editplot(post_mtype_stats_tidy, x, y, hue=hue)
        if span is not None:
            ax.set_xlim(*span)
            name += "-span"
        rotate_set_labels(ax)
        savefig(name, fig, folder="edit_replay_ordering")


# %%

# generate edit selections, ordered by distance

metaoperation_stats.sort_values("centroid_distance_to_nuc_um", inplace=True)
metaoperation_stats

edit_selections = {}
for i in range(len(metaoperation_stats) + 1):
    edit_selections[i] = metaoperation_stats.index[:i].tolist()


# %%
resolved_pre_synapses, resolved_post_synapses = resolve_synapses_from_edit_selections(
    nf, edit_selections, root_id, client
)

post_mtype_counts = count_synapses_by_sample(
    pre_synapses, resolved_pre_synapses, "post_mtype"
)

# %%

operation_feature_key = "centroid_distance_to_nuc_um"
operation_key = "metaoperation_added"
fillna = metaoperation_stats[operation_feature_key].min()


def wrangle_counts_by_edit_sample(
    metaoperation_stats, counts_table, operation_feature_key, operation_key, fillna
):
    post_mtype_stats_tidy = counts_table.reset_index().melt(
        var_name="post_mtype", value_name="count", id_vars="sample"
    )

    post_mtype_probs = counts_table / counts_table.sum(axis=1).values[:, None]
    post_mtype_probs.fillna(0, inplace=True)
    post_mtype_probs_tidy = post_mtype_probs.reset_index().melt(
        var_name="post_mtype", value_name="prob", id_vars="sample"
    )

    post_mtype_stats_tidy["metaoperation_added"] = (
        (post_mtype_stats_tidy["sample"] - 1)
        .map(metaoperation_stats.index.to_series().reset_index(drop=True))
        .fillna(-1)
        .astype(int)
    )

    post_mtype_stats_tidy["prob"] = post_mtype_probs_tidy["prob"]

    post_mtype_stats_tidy[operation_feature_key] = (
        post_mtype_stats_tidy[operation_key]
        .map(metaoperation_stats[operation_feature_key])
        .fillna(fillna)
    )

    if "time" in operation_feature_key:
        post_mtype_stats_tidy[operation_feature_key] = pd.to_datetime(
            post_mtype_stats_tidy[operation_feature_key]
        )

    return post_mtype_stats_tidy


post_mtype_stats_tidy = wrangle_counts_by_edit_sample(
    metaoperation_stats, post_mtype_counts, operation_feature_key, operation_key, fillna
)
post_mtype_stats_tidy

# %%

x = "centroid_distance_to_nuc_um"
hue = "post_mtype"
spans = [None]

for y in ["count", "prob"]:
    for span in spans:
        name = f"{y}_vs_{x}_by_{hue}-root_id={root_id}"
        fig, ax = editplot(post_mtype_stats_tidy, x, y, hue=hue)
        if span is not None:
            ax.set_xlim(*span)
            name += "-span"
        rotate_set_labels(ax)
        savefig(name, fig, folder="edit_replay_ordering")

# %%

import numpy as np

n_per = 25
proportions = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

edit_selections = {}
sample_proportions = {}
i = 0
for proportion in proportions:
    if proportion == 0:
        edit_selections[i] = []
        i += 1
    elif proportion == 1:
        edit_selections[i] = metaoperation_stats.index.tolist()
        i += 1
    else:
        for _ in range(n_per):
            select = np.random.choice(
                metaoperation_stats.index,
                size=int(np.round(proportion * len(metaoperation_stats))),
                replace=False,
            )
            edit_selections[i] = list(select)
            sample_proportions[i] = proportion
            i += 1

# %%
resolved_pre_synapses, resolved_post_synapses = resolve_synapses_from_edit_selections(
    nf, edit_selections, root_id, client
)

post_mtype_counts = count_synapses_by_sample(
    pre_synapses, resolved_pre_synapses, "post_mtype"
)

# %%

operation_feature_key = "centroid_distance_to_nuc_um"
operation_key = "metaoperation_added"
fillna = metaoperation_stats[operation_feature_key].min()

# %%
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

post_mtype_stats_tidy["proportion_used"] = post_mtype_stats_tidy["sample"].map(
    sample_proportions
)
# %%
x = "proportion_used"
hue = "post_mtype"
spans = [None]

for y in ["count", "prob"]:
    for span in spans:
        name = f"{y}_vs_{x}_by_{hue}-root_id={root_id}"
        fig, ax = editplot(post_mtype_stats_tidy, x, y, hue=hue)
        if span is not None:
            ax.set_xlim(*span)
            name += "-span"
        rotate_set_labels(ax)
        savefig(name, fig, folder="edit_replay_ordering")

# %%

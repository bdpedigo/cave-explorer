# %%
import caveclient as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib_venn import venn2
from tqdm.autonotebook import tqdm

# %%
client = cc.CAVEclient("minnie65_phase3_v1")

# %%

model_preds = client.materialize.query_table("aibs_metamodel_mtypes_v661_v2")

# TODO this one seems to match the paper better but is just for excitatory neurons
# model_preds = client.materialize.query_table("aibs_soma_nuc_exc_mtype_preds_v117")

nuclei = client.materialize.query_table("nucleus_detection_v0")

# TODO differences between querying on root_id, supervoxel_id, or target_id/id?
model_preds_root_ids = set(model_preds["target_id"].unique())
nuclei_root_ids = set(nuclei["id"].unique())

venn2([model_preds_root_ids, nuclei_root_ids], set_labels=["model_preds", "nuclei"])

model_preds.set_index("target_id", inplace=True)
nuclei.set_index("id", inplace=True)

# %%

model_df = model_preds.join(
    nuclei, how="inner", lsuffix="_model_preds", rsuffix="_nuclei"
)
duplicated = model_df.index.duplicated(keep=False)
model_df[duplicated]
# TODO looks like there are some (I think harmless) duplicated entries

# %%
model_df = model_df[~model_df.index.duplicated(keep="first")]

# %%
model_df["cell_type"].value_counts()

# %%

# TODO unsure how a few of the categories map onto his colors in the paper
# BC - basket cell
# MC - Martinotti cell
# NGC - neurogliaform cell
# BPC - bipolar cell (where are multipolar cells?)

# I think this annotates all of the neurons in the slanted tube
column_df = client.materialize.query_table("allen_v1_column_types_slanted_ref")

column_df["cell_type"].value_counts()

fig, ax = plt.subplots()

venn2(
    [set(model_df.index), set(column_df["target_id"])],
    set_labels=["model_preds", "column"],
    ax=ax,
)

column_df = column_df.query("target_id.isin(@model_df.index)")

column_df.set_index("target_id", inplace=True)

fig, ax = plt.subplots()

venn2(
    [set(model_df.index), set(column_df.index)],
    set_labels=["model_preds", "column"],
    ax=ax,
)
plt.show()


# %%

# remap positions (which come as a list) to their own xyz columns

column_model_df = model_df.loc[column_df.index]

positions = column_model_df["pt_position_nuclei"].explode().reset_index()


def to_xyz(order):
    if order % 3 == 0:
        return "x"
    elif order % 3 == 1:
        return "y"
    else:
        return "z"


positions["axis"] = positions.index.map(to_xyz)
positions = positions.pivot(
    index="target_id", columns="axis", values="pt_position_nuclei"
)

column_model_df = column_model_df.join(positions)

column_model_df[["x", "y", "z"]]

# %%
excitatory_df = column_model_df.query("classification_system == 'excitatory_neuron'")

excitatory_df = excitatory_df.sort_values("cell_type")

# FIGURE 3E
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.swarmplot(
    data=excitatory_df, x="cell_type", y="y", hue="cell_type", palette="tab20", s=1.75
)
ax.invert_yaxis()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

# TODO seems like the labels here don't match up with the labels in the paper

# TODO what is the transformation between xyz and soma depth?


# %%

source_ids = column_model_df.query("cell_type == 'PTC'")["pt_root_id_model_preds"]
target_ids = column_model_df.query("cell_type == 'L2a'")["pt_root_id_model_preds"]

# TODO this doesn't eat a pandas index object even though it says iterable
query_synapses = client.materialize.synapse_query(
    pre_ids=list(source_ids), post_ids=list(target_ids)
)
mean_synapses_per_target_cell = len(query_synapses) / len(target_ids)
mean_synapses_per_target_cell

# %%

exc_df = column_model_df.query("classification_system == 'excitatory_neuron'")
inh_df = column_model_df.query("classification_system == 'inhibitory_neuron'")

pbar = tqdm(total=exc_df["cell_type"].nunique() * inh_df["cell_type"].nunique())

rows = []
for source_name, source_df in inh_df.groupby("cell_type"):
    for target_name, target_df in exc_df.groupby("cell_type"):
        source_ids = source_df["pt_root_id_model_preds"]
        target_ids = target_df["pt_root_id_model_preds"]

        query_synapses = client.materialize.synapse_query(
            pre_ids=source_ids, post_ids=target_ids
        )
        mean_synapses_per_target_cell = len(query_synapses) / len(target_ids)
        rows.append(
            {
                "source": source_name,
                "target": target_name,
                "mean_syn_per_target": mean_synapses_per_target_cell,
            }
        )
        pbar.update(1)

pbar.close()

connection_df = pd.DataFrame(rows)

# %%
connection_df = connection_df.pivot(
    index="source", columns="target", values="mean_syn_per_target"
)
connection_df = connection_df.loc[["PTC", "DTC", "STC", "ITC"]]

# %%

# FIGURE 4C
fig, ax = plt.subplots(1, 1, figsize=(3, 9))
sns.heatmap(connection_df.T, annot=True, cmap="viridis", ax=ax, cbar=False)

# %%

client.materialize.query_table("allen_v1_column_types_slanted_ref")

# %%
query_synapses = client.materialize.synapse_query(pre_ids=column_model_df.index)

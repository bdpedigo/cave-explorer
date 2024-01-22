# %%

import caveclient as cc
from tqdm.auto import tqdm

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v507")
query_neurons.sort_values("id", inplace=True)


# %%
from pkg.edits import get_detailed_change_log

change_logs = []
for root_id in tqdm(query_neurons["pt_root_id"][:]):
    try:
        change_log = get_detailed_change_log(root_id, client, filtered=False)
        change_log["root_id"] = root_id
        change_logs.append(change_log)
    except Exception as e:
        print(e)
        print(root_id)
        continue

# %%
import pandas as pd

all_change_logs = pd.concat(change_logs, axis=0, ignore_index=True)

# %%

# %%
all_change_logs["datetime"] = pd.to_datetime(
    all_change_logs["timestamp"], format="ISO8601"
)
# %%
import numpy as np

time_orders = (
    all_change_logs.groupby("root_id")["datetime"].mean().sort_values(ascending=False)
)
id_map = zip(time_orders.index, np.arange(len(time_orders)))
all_change_logs["root_id_codes"] = all_change_logs["root_id"].map(dict(id_map))

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
sns.scatterplot(
    data=all_change_logs, x="datetime", y="root_id_codes", ax=ax, marker="|"
)
ax.set(ylabel="Root ID", xlabel="Time")

# %%

n = 30
sns.set_context("paper")
palette = sns.color_palette("husl", n)
fig, axs = plt.subplots(n, 1, figsize=(5, 15), sharex=True, sharey=True)
for i in range(n):
    neuron_change_logs = all_change_logs.query(f"root_id_codes=={i}")
    ax = axs[i]
    sns.histplot(
        data=neuron_change_logs,
        x="datetime",
        ax=ax,
        binwidth=7,
        color=palette[i],
        clip_on=False,
    )
    ax.set(ylabel="", yticks=[])
    ax.spines[["left", "right", "top"]].set_visible(False)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
)
ax.set(xlabel="Time")
# fig.subplots_adjust(hspace=-0.25)

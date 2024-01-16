# %%
import pandas as pd

import caveclient as cc

client = cc.CAVEclient("v1dd")
# %%
client = cc.CAVEclient("minnie65_phase3_v1")
mtypes: pd.DataFrame
mtypes = client.materialize.query_table("aibs_metamodel_mtypes_v661_v2").set_index(
    "target_id"
)
nucs = client.materialize.query_table("nucleus_detection_v0").set_index("id")
mtypes = mtypes.join(nucs["pt_root_id"], lsuffix="_current")

# %%

from pkg.edits import get_detailed_change_log
from tqdm.auto import tqdm

valid_nucs = nucs.query("pt_root_id != 0")

sample_root_ids = valid_nucs.sample(100, replace=False)["pt_root_id"]

change_logs = []
for root_id in tqdm(sample_root_ids):
    change_log = get_detailed_change_log(root_id, client)
    change_log["root_id"] = root_id
    if len(change_log) > 0:
        change_logs.append(change_log)

change_log = pd.concat(change_logs)

# %%

change_log.groupby("root_id").size().hist(bins=100)

# %%
import datetime

import numpy as np

n_edits_by_month = {}
for year in range(2020, 2024):
    for month in range(1, 13):
        old_roots, new_roots = client.chunkedgraph.get_delta_roots(
            datetime.datetime(year, month, 1),
            datetime.datetime(year, month, 28),
        )
        # roughly 1/3 the number of changed roots should be close to number of edits
        n_edits = np.round((len(old_roots) + len(new_roots)) / 3)
        print(f"{(year, month)}: {n_edits}")
        n_edits_by_month[(year, month)] = n_edits

# %%
n_edits_by_month = pd.Series(n_edits_by_month)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")
fig, ax = plt.subplots(figsize=(8, 3))
sns.barplot(
    x=n_edits_by_month.index.to_flat_index().to_series().apply(str).to_list(),
    y=n_edits_by_month.values,
    ax=ax,
)
ax.set(xlabel="Month", ylabel="Number of edits")
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
xticks = ax.get_xticks()
xticklabels = ax.get_xticklabels()
ax.set_xticks(xticks[::6])
ax.set_xticklabels(xticklabels[::6], rotation=45, ha="right")


# %%
cum_edits_by_month = n_edits_by_month.cumsum()
fig, ax = plt.subplots(figsize=(8, 3))
sns.lineplot(
    x=cum_edits_by_month.index.to_flat_index().to_series().apply(str).to_list(),
    y=cum_edits_by_month.values,
    ax=ax,
)
ax.set(xlabel="Month", ylabel="Cumulative edits")
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
xticks = ax.get_xticks()
xticklabels = ax.get_xticklabels()
ax.set_xticks(xticks[::6])
ax.set_xticklabels(xticklabels[::6], rotation=45, ha="right")
ax.set_ylim(0, 1_000_000)


def formatter(x, pos):
    if x >= 1_000_000:
        return f"{x/1_000_000:.0f}M"
    elif x == 0:
        return "0"
    else:
        return f"{x/1_000:.0f}K"


def month_formatter(x, pos):
    year, month = x.replace("(", "").replace(")", "").split(",")
    print(month)
    month = month.strip()
    if month == "1":
        month = "Jan"
    elif month == "7":
        month = "Jul"

    return f"{month} {year}"


ax.yaxis.set_major_formatter(formatter)
# ax.xaxis.set_major_formatter(month_formatter)
ax.set_xticklabels(
    [month_formatter(label.get_text(), 0) for label in ax.get_xticklabels()],
    rotation=45,
)

plt.savefig("cumulative_edits.svg", bbox_inches="tight")

# %%
# client = cc.CAVEclient(
#     datastack_name="aibs_v1dd", server_address="https://globalv1.em.brain.allentech.org"
# )


# %%
(mtypes["pt_root_id"] == mtypes["pt_root_id_current"]).mean()


# %%

mtypes.sample(1000)

# %%
from tqdm.auto import tqdm

n = 1
rows = []
for root_id in tqdm(mtypes.sample(n, random_state=8888)["pt_root_id"]):
    try:
        change_log = client.chunkedgraph.get_change_log(root_id)
        rows.append(
            {
                "root_id": root_id,
                "n_changes": len(change_log),
                "n_mergers": change_log["n_mergers"],
                "n_splits": change_log["n_splits"],
            }
        )
    except:
        print("Failed on", root_id)
edit_stats = pd.DataFrame(rows).set_index("root_id")
# %%
edit_stats["n_changes"] = edit_stats["n_mergers"] + edit_stats["n_splits"]
# %%
import matplotlib.pyplot as plt
import seaborn.objects as so

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

plot = (
    so.Plot(data=edit_stats, x="n_changes")
    .on(axs[0])
    .add(so.Bar(), so.Hist(cumulative=False, stat="proportion"))
    .label(x="Number of edits", y="Proportion of neurons")
)
plot.plot()

so.Plot(data=edit_stats, x="n_changes").on(axs[1]).add(
    so.Line(), so.Hist(cumulative=True, stat="proportion")
).label(x="Number of edits", y="Cumulative proportion").scale(x="log").show()
plot.save("n_edits.png", bbox_inches="tight")

# %%

# %%
import os

os.environ["LAZYCLOUD_USE_CLOUD"] = "True"
os.environ["LAZYCLOUD_RECOMPUTE"] = "False"
os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"

import pandas as pd

from pkg.paths import OUT_PATH

# %%
path = OUT_PATH / "access_time_ordered"

files = os.listdir(path)
diff_files = [f for f in files if "diffs" in f]

# %%
all_diffs = []
for f in diff_files:
    diffs = pd.read_csv(path / f, index_col=0)
    diffs["root_id"] = f.split("=")[1].split(".")[0]
    all_diffs.append(diffs)
all_diffs = pd.concat(all_diffs)

# %%
all_diffs
# %%

import matplotlib.pyplot as plt
import seaborn as sns

from pkg.plot import savefig

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.set_context("talk")
sns.lineplot(
    data=all_diffs,
    x="sample",
    y="diff",
    legend=False,
    units="root_id",
    estimator=None,
    linewidth=0.75,
    hue="root_id",
)
sns.lineplot(
    data=all_diffs, x="sample", y="diff", legend=False, color="black", zorder=-1
)
ax.set_ylabel("Difference from final")
ax.set_xlabel("Metaoperation added")
ax.set_xlim(0, 125)
savefig("all_diffs", fig, folder="access_time_ordered")

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.set_context("talk")
sns.lineplot(
    data=all_diffs,
    x="sample",
    y="diff",
    legend=False,
    units="root_id",
    estimator=None,
    linewidth=0.75,
    hue="root_id",
)
sns.lineplot(
    data=all_diffs, x="sample", y="diff", legend=False, color="black", zorder=-1
)
ax.set_ylabel("Difference from final")
ax.set_xlabel("Metaoperation added")
ax.set_xlim(0, 50)
savefig("all_diffs_cropped", fig, folder="access_time_ordered")

# %%

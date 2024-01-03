# %%
import caveclient as cc
import pandas as pd

client = cc.CAVEclient("minnie65_phase3_v1")
mtypes: pd.DataFrame
mtypes = client.materialize.query_table("aibs_metamodel_mtypes_v661_v2").set_index(
    "target_id"
)
nucs = client.materialize.query_table("nucleus_detection_v0").set_index("id")
mtypes = mtypes.join(nucs["pt_root_id"], lsuffix="_current")

# %%
(mtypes["pt_root_id"] == mtypes["pt_root_id_current"]).mean()


# %%

mtypes.sample(1000)

# %%
from tqdm.auto import tqdm

n = 1000
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

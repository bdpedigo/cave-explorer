# %%
import caveclient as cc

from pkg.neuronframe import load_neuronframe
from pkg.utils import load_manifest

client = cc.CAVEclient("minnie65_phase3_v1")
manifest = load_manifest()

# %%

root_id = manifest.query("is_sample").iloc[0]["current_root_id"]

# %%
nf = load_neuronframe(root_id, client)
final_nf = nf.set_edits(nf.edits.index, inplace=False)
final_nf.nodes = final_nf.nodes.copy()
# %%
l2_data = client.l2cache.get_l2data(final_nf.nodes.index)

# %%
import numpy as np
import pandas as pd

node_data = pd.DataFrame(l2_data).T
node_data.index = node_data.index.astype(int)
node_data["pca_val_1"] = node_data["pca_val"].apply(
    lambda x: np.nan if isinstance(x, float) else np.sqrt(x[1])
)
node_data["pca_val_2"] = node_data["pca_val"].apply(
    lambda x: np.nan if isinstance(x, float) else np.sqrt(x[2])
)

final_nf.nodes["size_nm3"] = np.nan
final_nf.nodes.loc[node_data.index, "size_nm3"] = node_data["size_nm3"]
final_nf.nodes["size_nm3"] = final_nf.nodes["size_nm3"].astype(float)

final_nf.nodes["max_dt_nm"] = np.nan
final_nf.nodes.loc[node_data.index, "max_dt_nm"] = node_data["max_dt_nm"]
final_nf.nodes["max_dt_nm"] = final_nf.nodes["max_dt_nm"].astype(float)

final_nf.nodes["mean_dt_nm"] = np.nan
final_nf.nodes.loc[node_data.index, "mean_dt_nm"] = node_data["mean_dt_nm"]
final_nf.nodes["mean_dt_nm"] = final_nf.nodes["mean_dt_nm"].astype(float)

final_nf.nodes["pca_val_1"] = np.nan
final_nf.nodes.loc[node_data.index, "pca_val_1"] = node_data["pca_val_1"]
final_nf.nodes["pca_val_1"] = final_nf.nodes["pca_val_1"].astype(float)

final_nf.nodes["pca_val_2"] = np.nan
final_nf.nodes.loc[node_data.index, "pca_val_2"] = node_data["pca_val_2"]
final_nf.nodes["pca_val_2"] = final_nf.nodes["pca_val_2"].astype(float)


# %%
out = final_nf.k_hop_aggregation(k=5, aggregations=["mean", "sum"])

out["radius_estimate"] = (
    np.sqrt(2)*out["pca_val_1_neighbor_mean"] + np.sqrt(2)*out["pca_val_2_neighbor_mean"]
) / 2



# %%

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
sns.scatterplot(
    x=node_data["mean_dt_nm"],
    y=node_data["pca_val"].apply(
        lambda x: np.nan if isinstance(x, float) else (x[1] + x[2]) / 2
    ),
)
ax.set_xlim(0, 450)

# %%
import pcg_skel

skel = pcg_skel.coord_space_skeleton(root_id=root_id, client=client)

# %%

from pathlib import Path

import caveclient as cc
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

import pcg_skel
import skeleton_plot

FIG_PATH = Path("cave-explorer/results/figs/visualize_edits")

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

# %%
# NOTE: the lowest level segmentation is done in 8x8x40,
# not 4x4x40 (client.info.viewer_resolution())
seg_res = client.chunkedgraph.segmentation_info["scales"][0]["resolution"]
res = client.info.viewer_resolution()
# %%
meta = client.materialize.query_table("allen_v1_column_types_slanted_ref")
meta = meta.sort_values("target_id")
nuc = client.materialize.query_table("nucleus_detection_v0").set_index("id")


def extract_operations(root_id):
    change_log = client.chunkedgraph.get_change_log(root_id)
    details = client.chunkedgraph.get_operation_details(change_log["operations_ids"])

    merges = {}
    splits = {}
    for operation, detail in details.items():
        operation = int(operation)
        source_coords = detail["source_coords"][0]
        sink_coords = detail["sink_coords"][0]
        x = (source_coords[0] + sink_coords[0]) / 2
        y = (source_coords[1] + sink_coords[1]) / 2
        z = (source_coords[2] + sink_coords[2]) / 2

        x *= seg_res[0]
        y *= seg_res[1]
        z *= seg_res[2]

        pt = [x, y, z]
        row = {"x": x, "y": y, "z": z, "pt": pt}
        if "added_edges" in detail:
            merges[operation] = row
        elif "removed_edges" in detail:
            splits[operation] = row

    merges = pd.DataFrame.from_dict(merges, orient="index")
    merges.index.name = "operation"
    splits = pd.DataFrame.from_dict(splits, orient="index")
    splits.index.name = "operation"
    return merges, splits


n_show = 20
sub_meta = meta.sample(n_show)
for i in tqdm(range(n_show)):
    target_id = sub_meta.iloc[i]["target_id"]
    root_id = nuc.loc[target_id]["pt_root_id"]
    latest = client.chunkedgraph.get_latest_roots(root_id)
    assert len(latest) == 1
    root_id = latest[0]

    change_log = client.chunkedgraph.get_change_log(root_id)

    merges, splits = extract_operations(root_id)

    meshwork = pcg_skel.coord_space_meshwork(
        root_id,
        client=client,
        # synapses="all",
        # synapse_table=client.materialize.synapse_table,
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    skeleton_plot.plot_tools.plot_mw_skel(
        meshwork,
        ax=ax,
        invert_y=True,
        plot_presyn=False,
        plot_postsyn=False,
        color="black",
        skel_alpha=0.5,
        x="x",
        y="y",
        presyn_size=5,
        postsyn_size=5,
    )

    if len(merges) > 0:
        ax.scatter(merges["x"], merges["y"], color="blue", s=20, marker="o")
    if len(splits) > 0:
        ax.scatter(splits["x"], splits["y"], color="red", s=20, marker="x")

    ax.autoscale()
    ax.axis("off")

    ax.set(title=f"root_id: {root_id}, cell_type: {sub_meta.iloc[i]['cell_type']}")

    plt.savefig(FIG_PATH / f"{root_id}.png", bbox_inches="tight")

# %%
meshwork = pcg_skel.coord_space_meshwork(
    root_id,
    client=client,
    root_point=nuc.loc[target_id]["pt_position"],
    root_point_resolution=res,
)
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
skeleton_plot.plot_tools.plot_mw_skel(
    meshwork,
    ax=ax,
    invert_y=True,
    plot_presyn=False,
    plot_postsyn=False,
    color="black",
    skel_alpha=0.5,
    x="x",
    y="y",
    presyn_size=5,
    postsyn_size=5,
    plot_soma=True,
)

# %%
